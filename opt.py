import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import optuna
import argparse
from pathlib import Path
import datetime
import logging
from functools import partial
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from contextlib import nullcontext

from src.mlflow_utils import (
    IS_OPTUNA_TAG,
    LR_SCHEDULER_TAG,
    SEARCH_SPACE_TAG,
    TASK_TAG,
    setup_mlflow,
    get_or_create_experiment,
)
from src.training.config import build_training_parser, validate_parsed_args
from src.training.data import PROCESSED_DATASETS, declared_column_types
from src.training.tracking import execution_tags
from src.training.types import task_spec
from src.training.types import DEFAULT_LR_SCHEDULER

# Import components from train.py
from train import main as train_main
from src.utils import set_global_seed

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class _DisabledMlflow:
    """Drop-in no-op used when an Optuna invocation disables tracking."""

    class _Run:
        class info:
            run_id = "disabled"

    def start_run(self, *args, **kwargs):
        return nullcontext(self._Run())

    def log_params(self, *args, **kwargs):
        return None

    def log_param(self, *args, **kwargs):
        return None

    def log_metric(self, *args, **kwargs):
        return None

    def log_metrics(self, *args, **kwargs):
        return None

    def set_tag(self, *args, **kwargs):
        return None


def define_search_space(
    trial, task: str = 'classification', profile: str = 'full', mixed_columns: bool = True
):
    """
    Define the hyperparameter search space for Optuna.

    ``profile`` is the search-space profile (ADR 0005): ``full`` samples every knob the
    task uses; ``reduced`` samples only the knobs that govern the task's own stage and the
    corruption it learns from. A key the profile does not return is *held*: the trial runs
    it at the task default, because the override replaces the base configuration.
    ``mixed_columns`` says whether the table has both column types, which is the only case
    in which the loss balance ``LAMBDA_NUM`` does anything.
    """
    if profile == 'reduced':
        params = {
            # Corruption rate in both stages: the decode stage re-rolls masks at this rate
            # every epoch, so the decoder learns from exactly these cells.
            'PROB_MASCARA': trial.suggest_float('PROB_MASCARA', 0.2, 0.6, step=0.1),
            # The stage that produces the output. The range reaches above the default
            # 1e-3, which the old 1e-5..1e-3 range had on its upper bound.
            'LR_DECODE': trial.suggest_float('LR_DECODE', 1e-4, 1e-2, log=True),
            'WEIGHT_DECAY_DECODE': trial.suggest_float('WEIGHT_DECAY_DECODE', 1e-5, 1e-2, log=True),
            # The one regulariser acting on both stages.
            'DROPOUT': trial.suggest_float('DROPOUT', 0.1, 0.5, step=0.1),
        }
        if mixed_columns:
            params['LAMBDA_NUM'] = trial.suggest_float('LAMBDA_NUM', 0.1, 10.0, log=True)
        return params

    # Attention splits the width across heads, so the width must divide evenly. Sampling
    # the two independently used to produce combinations that crash, which were then
    # scored as merely terrible hyperparameters (backlog B3).
    # What Optuna records is the per-head width, named as such; the model width the trial
    # trains with is derived below and is what the returned configuration carries.
    heads = trial.suggest_categorical('HEADS', [4, 8, 16])
    dim = trial.suggest_int('HEAD_DIM', 64 // heads, 256 // heads) * heads
    params = {
        'DIM': dim,
        'HEADS': heads,
        'HIDDEN_DIM': trial.suggest_int('HIDDEN_DIM', 8, 64, step=8),
        'LAYERS': trial.suggest_int('LAYERS', 1, 6, step=1),
        'DIM_FEED': trial.suggest_int('DIM_FEED', 16, 128, step=16),
        'DROPOUT': trial.suggest_float('DROPOUT', 0.1, 0.5, step=0.1),
        'EPOCHS_PRE': trial.suggest_int('EPOCHS_PRE', 20, 60, step=10),
        'BATCH': trial.suggest_categorical('BATCH', [64, 128, 256, 512]),
        'LR_PRE': trial.suggest_float('LR_PRE', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY_PRE': trial.suggest_float('WEIGHT_DECAY_PRE', 1e-5, 1e-2, log=True),
        # Governs training corruption only; the evaluation rate is deliberately absent,
        # since a trial free to hide fewer cells would win by making its own exam easier.
        'PROB_MASCARA': trial.suggest_float('PROB_MASCARA', 0.2, 0.6, step=0.1),
    }
    if task == 'imputation':
        params.update({
            'EPOCHS_DECODE': trial.suggest_int('EPOCHS_DECODE', 20, 60, step=10),
            # Reaches above the default 1e-3, which the old 1e-5..1e-3 range had on its
            # upper bound (ADR 0005, decision 2: a correction, so it applies here too).
            'LR_DECODE': trial.suggest_float('LR_DECODE', 1e-4, 1e-2, log=True),
            'WEIGHT_DECAY_DECODE': trial.suggest_float('WEIGHT_DECAY_DECODE', 1e-5, 1e-2, log=True),
        })
        # The loss balance weighs the numerical term against the categorical one, so on
        # a single-type table it is a pure scale and would only waste a dimension.
        if mixed_columns:
            params['LAMBDA_NUM'] = trial.suggest_float('LAMBDA_NUM', 0.1, 10.0, log=True)
    else:
        params.update({
            'EPOCH_FINE': trial.suggest_int('EPOCH_FINE', 20, 60, step=10),
            'LR_FINE': trial.suggest_float('LR_FINE', 1e-5, 1e-3, log=True),
            'WEIGHT_DECAY_FINE': trial.suggest_float('WEIGHT_DECAY_FINE', 1e-5, 1e-2, log=True),
        })
    return params


class ObjectiveFunctionWrapper:
    """
    Wrapper class for the Optuna objective function to maintain state
    """
    def __init__(self, dataset_name, seed=42, output_dir=".", optuna_dir=None):
        self.dataset_name = dataset_name
        self.seed = seed
        self.output_dir = output_dir
        self.optuna_dir = optuna_dir  # Directory to store all Optuna results
        self.base_dataset_name = dataset_name.split('_')[0]
        # Which task the study is optimising, and the metric and direction that go with
        # it. Set by run_hyperparameter_optimization before the study starts.
        self.task = 'classification'
        self.ranking_metric = 'f1_macro'
        self.direction = 'maximize'
        # Which knobs a trial samples, and whether the table has both column types (the
        # only case in which the loss balance is worth a dimension). Set by
        # run_hyperparameter_optimization once per study, never per trial.
        self.profile = 'full'
        self.mixed_columns = True
        self.best_score = float('-inf')
        self.best_params = None
        self.best_trial_number = None
        self.best_metrics = None
        # MLflow parent run ID — set by run_hyperparameter_optimization before calling study.optimize
        self.mlflow_parent_run_id: str | None = None
        self.mlflow_experiment_id: str | None = None
        self.lr_scheduler: str | None = None
        # trial number -> MLflow run id, filled as trials run.
        self.trial_run_ids: dict[int, str] = {}

    def __call__(self, trial):
        # Define hyperparameters for this trial
        params = define_search_space(
            trial, task=self.task, profile=self.profile, mixed_columns=self.mixed_columns
        )
        # Optuna's own record of the trial holds what was sampled (head count, per-head
        # width), not the configuration trained with. Keep that on the trial so the study
        # can hand on exactly what its winner ran, even after a resume.
        trial.set_user_attr("hyperparameters", params)

        # Create Args object to pass to train_main
        class Args:
            pass
        
        args = Args()
        args.dataset_name = self.dataset_name
        args.label_column = 'class'  # Default value
        
        # Use a temporary directory for trials - we will only save the best one
        temp_dir = Path(self.output_dir) / self.dataset_name / "temp_trials"
        args.output_dir = str(temp_dir)
        args.metrics_dir = getattr(self, "metrics_dir", "metrics")
        args.disable_mlflow = getattr(self, "disable_mlflow", False)
        
        args.plot_losses = False
        args.save_model = False
        args.seed = self.seed
        # This namespace is built here, not parsed, so the study's task has to be copied
        # on. Without it the runner falls back to classification, trains the wrong stage
        # and never produces the imputation ranking metric the objective then asks for.
        args.task = self.task
        args.hyperparams_override = params  # Add custom field for hyperparams
        # The schedule is not part of the search space; every trial uses the one
        # chosen on the command line (or the default) so trials stay comparable.
        args.lr_scheduler = self.lr_scheduler
        # The runner logs into this trial's run instead of opening a second
        # top-level run, which MLflow refuses while the trial run is active.
        args.mlflow_run_role = "optuna_trial"

        # Create temporary directory
        os.makedirs(args.output_dir, exist_ok=True)

        # One nested run per trial under the study run. The runner adds the
        # structured dataset/schedule/environment tags, the hyperparameters and
        # the final metrics; the tags below survive even if training never
        # reaches the runner.
        trial_run_name = f"optuna_trial_{trial.number}"
        trial_tags = {
            "trial_number": str(trial.number),
            "dataset": self.dataset_name,
            "run_type": "optuna_trial",
            "run_role": "optuna_trial",
            LR_SCHEDULER_TAG: self.lr_scheduler or DEFAULT_LR_SCHEDULER,
            # Both tags are dense on every run kind; a trial that crashes before the
            # runner sets its own would otherwise be a run of no known task.
            TASK_TAG: self.task,
            IS_OPTUNA_TAG: "true",
            SEARCH_SPACE_TAG: self.profile,
        }

        with mlflow.start_run(
            experiment_id=self.mlflow_experiment_id,
            run_name=trial_run_name,
            tags=trial_tags,
            nested=True,
        ) as trial_run:
            self.trial_run_ids[trial.number] = trial_run.info.run_id
            # A param as well as a tag, so the profile shows in the trial's parameter
            # table beside the knobs it explains.
            mlflow.log_param(SEARCH_SPACE_TAG, self.profile)
            try:
                # Run the training with these hyperparameters
                metrics = train_main(args, return_metrics=True)

                # Get the validation and test scores
                score = metrics[self.ranking_metric]

                # The objective under one key whatever the evaluation mode;
                # the full metric set is on the run as cv/test/* or test/*.
                mlflow.log_metric("optuna/objective_value", score)
                mlflow.set_tag("trial_status", "success")

                # Report intermediate values
                trial.report(score, step=0)

                # Check if this is the best score so far
                # Lower is better for an error ratio, so the comparison follows the
                # task rather than assuming bigger wins.
                improved = (
                    score > self.best_score
                    if self.direction == 'maximize'
                    else score < self.best_score
                )
                if improved:
                    self.best_score = score
                    self.best_params = params
                    self.best_trial_number = trial.number
                    self.best_metrics = metrics

                    # Save the best parameters found so far to datasets/hiperparams
                    self.save_best_params()

                return score

            except Exception as e:
                logger.error(f"Trial {trial.number} failed with error: {str(e)}")
                mlflow.set_tag("trial_status", "failed")
                mlflow.set_tag("error", str(e)[:250])  # tag truncated to 250 chars
                # Never score a failure. Returning 0.0 was "the worst possible score"
                # only while maximising macro F1; under a minimised error ratio it is the
                # best possible one, so the search would have hunted for crashes.
                raise optuna.TrialPruned() from e

    def save_best_params(self):
        """Save the best parameters found so far, somewhere the other task cannot feel.

        ``datasets/hiperparams/<base>/<dataset>.json`` is read by **both** tasks and names
        neither, so an imputation study writing there would silently retune every later
        classification run on that dataset. An imputation study therefore keeps its result
        inside its own study directory; promoting it is a deliberate copy.
        """
        if self.task == 'imputation':
            save_dir = Path(self.optuna_dir) if self.optuna_dir else Path('.')
        else:
            save_dir = Path('datasets/hiperparams') / self.base_dataset_name
        save_dir.mkdir(exist_ok=True, parents=True)

        save_path = save_dir / f"{self.dataset_name}.json"
        with open(save_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)

        logger.info(
            f"Saved best hyperparameters ({self.ranking_metric}={self.best_score:.4f}) "
            f"to {save_path}"
        )
        return save_path


def run_hyperparameter_optimization(args):
    """
    Run hyperparameter optimization with Optuna
    """
    logger.info(f"Starting hyperparameter optimization for {args.dataset_name}")
    logger.info(f"Number of trials: {args.n_trials}")
    
    # Configure MLflow unless the runtime has explicitly disabled every API call.
    global mlflow
    if getattr(args, "disable_mlflow", False):
        mlflow = _DisabledMlflow()
        experiment_id = "disabled"
    else:
        setup_mlflow()  # Honours MLFLOW_TRACKING_URI env var, falls back to sqlite:///mlflow.db
        experiment_id = get_or_create_experiment(args.dataset_name)

    # Create directory structure for Optuna results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    optuna_dir = Path(args.output_dir) / args.dataset_name / f"optuna_{timestamp}"
    optuna_dir.mkdir(exist_ok=True, parents=True)
    
    # Create the objective function
    objective = ObjectiveFunctionWrapper(
        dataset_name=args.dataset_name,
        seed=args.seed,
        output_dir=args.output_dir,
        optuna_dir=optuna_dir
    )
    objective.metrics_dir = getattr(args, "metrics_dir", "metrics")
    objective.disable_mlflow = getattr(args, "disable_mlflow", False)
    objective.lr_scheduler = getattr(args, "lr_scheduler", None)
    task = task_spec(getattr(args, "task", None) or "classification")
    objective.task = task.name
    objective.ranking_metric = task.ranking_metric
    objective.direction = task.direction
    objective.best_score = float("-inf") if task.direction == "maximize" else float("inf")
    # Unspecified, the profile follows the task: only imputation defines ``reduced`` so far.
    profile = getattr(args, "search_space", None) or (
        "reduced" if task.name == "imputation" else "full"
    )
    objective.profile = profile
    # Whether the table has both column types decides whether the loss balance is
    # sampled. Read once from the table's header and the pipeline's declaration, the way
    # the loader does it, never from dtypes (electricity's integer-coded ``day`` is
    # declared categorical) and never per trial.
    base_dataset_name = args.dataset_name.split('_')[0]
    header = pd.read_csv(
        PROCESSED_DATASETS / base_dataset_name / f"{args.dataset_name}.csv", nrows=0
    ).columns.tolist()
    categorical_columns, numerical_columns = declared_column_types(
        header, getattr(args, "label_column", None) or "class", base_dataset_name
    )
    objective.mixed_columns = bool(categorical_columns) and bool(numerical_columns)
    logger.info(
        f"Search-space profile: {profile}; table has both column types: "
        f"{objective.mixed_columns}"
    )
    
    # Create an Optuna study
    storage_name = f"sqlite:///{optuna_dir}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=f"TRIDENT_{args.dataset_name}",
        storage=storage_name,
        direction=task.direction,
        load_if_exists=True,
        # The seed that fixes every trial's training fixes the sampler too, so a study
        # can be rerun; an unseeded sampler made each study a one-off.
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    
    # Open the MLflow parent run for this Optuna study
    parent_run_name = f"optuna_{args.dataset_name}_{timestamp}"
    # The same helper every other run kind uses, so the study parent carries the dense
    # ``task`` and ``is_optuna`` tags a filter relies on (ADR 0004, decision 7).
    parent_tags = execution_tags(
        dataset_name=args.dataset_name,
        run_role="optuna_study",
        run_type="optuna_study",
        seed=args.seed,
        cv_folds=None,
        lr_scheduler=getattr(args, "lr_scheduler", None) or DEFAULT_LR_SCHEDULER,
        environment=None,
        extra_tags={"n_trials": str(args.n_trials), SEARCH_SPACE_TAG: profile},
        task=task.name,
        is_optuna=True,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
        tags=parent_tags,
    ) as parent_run:
        mlflow.log_params({
            "n_trials": args.n_trials,
            "dataset_name": args.dataset_name,
            "seed": args.seed,
            "optuna_storage": storage_name,
            SEARCH_SPACE_TAG: profile,
        })

        # Give the objective access to the parent run so it can open child runs
        objective.mlflow_parent_run_id = parent_run.info.run_id
        objective.mlflow_experiment_id = experiment_id

        # Run the optimization
        study.optimize(objective, n_trials=args.n_trials)

        # The configuration the winning trial actually trained with, not Optuna's record
        # of what it sampled (the study's own best params hold ``HEAD_DIM``, not ``DIM``).
        best_config = study.best_trial.user_attrs["hyperparameters"]

        # Log best trial summary to the parent run
        mlflow.log_metrics({
            "optuna/best_objective_value": study.best_value,
            "optuna/best_trial_number": float(study.best_trial.number),
        })
        mlflow.log_params({f"best_{k}": v for k, v in best_config.items()})
        mlflow.set_tag("best_trial_number", str(study.best_trial.number))

        # ---- end of parent MLflow run ----

    study_run_id = parent_run.info.run_id
    # Tag the winning trial once the study is complete, so the tag can never
    # go stale the way a running "best so far" marker would.
    best_trial_run_id = objective.trial_run_ids.get(study.best_trial.number)
    if best_trial_run_id is not None and not getattr(args, "disable_mlflow", False):
        MlflowClient().set_tag(best_trial_run_id, "best_trial", "true")
    
    # Report best parameters
    logger.info("\n\n" + "="*50)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {task.ranking_metric}: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in best_config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*50)
    
    # Save all Optuna results in the optuna_dir
    # 1. Save best hyperparameters
    best_params_path = optuna_dir / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    logger.info(f"Best hyperparameters saved to {best_params_path}")
    
    # 2. Also save to the standard hiperparams directory for model loading. That file is
    # read by both tasks and names neither, so an imputation study stays inside its own
    # study directory (same rule as ``save_best_params``); promoting it is a deliberate
    # copy until a task-keyed lookup exists.
    if task.name == 'imputation':
        logger.info(
            "Imputation study: not writing datasets/hiperparams, which classification reads"
        )
    else:
        base_dataset_name = args.dataset_name.split('_')[0]
        hiperparams_dir = Path('datasets/hiperparams') / base_dataset_name
        hiperparams_dir.mkdir(exist_ok=True, parents=True)
        hiperparams_path = hiperparams_dir / f"{args.dataset_name}.json"
        with open(hiperparams_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        logger.info(f"Saved for model loading: {hiperparams_path}")
    
    # Retrain the model with the best parameters and save results in optuna_dir
    if args.retrain_best:
        logger.info("\nRetraining with best parameters...")
        
        # Create Args object for retraining
        class Args:
            pass
        
        final_args = Args()
        final_args.dataset_name = args.dataset_name
        final_args.label_column = 'class'  # Default value
        
        # Save the final model in the optuna directory
        final_dir = optuna_dir / "best_model"
        final_dir.mkdir(exist_ok=True)
        final_args.output_dir = str(final_dir)
        final_args.metrics_dir = getattr(args, "metrics_dir", "metrics")
        final_args.disable_mlflow = getattr(args, "disable_mlflow", False)
        
        # Include plots when retraining
        final_args.plot_losses = True
        final_args.save_model = True
        final_args.seed = args.seed
        # Same hand-built namespace as a trial: the task travels with it.
        final_args.task = task.name
        final_args.hyperparams_override = best_config
        final_args.lr_scheduler = getattr(args, "lr_scheduler", None)
        # A normal comparable parent run, linked back to the study that chose it.
        final_args.mlflow_tags = {"optuna_study_run_id": study_run_id}
        
        # Run final training
        metrics = train_main(final_args, return_metrics=True)
        
        # Also save metrics directly in the optuna directory
        metrics_path = optuna_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Best model metrics saved to {metrics_path}")
        
        logger.info("Retraining completed!")
    
    # Clean up temporary directories
    temp_dir = Path(args.output_dir) / args.dataset_name / "temp_trials"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary directory: {str(e)}")
    
    logger.info(f"All Optuna results saved in: {optuna_dir}")


def build_parser() -> argparse.ArgumentParser:
    """The training parser, so a study launched here accepts every flag main.py accepts.

    opt.py used to keep a private parser that lacked --task, --lr_scheduler,
    --disable_mlflow and --metrics_dir, so a study launched through it could not be an
    imputation study at all. One parser for both entry points cannot drift (ADR 0005).
    """
    parser = build_training_parser()
    parser.description = "TRIDENT: hyperparameter optimization with Optuna"
    return parser


if __name__ == "__main__":
    arguments = validate_parsed_args(build_parser().parse_args())
    if getattr(arguments, "all", False):
        raise SystemExit("error: opt.py runs one dataset; pass --dataset_name")
    run_hyperparameter_optimization(arguments)
