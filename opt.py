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

from src.mlflow_utils import LR_SCHEDULER_TAG, setup_mlflow, get_or_create_experiment, build_run_tags
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


def define_search_space(trial):
    """
    Define the hyperparameter search space for Optuna
    """
    params = {
        'DIM': trial.suggest_int('DIM', 64, 256, step=32),
        'HIDDEN_DIM': trial.suggest_int('HIDDEN_DIM', 8, 64, step=8),
        'HEADS': trial.suggest_int('HEADS', 4, 16, step=4),
        'LAYERS': trial.suggest_int('LAYERS', 1, 6, step=1),
        'DIM_FEED': trial.suggest_int('DIM_FEED', 16, 128, step=16),
        'DROPOUT': trial.suggest_float('DROPOUT', 0.1, 0.5, step=0.1),
        'EPOCHS_PRE': trial.suggest_int('EPOCHS_PRE', 20, 60, step=10),
        'BATCH': trial.suggest_categorical('BATCH', [64, 128, 256, 512]),
        'LR_PRE': trial.suggest_float('LR_PRE', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY_PRE': trial.suggest_float('WEIGHT_DECAY_PRE', 1e-5, 1e-2, log=True),
        'PROB_MASCARA': trial.suggest_float('PROB_MASCARA', 0.2, 0.6, step=0.1),
        'EPOCH_FINE': trial.suggest_int('EPOCH_FINE', 20, 60, step=10),
        'LR_FINE': trial.suggest_float('LR_FINE', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY_FINE': trial.suggest_float('WEIGHT_DECAY_FINE', 1e-5, 1e-2, log=True),
    }
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
        self.best_score = 0
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
        params = define_search_space(trial)
        
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
        }

        with mlflow.start_run(
            experiment_id=self.mlflow_experiment_id,
            run_name=trial_run_name,
            tags=trial_tags,
            nested=True,
        ) as trial_run:
            self.trial_run_ids[trial.number] = trial_run.info.run_id
            try:
                # Run the training with these hyperparameters
                metrics = train_main(args, return_metrics=True)

                # Get the validation and test scores
                f1_macro = metrics['f1_macro']

                # The objective under one key whatever the evaluation mode;
                # the full metric set is on the run as cv/test/* or test/*.
                mlflow.log_metric("optuna/objective_value", f1_macro)
                mlflow.set_tag("trial_status", "success")

                # Report intermediate values
                trial.report(f1_macro, step=0)

                # Check if this is the best score so far
                if f1_macro > self.best_score:
                    self.best_score = f1_macro
                    self.best_params = params
                    self.best_trial_number = trial.number
                    self.best_metrics = metrics

                    # Save the best parameters found so far to datasets/hiperparams
                    self.save_best_params()

                return f1_macro

            except Exception as e:
                logger.error(f"Trial {trial.number} failed with error: {str(e)}")
                mlflow.set_tag("trial_status", "failed")
                mlflow.set_tag("error", str(e)[:250])  # tag truncated to 250 chars
                return 0.0  # Return worst possible score on failure

    def save_best_params(self):
        """
        Save the best parameters found so far to the correct location
        """
        # Create directory structure if it doesn't exist
        save_dir = Path('datasets/hiperparams') / self.base_dataset_name
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the hyperparameters
        save_path = save_dir / f"{self.dataset_name}.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        logger.info(f"Saved best hyperparameters (F1={self.best_score:.4f}) to {save_path}")


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
    
    # Create an Optuna study
    storage_name = f"sqlite:///{optuna_dir}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=f"TRIDENT_{args.dataset_name}",
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    # Open the MLflow parent run for this Optuna study
    parent_run_name = f"optuna_{args.dataset_name}_{timestamp}"
    parent_tags = build_run_tags(
        dataset_name=args.dataset_name,
        run_type="optuna_study",
        seed=args.seed,
    )
    parent_tags["n_trials"] = str(args.n_trials)
    parent_tags["run_role"] = "optuna_study"
    parent_tags[LR_SCHEDULER_TAG] = getattr(args, "lr_scheduler", None) or DEFAULT_LR_SCHEDULER

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
        })

        # Give the objective access to the parent run so it can open child runs
        objective.mlflow_parent_run_id = parent_run.info.run_id
        objective.mlflow_experiment_id = experiment_id

        # Run the optimization
        study.optimize(objective, n_trials=args.n_trials)

        # Log best trial summary to the parent run
        mlflow.log_metrics({
            "optuna/best_objective_value": study.best_value,
            "optuna/best_trial_number": float(study.best_trial.number),
        })
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
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
    logger.info(f"Best F1 macro: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*50)
    
    # Save all Optuna results in the optuna_dir
    # 1. Save best hyperparameters
    best_params_path = optuna_dir / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best hyperparameters saved to {best_params_path}")
    
    # 2. Also save to the standard hiperparams directory for model loading
    base_dataset_name = args.dataset_name.split('_')[0]
    hiperparams_dir = Path('datasets/hiperparams') / base_dataset_name
    hiperparams_dir.mkdir(exist_ok=True, parents=True)
    hiperparams_path = hiperparams_dir / f"{args.dataset_name}.json"
    with open(hiperparams_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
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
        final_args.hyperparams_override = study.best_params
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRIDENT: Hyperparameter optimization with Optuna')
    
    # Required parameters
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (without the .csv extension)')
    
    # Optional parameters
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials to run (default: 50)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for Optuna results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generation')
    parser.add_argument('--retrain_best', action='store_true',
                        help='Retrain the model with the best parameters after optimization')
    
    args = parser.parse_args()
    run_hyperparameter_optimization(args)
