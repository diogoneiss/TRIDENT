"""Translation from legacy argparse namespaces to typed training requests."""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

from .types import (
    DEFAULT_TASK,
    LR_SCHEDULER_NAMES,
    SEARCH_SPACE_PROFILES,
    TASK_NAMES,
    DatasetSpec,
    Hyperparameters,
    RuntimeOptions,
    TrainingRequest,
)


def load_hyperparameters(args: argparse.Namespace) -> tuple[Hyperparameters, str]:
    """The configuration a run trains with, and where it came from.

    The source is ``"override"`` (a programmatic mapping, as an Optuna trial passes),
    the repository-relative posix path of the file that was loaded, or ``"defaults"``.
    It is logged on every run as the ``config_source`` param so tuned and default runs
    can be told apart without a new tag (ADR 0005, decision 5).
    """
    hyperparameters, source = _load_base_hyperparameters(args)
    # ``--lr_scheduler`` wins over whatever the override mapping or JSON file said,
    # so one invocation can re-run any stored configuration under another schedule.
    lr_scheduler = getattr(args, "lr_scheduler", None)
    if lr_scheduler is not None:
        hyperparameters = dataclasses.replace(hyperparameters, lr_scheduler=lr_scheduler)
    return hyperparameters, source


def _load_base_hyperparameters(args: argparse.Namespace) -> tuple[Hyperparameters, str]:
    override = getattr(args, "hyperparams_override", None)
    if override is not None:
        return Hyperparameters.from_mapping(override), "override"

    dataset_name = getattr(args, "dataset_name")
    task = getattr(args, "task", None) or DEFAULT_TASK
    # The shared file names no task, so a promoted imputation configuration lives beside
    # it under a task-keyed name and is read first; the fallback keeps a variant with no
    # promoted configuration behaving as before. Classification never reads the
    # task-keyed file, so promoting an imputation result cannot retune it (ADR 0005).
    candidates = [hyperparameter_file(dataset_name, task)]
    if task != DEFAULT_TASK:
        candidates.append(hyperparameter_file(dataset_name, DEFAULT_TASK))
    for hyperparameters_path in candidates:
        if hyperparameters_path.exists():
            values = json.loads(hyperparameters_path.read_text())
            if isinstance(values, Mapping):
                return Hyperparameters.from_mapping(values), hyperparameters_path.as_posix()

    return Hyperparameters(), "defaults"


def hyperparameter_file(dataset_name: str, task: str) -> Path:
    """Where a task's promoted configuration for a dataset variant lives.

    Classification keeps the shared ``<dataset>.json`` every run read before ADR 0005;
    any other task is keyed into the name, ``<dataset>.<task>.json``, beside it.
    """
    base_dataset_name = dataset_name.split("_", 1)[0]
    suffix = ".json" if task == DEFAULT_TASK else f".{task}.json"
    return Path("datasets/hiperparams") / base_dataset_name / f"{dataset_name}{suffix}"


def logged_hyperparameters(request: TrainingRequest) -> dict[str, object]:
    """The parameters this run actually used, under their config-file names."""
    return complete_configuration(request.hyperparameters, request.task)


def complete_configuration(values: Hyperparameters, task: str) -> dict[str, object]:
    """A task's full key set with resolved values, under the config-file names.

    Only the ones the task consumes: an imputation run has no classifier, so the
    fine-tuning rates and the label count would describe nothing, and a classification run
    has no decoder. A parameter recorded but never used misleads whoever reads the run
    later, which is exactly the complaint backlog item C3 makes about ``LABELS``. The
    same key set is what a promoted configuration file holds, so a promoted file names
    every value the task will train with, held ones included.
    """
    shared: dict[str, object] = {
        "DIM": values.dimension, "HIDDEN_DIM": values.hidden_dimension,
        "HEADS": values.heads, "LAYERS": values.layers,
        "DIM_FEED": values.feedforward_dimension, "DROPOUT": values.dropout,
        "EPOCHS_PRE": values.pretraining_epochs, "BATCH": values.batch_size,
        "LR_PRE": values.pretraining_learning_rate,
        "WEIGHT_DECAY_PRE": values.pretraining_weight_decay,
        "PROB_MASCARA": values.mask_probability,
        "LR_SCHEDULER": values.lr_scheduler,
    }
    if task == "imputation":
        return {
            **shared,
            "EPOCHS_DECODE": values.decode_epochs,
            "LR_DECODE": values.decode_learning_rate,
            "WEIGHT_DECAY_DECODE": values.decode_weight_decay,
            "LAMBDA_NUM": values.lambda_num,
            "EVAL_MASK_RATE": values.eval_mask_rate,
        }
    return {
        **shared,
        "EPOCH_FINE": values.finetuning_epochs,
        "LR_FINE": values.finetuning_learning_rate,
        "WEIGHT_DECAY_FINE": values.finetuning_weight_decay,
        "LABELS": values.labels,
    }


def resolve_training_request(args: argparse.Namespace) -> TrainingRequest:
    hyperparameters, config_source = load_hyperparameters(args)
    return TrainingRequest(
        dataset=DatasetSpec.from_name(args.dataset_name, getattr(args, "label_column", None)),
        hyperparameters=hyperparameters,
        runtime=RuntimeOptions(
            output_dir=Path(getattr(args, "output_dir", "results")),
            metrics_dir=Path(getattr(args, "metrics_dir", "metrics")),
            tracking_enabled=not getattr(args, "disable_mlflow", False),
            # Programmatic only (set by opt.py), like ``hyperparams_override``.
            tracking_run_role=getattr(args, "mlflow_run_role", None) or "parent",
            tracking_tags=dict(getattr(args, "mlflow_tags", None) or {}),
        ),
        seed=getattr(args, "seed", 42),
        cv_folds=getattr(args, "cv_folds", None),
        plot_losses=getattr(args, "plot_losses", False),
        save_model=getattr(args, "save_model", False),
        task=getattr(args, "task", None) or DEFAULT_TASK,
        score_null_path=getattr(args, "score_null_path", False),
        # Programmatic only (set by opt.py), like ``hyperparams_override``.
        score_search_objective=getattr(args, "score_search_objective", False),
        config_source=config_source,
    )


def validate_parsed_args(args: argparse.Namespace) -> argparse.Namespace:
    """Post-parse validation for mutually exclusive and dependent flags."""
    run_all = getattr(args, "all", False)
    dataset_name = getattr(args, "dataset_name", None)

    # Must specify exactly one of --all or --dataset_name.
    if not run_all and not dataset_name:
        raise SystemExit("error: one of --all or --dataset_name is required")

    # --limit and --nan_level only make sense with --all.
    if not run_all:
        if getattr(args, "limit", None) is not None:
            raise SystemExit("error: --limit can only be used with --all")
        if getattr(args, "nan_level", 0) != 0:
            raise SystemExit("error: --nan_level can only be used with --all")

    # The null-path diagnostic scores what a decoder reconstructs, and classification has
    # no decoder, so there would be nothing to score. Caught here rather than mid-run.
    if getattr(args, "score_null_path", False) and getattr(args, "task", DEFAULT_TASK) != "imputation":
        raise SystemExit("error: --score_null_path requires --task imputation")

    # One fold is not cross-validation: build_folds derives the validation ratio as
    # 0.1 / (1 - 1/cv_folds), which divides by zero at 1 (backlog B2). Omitting the flag
    # is the single-split mode, so the message points there rather than just refusing.
    cv_folds = getattr(args, "cv_folds", None)
    if cv_folds is not None and cv_folds < 2:
        raise SystemExit(
            "error: --cv_folds must be 2 or more; omit it to use the predefined split"
        )

    # --all is incompatible with --use_optuna.
    if run_all and getattr(args, "use_optuna", False):
        raise SystemExit("error: --all and --use_optuna are mutually exclusive")

    # The reduced profile holds the shared knobs and samples the decode stage; for
    # classification it would sample a space no one has asked to run (ADR 0005). Refused
    # here rather than mid-study, like --score_null_path.
    if (
        getattr(args, "search_space", None) == "reduced"
        and getattr(args, "task", DEFAULT_TASK) != "imputation"
    ):
        raise SystemExit("error: --search_space reduced requires --task imputation")

    return args


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIDENT: Training and evaluation of the model")

    # Dataset selection: --all and --dataset_name are mutually exclusive.
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset_name", type=str, default=None)
    dataset_group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run training on all available datasets.",
    )

    # Companion flags for --all mode.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When used with --all, run only the first N datasets (alphabetical).",
    )
    parser.add_argument(
        "--nan_level",
        type=int,
        default=0,
        choices=[0, 20, 40, 60, 80],
        help="When used with --all, select the missingness level (default: 0).",
    )

    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    parser.add_argument("--disable_mlflow", action="store_true")
    parser.add_argument("--plot_losses", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=None)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=LR_SCHEDULER_NAMES,
        help=(
            "Learning-rate schedule for both training stages. Overrides LR_SCHEDULER from the "
            "hyperparameter file. Default: cosine_legacy (the schedule of every run before ADR 0003)."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=TASK_NAMES,
        help=(
            "What to train. 'classification' predicts the label from the [CLS] token and is "
            "what every run did before ADR 0004. 'imputation' replaces the classifier with a "
            "decoder that reconstructs hidden cell values. Command line only: a per-dataset "
            "config choosing the task would make --all train different tasks per dataset."
        ),
    )
    parser.add_argument(
        "--score_null_path",
        action="store_true",
        help=(
            "Also score induced-missing cells with the model seeing [NULL] rather than "
            "[MASK], as a diagnostic. Requires --task imputation. Never ranks folds."
        ),
    )
    parser.add_argument("--use_optuna", action="store_true")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--retrain_best", action="store_true")
    parser.add_argument(
        "--promote_best",
        action="store_true",
        help=(
            "After an Optuna study, publish the winning configuration where the task's "
            "runs read it: datasets/hiperparams/<base>/<dataset>.json for classification, "
            "<dataset>.imputation.json for imputation (ADR 0005). Without it a study "
            "writes only inside its own results directory. Independent of --retrain_best."
        ),
    )
    parser.add_argument(
        "--search_space",
        type=str,
        default=None,
        choices=SEARCH_SPACE_PROFILES,
        help=(
            "Which hyperparameters an Optuna study samples (ADR 0005). 'full' samples every "
            "knob the task uses; 'reduced' samples only the decode-stage knobs, the mask "
            "rate and dropout, holding the rest at the task default. Unspecified resolves "
            "to 'reduced' for --task imputation and 'full' for classification; 'reduced' is "
            "defined only for the imputation task."
        ),
    )
    return parser
