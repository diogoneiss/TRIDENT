"""Translation from legacy argparse namespaces to typed training requests."""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

from .types import (
    DEFAULT_BASELINE_CACHE_DIR,
    DECODER_HEADS,
    DEFAULT_TASK,
    LR_SCHEDULER_NAMES,
    PRETRAINING_OBJECTIVES,
    SEARCH_OBJECTIVE_POPULATIONS,
    SEARCH_SPACE_PROFILES,
    TASK_NAMES,
    DatasetSpec,
    Hyperparameters,
    RuntimeOptions,
    TrainingRequest,
    task_defaults,
)


# The command-line flags that override a configuration's values in ``load_hyperparameters``.
# A study builds each trial's namespace itself, so it copies these on, or a flag given to the
# study would silently not reach its trials.
CONFIGURATION_FLAGS: tuple[str, ...] = (
    "lr_scheduler", "decode_patience", "decoder_heads", "pretrain_objective", "decode_epochs",
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
    # ``--decode_patience`` wins the same way.
    decode_patience = getattr(args, "decode_patience", None)
    if decode_patience is not None:
        hyperparameters = dataclasses.replace(hyperparameters, decode_patience=decode_patience)
    # ``--decoder_heads`` wins the same way.
    decoder_heads = getattr(args, "decoder_heads", None)
    if decoder_heads is not None:
        hyperparameters = dataclasses.replace(hyperparameters, decoder_heads=decoder_heads)
    # ``--pretrain_objective`` wins the same way (ADR 0011).
    pretraining_objective = getattr(args, "pretrain_objective", None)
    if pretraining_objective is not None:
        hyperparameters = dataclasses.replace(
            hyperparameters, pretraining_objective=pretraining_objective
        )
    # ``--decode_epochs`` wins the same way (ADR 0013).
    decode_epochs = getattr(args, "decode_epochs", None)
    if decode_epochs is not None:
        hyperparameters = dataclasses.replace(hyperparameters, decode_epochs=decode_epochs)
    return hyperparameters, source


def _load_base_hyperparameters(args: argparse.Namespace) -> tuple[Hyperparameters, str]:
    task = getattr(args, "task", None) or DEFAULT_TASK
    # Beneath every source, so a key a source leaves out takes the task's value wherever the
    # run came from: a plain run, a promoted file, or an Optuna trial holding a knob (ADR 0013).
    defaults = task_defaults(task)
    override = getattr(args, "hyperparams_override", None)
    if override is not None:
        return Hyperparameters.from_mapping({**defaults, **override}), "override"

    dataset_name = getattr(args, "dataset_name")
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
                return (
                    Hyperparameters.from_mapping({**defaults, **values}),
                    hyperparameters_path.as_posix(),
                )

    return Hyperparameters.from_mapping(defaults), "defaults"


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
            "PRETRAIN_OBJECTIVE": values.pretraining_objective,
            "EPOCHS_DECODE": values.decode_epochs,
            "LR_DECODE": values.decode_learning_rate,
            "WEIGHT_DECAY_DECODE": values.decode_weight_decay,
            "LAMBDA_NUM": values.lambda_num,
            "EVAL_MASK_RATE": values.eval_mask_rate,
            "DECODE_PATIENCE": values.decode_patience,
            "DECODER_HEADS": values.decoder_heads,
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
            mirror_runs=not (
                getattr(args, "disable_mlflow", False) or getattr(args, "disable_mirror", False)
            ),
            # Programmatic only (set by opt.py), like ``hyperparams_override``.
            tracking_run_role=getattr(args, "mlflow_run_role", None) or "parent",
            tracking_tags=dict(getattr(args, "mlflow_tags", None) or {}),
            baseline_cache_dir=(
                None if getattr(args, "no_baseline_cache", False) else DEFAULT_BASELINE_CACHE_DIR
            ),
        ),
        seed=getattr(args, "seed", 42),
        cv_folds=getattr(args, "cv_folds", None),
        plot_losses=getattr(args, "plot_losses", False),
        save_model=getattr(args, "save_model", False),
        task=getattr(args, "task", None) or DEFAULT_TASK,
        score_null_path=getattr(args, "score_null_path", False),
        score_column_wise=getattr(args, "score_column_wise", False),
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
    if getattr(args, "score_column_wise", False) and getattr(args, "task", DEFAULT_TASK) != "imputation":
        raise SystemExit("error: --score_column_wise requires --task imputation")

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

    # Only imputation has a decode stage to stop early.
    if (
        getattr(args, "decode_patience", None) is not None
        and getattr(args, "task", DEFAULT_TASK) != "imputation"
    ):
        raise SystemExit("error: --decode_patience requires --task imputation")

    # Only imputation has a decode stage to size.
    if (
        getattr(args, "decode_epochs", None) is not None
        and getattr(args, "task", DEFAULT_TASK) != "imputation"
    ):
        raise SystemExit("error: --decode_epochs requires --task imputation")

    # Only imputation has validation populations to choose between; classification ranks
    # by macro F1 (ADR 0008). Refused here rather than mid-study.
    if (
        getattr(args, "search_objective", None) is not None
        and getattr(args, "task", DEFAULT_TASK) != "imputation"
    ):
        raise SystemExit("error: --search_objective requires --task imputation")

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
    parser.add_argument(
        "--disable_mirror",
        action="store_true",
        help="Do not mirror the finished run into TRIDENT/mirror/<task> (ADR 0006).",
    )
    parser.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help=(
            "Compute the baseline imputers' scores every time instead of reusing the ones "
            "cached under results/baseline_cache/ for the same fold and cells."
        ),
    )
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
            "hyperparameter file. Default: cosine_legacy (the schedule of every run before ADR 0003) "
            "for classification, cosine for imputation (ADR 0013)."
        ),
    )
    parser.add_argument(
        "--decode_patience",
        type=int,
        default=None,
        help=(
            "Imputation only: stop the decode stage once its validation loss has not improved "
            "for this many epochs, keeping the best epoch. Overrides DECODE_PATIENCE from the "
            "hyperparameter file. Default: 0, every epoch trains."
        ),
    )
    parser.add_argument(
        "--decode_epochs",
        type=int,
        default=None,
        help=(
            "Imputation only: how many epochs the decode stage trains. Overrides EPOCHS_DECODE "
            "from the hyperparameter file. Default: 450 since ADR 0013; 150 is the decode "
            "stage of every run before it."
        ),
    )
    parser.add_argument(
        "--decoder_heads",
        type=str,
        default=None,
        choices=DECODER_HEADS,
        help=(
            "How the decoder applies its per-column heads: 'batched' (the same arithmetic in a "
            "few operations, faster; the default since 2026-10-01, ADR 0012) or 'per_column' "
            "(one operation per column, which reproduces every earlier run to the last digit). "
            "Overrides DECODER_HEADS. Default: batched."
        ),
    )
    parser.add_argument(
        "--pretrain_objective",
        type=str,
        default=None,
        choices=PRETRAINING_OBJECTIVES,
        help=(
            "What pre-training reconstructs at a masked cell (ADR 0011). Overrides "
            "PRETRAIN_OBJECTIVE from the hyperparameter file. Default: embedding (the cell's "
            "clean embedding, as every run before ADR 0011) for classification, "
            "embedding_normalized for imputation (ADR 0013)."
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
    parser.add_argument(
        "--score_column_wise",
        action="store_true",
        help=(
            "Also score induced-missing cells one gap column at a time: that column's gaps "
            "as [MASK], every other gap as [NULL], the row shape the decode stage trains on. "
            "A diagnostic (task T02). Requires --task imputation. Never ranks folds."
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
    parser.add_argument(
        "--search_objective",
        type=str,
        default=None,
        choices=SEARCH_OBJECTIVE_POPULATIONS,
        help=(
            "Which validation population an imputation study ranks its trials by (ADR 0008): "
            "'masked', the cells hidden on the validation split, which every study before "
            "ADR 0008 used; or 'induced', the validation rows' own gaps scored against the "
            "complete _00nan sibling, the population the headline test score is on. "
            "Unspecified means 'masked'. 'induced' needs a variant with gaps."
        ),
    )
    return parser
