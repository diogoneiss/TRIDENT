"""Immutable values exchanged between the training workflow stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeAlias

# Learning-rate schedule names accepted by ``Hyperparameters.lr_scheduler`` and
# the ``--lr_scheduler`` flag. ``cosine_legacy`` is the schedule every run before
# ADR 0003 used, so it stays the default to keep old configs reproducible.
LR_SCHEDULER_NAMES: tuple[str, ...] = (
    "cosine_legacy",
    "cosine",
    "warmup_cosine",
    "constant",
    "plateau",
)
DEFAULT_LR_SCHEDULER = "cosine_legacy"

# What pre-training regresses a masked cell onto (ADR 0011), accepted by
# ``Hyperparameters.pretraining_objective`` and the ``--pretrain_objective`` flag.
# ``embedding`` is the cell's detached clean embedding, what every run before that decision
# trained on, so it stays the default. ``value`` is the cell's value itself, through fresh
# decoder heads that the stage discards; ``embedding_normalized`` keeps the embedding target
# but layer-normalises it and reads the transformer through a predictor head.
PRETRAINING_OBJECTIVES: tuple[str, ...] = ("embedding", "value", "embedding_normalized")
DEFAULT_PRETRAINING_OBJECTIVE = "embedding"

# Where runs keep the baseline imputers' cached scores unless told otherwise, relative to the
# checkout root every run starts from (ADR 0009). Never under a run's own output directory:
# the study runners give every cell its own, and a cache there would never be shared.
DEFAULT_BASELINE_CACHE_DIR = Path("results") / "baseline_cache"

# Tasks a training request can run (ADR 0004). ``classification`` is every run before
# that decision, so it stays the default; ``imputation`` trains the decode stage in
# place of the classifier. Each task declares the fold-ranking metric and the
# direction in which a fold is best: the pair travels together because best/worst
# selection inverts without the direction.
TASK_NAMES: tuple[str, ...] = ("classification", "imputation")
DEFAULT_TASK = "classification"

# Search-space profiles for a hyper-parameter search (ADR 0005). ``full`` samples every
# knob the task uses; ``reduced`` samples only the knobs that govern the task's own stage
# and the corruption it learns from, holding the rest at the task default. Only the
# imputation task defines ``reduced`` so far; the flag resolves per task when unspecified.
SEARCH_SPACE_PROFILES: tuple[str, ...] = ("full", "reduced")
# Which validation population an imputation search ranks its trials by (ADR 0008): the
# masked cells every study before it used, or the validation rows' own gaps, scored
# against the complete sibling like the induced test population.
SEARCH_OBJECTIVE_POPULATIONS: tuple[str, ...] = ("masked", "induced")
DEFAULT_SEARCH_OBJECTIVE_POPULATION = "masked"


def validation_objective_key(population: str) -> str:
    """The validation metric an imputation search ranks by, for a population."""
    if population not in SEARCH_OBJECTIVE_POPULATIONS:
        raise ValueError(
            f"Unknown search objective population {population!r}; expected one of "
            f"{SEARCH_OBJECTIVE_POPULATIONS}."
        )
    return f"validation/impute/{population}/impute_score"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    ranking_metric: str
    direction: str
    # Metric prefix of the task's second training stage (``finetune`` or ``decode``).
    # Pre-training is shared, so a task's loss bands are the pre-training pair plus
    # this stage's pair.
    stage: str
    # What a hyper-parameter search ranks its trials by (ADR 0005). Not necessarily the
    # fold-ranking metric: the imputation task scores its search on the validation split
    # so the test split never chooses hyperparameters; classification keeps the
    # test-split macro F1 it always used. The direction is the ranking metric's.
    search_objective: str

    @property
    def loss_keys(self) -> tuple[str, ...]:
        return (
            "pretrain/train_loss",
            "pretrain/val_loss",
            f"{self.stage}/train_loss",
            f"{self.stage}/val_loss",
        )

    @property
    def ranking_metric_short_name(self) -> str:
        """The last path segment, used where slashes make poor keys (manifest JSON)."""
        return self.ranking_metric.rsplit("/", 1)[-1]


_TASK_SPECS: Mapping[str, TaskSpec] = {
    "classification": TaskSpec("classification", "f1_macro", "maximize", "finetune", "f1_macro"),
    "imputation": TaskSpec(
        "imputation",
        "impute/masked/impute_score",
        "minimize",
        "decode",
        "validation/impute/masked/impute_score",
    ),
}
CLASSIFICATION = _TASK_SPECS["classification"]
IMPUTATION = _TASK_SPECS["imputation"]


def task_spec(name: str) -> TaskSpec:
    """Return the ranking contract for a task name, rejecting unknown names."""
    try:
        return _TASK_SPECS[name]
    except KeyError:
        raise ValueError(
            f"Unknown task {name!r}; expected one of {', '.join(TASK_NAMES)}"
        ) from None


#TODO store those defaults elsewhere
@dataclass(frozen=True)
class Hyperparameters:
    dimension: int = 128
    hidden_dimension: int = 16
    heads: int = 16
    layers: int = 2
    feedforward_dimension: int = 32
    dropout: float = 0.2
    pretraining_epochs: int = 300
    finetuning_epochs: int = 150

    batch_size: int = 256
    pretraining_learning_rate: float = 0.00034
    pretraining_weight_decay: float = 0.005
    mask_probability: float = 0.5
    finetuning_learning_rate: float = 0.001
    finetuning_weight_decay: float = 0.0019
    labels: int = 4
    lr_scheduler: str = DEFAULT_LR_SCHEDULER
    pretraining_objective: str = DEFAULT_PRETRAINING_OBJECTIVE

    # The decode stage, used only by the imputation task. Defaulted so that every config
    # written before that task existed keeps loading unchanged; the fine-tuning values are
    # mirrored because the decode stage is fine-tuning's counterpart.
    decode_epochs: int = 150
    decode_learning_rate: float = 0.001
    decode_weight_decay: float = 0.0019
    # Weight on the numerical reconstruction term, each term already averaged over its own
    # hidden cells.
    lambda_num: float = 1.0
    # Nominal share of cells hidden when scoring. Nominal because the masking helper
    # scales it down by each row's null density: asking for 0.2 hides about 20% of a
    # complete variant but about 5% of an 80%-missing one.
    eval_mask_rate: float = 0.2
    # Extra nominal rates to score the test fold at, as diagnostics only. The primary rate
    # above is what validation and fold ranking use.
    eval_mask_rates_extra: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.lr_scheduler not in LR_SCHEDULER_NAMES:
            raise ValueError(
                f"Unknown learning-rate scheduler {self.lr_scheduler!r}; "
                f"expected one of {', '.join(LR_SCHEDULER_NAMES)}"
            )
        if self.pretraining_objective not in PRETRAINING_OBJECTIVES:
            raise ValueError(
                f"Unknown pre-training objective {self.pretraining_objective!r}; "
                f"expected one of {', '.join(PRETRAINING_OBJECTIVES)}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Hyperparameters":
        return cls(
            dimension=int(values.get("DIM", values.get("dimension", 128))),
            hidden_dimension=int(values.get("HIDDEN_DIM", values.get("hidden_dimension", 16))),
            heads=int(values.get("HEADS", values.get("heads", 16))),
            layers=int(values.get("LAYERS", values.get("layers", 2))),
            feedforward_dimension=int(values.get("DIM_FEED", values.get("feedforward_dimension", 32))),
            dropout=float(values.get("DROPOUT", values.get("dropout", 0.2))),
            pretraining_epochs=int(values.get("EPOCHS_PRE", values.get("pretraining_epochs", 300))),
            batch_size=int(values.get("BATCH", values.get("batch_size", 256))),
            pretraining_learning_rate=float(
                values.get("LR_PRE", values.get("pretraining_learning_rate", 0.00034))
            ),
            pretraining_weight_decay=float(
                values.get("WEIGHT_DECAY_PRE", values.get("pretraining_weight_decay", 0.005))
            ),
            mask_probability=float(values.get("PROB_MASCARA", values.get("mask_probability", 0.5))),
            finetuning_epochs=int(values.get("EPOCH_FINE", values.get("finetuning_epochs", 150))),
            finetuning_learning_rate=float(
                values.get("LR_FINE", values.get("finetuning_learning_rate", 0.001))
            ),
            finetuning_weight_decay=float(
                values.get("WEIGHT_DECAY_FINE", values.get("finetuning_weight_decay", 0.0019))
            ),
            labels=int(values.get("LABELS", values.get("labels", 4))),
            lr_scheduler=str(
                values.get("LR_SCHEDULER", values.get("lr_scheduler", DEFAULT_LR_SCHEDULER))
            ),
            pretraining_objective=str(
                values.get(
                    "PRETRAIN_OBJECTIVE",
                    values.get("pretraining_objective", DEFAULT_PRETRAINING_OBJECTIVE),
                )
            ),
            decode_epochs=int(values.get("EPOCHS_DECODE", values.get("decode_epochs", 150))),
            decode_learning_rate=float(
                values.get("LR_DECODE", values.get("decode_learning_rate", 0.001))
            ),
            decode_weight_decay=float(
                values.get("WEIGHT_DECAY_DECODE", values.get("decode_weight_decay", 0.0019))
            ),
            lambda_num=float(values.get("LAMBDA_NUM", values.get("lambda_num", 1.0))),
            eval_mask_rate=float(values.get("EVAL_MASK_RATE", values.get("eval_mask_rate", 0.2))),
            eval_mask_rates_extra=tuple(
                float(rate)
                for rate in values.get(
                    "EVAL_MASK_RATES_EXTRA", values.get("eval_mask_rates_extra", ())
                )
            ),
        )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    base_dataset_name: str
    label_column: str

    @classmethod
    def from_name(cls, dataset_name: str, label_column: str | None) -> "DatasetSpec":
        return cls(
            dataset_name=dataset_name,
            base_dataset_name=dataset_name.split("_", 1)[0],
            label_column=label_column or "class",
        )


# How the top-level MLflow run of one training execution is recorded.
#   parent        one comparable training execution: a new top-level run with
#                 the full ADR 0002 record (summary, loss bands, artifacts,
#                 lineage, diagnostic children).
#   optuna_trial  one hyper-parameter search trial: logs a lightweight record
#                 (tags, parameters, final summary metrics) into the trial run
#                 the caller already has active, so trials never crowd the
#                 ``run_role = parent`` comparison table.
TRACKING_RUN_ROLES: tuple[str, ...] = ("parent", "optuna_trial")


@dataclass(frozen=True)
class RuntimeOptions:
    output_dir: Path = Path("results")
    metrics_dir: Path = Path("metrics")
    tracking_enabled: bool = True
    tracking_run_role: str = "parent"
    # Extra tags merged onto the top-level run, e.g. the study that produced a
    # retrained configuration.
    tracking_tags: Mapping[str, str] = field(default_factory=dict)
    # Mirror the finished run tree into ``TRIDENT/mirror/<task>`` (ADR 0006). On by the
    # user's explicit decision; ``--disable_mirror`` turns it off, and so does disabling
    # tracking, since there is then nothing to mirror.
    mirror_runs: bool = True
    # Where an imputation run reads and writes the baseline imputers' cached scores
    # (``src/training/baseline_cache.py``); None, from ``--no_baseline_cache``, computes
    # them every time. On by the user's decision of 2026-09-30: a hit is the computation.
    baseline_cache_dir: Path | None = DEFAULT_BASELINE_CACHE_DIR

    def __post_init__(self) -> None:
        if self.tracking_run_role not in TRACKING_RUN_ROLES:
            raise ValueError(
                f"Unknown tracking run role {self.tracking_run_role!r}; "
                f"expected one of {', '.join(TRACKING_RUN_ROLES)}"
            )


@dataclass(frozen=True)
class FoldSplit:
    train_indices: Sequence[int]
    validation_indices: Sequence[int]
    test_indices: Sequence[int]


@dataclass(frozen=True)
class PreparedDataset:
    frame: Any
    label_column: str
    categorical_columns: Sequence[str]
    numerical_columns: Sequence[str]
    label_classes: Sequence[Any]
    source_path: Path
    splits_path: Path
    # The scaler that produced ``frame``'s numerical columns, kept so a preview can show
    # a credit amount rather than a z-score. Retaining it does not move when scaling
    # happens; ``None`` when the dataset has no numerical columns.
    scaler: Any = None
    # ``frame``'s numerical columns as they were before scaling overwrote them, kept so a
    # preview can report a known truth exactly instead of one that round-tripped through
    # the model's float32 (ticket 0004). ``None`` when the dataset has no numerical
    # columns. Retaining it does not move when scaling happens.
    raw_numerical: Any = None


@dataclass(frozen=True)
class PretrainingOutcome:
    model: Any
    train_losses: Sequence[float]
    validation_losses: Sequence[float]


# How a fold identifies itself. Ordinarily its 1-based number, but the single-split path
# reports the literal ``"single_split"`` (``finetuning.py``), so anything keyed by fold has
# to admit both.
FoldKey: TypeAlias = int | str


@dataclass(frozen=True)
class FoldResult:
    fold: FoldKey
    dataset_name: str
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class LoggedMetric:
    key: str
    value: float
    step: int | None


@dataclass(frozen=True)
class LoggedArtifact:
    path: Path
    artifact_path: str | None


@dataclass(frozen=True)
class FoldTrackingRecord:
    result: FoldResult
    metric_events: Sequence[LoggedMetric]
    artifacts: Sequence[LoggedArtifact]


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci95_lower: float
    ci95_upper: float
    std: float
    minimum: float
    maximum: float
    fold_count: int


@dataclass(frozen=True)
class LossBand:
    step: int
    mean: float
    ci95_lower: float
    ci95_upper: float


@dataclass(frozen=True)
class CrossValidationSummary:
    metrics: Mapping[str, MetricSummary]
    loss_bands: Mapping[str, Sequence[LossBand]]
    diagnostic_roles: Mapping[FoldKey, str]
    # Wall-clock stage timings summarized across folds, keyed without the
    # ``time/`` prefix (``pretrain_seconds``, ``finetune_seconds``,
    # ``total_seconds``). Empty when the folds logged no timing events.
    timings: Mapping[str, MetricSummary] = field(default_factory=dict)
    # Per scored population (``impute/induced``), the baseline imputer whose mean
    # ``impute_score`` over the folds was lowest (ADR 0007). Empty for classification.
    best_baselines: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingArtifactPaths:
    raw_fold_metrics_csv: Path
    summary_json: Path
    manifest_json: Path
    provenance_json: Path


@dataclass(frozen=True)
class FinetuningOutcome:
    model: Any
    result: FoldResult
    train_losses: Sequence[float]
    validation_losses: Sequence[float]


@dataclass(frozen=True)
class DecodingOutcome:
    """What one fold's decode stage produced.

    ``scored_cells`` holds one row per cell the model was asked to fill on the test
    split, which is what both the imputation metrics and the preview artifacts read.
    """

    model: Any
    result: Any
    train_losses: Sequence[float]
    validation_losses: Sequence[float]
    scored_cells: Any
    # Per population (``masked``, ``induced``), every baseline imputer's per-column errors
    # in long form (``column``, ``metric``, ``value``), for the per-column artifact.
    baseline_per_column: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    fold_results: Sequence[FoldResult]
    mean_metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingRequest:
    dataset: DatasetSpec
    hyperparameters: Hyperparameters
    runtime: RuntimeOptions
    seed: int
    cv_folds: int | None
    plot_losses: bool
    save_model: bool
    # Defaulted, so every existing construction site -- including the reviewed
    # vehicle_00nan regression fixture -- keeps meaning what it meant.
    task: str = DEFAULT_TASK
    score_null_path: bool = False
    # Programmatic only, set by a hyper-parameter search (ADR 0005): score the validation
    # split too, so the search ranks trials on it and never on the test split. No flag
    # reaches it, so an ordinary run never carries a validation score.
    score_search_objective: bool = False
    # Where the hyperparameters came from: "defaults", "override", or the
    # repository-relative path of the file that was loaded. Logged on every run as the
    # ``config_source`` param (ADR 0005, decision 5).
    config_source: str = "defaults"

    def __post_init__(self) -> None:
        if self.task not in TASK_NAMES:
            raise ValueError(
                f"Unknown task {self.task!r}; expected one of {', '.join(TASK_NAMES)}"
            )


class TrainingTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics at an optional training step."""
