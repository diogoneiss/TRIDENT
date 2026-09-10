# Curated MLflow Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each MLflow parent run a clean, comparable TRIDENT execution record while retaining only the `f1_macro` best and worst folds as diagnostic child runs.

**Architecture:** A pure summary module calculates final-metric Student-t summaries, epoch loss bands, and diagnostic roles from buffered fold records. The MLflow tracker is the single tracking seam: it buffers fold events, logs the parent, then replays selected records into nested runs after ranking is known. The runner retains its orchestration and return shape.

**Tech Stack:** Python 3.10, PyTorch, pandas, SciPy, MLflow 3.14+, pytest, uv.

## Global Constraints

- Use Python 3.10 through `uv run --python 3.10 <command>`.
- Do not alter split/scaling order, seed behavior, model architecture, optimization loops, scheduler cadence, metric calculation, CLI compatibility, or `train.main(args, return_metrics=False)` return shape.
- Use `f1_macro` to select `best_fold` and `worst_fold`; ties choose the lowest fold number and yield one `best_and_worst` child when selections coincide.
- Parent final metrics log mean, 95% Student-t bounds, sample standard deviation, observed min/max, and fold count; loss series log only mean and bounds per epoch.
- Keep `TRIDENT/<base_dataset>` experiments. A non-CV execution is one top-level run; only retained CV diagnostics become child runs.
- Disabled tracking must call no MLflow API. Preserve unrelated worktree changes.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Direct SciPy dependency for exact Student-t critical values. |
| `src/training/types.py` | Immutable fold-event, artifact, provenance, and summary values. |
| `src/training/summary.py` | Pure validation, statistics, loss bands, and diagnostic selection. |
| `src/training/artifacts.py` | Parent CSV/JSON artifact writers. |
| `src/training/data.py` | Typed dataset source/split lineage. |
| `src/training/tracking.py` | MLflow/no-op adapters; buffering, parent logging, and replay. |
| `src/training/runner.py` | Per-fold tracker use and tracking finalization. |
| `tests/unit/test_training_summary.py` | Statistical and validation tests. |
| `tests/unit/test_training_artifacts.py` | Provenance and parent-artifact tests. |
| `tests/unit/test_training_tracking.py` | Temporary-backend hierarchy, lineage, artifact, and no-op tests. |
| `README.md` | MLflow comparison workflow. |

## Task 1: Add typed CV-summary values and pure statistical aggregation

**Files:**

- Modify: `pyproject.toml`, `uv.lock`, `src/training/types.py`, `src/training/summary.py`
- Create: `tests/unit/test_training_summary.py`

**Interfaces:**

- Consumes: completed `FoldResult` values and buffered raw metric events.
- Produces: `CrossValidationSummary` with per-metric statistics, four loss-band histories, and diagnostic roles.

- [ ] **Step 1: Write failing summary tests**

Create helpers and tests with this intended interface:

~~~
from math import sqrt

import pytest

from src.training.summary import summarize_cross_validation
from src.training.types import FoldResult, FoldTrackingRecord, LoggedMetric


def _record(fold: int, f1_macro: float) -> FoldTrackingRecord:
    return FoldTrackingRecord(
        result=FoldResult(fold, "vehicle_00nan", {
            "accuracy": 0.5 + f1_macro / 2, "f1_macro": f1_macro,
        }),
        metric_events=(
            LoggedMetric("pretrain/train_loss", 1.0 + fold, 0),
            LoggedMetric("pretrain/val_loss", 2.0 + fold, 0),
            LoggedMetric("finetune/train_loss", 3.0 + fold, 0),
            LoggedMetric("finetune/val_loss", 4.0 + fold, 0),
        ),
        artifacts=(),
    )


def test_summarize_cross_validation_logs_statistics_and_loss_bands() -> None:
    summary = summarize_cross_validation([_record(1, 0.4), _record(2, 0.8)])

    stats = summary.metrics["f1_macro"]
    assert stats.mean == pytest.approx(0.6)
    assert stats.std == pytest.approx(sqrt(0.08))
    assert stats.fold_count == 2
    assert stats.minimum == pytest.approx(0.4)
    assert stats.maximum == pytest.approx(0.8)
    assert stats.ci95_lower < stats.mean < stats.ci95_upper
    assert summary.diagnostic_roles == {1: "worst_fold", 2: "best_fold"}
    assert summary.loss_bands["finetune/val_loss"][0].mean == pytest.approx(5.5)


def test_summarize_cross_validation_uses_one_role_for_a_tie() -> None:
    summary = summarize_cross_validation([_record(1, 0.5), _record(2, 0.5)])

    assert summary.diagnostic_roles == {1: "best_and_worst"}
~~~

Add independent tests rejecting fewer than two folds, a missing/non-finite final metric, and loss events with mismatched epoch sets.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_summary.py -q
~~~

Expected: collection fails because the summary function and typed records do not exist.

- [ ] **Step 3: Add the direct dependency**

Run:

~~~powershell
uv add scipy>=1.10
~~~

Verify both `pyproject.toml` and `uv.lock` change. Do not rely on SciPy being transitively present through scikit-learn.

- [ ] **Step 4: Implement the minimal summary interface**

Add these frozen values to `src/training/types.py`:

~~~
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
    diagnostic_roles: Mapping[int, str]
~~~

Implement `summarize_cross_validation(records: Sequence[FoldTrackingRecord]) -> CrossValidationSummary`. Validate a common finite numeric metric set and exact common steps for the four loss keys. Use `scipy.stats.t.ppf(0.975, df=n - 1)`, `numpy.std(values, ddof=1)`, and `sqrt(n)`. Rank `result.metrics["f1_macro"]` by `(score, -fold)`. Preserve `compute_cv_summary()` unchanged for the legacy console/return behavior.

- [ ] **Step 5: Run the summary tests to verify they pass**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_summary.py tests/unit/test_training_metrics.py -q
~~~

Expected: all tests pass, including the existing legacy summary test.

- [ ] **Step 6: Commit**

~~~powershell
git add pyproject.toml uv.lock src/training/types.py src/training/summary.py tests/unit/test_training_summary.py
git commit -m "feat: summarize cross-validation tracking"
~~~

## Task 2: Write parent artifacts and typed provenance

**Files:**

- Modify: `src/training/types.py`, `src/training/data.py`, `src/training/artifacts.py`
- Create: `tests/unit/test_training_artifacts.py`

**Interfaces:**

- Consumes: `PreparedDataset`, raw `FoldResult` values, and `CrossValidationSummary`.
- Produces: parent artifact paths and serializable provenance.

- [ ] **Step 1: Write failing artifact/provenance tests**

Use a prepared dataset with `source_path=Path("datasets/processed_datasets/vehicle/vehicle_00nan.csv")` and assert:

~~~
paths = writer.write_cv_tracking_artifacts(fold_results, summary, dataset, seed=42, cv_folds=2)

assert paths.raw_fold_metrics_csv.exists()
assert json.loads(paths.summary_json.read_text())["interval"] == "two-sided 95% Student-t"
assert json.loads(paths.provenance_json.read_text())["source_path"].endswith("vehicle_00nan.csv")
assert json.loads(paths.manifest_json.read_text())["diagnostic_roles"] == {
    "1": "worst_fold", "2": "best_fold",
}
~~~

- [ ] **Step 2: Verify the test fails**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_artifacts.py -q
~~~

Expected: `PreparedDataset` lacks typed lineage fields and no CV-tracking writer exists.

- [ ] **Step 3: Implement the artifact/provenance contract**

Extend `PreparedDataset` with `source_path: Path` and `splits_path: Path`; populate both in `prepare_dataset()` while retaining the existing `frame.attrs["splits_path"]` compatibility value. Add `TrackingArtifactPaths` and `ArtifactWriter.write_cv_tracking_artifacts(fold_results, summary, dataset, seed, cv_folds)`, writing exactly:

~~~text
metrics/raw_fold_metrics.csv
metrics/cv_summary.json
tracking/diagnostic_manifest.json
data/provenance.json
~~~

Use `json.dumps(payload, indent=2)`, converting Path and NumPy values to built-ins. Keep the current local `metrics.csv` and `hyperparameters.json` behavior.

- [ ] **Step 4: Verify the tests pass**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_artifacts.py tests/unit/test_training_data.py -q
~~~

- [ ] **Step 5: Commit**

~~~powershell
git add src/training/types.py src/training/data.py src/training/artifacts.py tests/unit/test_training_artifacts.py
git commit -m "feat: record cross-validation tracking artifacts"
~~~

## Task 3: Buffer folds and replay only curated MLflow diagnostics

**Files:**

- Modify: `src/training/tracking.py`, `src/mlflow_utils.py`, `tests/unit/test_training_runtime.py`
- Create: `tests/unit/test_training_tracking.py`

**Interfaces:**

- Consumes: `FoldTrackingRecord`, `CrossValidationSummary`, `PreparedDataset`, and `TrackingArtifactPaths`.
- Produces: parent metrics/tags/input lineage and zero, one, or two nested diagnostic runs.

- [ ] **Step 1: Write failing temporary-backend tests**

Set `MLFLOW_TRACKING_URI` to a temporary SQLite URI and run from a temporary working directory. Exercise the intended API:

~~~
with tracker.parent_run(
    dataset_name="vehicle_00nan", seed=42, cv_folds=2, hyperparameters={}
) as active_tracker:
    with active_tracker.fold_run(
        fold=1, cv_folds=2, dataset_name="vehicle_00nan"
    ) as fold_tracker:
        fold_tracker.log_metrics({"pretrain/train_loss": 1.0}, step=0)
        fold_tracker.log_metrics({"test/f1_macro": 0.4})
~~~

Use `MlflowClient` to assert a top-level `run_role=parent`, metrics such as `cv/test/f1_macro/mean`, one logged dataset input, and exactly two children tagged `best_fold`/ `worst_fold`. Add tied, single-split, parent-artifact, and disabled-tracker tests.

- [ ] **Step 2: Verify the tests fail**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_tracking.py tests/unit/test_training_runtime.py -q
~~~

Expected: the existing tracker creates an eager child run and lacks collection/finalization APIs.

- [ ] **Step 3: Implement the tracker seam**

Make `fold_run()` yield a fold-scoped buffer:

~~~
class BufferedFoldTracker:
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metric_events.extend(LoggedMetric(key, float(value), step) for key, value in metrics.items())

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append(LoggedArtifact(Path(path), artifact_path))

    def to_record(self, result: FoldResult) -> FoldTrackingRecord:
        return FoldTrackingRecord(result, tuple(self.metric_events), tuple(self.artifacts))

~~~

`MlflowTracker.parent_run()` keeps the existing keyword-only
`dataset_name`, `seed`, `cv_folds`, and `hyperparameters` inputs and yields the
active parent tracker. `MlflowTracker.fold_run()` takes `fold`, `cv_folds`, and
`dataset_name`, and yields `BufferedFoldTracker`. Its
`finalize_cross_validation(records, summary, dataset, artifact_paths)` method
must call, in order, `log_prepared_dataset(dataset)`,
`_log_parent_summary(summary, artifact_paths)`, and
`_log_diagnostic_children(records, summary.diagnostic_roles)`.

Do not call generic `mlflow.autolog()`. Parent tags include `run_role=parent`, `dataset_variant`, parsed `missingness_percent`, and `evaluation_mode`. Add `log_prepared_dataset(dataset)` that creates `dataset_input = mlflow.data.from_pandas(dataset.frame, source=str(dataset.source_path), name=dataset.frame.attrs["dataset_name"])` and calls `mlflow.log_input(dataset_input, context="training")`.

`finalize_cross_validation()` logs `cv/test/<metric>/<field>`, logs loss bands at their original epoch steps, and logs parent artifacts. It then opens one nested run per `summary.diagnostic_roles`, replays every buffered metric event, and logs every retained artifact. The disabled adapter uses an in-memory buffer and no-ops for finalization/lineage.

- [ ] **Step 4: Verify the tracking tests pass**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_tracking.py tests/unit/test_training_runtime.py -q
~~~

- [ ] **Step 5: Commit**

~~~powershell
git add src/training/tracking.py src/mlflow_utils.py tests/unit/test_training_tracking.py tests/unit/test_training_runtime.py
git commit -m "feat: curate MLflow cross-validation runs"
~~~

## Task 4: Integrate finalization without changing training behavior

**Files:**

- Modify: `src/training/runner.py`, `src/training/artifacts.py`, `src/training/types.py`
- Test: `tests/unit/test_training_runtime.py`, `tests/integration/test_vehicle_regression.py`

**Interfaces:**

- Consumes: Task 3 fold trackers and existing training outcomes.
- Produces: the same `TrainingResult`, local artifacts, and finalized MLflow records.

- [ ] **Step 1: Write a failing runner-orchestration test**

Replace `create_tracker()` with a fake whose `fold_run()` yields a distinct fold tracker. Assert existing training functions receive the yielded object, CV finalization receives both records once, and a single split creates no child:

~~~
assert fake_tracker.fold_tracker_ids == fake_training_function_tracker_ids
assert fake_tracker.finalized_records == [1, 2]
assert fake_tracker.single_split_children == 0
~~~

- [ ] **Step 2: Verify it fails**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_runtime.py -q
~~~

Expected: the runner currently passes the parent tracker to every fold and performs no finalization.

- [ ] **Step 3: Implement minimal orchestration**

Bind and use the fold tracker:

~~~
with tracker.fold_run(
    fold=ordinal, cv_folds=request.cv_folds, dataset_name=dataset_name
) as fold_tracker:
    pretraining = train_pretrainer(
        dataset, fold, request.hyperparameters, device, fold_tracker
    )
    finetuning = train_and_evaluate_classifier(
        dataset, fold, pretraining, request.hyperparameters, device, fold_tracker
    )
    records.append(fold_tracker.to_record(finetuning.result))
~~~

Keep plot/model writes in place, but buffer their `log_artifact()` calls. Add `ArtifactWriter.write_tracking_provenance(dataset, seed, cv_folds) -> Path` for the parent provenance JSON; `write_cv_tracking_artifacts()` reuses that writer rather than duplicating the payload. For both evaluation modes, while the parent is active, log prepared-dataset lineage and write/log hyperparameters, raw metrics, and provenance artifacts. For CV additionally write the CV summary/manifest artifacts, call `summary = summarize_cross_validation(records)`, then call `tracker.finalize_cross_validation(records, summary, dataset, artifact_paths)`. For a single split, replay its fold buffer directly into the active parent without CV summary, manifest, or diagnostic-child creation. Keep current `TrainingResult(results, mean_metrics)` construction and console `compute_cv_summary()` output exactly.

- [ ] **Step 4: Verify targeted behavior and regression**

~~~powershell
uv run --python 3.10 pytest tests/unit/test_training_runtime.py tests/unit/test_training_metrics.py -q
uv run --python 3.10 pytest -m integration
~~~

Expected: unit tests pass and the reviewed `vehicle_00nan` fixture passes unchanged.

- [ ] **Step 5: Commit**

~~~powershell
git add src/training/runner.py src/training/artifacts.py src/training/types.py tests/unit/test_training_runtime.py
git commit -m "feat: finalize curated training tracking"
~~~

## Task 5: Document comparison and run complete verification

**Files:**

- Modify: `README.md`
- Test: full unit and integration suites

**Interfaces:**

- Consumes: completed tracking layout.
- Produces: a reproducible MLflow UI workflow and accurate interpretation.

- [ ] **Step 1: Add documentation acceptance assertions**

The README must state:

~~~text
Filter tags.run_role = parent before comparing executions.
Compare cv/test/f1_macro/mean with ci95_lower, ci95_upper, and std.
Open best_fold/worst_fold children only for diagnosis.
The 95% bounds are internal CV uncertainty, not an independent-test guarantee.
~~~

- [ ] **Step 2: Update README and check it**

Document parent/diagnostic roles, loss-band names, dataset lineage, and single-split layout. Link the ADR and deferred logged-model ticket.

~~~powershell
rg -n "run_role|cv/test/f1_macro/mean|internal CV|best_fold|logged-model" README.md
~~~

Expected: every acceptance term appears in the CV documentation.

- [ ] **Step 3: Run complete verification**

~~~powershell
uv run --python 3.10 pytest -m "not integration"
uv run --python 3.10 pytest -m integration
git diff --check
~~~

Expected: both suites exit 0 and Git reports no whitespace errors.

- [ ] **Step 4: Review specification coverage**

Check every item in `docs/superpowers/specs/2026-08-03-curated-mlflow-cross-validation-design.md`:

- Parent comparison metrics and loss bands exist.
- Selected child runs and the tie case exist.
- Single splits are top-level only.
- Parent artifacts and lineage are present.
- Disabled tracking remains inert.
- Regression behavior and public return shape are unchanged.
- README, ADR, and deferred lifecycle ticket are linked.

- [ ] **Step 5: Commit**

~~~powershell
git add README.md
git commit -m "docs: explain curated MLflow comparisons"
~~~
