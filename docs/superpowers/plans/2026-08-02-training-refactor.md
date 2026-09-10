# Training Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split TRIDENT training into focused modules without changing default behavior and add an isolated two-fold regression test.

**Architecture:** Keep `train.py` as the API and direct-execution façade, but move orchestration into `src/training/`. Typed values cross module boundaries; the runner selects either real MLflow tracking or no-op tracking from `RuntimeOptions`.

**Tech Stack:** Python 3.10+, PyTorch, pandas, scikit-learn, MLflow, pytest, uv.

## Global Constraints

- Preserve current seed use, automatic device selection, output defaults, MLflow defaults, metric names, and Optuna result dictionary shape.
- Do not alter scaler fitting order, scheduler cadence, or other training-correctness behavior.
- Keep `train.main(args, return_metrics=False)`, `uv run main.py`, `uv run train.py`, and `hyperparams_override` compatible.
- Add public `--metrics_dir` and `--disable_mlflow`; defaults remain `metrics` and enabled tracking.
- Store work tracking, decisions, and fixtures only in this repository.
- The integration baseline uses `vehicle_00nan`, seed 42, two folds, two pre-training epochs, two fine-tuning epochs, automatic device selection, temporary outputs, and disabled MLflow.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/training/types.py` | Typed request, runtime, fold, and result values. |
| `src/training/config.py` | Hyperparameter resolution, dataset paths, namespace translation, and shared parser. |
| `src/training/data.py` | Dataset preparation and legacy/CV fold construction. |
| `src/training/pretraining.py` | Pre-training model creation and dynamic-mask loop. |
| `src/training/finetuning.py` | Classifier training, evaluation, and fold metrics. |
| `src/training/tracking.py` | MLflow and no-op tracking adapters. |
| `src/training/artifacts.py` | Plots, models, metric CSVs, and hyperparameter JSON. |
| `src/training/runner.py` | Run and fold orchestration. |
| `src/training/cli.py` | Shared CLI dispatch. |
| `train.py` | Compatibility exports and executable façade. |

### Task 1: Establish typed configuration and test infrastructure

**Files:**

- Create: `src/training/__init__.py`, `src/training/types.py`, `src/training/config.py`
- Create: `tests/unit/test_training_config.py`, `pytest.ini`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**

- `Hyperparameters`, `DatasetSpec`, `RuntimeOptions`, `FoldSplit`, `PreparedDataset`, `PretrainingOutcome`, `FoldResult`, `TrainingResult`, and `TrainingRequest` are frozen dataclasses in `types.py`.
- `TrainingTracker` is a `typing.Protocol` in `types.py` with `log_metrics(metrics: dict[str, float], step: int | None = None) -> None`; later tasks provide its concrete adapters.
- `resolve_training_request(args: argparse.Namespace) -> TrainingRequest` and `build_training_parser() -> argparse.ArgumentParser` are in `config.py`.

- [ ] **Step 1: Write the failing configuration tests.**

```python
def test_legacy_namespace_uses_legacy_runtime_defaults() -> None:
    args = Namespace(dataset_name="vehicle_00nan", label_column="class", output_dir="results", seed=42,
                     plot_losses=False, save_model=False, cv_folds=None)
    request = resolve_training_request(args)
    assert request.runtime == RuntimeOptions(Path("results"), Path("metrics"), True)


def test_parser_exposes_runtime_options() -> None:
    args = build_training_parser().parse_args(
        ["--dataset_name", "vehicle_00nan", "--metrics_dir", "tmp/metrics", "--disable_mlflow"]
    )
    assert args.metrics_dir == "tmp/metrics"
    assert args.disable_mlflow is True
```

- [ ] **Step 2: Verify red.**

Run: `uv run pytest tests/unit/test_training_config.py -v`

Expected: collection fails because `src.training` does not exist.

- [ ] **Step 3: Implement the minimal typed configuration.**

```python
@dataclass(frozen=True)
class RuntimeOptions:
    output_dir: Path = Path("results")
    metrics_dir: Path = Path("metrics")
    tracking_enabled: bool = True


def resolve_training_request(args: Namespace) -> TrainingRequest:
    return TrainingRequest(
        dataset=DatasetSpec.from_name(args.dataset_name, getattr(args, "label_column", None)),
        hyperparameters=load_hyperparameters(args),
        runtime=RuntimeOptions(Path(getattr(args, "output_dir", "results")),
                               Path(getattr(args, "metrics_dir", "metrics")),
                               not getattr(args, "disable_mlflow", False)),
        seed=getattr(args, "seed", 42), cv_folds=getattr(args, "cv_folds", None),
        plot_losses=getattr(args, "plot_losses", False), save_model=getattr(args, "save_model", False),
    )
```

Add `pytest>=8.0` to the uv development dependency group and declare an `integration` marker in `pytest.ini`.

- [ ] **Step 4: Verify green.**

Run: `uv run pytest tests/unit/test_training_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add pyproject.toml uv.lock pytest.ini tests/unit/test_training_config.py src/training/__init__.py src/training/types.py src/training/config.py
git commit -m "feat: add typed training configuration"
```

### Task 2: Extract behavior-preserving data and training stages

**Files:**

- Create: `src/training/data.py`, `src/training/summary.py`, `src/training/pretraining.py`, `src/training/finetuning.py`
- Create: `tests/unit/test_training_data.py`, `tests/unit/test_training_metrics.py`

**Interfaces:**

- `prepare_dataset(spec: DatasetSpec) -> PreparedDataset` preserves label encoding, categorical lookup, numerical scaling, and derived columns.
- `build_folds(frame: pd.DataFrame, label_column: str, cv_folds: int | None, seed: int) -> list[FoldSplit]` copies the current predefined-split and cross-validation algorithms.
- `compute_cv_summary(df_or_path: pd.DataFrame | str | Path) -> dict[str, dict[str, float | str]]` retains the existing summary format.
- `train_pretrainer(dataset: PreparedDataset, fold: FoldSplit, hyperparameters: Hyperparameters, device: torch.device, tracker: TrainingTracker) -> PretrainingOutcome` and `train_and_evaluate_classifier(dataset: PreparedDataset, fold: FoldSplit, pretraining: PretrainingOutcome, hyperparameters: Hyperparameters, device: torch.device, tracker: TrainingTracker) -> FoldResult` preserve the current loops.

- [ ] **Step 1: Write failing data and metric tests.**

```python
def test_build_folds_creates_two_disjoint_test_splits() -> None:
    frame = pd.DataFrame({"feature": range(12), "class": [0, 1] * 6})
    folds = build_folds(frame, label_column="class", cv_folds=2, seed=42)
    assert len(folds) == 2
    assert set(folds[0].test_indices).isdisjoint(set(folds[1].test_indices))


def test_build_fold_result_retains_classification_metrics() -> None:
    result = build_fold_result(1, "vehicle_00nan", np.array([0, 1]), np.array([0, 1]), 0.2)
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["f1_micro"] == 1.0
    assert result.metrics["f1_macro"] == 1.0
```

- [ ] **Step 2: Verify red.**

Run: `uv run pytest tests/unit/test_training_data.py tests/unit/test_training_metrics.py -v`

Expected: collection fails because the data and fine-tuning modules do not exist.

- [ ] **Step 3: Extract the existing algorithms without altering them.**

```python
def train_pretrainer(dataset: PreparedDataset, fold: FoldSplit,
                     hyperparameters: Hyperparameters, device: torch.device,
                     tracker: TrainingTracker) -> PretrainingOutcome:
    return _run_pretraining_epochs(dataset, fold, hyperparameters, device, tracker)


def train_and_evaluate_classifier(dataset: PreparedDataset, fold: FoldSplit,
                                  pretraining: PretrainingOutcome,
                                  hyperparameters: Hyperparameters, device: torch.device,
                                  tracker: TrainingTracker) -> FoldResult:
    return _run_finetuning_epochs(dataset, fold, pretraining, hyperparameters, device, tracker)
```

Keep scaler fitting before fold construction and preserve the existing KFold, StratifiedKFold, StratifiedShuffleSplit, and ShuffleSplit branch conditions.

- [ ] **Step 4: Verify green.**

Run: `uv run pytest tests/unit/test_training_data.py tests/unit/test_training_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/unit/test_training_data.py tests/unit/test_training_metrics.py src/training/data.py src/training/summary.py src/training/pretraining.py src/training/finetuning.py
git commit -m "refactor: extract training stages"
```

### Task 3: Add runtime adapters, compatibility façade, and integration baseline

**Files:**

- Create: `src/training/tracking.py`, `src/training/artifacts.py`, `src/training/runner.py`, `src/training/cli.py`
- Create: `tests/unit/test_training_runtime.py`, `tests/integration/test_vehicle_regression.py`, `tests/fixtures/vehicle_00nan_regression.json`
- Modify: `train.py`, `main.py`, `opt.py`, `README.md`, `docs/tickets/0001-training-refactor.md`

**Interfaces:**

- `create_tracker(enabled: bool) -> TrainingTracker` returns a real MLflow wrapper or a no-op wrapper. The no-op wrapper calls no MLflow API.
- `run_training(request: TrainingRequest) -> TrainingResult` returns individual `FoldResult` values and their summary.
- `run_from_namespace(args: argparse.Namespace, return_metrics: bool = False) -> dict | None` preserves the existing Optuna-compatible return shape.
- `train.main` delegates to `run_from_namespace`; direct execution and `main.py` invoke the same shared parser.

- [ ] **Step 1: Write failing runtime and integration tests.**

```python
def test_disabled_tracking_never_starts_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.training.tracking.mlflow.start_run", pytest.fail)
    with create_tracker(enabled=False).start_parent("dataset", {}, {}):
        pass


@pytest.mark.integration
def test_vehicle_two_fold_metrics_match_reviewed_baseline(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    request = resolve_training_request(Namespace(**fixture["configuration"],
        output_dir=str(tmp_path / "results"), metrics_dir=str(tmp_path / "metrics"), disable_mlflow=True))
    result = run_training(request)
    assert_metric_series_within_tolerance(result.fold_results, fixture["folds"], fixture["tolerances"]["fold"])
    assert_mean_metrics_within_tolerance(result.mean_metrics, fixture["means"], fixture["tolerances"]["mean"])
```

- [ ] **Step 2: Verify red.**

Run: `uv run pytest tests/unit/test_training_runtime.py -v; uv run pytest -m integration -v`

Expected: runtime collection fails because adapters do not exist; integration fails because the fixture does not exist.

- [ ] **Step 3: Implement the adapters and façade.**

```python
def main(args: argparse.Namespace, return_metrics: bool = False) -> dict | None:
    return run_from_namespace(args, return_metrics=return_metrics)


if __name__ == "__main__":
    run_training_cli()
```

The real tracker preserves existing `setup_mlflow`, experiment, parent/nested run, tag, metric, and artifact behavior. When tracking is disabled, do not call `mlflow.autolog`, `setup_mlflow`, `start_run`, or `mlflow.log_*`. Write project metrics under `request.runtime.metrics_dir`, retain run artifacts under `request.runtime.output_dir`, and use parser defaults for legacy callers. Add `--metrics_dir` and `--disable_mlflow` to the shared parser. Set those fields explicitly in Optuna trial and retraining namespaces.

- [ ] **Step 4: Generate and commit the reviewed fixture.**

Run the configured two-fold job three times, with automatic device selection, disabled MLflow, and temporary output paths. Record fold `accuracy`, `f1_micro`, `f1_macro`, and their means. Set each tolerance to the maximum observed absolute deviation plus `0.005`, with a minimum of `0.01`; record Python, torch, CUDA availability, and selected device. Write that reviewed data to `tests/fixtures/vehicle_00nan_regression.json`.

- [ ] **Step 5: Verify green and document the public controls.**

Run: `uv run pytest -v; uv run pytest -m integration -v; uv run main.py --help; uv run train.py --help`

Expected: all tests pass; integration leaves no repository `results/`, `metrics/`, or MLflow artifacts; both help outputs list `--metrics_dir` and `--disable_mlflow`.

Add README examples for the two public controls and the integration command. Change the internal ticket status to `Done` after verification passes.

- [ ] **Step 6: Commit.**

```bash
git add tests src/training/tracking.py src/training/artifacts.py src/training/runner.py src/training/cli.py train.py main.py opt.py README.md docs/tickets/0001-training-refactor.md
git commit -m "refactor: modularize training workflow"
```

## Plan Self-Review

- The three tasks cover every approved requirement: package boundaries, typed values, legacy compatibility, shared CLI, public isolated-runtime controls, no-op MLflow, checked-in regression fixture, temporary artifacts, automatic device selection, and internal documentation.
- The training algorithm remains untouched; all movement is constrained to behavior preservation.
- All named types and functions are defined before later tasks consume them.
- The plan has no unfinished requirements, ambiguous file ownership, or deferred implementation references.
