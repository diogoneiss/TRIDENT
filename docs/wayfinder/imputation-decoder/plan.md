# Imputation Decoder Task Implementation Plan

> **How to execute:** Each task below has an execution ticket under
> [`docs/tickets/imputation-decoder/`](../../tickets/imputation-decoder/spec.md); pick a
> ticket by path and work it with `mattpocock-skills:tdd`. Every task is already shaped as one red-green cycle: write the failing tests, run them to see them fail, implement the minimum, run them to see them pass, commit. Respect the task order, because Task 1 must land before Task 6. Steps use checkbox (`- [ ]`) syntax for tracking. The wayfinder map that produced this plan is [`map.md`](map.md); ADR 0004 is the specification.

**Goal:** Add `--task imputation`, a decode stage that reconstructs masked cell values with per-column heads, imputation metrics with a per-task fold-ranking contract, MLflow identity tags with backfill, preview artifacts, Optuna support, and a second regression fixture, while every classification run stays bit-identical.

**Architecture:** [ADR 0004](../../adr/0004-imputation-decoder-task.md) is the specification; this plan orders its fourteen decisions so nothing can crash mid-run. The ranking-metric plumbing lands **before** the decode stage, because the cross-validation summariser raises when the ranking metric is absent and it runs after every fold has trained. A frozen `TaskSpec` registry is the single dispatch point; every new argument is defaulted to classification so existing call sites and tests pass unedited.

**Tech Stack:** Python 3.10, PyTorch, pandas, scikit-learn, MLflow, Optuna, pytest, uv.

## Global Constraints

- Use Python 3.10 through `uv run --python 3.10 <command>`.
- Do not alter the classification path: split/scaling order, seed behaviour, the pre-training loop and its Xavier sweep, scheduler cadence, metric calculation, CLI compatibility, or `train.main(args, return_metrics=False)`. `tests/integration/test_vehicle_regression.py` must pass **with no edit**.
- Every new function argument and dataclass field defaults to the classification behaviour.
- The decode stage must **not** call `model.apply(initialize_weights)`; that sweep would erase the pretrained encoder.
- `[MASK]`, `[NULL]` and the literal `"nan"` are excluded from every categorical head's output space. Targets are encoded from `preprocess_table(..., fine_tunning=True)`, never the raw frame.
- Training never reads the `_00nan` sibling; it is read for test-fold scoring only, after an alignment assertion.
- Preserve unrelated worktree changes, especially `metrics/`, `results/`, MLflow files and user edits.
- Commit after each task. Commit messages end with the session's attribution lines.

---

## File structure

| File | Responsibility |
|---|---|
| `src/training/types.py` | `TaskSpec` + registry, `TASK_NAMES`, new `Hyperparameters` fields, `TrainingRequest.task` / `score_null_path`, `DecodingOutcome`, `PreparedDataset.scaler`. |
| `src/training/summary.py` | Direction-aware `_diagnostic_roles` and `_validate_final_metrics`; decode keys in `_LOSS_KEYS`; `TIMING_METRIC_KEYS` gains `time/decode_seconds`. |
| `src/training/artifacts.py` | Manifest keyed by the ranking metric's short name; preview, cell ledger and per-column writers. |
| `src/training/imputation_metrics.py` | Pure metric functions, baselines, `impute_score`, zero-baseline guard. |
| `src/models.py` | `TridentDecoder`: per-column heads, output-space exclusion, batched numerical heads, loss. |
| `src/training/data.py` | Retain the scaler; sibling lookup and alignment assertion; fixed evaluation mask draw. |
| `src/training/decoding.py` | `train_and_evaluate_decoder`, both populations, extra rates, null path. |
| `src/training/runner.py` | Task dispatch, decode timing, artifact calls, task-specific parameter logging. |
| `src/training/config.py` | `--task`, `--score_null_path`, parse-time rejection, new JSON keys. |
| `src/training/tracking.py` | `task` and `is_optuna` tags at both call sites; `impute_` run-name prefix. |
| `src/mlflow_utils.py` | Tag constants. |
| `scripts/backfill_run_tags.py` | Generalised backfill (keeps the `lr_scheduler` invocation working). |
| `opt.py` | Task-switched search space, `TrialPruned`, `HEADS` divides `DIM`, direction-aware best, study-dir-only writes, `is_optuna` on the study parent. |
| `main.py` | Batch summary prints the task's ranking metric. |
| `tests/unit/test_training_tasks.py` | `TaskSpec`, ranking direction, manifest key naming. |
| `tests/unit/test_imputation_metrics.py` | Metric functions, degenerate compositions, zero baseline. |
| `tests/unit/test_decoder_model.py` | Head shapes, exclusion, loss, no-NaN targets. |
| `tests/unit/test_training_decoding.py` | Decode stage on a tiny synthetic frame; classification never builds a decoder. |
| `tests/unit/test_backfill_run_tags.py` | Generalised backfill, both tags, idempotency. |
| `tests/fixtures/credit-g_20nan_imputation_regression.json`, `tests/integration/test_credit_g_imputation_regression.py` | Second regression baseline. |
| `README.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`, `CONTEXT.md`, `docs/BACKLOG.md` | Flags, schema, MLflow filters, decode-stage section, both fixtures named, B3 marked fixed. |

## Task 1: Per-task ranking contract, classification byte-identical

**Files:**

- Modify: `src/training/types.py`, `src/training/summary.py`, `src/training/artifacts.py`
- Create: `tests/unit/test_training_tasks.py`

**Interfaces:**

- Consumes: `FoldTrackingRecord` sequences and a `TaskSpec`.
- Produces: `CrossValidationSummary.diagnostic_roles` chosen in the task's direction; a manifest whose ranking keys are `<short>_ranking` and `<short>`.

- [ ] **Step 1: Write failing tests**

~~~
from src.training.types import CLASSIFICATION, IMPUTATION, TaskSpec, task_spec
from src.training.summary import summarize_cross_validation


def test_registry_has_both_tasks_with_directions() -> None:
    assert task_spec("classification").ranking_metric == "f1_macro"
    assert task_spec("classification").direction == "maximize"
    assert task_spec("imputation").ranking_metric == "impute/masked/impute_score"
    assert task_spec("imputation").direction == "minimize"


def test_lower_is_better_selects_lowest_fold_as_best() -> None:
    records = [_record(1, {"impute/masked/impute_score": 0.9}),
               _record(2, {"impute/masked/impute_score": 0.4})]
    summary = summarize_cross_validation(records, task=task_spec("imputation"))
    assert summary.diagnostic_roles == {2: "best_fold", 1: "worst_fold"}


def test_default_task_keeps_classification_selection() -> None:
    # No task argument: identical to today's behaviour.
    ...
~~~

Add a manifest test in `tests/unit/test_training_artifacts.py` asserting that with the default task the JSON still contains `"f1_macro_ranking"` and `selected_folds[*]["f1_macro"]`, and with the imputation spec contains `"impute_score_ranking"`.

- [ ] **Step 2: Run the tests to verify they fail**

`uv run --python 3.10 pytest tests/unit/test_training_tasks.py -q`

- [ ] **Step 3: Implement the contract**

In `types.py`: `TASK_NAMES = ("classification", "imputation")`, a frozen `TaskSpec(name, ranking_metric, direction)`, module constants `CLASSIFICATION` and `IMPUTATION`, and `task_spec(name)` raising on unknown names. `ranking_metric` short name = last path segment.

In `summary.py`: `summarize_cross_validation(records, task=CLASSIFICATION)`; `_validate_final_metrics` requires `task.ranking_metric` instead of the literal; `_diagnostic_roles(records, task)` negates the sort key according to `task.direction`. Add `decode/train_loss`, `decode/val_loss` to `_LOSS_KEYS` and `time/decode_seconds` to `TIMING_METRIC_KEYS`.

In `artifacts.py`: `write_cv_tracking_artifacts(..., task=CLASSIFICATION)`; build the ranking dict and `selected_folds` entries from `f"{short}_ranking"` and `short`.

- [ ] **Step 4: Verify the whole existing suite still passes unedited**

`uv run --python 3.10 pytest -m "not integration" -q`

- [ ] **Step 5: Commit**

## Task 2: Request, hyperparameters and command line

**Files:**

- Modify: `src/training/types.py`, `src/training/config.py`, `src/training/runner.py`, `tests/unit/test_training_config.py`, `tests/unit/test_training_runtime.py`

**Interfaces:**

- Consumes: argparse namespace and JSON mappings.
- Produces: `TrainingRequest` with `task` and `score_null_path`; `Hyperparameters` with the five new fields and `eval_mask_rates_extra`; task-specific parameter dict for MLflow.

- [ ] **Step 1: Write failing tests**

Cover: `Hyperparameters.from_mapping({})` yields the five defaults; a mapping lacking every new key still loads; `--task` defaults to `classification` and rejects other values; `--score_null_path` without `--task imputation` raises `SystemExit`; `--score_null_path --task imputation` parses; `_mlflow_hyperparameters` for an imputation request omits `EPOCH_FINE`, `LR_FINE`, `WEIGHT_DECAY_FINE`, `LABELS` and includes the decode keys; for classification it is unchanged; `task` is logged as a run parameter.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

`Hyperparameters`: `decode_epochs=150`, `decode_learning_rate=0.001`, `decode_weight_decay=0.0019`, `lambda_num=1.0`, `eval_mask_rate=0.2`, `eval_mask_rates_extra: tuple[float, ...] = ()`; JSON keys `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `LAMBDA_NUM`, `EVAL_MASK_RATE`, `EVAL_MASK_RATES_EXTRA`. `TrainingRequest`: `task: str = "classification"`, `score_null_path: bool = False`, validated in `__post_init__`. Parser: `--task` with `choices=TASK_NAMES`, `--score_null_path` store-true; `validate_parsed_args` rejects the flag without the task. `resolve_training_request` passes both. `_mlflow_hyperparameters(request)` branches on task; `_log_execution_params` logs `task`.

- [ ] **Step 4: Verify tests pass; run the full unit suite**

- [ ] **Step 5: Commit**

## Task 3: Imputation metrics as pure functions

**Files:**

- Create: `src/training/imputation_metrics.py`, `tests/unit/test_imputation_metrics.py`

**Interfaces:**

- Consumes: numpy arrays of actual and imputed values per scored cell, with column type and column id; training-fold column means and modes.
- Produces: `dict[str, float]` with `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat`, `n_num_cells`, `n_cat_cells`, `impute_score`, plus per-column rows for the artifact.

- [ ] **Step 1: Write failing tests**

Synthetic inputs: mixed table; all-numerical (`w_cat == 0`, score equals the numerical ratio); all-categorical (`w_num == 0`); a constant column giving a zero baseline (guard yields a finite score and no exception); `macro_f1_cat` with an absent class (`zero_division=0`); pooled versus per-column agreement under uniform counts; baselines computed from the *training* values passed in, not from the scored cells.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

`score_cells(cells, baselines) -> ImputationScores` and `mean_mode_baselines(train_frame, numerical_columns, categorical_columns)`. Zero-baseline guard: when a baseline error is zero, that term's ratio is defined as `1.0` if the model's error is also zero, else the raw model error, and the choice is documented in the docstring.

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 4: The decoder model

**Files:**

- Modify: `src/models.py`
- Create: `tests/unit/test_decoder_model.py`

**Interfaces:**

- Consumes: the pretrained `TabularEmbedder` and `TabularTransformerEncoder`; a masked `EncodedTable` and a clean-target `EncodedTable`.
- Produces: `TridentDecoder.forward(masked, targets) -> (loss, metrics)` and `TridentDecoder.predict(encoded) -> (cat_logits_per_column, num_values)`.

- [ ] **Step 1: Write failing tests**

Build a tiny frame with one categorical column containing a NaN (so its vocabulary has `[MASK]`, `[NULL]` and `"nan"`) and two numerical columns. Assert: the categorical head has `V - 3` outputs; `valid_ids` excludes exactly the three special ids; `local_of` maps them to `-1`; numerical heads are batched (one `baddbmm` path, per-column parameters still present in `state_dict`); the loss on a hand-built batch equals the manual `CE + lambda_num * MSE` with per-type count averaging; targets from the processed frame contain no NaN; `predict` never returns a special id.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement `TridentDecoder`**

Per-column `cat_heads[key] = Linear(d, V_col - n_special)`, buffers `valid_ids` / `local_of`, per-column `num_heads` mirrored from the embedder's MLP and applied with stacked weights. Loss as ADR decision 4. No initialisation sweep. Missing-cell inference path: caller substitutes `[MASK]`, model decodes with `argmax` over real ids and scaled scalars.

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 5: Data support for evaluation

**Files:**

- Modify: `src/training/data.py`, `src/training/types.py`, `tests/unit/test_training_data.py`

**Interfaces:**

- `prepare_dataset` retains the scaler on `PreparedDataset.scaler` (default `None`).
- `load_complete_sibling(spec) -> pd.DataFrame | None` finds `<base>_00nan.csv`, asserts alignment (shape, columns, every observed cell equal), returns `None` when absent or when the variant is `_00nan`.
- `evaluation_mask(frame, rate, seed, fold) -> pd.DataFrame` draws one `preprocess_table` mask from a generator seeded by `(seed, fold)` without disturbing the global RNG state.

- [ ] **Step 1: Write failing tests**

Scaler present after `prepare_dataset` and the split-and-scale order unchanged (existing tests untouched); sibling found for a `_20nan` fixture frame, `None` for `_00nan`; alignment assertion raises on a deliberately altered observed cell; evaluation mask identical across two calls with the same `(seed, fold)` and different across folds; global `np.random` state untouched after the call.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

`preprocess_table` uses global `np.random`; wrap the evaluation draw in a `np.random.get_state()` / `set_state()` guard around a local reseed so the fixed draw consumes no global draws.

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 6: The decode stage

**Files:**

- Create: `src/training/decoding.py`, `tests/unit/test_training_decoding.py`

**Interfaces:**

- `train_and_evaluate_decoder(dataset, fold, pretraining, hyperparameters, device, tracker, seed, score_null_path=False) -> DecodingOutcome`
- `DecodingOutcome(model, result, train_losses, validation_losses, scored_cells)`

- [ ] **Step 1: Write failing tests**

On a tiny in-memory `PreparedDataset` (mixed columns, a few NaNs, two decode epochs): per-epoch metrics `decode/train_loss`, `decode/val_loss`, `decode/learning_rate`, `decode/val_rmse_num_z`, `decode/val_acc_cat` are logged; `result.metrics` contains `impute/masked/*` always and `impute/induced/*` only when a sibling is supplied; the best-validation checkpoint is restored (validation loss non-increasing in the restored state); `scored_cells` has one row per scored cell with the required columns; extra rates add `impute/masked/rate_<pct>/*` keys; `score_null_path=True` adds `impute/induced/null_token/*` and a `null_path_imputed` column. **Classification guard**: a test that runs `run_training` with the default task under a monkeypatched `TridentDecoder.__init__` that raises, proving the classification path never constructs a decoder.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

Mirror `finetuning.py` structurally: build the decoder from `pretraining.model.embedder` / `.transformer`, AdamW with the decode rate and decay, `StageScheduler` on the run's schedule with `decode_epochs`, per-epoch re-rolled training masks at `mask_probability`, one fixed validation and one fixed test draw at `eval_mask_rate`, best-validation-loss restore. Scoring: encode the test frame with `[MASK]` substituted at the evaluation cells (and, for induced-missing, at the NaN cells), decode, gather actual/imputed per cell, call `score_cells`. No `initialize_weights`.

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 7: Artifacts

**Files:**

- Modify: `src/training/artifacts.py`, `tests/unit/test_training_artifacts.py`

**Interfaces:**

- `write_imputation_preview(name, scored_cells, seed, fold, n_rows, scaler, label_encoders) -> tuple[Path, Path]` writes the triptych markdown and the cell-ledger CSV from **one** sample draw and stamps `in_preview`.
- `write_per_column_imputation(records) -> Path` writes the long-format parent table.

- [ ] **Step 1: Write failing tests**

Preview contains exactly the sampled rows; the CSV contains every scored cell, is a superset of the preview, and `in_preview` is true for exactly the previewed cells; original units appear in the preview (inverse-transformed) while the CSV carries both scalings and confidence; the model-saw line distinguishes `[NULL]->[MASK]` from `[MASK]`; the per-column table has one row per (fold, column, population, metric).

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement** (the prototype on branch `prototype/imputation-preview` is the visual reference; rewrite it properly rather than promoting it)

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 8: Runner integration, MLflow identity, batch summary

**Files:**

- Modify: `src/training/runner.py`, `src/training/tracking.py`, `src/mlflow_utils.py`, `main.py`, `tests/unit/test_training_tracking.py`, `tests/unit/test_training_runtime.py`

**Interfaces:**

- Runner dispatches on `task_spec(request.task)`; times the decode stage under `time/decode_seconds`; calls the new writers; logs the preview and ledger on the fold tracker and the per-column table on the parent.
- `MlflowTracker.parent_run(..., task: str, is_optuna: bool = False)`; `_execution_tags` and `_log_diagnostic_children` both set `task` and `is_optuna`; run name `impute_<dataset>_<timestamp>` for the imputation task.
- `main.run_all` prints the task's ranking metric under a header naming it.

- [ ] **Step 1: Write failing temporary-backend tests**

Parent, both diagnostic children and (via the existing Optuna tracking test) study and trial runs carry `task` and `is_optuna`; an imputation parent's run name starts with `impute_`; the parent holds `cv/test/impute/masked/impute_score/mean` and the per-column artifact; children hold the preview and ledger; a classification run's tags gain `task=classification`, `is_optuna=false` and nothing else changes.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the unit suite and the classification regression**

`uv run --python 3.10 pytest -m "not integration" -q` then `uv run --python 3.10 pytest -m integration -q`. The vehicle fixture must pass unedited.

- [ ] **Step 5: Commit**

## Task 9: Backfill

**Files:**

- Create: `scripts/backfill_run_tags.py`, `tests/unit/test_backfill_run_tags.py`
- Modify: `scripts/backfill_lr_scheduler_tag.py` (thin wrapper or documented invocation), `tests/unit/test_backfill_lr_scheduler_tag.py`

- [ ] **Step 1: Write failing tests**

Generalised entry point takes `(tag, value, marker_tag | None)`; `task=classification` with `task_backfilled=true` stamps only runs lacking `task`; `is_optuna=false` with no marker; runs already carrying a tag are untouched; deleted runs are reported not written; dry run by default; the `lr_scheduler` invocation still produces its previous output.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Verify tests pass; dry-run against `mlflow.db`, then apply**

`uv run --python 3.10 python scripts/backfill_run_tags.py --tag task --value classification --marker task_backfilled`
`uv run --python 3.10 python scripts/backfill_run_tags.py --tag is_optuna --value false`
then each with `--apply`. Expect 251 runs each. Record the applied date in ADR 0004's status.

- [ ] **Step 5: Commit**

## Task 10: Optuna

**Files:**

- Modify: `opt.py`, `tests/unit/test_opt_tracking.py`

- [ ] **Step 1: Write failing tests**

`define_search_space(trial, task="imputation")` samples the decode keys and `LAMBDA_NUM`, never `EPOCH_FINE`/`LR_FINE`/`WEIGHT_DECAY_FINE`, never either evaluation rate; every sampled `(DIM, HEADS)` satisfies `DIM % HEADS == 0`; a failing trial raises `optuna.TrialPruned` and still sets `trial_status=failed`; best-tracking improves on a *lower* score under the imputation task; the study parent carries `task` and `is_optuna=true`; `save_best_params` under the imputation task writes only inside the study directory and leaves `datasets/hiperparams/` untouched; `study.direction` follows the task.

- [ ] **Step 2: Verify they fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

## Task 11: Regression fixture, documentation, complete verification

**Files:**

- Create: `tests/fixtures/credit-g_20nan_imputation_regression.json`, `tests/integration/test_credit_g_imputation_regression.py`
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`, `CONTEXT.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Generate the fixture**

Configuration from ADR decision 14 (`DIM` 16, `HIDDEN_DIM` 8, `HEADS` 4, `LAYERS` 1, `DIM_FEED` 16, `DROPOUT` 0.1, `EPOCHS_PRE` 2, `EPOCHS_DECODE` 2, `BATCH` 64, `LAMBDA_NUM` 1.0, `EVAL_MASK_RATE` 0.2, `cv_folds` 2, seed 42, tracking disabled). Run three times, record `observed_max_deviation`, set `tolerance` well above it (the existing fixture uses 0.01 over an observed 0.0), and record the environment block. Pin `impute_score`, `rmse_num_z`, `acc_cat` for both populations, per fold and mean. Put the regeneration command in the test docstring.

- [ ] **Step 2: Write the integration test** (`@pytest.mark.integration`, same shape as the vehicle test)

- [ ] **Step 3: Update documentation**

README: `--task`, `--score_null_path`, the six new keys with the `PROB_MASCARA` / `EVAL_MASK_RATE` pairing, the `task` and `is_optuna` filters in "MLflow Cross-Validation Comparisons", the artifact inventory, the `--lr_scheduler cosine` recommendation. ARCHITECTURE: a "TridentDecoder" section with theory and implementation diagrams and the decode stage in the orchestration diagram. AGENTS.md: name both fixtures. CONTEXT.md: confirm the imputation entries. BACKLOG: mark B3 fixed via ADR 0004.

- [ ] **Step 4: Run complete verification**

`uv run --python 3.10 pytest -m "not integration" -q` and `uv run --python 3.10 pytest -m integration -q`; run `graphify update .`.

- [ ] **Step 5: Review specification coverage**

Walk ADR 0004's fourteen decisions and confirm each has a test or a documented manual check.

- [ ] **Step 6: Commit**
