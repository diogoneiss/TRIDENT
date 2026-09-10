# Map: Imputation decoder task

Label: wayfinder:map
Charted: 2026-09-10
**Status: destination reached 2026-09-10.** All fourteen tickets are resolved. The
handoff is [ADR 0004](../../adr/0004-imputation-decoder-task.md) and the
[implementation plan](plan.md); this
map and its tickets remain as the record of how each decision was reached.
Tracker: local markdown. `AGENTS.md` keeps tickets, plans and decisions under
`docs/`, so this effort lives at `docs/wayfinder/imputation-decoder/` instead of the
plugin's default `.scratch/`. Tickets are `issues/NN-<slug>.md`; each carries
`Type:`, `Status:` (`open` / `claimed` / `resolved`) and `Blocked by:` lines. The
frontier is every open, unblocked, unclaimed ticket, lowest number first.

Work it with `/wayfinder docs/wayfinder/imputation-decoder/map.md` (optionally naming a
ticket). One decision per session.

## Destination

A decided design for an **imputation task** in TRIDENT: the model is trained with the
same dynamic-masking technique the classification pipeline already uses, but a
**decoder** reconstructs the actual masked cell values instead of a `[CLS]` class. The
design is recorded as an ADR under `docs/adr/` plus an implementation plan beside this
map at `plan.md`, precise enough that implementation can start without any
further design decision. It covers: the decoder architecture and the stage that trains
it, the loss and the error metrics against the ground truth, the MLflow identity of
imputation runs (tag, run name, experiment, backfill), the CLI and hyperparameter
surface, the human-friendly imputed-vs-actual preview artifact, and the isolation
strategy that keeps every classification run bit-identical to today.

Planning only: this map produces the decisions and the plan, not the implementation.

## Notes

- **Domain**: TRIDENT. Read `README.md` (paper, CLI, config schema),
  `docs/ARCHITECTURE.md` (tensor-level walkthrough) and `CONTEXT.md` (glossary) before
  a ticket. For codebase questions run `graphify query "<question>"` first.
- **Skills each session should call**: `mattpocock-skills:grilling` and
  `mattpocock-skills:domain-modeling` for every grilling ticket;
  `mattpocock-skills:prototype` for the preview ticket; `mattpocock-skills:research`
  for research tickets. Update `CONTEXT.md` inline as terms settle. Terms expected to
  land in the glossary: *imputation task*, *decode stage*, *decoder*, *imputation
  ground truth*, and a per-task *fold-ranking metric* (today the glossary fixes it to
  `f1_macro`).
- **Standing preferences** (from `AGENTS.md` and prior decisions):
  - Classification stays the default; every behaviour change is gated behind a flag,
    carried as an MLflow tag, and backfilled on existing runs so old and new runs stay
    comparable (the ADR 0003 pattern).
  - Do not change classification training behaviour: split/scaling order, scheduler
    cadence, and the seeded random-draw sequence are protected. The `vehicle_00nan`
    integration fixture is the regression baseline and must keep passing unchanged.
  - Use `uv run --python 3.10 ...`; unit tests are `pytest -m "not integration"`.
  - Backlog items C1 (scaler fit before split) and C2 (`"nan"` is a category next to
    `[NULL]`) are protected; the imputation design must work around them, not fix them.
- **Research subagents**: findings live on `research/<name>` branches at
  `docs/research/<name>.md`; read them with
  `git show research/<name>:docs/research/<name>.md`. Two caveats learned while
  charting: agent worktrees are created from an old commit rather than the working
  branch, so a research agent must create its branch from an explicit commit hash; and
  an agent that fans out into further sub-agents trips the API rate limit, so forbid
  sub-agents in research prompts.
- **Facts established while charting** (verified against code and data, so sessions
  need not re-derive them):
  - The `_20nan` / `_40nan` / `_60nan` / `_80nan` CSVs are row-aligned with `_00nan`:
    same shape and columns, every observed cell equal to the `_00nan` value, and
    `_00nan` equals the raw CSV. `datasets/generate_splits.py::inject_nans` NaNs
    ~pct of the rows of each feature column independently (MCAR per column); the label
    column is never NaN. So true imputation ground truth exists for every induced-missing
    cell.
  - `TridentPretrainer` (`src/models.py`) has no decoder: it regresses the transformer
    output at `[MASK]` positions against the detached clean embedding (embedding-space
    MSE). Nothing in the repo maps a token back to a value.
  - `EncodedTable.masked_positions` (`src/embedder.py`) is a `(rows, n_columns)` bool
    matrix in categorical-then-numerical order; `num_values` holds the scaled numerical
    targets; `cat_indices` holds LabelEncoder indices whose vocabulary also contains
    `[MASK]`, `[NULL]` and, for columns with real NaNs, the literal string `"nan"`.
  - `preprocess_table` (`src/utils.py`) never masks an already-null cell, so real-NULL
    cells never enter the current loss, and numerical NaN targets are never selected.
  - `f1_macro` is hardcoded as the fold-ranking metric in `src/training/summary.py`
    (raises if absent), `src/training/artifacts.py` (diagnostic manifest) and `opt.py`
    (objective, `direction="maximize"`).
  - `prepare_dataset` (`src/training/data.py`) discards the fitted `StandardScaler`;
    showing imputed numbers in original units needs it retained (a non-behavioural
    addition to `PreparedDataset`).
  - MLflow identity today: experiment `TRIDENT/<base>`, run name
    `train_<dataset>_<timestamp>`, tags `run_role`, `run_type`, `dataset_variant`,
    `missingness_percent`, `evaluation_mode`, `lr_scheduler`, plus environment tags
    (`src/training/tracking.py::_execution_tags`).

## Decisions so far

<!-- one line per resolved ticket: gist, then the link for detail -->

- [02. Research: masked-cell decoder heads](issues/02-masked-cell-decoder-heads.md):
  the literature decodes with per-column heads (CE over the real categories, MSE on
  standardised scalars) combined by a plain sum with count-based means; three candidate
  designs for TRIDENT (A per-column heads, B tied heads, C MAE-style decoder) are on
  branch `research/masked-cell-decoder-heads`. The note's "d = 4" capacity caveat is a
  misread: training `DIM` defaults to 128.
- [01. Research: imputation evaluation protocol](issues/01-imputation-evaluation-protocol.md):
  the literature scores only artificially masked observed cells against a complete
  table, uses pooled RMSE in a scaled space for numericals (MAE nowhere), error rate or
  macro-F1 for categoricals, no combined scalar, MCAR at 20% or {0.1, 0.3, 0.5, 0.7},
  fixed budgets rather than early stopping. Recommended TRIDENT default (`rmse_num_z`,
  `acc_cat`, a proposed `impute_score` ranking ratio, induced-missing test cells as the
  ground truth) is on branch `research/imputation-eval-protocol`.
- [03. Imputation ground truth](issues/03-imputation-ground-truth.md): score both
  self-masked cells (every variant) and induced-missing cells against the `_00nan`
  sibling (headline where it exists), test fold only; induced-missing cells are fed as
  `[MASK]`; training never reads the sibling; the evaluation mask is one fixed
  `preprocess_table` draw per fold at a new evaluation mask rate defaulting to 0.2;
  validation selects the checkpoint. Three glossary terms added to `CONTEXT.md`.
- [04. Decoder architecture and stage](issues/04-decoder-architecture-and-stage.md): a
  decode stage replaces fine-tuning (pre-training untouched); per-column heads emit only
  the real categories, excluding `[MASK]`, `[NULL]` and the dead `"nan"` entry, with
  targets encoded from the processed frame; numerical heads mirror the embedder's MLP;
  loss is per-type count-averaged CE + `lambda_num` * MSE; encoder fine-tuned; training
  re-rolls masks at `PROB_MASCARA` while evaluation uses the fixed draw; shared
  `lr_scheduler`; best-validation-loss checkpoint; feature-only frame. Hazard recorded:
  never re-run pre-training's Xavier sweep after pre-training.
- [05. Loss, error metrics and fold ranking](issues/05-loss-and-error-metrics.md): report
  pooled `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat` and per-type cell counts for
  both populations; rank folds by `impute_score`, the cell-fraction-weighted sum of each
  error divided by its mean/mode baseline from the training fold, lower better, which
  degrades correctly on the all-numerical and all-categorical datasets; the ranking metric
  and its direction become an explicit per-task property; metrics are namespaced
  `impute/masked/...` and `impute/induced/...`; per-column detail goes to an artifact,
  not to tracked metrics.
- [06. MLflow run identity](issues/06-mlflow-run-identity.md): a new dense `task` tag
  (`classification` / `imputation`) on every run kind, backfilled with a
  `task_backfilled` marker; plus a new dense `is_optuna` boolean, true only on study
  parents and trials, backfilled to `false` with no marker because the store proves the
  value; same `TRIDENT/<base>` experiment, filtered by tag; run names prefixed
  `impute_<dataset>_<timestamp>`; the existing backfill script is generalised, not copied.
- [07. Imputation preview artifact](issues/07-imputation-preview-artifact.md): a
  prototype over real `credit-g` rows chose the row triptych (actual / model saw /
  imputed) as the human preview, a cell-ledger CSV of every scored cell as its
  superset companion, and dropped the column report as duplicated by ticket 05. Fixed-seed
  row sampling, both populations in one preview, original units in the preview and both
  scalings in the CSV, attached where loss plots already attach. Prototype captured on
  branch `prototype/imputation-preview`.
- [08. CLI and hyperparameter surface](issues/08-cli-and-hyperparameter-surface.md):
  `--task {classification,imputation}` defaulting to classification, command-line only and
  never from JSON, logged as a run parameter; five new defaulted keys (`EPOCHS_DECODE`,
  `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `LAMBDA_NUM`, `EVAL_MASK_RATE`) that leave every
  existing config valid; only the parameters a task uses get logged, avoiding backlog C3;
  every flag combination stays legal; the batch summary prints the task's ranking metric.
- [09. Isolation strategy](issues/09-isolation-strategy.md): a frozen `TaskSpec` (name,
  ranking metric, direction) in a registry with `if`/`elif` dispatch, matching the
  `create_tracker` pattern; the spec reaches the summariser, artifact writer and optimiser
  as a **defaulted** argument, and `TrainingRequest.task` is defaulted and last, so the
  protected regression fixture needs no edit; the manifest is keyed by the ranking
  metric's short name, keeping the classification artifact byte-identical; a new
  `src/training/decoding.py` returns a `DecodingOutcome` carrying the scored-cell table
  while `ArtifactWriter` draws the preview sample and writes both files; `PreparedDataset`
  keeps the fitted scaler without moving the split-and-scale order.
- [11. Multi-rate evaluation](issues/11-multi-rate-evaluation.md): sweeping happens within
  one run, for self-masked cells only, because the `_20nan`..`_80nan` ladder already is the
  across-runs sweep for induced-missing cells. One primary rate drives both validation
  checkpoint selection and fold ranking (the rate is *not* evaluation-only, since
  validation uses it); extra rates score the same checkpoint as test-only diagnostics,
  named by integer percent, leaving ticket 05's names untouched. `EVAL_MASK_RATE` stays
  scalar with a separate optional list; no sweep by default.
- [12. Null-path diagnostic](issues/12-null-path-diagnostic.md): shipped behind a
  `--score_null_path` switch, off by default, with a pre-registered criterion for
  overturning ticket 03; metrics take a `null_token` segment and never rank; the cell
  ledger gains a column and the preview is untouched. The score measures whether a readout
  trained on `[MASK]` transfers to `[NULL]`, not whether the null embedding is informative,
  because null cells never enter the decode loss. Amends ticket 08 (one parse-time
  rejection) and corrects ticket 09's field-order wording.
- [13. Optuna for imputation](issues/13-optuna-for-imputation.md): one search space with a
  task switch, swapping the three fine-tuning parameters for the decode ones plus
  `LAMBDA_NUM`, with both evaluation rates never sampled; the failure path raises
  `TrialPruned` instead of returning `0.0`, which under a minimised objective would have
  scored crashes as perfect; `HEADS` constrained to divisors of `DIM` (backlog B3);
  best-tracking takes its comparison from the task direction; and an imputation study
  writes only to its own study directory, never to the shared per-dataset config, which
  would otherwise clobber that dataset's classification hyperparameters.
- [14. Imputation regression fixture](issues/14-imputation-regression-fixture.md): one
  integration fixture on `credit-g_20nan`, the only variant exercising both column types
  and the induced-missing population, pinning `impute_score`, `rmse_num_z` and `acc_cat`
  per population; it copies the existing fixture's pattern exactly, including the
  environment block and a tolerance measured well above observed deviation. Degenerate
  compositions (all-numerical, all-categorical, zero baseline) are unit tests, not extra
  integration runs. `AGENTS.md` gains the second fixture by name.
- [10. Record the ADR and plan](issues/10-record-adr-and-plan.md): the destination.
  `docs/adr/0004-imputation-decoder-task.md` synthesises the fourteen decisions with links
  back to every ticket; `docs/wayfinder/imputation-decoder/plan.md` orders
  eleven TDD tasks so the ranking contract lands before the decode stage; `CONTEXT.md`
  holds six imputation terms and the generalised fold-ranking metric.

## Not yet specified

- **Saved-model shape**: what `--save_model` writes for a decoder run, and how it
  interacts with backlog H3 (whole-object pickling). Adjacent and equally unsharp: whether
  an imputation run should be able to write out a fully imputed copy of a dataset, which
  is the natural product of an imputer but is not part of the destination as drawn.

## Out of scope

- **Impute-then-classify**: feeding the decoder's reconstructions into the classifier
  is a different experiment with its own map.
- **Benchmarking against external imputers** (GAIN, MIWAE, HyperImpute, ...): the
  research ticket borrows their protocol so numbers are comparable later; running them
  is not part of this effort.
- **Fixing backlog C1/C2** (scaler leakage, `"nan"` category): protected behaviour; the
  design routes around them.
- **MLflow logged-model lifecycle**: already deferred to
  [Ticket 0002](../../tickets/0002-mlflow-logged-model-lifecycle.md).
- **Mask-versus-null input for the *classifier***: whether classification scores differ
  when missing cells arrive as `[MASK]` rather than `[NULL]`. This tests the paper's actual
  claim that null embeddings carry predictive signal, needs no decoder, and surfaced while
  resolving [ticket 12](issues/12-null-path-diagnostic.md). A separate experiment with its
  own map.
- **The ablations ticket 04 named**: frozen encoder, tied heads (Design B), MAE-style
  decoder (Design C), label-as-context, and the joint / value-only pre-training
  objectives. These are experiments to run once the task exists, not decisions on the
  route to it; the ADR records them as considered options and future work.
