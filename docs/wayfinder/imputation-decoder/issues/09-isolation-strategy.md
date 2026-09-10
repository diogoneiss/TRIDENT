# 09. How does the training package branch per task without touching classification?

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 04, 05

## Question

Fix the seams so the classification path is provably unchanged:

- **Request**: `TrainingRequest.task` (and `RuntimeOptions` / `Hyperparameters` fields)
  defaulting to classification.
- **Stages**: a new `src/training/decoding.py` with `train_and_evaluate_decoder`
  alongside `finetuning.py`; `runner.py` calls one or the other after the shared
  `train_pretrainer`; a `DecodingOutcome` type mirroring `FinetuningOutcome`.
- **Metrics plumbing**: `FoldResult.metrics` is task-specific; `summary.py`,
  `artifacts.py` and `opt.py` take the fold-ranking metric (and its direction) from the
  task instead of hardcoding `f1_macro`; `_LOSS_KEYS` and `TIMING_METRIC_KEYS` gain the
  names of the decode stage without renaming the existing ones.
- **Reproducibility proof**: the `vehicle_00nan` integration fixture passes unchanged;
  a unit test asserts that a classification request never constructs a decoder; the
  seeded draw sequence of the classification path is untouched.
- **Docs**: which sections of `docs/ARCHITECTURE.md`, `README.md` and `CONTEXT.md` the
  plan must update.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted, with the
decode outcome's shape revised during the discussion (see decision 4).

**Facts verified while resolving this ticket:**

- `TrainingRequest` has seven fields and **none is defaulted**, so a `task` field must
  carry a default and sit last, or every construction site breaks -- including
  `tests/integration/test_vehicle_regression.py`, which `AGENTS.md` protects.
- `_diagnostic_roles` (`summary.py:207-217`) computes the worst fold as a `min` over the
  metric and the best as a `min` over its **negation**. A lower-is-better metric
  therefore **silently swaps best and worst** rather than failing.
- `artifacts.py` embeds the metric name as a literal **JSON key**, not merely as a
  lookup: `"f1_macro_ranking"` at line 116 and `"f1_macro"` inside `selected_folds` at
  line 129. The manifest's schema is itself metric-named.
- Twelve literal `f1_macro` references plus one `direction="maximize"` constant span four
  modules: `main.py` (5, ticket 08's), `opt.py` (2), `summary.py` (3), `artifacts.py` (3).

**Decisions:**

1. **The task abstraction follows the house pattern**, which this codebase already uses
   twice: a names tuple, validation in the frozen dataclass's `__post_init__`, and a
   factory dispatching with an `if`/`elif` chain -- exactly how `create_tracker` and
   `StageScheduler` work. So a frozen `TaskSpec` holding the task name, its ranking metric
   and that metric's direction, held in a small registry, with `runner.py` dispatching the
   stage call the same way. **No polymorphic strategy objects**; this codebase uses them
   nowhere.

2. **The spec reaches the other modules as a defaulted argument.** `summarize_cross_validation`,
   `write_cv_tracking_artifacts` and `opt.py` gain a parameter defaulting to the
   classification spec, so every existing call site and unit test compiles and passes
   untouched. `TrainingRequest.task` follows the same rule -- defaulted, and last in field
   order, which is forced anyway. This is the difference between a change that needs
   auditing and one that needs only reading.

3. **The manifest is parameterised by the ranking metric's short name**, not given a
   neutral key. For classification that regenerates `"f1_macro_ranking"` and the
   `"f1_macro"` entries **exactly**, so the artifact stays byte-identical and
   `test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values`
   needs no edit; for imputation it yields `"impute_score_ranking"`. The short name is the
   ranking metric key's last path segment, since the imputation key is
   `impute/masked/impute_score` and slashes make poor JSON keys. A neutral key would be
   tidier in the abstract and would change a reviewed artifact for no benefit.

4. **`DecodingOutcome` mirrors `FinetuningOutcome` plus the scored-cell table:**

   ```python
   @dataclass(frozen=True)
   class DecodingOutcome:
       model: Any
       result: FoldResult
       train_losses: Sequence[float]
       validation_losses: Sequence[float]
       scored_cells: Any   # one row per scored cell: row, column, type,
                           # population, actual, imputed, confidence, error
   ```

   No training stage in this codebase touches the filesystem; `ArtifactWriter` produces
   every file. Keeping that means the decode stage returns data and stays testable without
   a filesystem, and every rendering decision lives beside the other rendering code.

   **Revised during the round:** the outcome does **not** carry the sampled row
   identifiers. Both files of ticket 07 must agree on which rows were sampled -- the
   preview shows them, the cell file flags them with `in_preview` -- so whoever samples
   must write both. `ArtifactWriter` writes both, so it draws the sample itself from the
   run seed and fold ordinal the runner passes in. One fewer field, and the two files
   cannot disagree.

   Size: the widest case is `electricity`, whose three-fold test split is roughly 15k rows
   over 8 columns, giving a table in the tens of thousands of rows -- a few megabytes, held
   for one fold.

   **`PreparedDataset` gains an optional scaler field, defaulted to absent.**
   `prepare_dataset` currently fits `StandardScaler`, applies it, and lets it fall out of
   scope, so nothing downstream can render original units. Retaining the reference does
   **not** move the split-and-scale order `AGENTS.md` protects: the scaler is still fit on
   the whole frame before any fold exists and applied at the same moment. The categorical
   direction needs nothing new, since the embedder already owns the `LabelEncoder`s.

5. **Proof that classification is untouched**, three parts:
   - `tests/integration/test_vehicle_regression.py` passes with **no edit at all**, which
     the defaulted `task` field guarantees. This is the numeric proof.
   - A unit test asserting a classification request never constructs decoder heads, so the
     new path cannot be entered by accident.
   - Unit tests for the new metric functions, including the zero-baseline guard from
     ticket 05, since that one divides.

**New module layout**: `src/training/decoding.py` holding `train_and_evaluate_decoder`,
beside `finetuning.py`; `runner.py` calls one or the other after the shared
`train_pretrainer`. `summary._LOSS_KEYS` gains the decode stage's per-epoch loss keys and
`TIMING_METRIC_KEYS` its stage timing, without renaming the existing ones.

**Docs**: `docs/ARCHITECTURE.md` gains a decode-stage section and diagram in both its
theory and implementation flavours; `README.md` gains the flag, schema and MLflow updates
already listed on tickets 06 and 08; `CONTEXT.md`'s **Fold-ranking metric** entry was
generalised as part of this resolution to name the metric and direction per task, since it
previously asserted the metric was `f1_macro` outright.

Consequence: unblocks ticket 14.

## Comments

- 2026-09-10 (ticket 12): the phrase "last in field order" in decision 2 is corrected. The
  constraint is that defaulted fields follow undefaulted ones; `TrainingRequest` now ends
  with two defaulted fields, `task` and `score_null_path`, whose order between themselves
  is free.
