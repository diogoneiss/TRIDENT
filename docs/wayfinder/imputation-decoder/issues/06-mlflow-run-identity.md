# 06. How are imputation runs identified in MLflow?

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: none

## Question

The user suggested a `_decode` suffix. Decide the run identity so classification and
imputation runs never mix in a comparison table, and so existing runs stay comparable:

- **Tag**: a new `task` tag (`classification` | `imputation`), with a backfill script
  stamping `task = classification` (+ `task_backfilled = true`) on every existing run,
  the ADR 0003 pattern. Alternative: overload the existing `run_type` tag.
- **Experiment**: same `TRIDENT/<base>` experiment (filter by tag) vs a separate
  `TRIDENT/<base>_decode` experiment.
- **Run name**: `decode_<dataset>_<timestamp>` vs `train_<dataset>_<timestamp>_decode`.
- **Where it applies**: parent runs, diagnostic children, Optuna study and trial runs.
- Whether the `README.md` section "MLflow Cross-Validation Comparisons" gains
  "filter by `tags.task`" next to `run_role` and `lr_scheduler`.

Relevant code: `_execution_tags` in `src/training/tracking.py`, `build_run_tags` in
`src/mlflow_utils.py`, `scripts/backfill_lr_scheduler_tag.py` (template for the backfill).

Recommended starting answer: `task` tag with backfill, same experiment, run name
`decode_<dataset>_<timestamp>`.

## Answer

Resolved 2026-09-10 over two grilling rounds. The Optuna tag in decision 4 is the user's
amendment to the recommended answer.

**Store survey taken while resolving this ticket** (read-only query against `mlflow.db`):

| Fact | Value |
|---|---|
| Active runs, across 9 `TRIDENT/<base>` experiments | 251 |
| Parents / diagnostic children / legacy runs without a role | 87 / 152 / 12 |
| Runs carrying `dataset` and `lr_scheduler` | 251 each |
| Runs carrying `run_role` | 239 |
| Runs carrying `run_type` | 90 |
| Runs carrying environment tags (`device`, ...) | 2 |
| Optuna runs of any kind | 0 |

The store's tags come in two shapes, and the distinction drives every decision below.
**Dense enums** (`run_role`, `run_type`, `lr_scheduler`, `dataset`) sit on runs and hold a
value from a fixed set; they can be filtered on. **Sparse markers** (`best_trial`,
`lr_scheduler_backfilled`) are written as the string `"true"` or are simply absent; there
is no `"false"` to match, so they can only answer "show me the true ones". MLflow tag
values are always strings, so no tag is a real boolean.

**Decisions:**

1. **A new dense `task` tag**, values `classification` and `imputation`, rather than
   overloading `run_type`. Run type describes the kind of execution (train, optuna study,
   optuna trial) and is orthogonal to the task; it also sits on only 90 of 251 runs, so
   every diagnostic child would go unlabelled. This follows the `lr_scheduler` precedent
   from ADR 0003, which reached all 251 runs.
2. **Same experiment.** Imputation runs stay in `TRIDENT/<base_dataset>` and are separated
   by tag, consistent with ADR 0002's one-experiment-per-base-dataset rule and its
   filter-first comparison workflow. Recorded cost: the two tasks report disjoint metric
   keys, so an unfiltered table shows many empty columns. Splitting would have doubled the
   experiment list to 18 and cut each dataset's history in half.
3. **Naming.** Tag value `imputation`; run name **prefixed**, giving
   `impute_<dataset>_<timestamp>` beside today's `train_<dataset>_<timestamp>`. A prefix
   is what a human reads first in a run list and sorts task-alike runs together; the
   originally suggested `_decode` suffix would hide behind the timestamp.
4. **Scope: every run kind** carries `task` -- parents, both diagnostic children, and the
   Optuna study and trial runs. A tag present on only some run kinds cannot be filtered
   safely.

   **Plus a second new dense tag, `is_optuna`**, holding the string `"true"` or `"false"`
   on every run, so Optuna runs can be filtered out of comparison tables in future. It is
   `"true"` on Optuna study parents and trial runs only. A `--retrain_best` run stays
   `"false"`: it is a full comparison parent that happens to be seeded from a study, and
   it already carries `optuna_study_run_id` to preserve that provenance. Filtering
   `is_optuna = 'false'` therefore means "runs meant for comparison".

   Note the deliberate contrast with `best_trial`, which is a sparse marker. A dense
   boolean distinguishes "not Optuna" from "recorded before the tag existed"; a sparse one
   cannot.
5. **Backfill: generalise `scripts/backfill_lr_scheduler_tag.py`** rather than copy it.
   Only its two tag-key constants are hardcoded; the run traversal, dry-run default,
   idempotency and deleted-run handling are already reviewed and covered by
   `tests/unit/test_backfill_lr_scheduler_tag.py`. The existing invocation must keep
   working so the applied schedule backfill stays reproducible. The backfill stamps all
   251 runs with:

   - `task = classification` **and** `task_backfilled = true`. The marker is warranted:
     nothing on an old run records which task it trained, so the value is inferred from
     outside knowledge (the imputation task did not exist).
   - `is_optuna = false`, with **no** marker. Unlike the task value, this one is provable
     from the store itself: no run carries an Optuna role or type, and Optuna with
     tracking enabled crashed on every trial until the fix recorded in backlog item B4, so
     no trial could have been recorded. A reader can re-derive it, so a marker would carry
     no information. Evidence recorded here rather than in a tag.

**Wiring the two tags** (detail for tickets 09 and 10). Three call sites set tags today
and all three need both new tags:

- `_execution_tags` in `src/training/tracking.py`, which covers parent runs and, through
  `OptunaTrialTracker`, the trial runs.
- `_log_diagnostic_children` in the same module, which builds child tags separately from
  `build_fold_tags` plus a manual update.
- The study parent's `parent_tags` in `opt.py`.

Stated rather than asked: README's "MLflow Cross-Validation Comparisons" section gains
the `task` filter beside `run_role` and `lr_scheduler`, and documents `is_optuna`.

Consequence for ticket 13: an Optuna study for the imputation task produces runs tagged
`task = imputation` and `is_optuna = true`, so its trials are filterable on both axes.

## Comments
