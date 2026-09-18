# Mirror Experiments: Implementation Plan

> **How to execute:** work the slices in order with `mattpocock-skills:tdd`; each is one
> or more red-green cycles at the seams named below. [ADR 0006](../../adr/0006-mirror-experiments-per-task.md)
> is the specification; the wayfinder map that produced this plan is [`map.md`](map.md).
> Steps use checkbox syntax for tracking.

**Goal:** every run in every experiment family has a mirror run in `TRIDENT/mirror/<task>`,
nesting preserved, written live when a root run closes and by an idempotent script that
backfills the history and repairs gaps; every run carries the dense `is_mirror` tag.

**Architecture:** one function, `mirror_run_tree`, in a new strict-mypy module
`src/training/mirroring.py`, copies a root run and everything nested under it, parents
before children, through the public `MlflowClient` API only. The trainer and the script
both call it, so the two paths cannot diverge. The live hook sits after the
`with mlflow.start_run(...)` block in `MlflowTracker.parent_run` and after the study
block in `opt.py`, and is wrapped so a mirroring failure is a warning, never a failed
training. The script scans the store, stamps `is_mirror=false` on untagged sources,
mirrors every root that is not `RUNNING`, and deletes mirrors whose source is gone.

## Global constraints

- Python 3.11 through `uv run --python 3.11`. The unit suite type-checks the tree; the
  new module is strict, and `src/mlflow_utils.py` stays on its ramp block unchanged in
  width (the new functions there are fully annotated).
- Classification training behaviour untouched; `tests/integration/test_vehicle_regression.py`
  and `tests/integration/test_credit_g_imputation_regression.py` pass unedited (both run
  with tracking disabled, so mirroring never reaches them).
- Every new field and flag defaults to the ADR's decided behaviour: `mirror_runs=True`,
  `--disable_mirror` absent. Tracking disabled implies mirroring disabled.
- Real runs go to a scratch `MLFLOW_TRACKING_URI=sqlite:///C:/...` and scratch
  `--output_dir` / `--metrics_dir`; their `mlruns/` folders are removed afterwards. The
  shared `mlflow.db` is touched only by the backfill, after a copy of the file is taken
  to the scratchpad.
- Preserve unrelated worktree changes.

## Seams under test, and why each earns its place

A seam is kept only when it catches a silent-wrongness risk; the rest is cut.

| Seam | Silent wrongness it catches |
|---|---|
| `mirror_run_tree` on a temp store: a parent with two diagnostic children built by the real `MlflowTracker`, then params, tags, every metric history, inputs, status, start/end times compared field by field, `mlflow.parentRunId` remapped, `is_mirror=true` and `source_run_id` present | A mirror that looks right in the UI but differs from its source in a value or a series, which would be read as a real number |
| Idempotency: `mirror_run_tree` twice yields the same run count and the same metric-history lengths | Duplicate runs or duplicate metric rows on every sync, which the store would accept without a word |
| `sync_store`: `RUNNING` skipped, `FAILED` mirrored with its status, deleted source deletes its mirror, hand-deleted mirror is restored, `TRIDENT/mirror/*` never a source, `is_mirror=false` stamped on untagged active sources, dry run writes nothing | Stale or orphaned mirrors, a mirror of a mirror, and a "dry run" that writes |
| Dense tag: the two existing every-run-kind tests assert `is_mirror == "false"` | The exact gap the dense-tag rule exists to prevent: one run kind without the tag cannot be excluded |
| Live hook: after `MlflowTracker.parent_run` a mirror exists by default, none with `mirror_runs=False`, none when the body raised, and a mirror function that raises leaves the run finished and the caller unharmed | A mirror failure losing a six-hour run, or the flag not doing what it says |
| `opt.py`: after a study, the study parent and its trials have mirrors; none under `--disable_mlflow` or `--disable_mirror` | The study path silently skipping the mirror while the trainer path does it |
| Config: `--disable_mirror` resolves to `mirror_runs=False`; the default resolves to `True` | The flag parsed but never reaching the tracker |

Cut: a test that the mirror experiment name is `TRIDENT/mirror/<task>` on its own (the
tree test reads it), any test of `MlflowClient` behaviour the store already guarantees
(dedup is asserted through the idempotency seam, not re-tested), and artifact-related
tests (nothing is copied).

## Slices

- [x] **1. Mirror experiment helper and tag names.** `IS_MIRROR_TAG`, `SOURCE_RUN_ID_TAG`,
  `MIRROR_EXPERIMENT_PREFIX`, `get_or_create_mirror_experiment(task)`,
  `is_mirror_experiment(name)` in `src/mlflow_utils.py`. Covered by the tree test.
- [x] **2. Dense `is_mirror=false`.** Extend the two every-run-kind tests; add the tag in
  `execution_tags` and `_log_diagnostic_children`.
- [x] **3. `mirror_run_tree`.** `src/training/mirroring.py`: copy a root and its nested
  runs through `create_run(start_time=…)`, `log_batch` (1000 metrics / 100 params /
  100 tags and 1000 entries per call), `log_inputs`, `set_terminated(status, end_time)`;
  `mlflow.runName` stripped from the copied tags and passed as `run_name`;
  `mlflow.parentRunId` remapped. Mypy gate after this slice.
- [x] **4. Idempotency and update.** Second call reuses the mirror found by
  `source_run_id`, re-sets tags, removes tags gone from the source (its own two tags and
  `mlflow.parentRunId` excepted), reports a differing param instead of forcing it,
  restores a deleted mirror whose source is active.
- [x] **5. `sync_store`.** Scan every non-mirror experiment, build the
  `source_run_id → mirror` index once per mirror experiment, stamp `is_mirror=false`,
  mirror every non-`RUNNING` root, delete orphaned mirrors; `MirrorReport` with created /
  updated / deleted / skipped / unmirrored / param conflicts; `apply=False` writes nothing.
- [x] **6. Live hook and flag.** `RuntimeOptions.mirror_runs: bool = True`,
  `--disable_mirror`, `resolve_training_request`, `create_tracker(..., mirror_runs)`,
  `runner.py`; the call after the `with` block in `MlflowTracker.parent_run`, wrapped in
  `mirror_root_safely` (warning on failure). `tracking.py` gains `import logging`.
- [x] **7. Study hook.** `opt.py` mirrors `study_run_id` after the study block, guarded by
  `disable_mlflow` and `disable_mirror`.
- [x] **8. `scripts/mirror_runs.py`.** Dry run by default, `--apply`, `--tracking_uri`,
  `--task`, `--experiment`; prints the report per experiment.
- [x] **9. Real run.** A short `--task imputation` training against a scratch store shows
  the mirror run in `TRIDENT/mirror/imputation` with the same metrics as its source; the
  script's dry run against the real store reports 1046 sources and the expected roots;
  then the backfill with `--apply` after copying `mlflow.db` to the scratchpad.
  _Done 2026-09-17: credit-g_20nan, 2 folds, 2+2 epochs, 139 + 24 + 24 metric keys equal
  series for series; dry run 1044 to create / 1046 to stamp / 1 RUNNING skipped in 17 s;
  apply 3 min 19 s, 200.9 MB to 405.6 MB, 2090 runs; parity check zero mismatches over
  1044 pairs; second dry run 0 created / 1044 refreshed in 30 s. Outcome in ADR 0006._
- [x] **10. Docs.** `CLAUDE.md` filter, README "MLflow Cross-Validation Comparisons",
  ADR 0005 "How to compare", ADR 0006 status → Accepted with any deviation noted;
  `graphify update .`.
