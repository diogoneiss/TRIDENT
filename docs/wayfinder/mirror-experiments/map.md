# Map: Mirror every run into one derived MLflow experiment per task

Label: wayfinder:map
Charted: 2026-09-17
**Status: destination reached 2026-09-17, in the grilling session that charted it;
execution landed and the backfill was applied the same day (plan, all ten slices ticked).**
Every decision was settled on the spot and recorded in
[ADR 0006](../../adr/0006-mirror-experiments-per-task.md); the glossary terms *mirror
experiment* and *mirror run* are in `CONTEXT.md`. No decision ticket exists because the
ADR holds every decision and its reasoning. The execution is the
[implementation plan](plan.md) beside this map.

Tracker: local markdown, as `AGENTS.md` requires. This effort lives at
`docs/wayfinder/mirror-experiments/`.

## Destination

Every run in every `TRIDENT/<base_dataset>` experiment family has a mirror run in
`TRIDENT/mirror/<task>`, nesting preserved, written live by the trainer when a root run
closes (default on, `--disable_mirror` to opt out) and by an idempotent
`scripts/mirror_runs.py` that backfills the 1046 existing runs and repairs whatever the
live path missed. Every run carries the dense `is_mirror` tag, and every documented
comparison recipe filters on it.

## Notes

- **Domain**: TRIDENT. Read `CONTEXT.md` (*experiment family*, *mirror experiment*,
  *mirror run*, *diagnostic fold run*, *Optuna trial run*), ADR 0002 (the family layout
  the mirror reflects) and ADR 0006 (the specification). Code: `src/mlflow_utils.py`,
  `src/training/tracking.py`, `src/training/types.py`, `src/training/config.py`,
  `src/training/runner.py`, `opt.py`, `scripts/backfill_run_tags.py` (the script's
  mould). For codebase questions run `graphify query "<question>"` first.
- **Skills**: `mattpocock-skills:tdd` for every slice, followed by the real run the plan
  names.
- **Standing preferences**: classification training behaviour untouched and both
  regression fixtures pass unedited; the mirror default is **on** by the user's explicit
  decision (ADR 0006, decision 4), the one exception to "old behaviour stays default";
  `uv run --python 3.11 ...`; the shared `mlflow.db` is touched only by the backfill,
  after a copy of the file is taken.
- **Facts established while charting** (verified 2026-09-17 against `mlflow.db` and
  MLflow 3.14.0; all in the ADR's Context): the census (1046 runs, 603,659 metric rows),
  the client API used for replay, the store's idempotency on replayed metrics, params
  and inputs, the UI collapsing nested runs, `get_or_create_experiment` being unusable
  for the mirror name, and the `compare-experiments` UI route existing.
- **Three code sites stamp the dense tag**: `execution_tags` serves the parent, the
  trial's runner-side tags and the study parent; `_log_diagnostic_children` builds its
  tags by hand; and `opt.py` pre-stamps a trial's tags before the runner opens, so a trial
  that crashes early still carries them (found red-green: the stubbed-trainer test never
  reaches the runner).
- **Live hook ordering**: `MlflowTracker.parent_run` yields inside
  `with mlflow.start_run(...)`; the mirror call goes after that block, so an exception
  through the `yield` skips it and "success path only" costs nothing. `--retrain_best`
  runs after the study's `with` closes (`opt.py`), so it is a root of its own.
