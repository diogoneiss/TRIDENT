# Mirror every run into one derived MLflow experiment per task

## Status

Accepted (2026-09-17). Decided in a grilling session on 2026-09-17; the glossary terms
*mirror experiment* and *mirror run* were recorded in `CONTEXT.md` as the decisions
settled, and the implementation landed the same day (map and plan under
[`docs/wayfinder/mirror-experiments/`](../wayfinder/mirror-experiments/map.md)). Two
details differ from the text below as first written, both stated where they apply: the
flag's programmatic home is `RuntimeOptions.mirror_runs` (beside `tracking_enabled`),
and the dense tag has a third creation site.

Extends [ADR 0002](0002-curated-cross-validation-mlflow-runs.md), which keeps every
variant of a base dataset in `TRIDENT/<base_dataset>`; that layout is untouched and
stays the source of truth. Adds a filter clause to the comparison recipes of ADR 0002,
[ADR 0003](0003-selectable-learning-rate-schedule.md) and
[ADR 0005](0005-reduced-optuna-search-for-imputation.md).

### Outcome (2026-09-17)

The backfill ran the same evening against the real store, after a copy of `mlflow.db`
was taken to the session's scratchpad (session-local, so the rollback window is that
session; the copy was byte-identical to the 200,900,608-byte file). Dry run first:
1044 mirror runs to create, 1046 sources to stamp, one `RUNNING` root skipped with its
child (the census's two zombies), 17 s. `--apply`: 1044 created, 1046 stamped, 3 min 19 s;
the file grew from 200.9 MB to 405.6 MB and the store holds 2090 runs, 251 in
`TRIDENT/mirror/classification` and 793 in `TRIDENT/mirror/imputation`. A read-only
parity check over all 1044 pairs found zero mismatches in metric rows, latest metrics,
params, inputs, tags, nesting, status, timestamps and names. A second dry run reported
0 created, 1044 refreshed, 0 stamped in 30 s: the sync is idempotent on real data, and
30 s is the cost of a repair pass. Before that, a short real `--task imputation` training
against a scratch store mirrored its tree live: 139 metric keys on the parent and 24 per
diagnostic child matched series for series. Both regression fixtures passed unedited.

## Context

Every training run lives in its experiment family, so reading one task across datasets
means multi-selecting experiments in the MLflow UI or calling
`mlflow.search_runs(search_all_experiments=True)`. Both work today and both are how the
2026-09-15 addendum was computed. The user wants a persistent experiment per task in
which every dataset lands side by side, without giving up the per-dataset families.

Facts verified while deciding, against a read-only `mlflow.db` and the installed MLflow
3.14.0:

- **Census, 2026-09-17.** Nine families and 1046 runs: 123 comparison parents, 212
  diagnostic fold runs, 18 study parents, 681 Optuna trials, and 12 runs from July and
  August 2026 in `TRIDENT/vehicle` that predate ADR 0002 and carry no `run_role`
  (three `train_vehicle_00nan_*` roots with `fold_N_of_3` children). 603,659 metric
  rows, of which the parents hold 295,095 and the fold runs 290,004; 30,611 latest
  metrics; 17,588 params; 25,301 tags; 115 dataset inputs, all on parents; no logged
  models. Two runs are `RUNNING` (dead processes) and eight `FAILED`. The file is 200 MB.
- **Dense tags.** `dataset`, `task`, `is_optuna` and `lr_scheduler` are on all 1046
  runs. `base_dataset`, `seed` and `cv_folds` are on 825: no diagnostic fold run carries
  them. Run names already spell task and dataset (`impute_credit-g_20nan_<timestamp>`,
  `train_vehicle_00nan_<timestamp>`).
- **A run belongs to exactly one experiment** and MLflow has no API to move or copy
  one. Mirroring is creating a second run and replaying the first into it:
  `MlflowClient.create_run(experiment_id, start_time, tags, run_name)`,
  `log_batch` (at most 1000 metrics, 100 params and 100 tags per call, metric
  timestamps and steps preserved), `log_inputs` with a `Dataset` rebuilt from the
  source's `name`, `digest`, `source_type`, `source`, `schema` and `profile`, and
  `set_terminated(run_id, status, end_time)`.
- **Replay is idempotent at the store.** The SQLAlchemy store drops a replayed metric
  whose key, value, timestamp and step already exist (`_log_metrics` recovers from the
  primary-key collision by inserting only the missing rows) and treats a replayed param
  with the same value as a no-op; a param with a *different* value raises. Tags are
  overwritten by `set_tag`.
- **The UI collapses nested runs under their parent.** A mirror that preserves nesting
  shows, by default, exactly what a family shows: roots only.
- `get_or_create_experiment` derives the experiment name with `dataset_name.split("_")[0]`,
  so it cannot name a mirror experiment.
- The UI's `compare-experiments/:searchString` route exists in the installed bundle.

## Decision

1. **Two mirror experiments, one per task, named `TRIDENT/mirror/<task>`** with
   `<task>` the literal value of the `task` tag (`TRIDENT/mirror/classification`,
   `TRIDENT/mirror/imputation`). A new helper `get_or_create_mirror_experiment(task)` in
   `src/mlflow_utils.py`, beside `get_or_create_experiment` and independent of it. A
   mirror experiment is never a run's home: nothing trains into it. One per task rather
   than per task and schedule, because `lr_scheduler` is dense and filters.

2. **Every run is mirrored, with nesting preserved.** Comparison parents, diagnostic
   fold runs, study parents, trials and the 12 legacy `vehicle` runs alike: the
   definition is *every run of the family*, with no exception list. A mirror run
   carries the source's params, tags, complete metric histories, dataset inputs, run
   name, status, start and end times. `mlflow.parentRunId` is the only tag rewritten:
   it points at the mirror run of the source's parent, so the UI nests mirrors as it
   nests sources. Every other `mlflow.*` tag (`runName`, `user`, `source.name`,
   `source.git.commit`) is copied verbatim, because it describes the training that
   happened. Artifacts are not copied; a mirror run's artifact root stays empty and the
   source's `source_run_id` leads to them.

3. **Identity through two tags.** `source_run_id` on every mirror run is its unique
   key and the way back. `is_mirror` is **dense**: `true` on every mirror run, `false`
   on every other run, stamped wherever a run's tags are built (`execution_tags`, which
   serves the parent, the trial's runner-side tags and the study parent;
   `_log_diagnostic_children`; and the trial tags `opt.py` pre-stamps before the runner
   opens, so a trial that crashes early still carries them) and backfilled as `false`
   onto the 1046 existing runs by the mirror script itself, which visits every source
   anyway. It is dense for the reason `task` and `is_optuna` are: a copied
   `run_role = parent` is indistinguishable from its source in any cross-experiment
   query, and a sparse tag would not exclude it. Every documented comparison filter
   gains `tags.is_mirror = 'false'`.

4. **Two writers, one function.** The same function mirrors a run tree whether the
   trainer or the script calls it, so the two paths produce identical mirror runs.
   - *Live.* The trainer mirrors when a **root** run closes: `MlflowTracker.parent_run`
     on exit (the parent and its diagnostic children are all logged by then) and the
     study run in `opt.py` on exit (the study parent and its trials). A nested run never
     triggers a mirror on its own; its root does, which is the only order in which the
     parent already has a mirror when the child's `parentRunId` is rewritten. A
     `--retrain_best` parent runs after the study closes and is a root of its own. The
     live path runs only when the source finished normally; a failed or interrupted run
     is left to the script, so the trainer's failure path is not touched and a
     `Ctrl+C` is not followed by a replay. Mirroring **never fails the training**: an
     exception is logged as a warning naming the source run id and the run stays intact
     in its family, unmirrored, for the script to repair.
   - *Flag.* `--disable_mirror` on the shared parser and `RuntimeOptions.mirror_runs:
     bool = True` (reached as `request.runtime.mirror_runs`, beside `tracking_enabled`),
     so `opt.py` and the tests control it without the CLI.
     `--disable_mlflow` implies no mirror. **Default on is the user's explicit
     decision**, a deliberate exception to the rule that old behaviour stays the
     default; training numbers do not change, only what the store contains.
   - *Script.* `scripts/mirror_runs.py`, in the shape of `backfill_run_tags.py`: a dry
     run by default, `--apply` to write, honours `MLFLOW_TRACKING_URI`, optional
     `--task` and `--experiment` scoping, and a report counting mirror runs created,
     updated, deleted and skipped, plus the roots still unmirrored. It covers the
     history and repairs whatever the live path missed. The source set is every
     experiment whose name is not `TRIDENT/mirror/*` and every run whose `is_mirror` is
     not `true`, so a mirror is never mirrored again.

5. **Sync is an upsert by replay.** For a source with an existing mirror (found by
   `source_run_id`), tags are re-set and tags that vanished from the source are removed
   from the mirror (its own two tags excepted), params are replayed (a differing value
   is reported, never forced, since a source never rewrites its params), metric
   histories and inputs are replayed and deduplicated by the store, and status and end
   time are re-applied. A `RUNNING` source is skipped. A `FAILED` source is mirrored
   with its status. A source whose `lifecycle_stage` is `deleted`, or that no longer
   exists, has its mirror deleted; a mirror deleted by hand is recreated. The script
   replays everything every time rather than detecting change; idempotency comes from
   the store, and the report is the evidence.

6. **Documentation.** The "MLflow run analysis" rule in `CLAUDE.md`, the "MLflow
   Cross-Validation Comparisons" section of the README and the "How to compare"
   sections of ADR 0005 gain the `is_mirror = false` clause; this ADR carries the
   mirror's own recipe below.

## How to compare

Open `TRIDENT/mirror/<task>`, or `search_runs(experiment_names=["TRIDENT/mirror/<task>"])`.
Filter `tags.run_role = 'parent'` and `tags.is_optuna = 'false'`, and compare within
one `tags.lr_scheduler`, exactly as in a family; `tags.base_dataset` and
`tags.dataset_variant` group the rows. The mirror never holds a number its family lacks:
cite the source run (`tags.source_run_id`) and read artifacts there. Any query that
spans experiments (`search_all_experiments=True`) must add `tags.is_mirror = 'false'`
or every source run is counted twice.

## Considered options

- **No mirror: multi-select experiments in the UI, or `search_all_experiments`.**
  Recommended as the cheapest thing that compares across datasets; rejected by the user,
  who wants a persistent per-task table in the store.
- **One experiment for everything, moving runs with `UPDATE runs.experiment_id` on the
  SQLite file.** Rejected: it redefines the experiment family, rewrites every documented
  query and bypasses MLflow's API.
- **Mirror only the comparison parents.** Recommended first on the grounds that they
  are the one layer with aligned columns; rejected by the user: the mirror aggregates
  nothing, each run stays a distinct row, nested runs stay collapsed, and *every run* is
  a simpler rule than a list of roles.
- **One mirror per task and schedule.** Rejected: `lr_scheduler` is dense and filters.
- **Script only, no live path.** Recommended, one mechanism for history and future
  alike; rejected by the user, who wants runs mirrored as they happen, with the script
  as the repair and the backfill.
- **Mirroring failure fails the training.** Rejected: the last step of a six-hour run
  must not be able to lose it.
- **Copy artifacts.** Rejected: size, and `source_run_id` leads to them.
- **A sparse `is_mirror`, or relying on experiment scoping alone.** Rejected: every
  existing cross-experiment recipe would double-count silently.

## Consequences

- The store roughly doubles: 2090 runs after the backfill, about 1.2 million metric
  rows, a 405.6 MB file. The default UI view of any experiment is unchanged, because
  nested runs stay collapsed.
- Classification training behaviour is untouched and both regression fixtures stay
  unedited: mirroring happens after the numbers exist and copies them.
- New: `TRIDENT/mirror/<task>`, `get_or_create_mirror_experiment`, the dense
  `is_mirror` tag and `source_run_id`, `--disable_mirror`, `RuntimeOptions.mirror_runs`,
  `src/training/mirroring.py` (`mirror_run_tree`, `sync_store`), the trainer's
  `mirror_root_safely`, and `scripts/mirror_runs.py`.
- The unit tests that assert every run kind carries `task` and `is_optuna`
  (`test_every_run_says_which_task_it_trained_and_whether_a_search_made_it` and its
  `opt.py` twin) are the seam for `is_mirror`; tracking tests on a real temporary store
  will observe a second experiment once the default is on.
- Known limits: a tag removed from a source is only removed from its mirror by the
  script, not live; the two `RUNNING` zombies stay unmirrored until they are cleaned up;
  a *deleted* root loses its mirror but its still-active children are never visited (the
  walk stops at the deleted parent), so their mirrors stay, collapsed under a deleted
  mirror parent, until the children are deleted at the source too; cross-dataset
  comparison of raw scores is still the reader's responsibility, the mirror only puts
  the rows in one table.
- Out of scope: experiment-level tags, artifacts, normalising metrics across datasets,
  and any mirror of the mirror.
