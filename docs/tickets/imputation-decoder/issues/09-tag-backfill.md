# 09. Tag backfill

Status: ready-for-agent
Blocked by: 08
Plan task: 9. ADR 0004 decision 7. Wayfinder ticket 06.

## Goal

Generalise `scripts/backfill_lr_scheduler_tag.py` into a parameterised backfill and apply
it to the 251 existing runs: `task=classification` with a `task_backfilled=true` marker,
and `is_optuna=false` with no marker.

## Seams under test

- `scripts/backfill_run_tags.py` entry point: `--tag`, `--value`, optional `--marker`,
  dry run by default, `--apply` to write.
- The existing `lr_scheduler` invocation still produces its previous output (thin wrapper
  or documented equivalent command); `tests/unit/test_backfill_lr_scheduler_tag.py` stays
  green.

## Acceptance criteria

- [ ] Stamps only runs lacking the tag; runs already carrying it are untouched
      (idempotent).
- [ ] With `--marker`, the marker is written beside the value; without it, only the value.
- [ ] Deleted runs are reported, never written.
- [ ] Dry run by default; the report names the count.
- [ ] Applied to `mlflow.db`: 251 runs for each tag; the applied date recorded in ADR
      0004's status.

## Constraints

- `is_optuna=false` gets no marker: the store proves the value (no Optuna run exists).
- Do not touch `mlflow.db.bak`.

## Comments
