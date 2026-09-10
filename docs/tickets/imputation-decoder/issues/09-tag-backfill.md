# 09. Tag backfill

Status: awaiting-approval (script done and dry-run verified; `--apply` needs the user)
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

- [x] Stamps only runs lacking the tag; runs already carrying it are untouched
      (idempotent).
- [x] With `--marker`, the marker is written beside the value; without it, only the value.
- [x] Deleted runs are reported, never written.
- [x] Dry run by default; the report names the count.
- [ ] Applied to `mlflow.db`: 251 runs for each tag; the applied date recorded in ADR
      0004's status.

## Constraints

- `is_optuna=false` gets no marker: the store proves the value (no Optuna run exists).
- Do not touch `mlflow.db.bak`.

## Comments

- 2026-09-10, four red-green slices in `tests/unit/test_backfill_run_tags.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | A run that already says it keeps what it said (and re-running changes nothing) |
  | 2 | An inferred value says that it was inferred |
  | 3 | A value the store itself proves needs no marker |
  | 4 | A dry run reports without writing |

  `scripts/backfill_run_tags.py` is the general form; `scripts/backfill_lr_scheduler_tag.py`
  and its tests are untouched and still green, so the applied ADR 0003 backfill stays
  reproducible.

  **Dry run against the real `mlflow.db`**, both tags, agreeing exactly with ticket 06's
  store survey:

  | | runs to tag | already tagged | deleted |
  |---|---|---|---|
  | `task=classification` (+`task_backfilled`) | 251 | 0 | 0 |
  | `is_optuna=false` (no marker) | 251 | 0 | 0 |

  Spread across the nine experiments as 62 vehicle, 43 credit-g, 31 letter, 20 biodeg and
  19 each for the rest.

  **Not applied.** Writing 502 tags across 251 real runs is the user's call, so the
  script stops at the dry run. To apply:

  ```
  uv run --python 3.10 python scripts/backfill_run_tags.py --tag task       --value classification --marker task_backfilled --apply
  uv run --python 3.10 python scripts/backfill_run_tags.py --tag is_optuna --value false --apply
  ```

  Record the applied date in ADR 0004's status afterwards.
