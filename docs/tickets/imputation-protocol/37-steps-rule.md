# A steps-aware training batch on the small tables (E100, T14; pre-registered 2026-10-30 17:31 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** T14. Every table trains at BATCH 256. With five folds the training split is 70% of the
  rows, so an epoch of pre-training or decode takes 2 steps on kc2, 3 on credit-g, vehicle and
  biodeg, against 9–124 elsewhere. E28 found the larger batch helping only where an epoch kept
  enough steps (pendigits, 8). Does giving the small tables at least 8 steps an epoch help them?
- **The user's go:** "ok, faça" (2026-10-30, about 17:30 GMT-3), on T14 as the run for gorgona8,
  in parallel with medusa1.
- **The code,** behind a flag with the old behaviour as the default (05b6e59):
  - **`--min_steps_per_epoch N`** (`MIN_STEPS_PER_EPOCH`, imputation only): the training batch of
    pre-training and decode becomes min(BATCH, floor(training rows / N)), so an epoch takes at least
    N steps; validation batches and the learning rates do not change. With N = 8: kc2 45, credit-g
    87, vehicle 73, biodeg 92.
  - **The tag** `min_steps_per_epoch` on imputation runs. A run without it ran at 0; a backfill
    waits on the user's go (this study logs nothing, below).
  - **Checks:** the unit suite with the typing gate passes; both integration fixtures pass
    unchanged.

## Design (fixed before the first cell)

- **Cells:** 16 variants (kc2, credit-g, vehicle and biodeg at 20, 40, 60 and 80nan) × seeds 42, 7
  and 13 × two arms, five folds each, today's default:
  - **M:** as it is;
  - **S8:** with `--min_steps_per_epoch 8`.
- **Running:** `steps_rule.py` and `run_steps_rule_queue.sh`, MLflow off, outputs in
  `trident-2026-10-30-steps-rule/` on each machine. Seed 42 on gorgona8, seeds 7 and 13 on medusa1
  (ADR 0035); both arms of a variant and seed run on one machine, in one queue. No cap: the cells
  are small.

## Measures and rule, fixed now (`steps_rule.py --tally`)

- **Per variant:** S8 − M on the default output's induced `impute_score` (lower is better),
  relative to M, over 15 fold pairs (5 folds × 3 seeds), with the project's rule and the 1% SESOI.
- **Primary,** over the 16 variants:
  - **supported:** S8 better on 4 or more and worse on at most 1;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **Descriptive:** the same difference on the masked `impute_score`.
- **What each reading leads to:** no default changes from this study.
  - **supported:** a fresh-seed confirmation, then an ADR for the user.
  - **refuted:** T14 closes; BATCH stays one value for every table.
  - **mixed:** reported, and the user chooses.

## Outcome

(open)
