# The large batch and longer decode on letter and electricity (E103; pre-registered 2026-10-31 00:17 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E101 ([38](38-large-batch.md)) found P600 (BATCH 1024, LR_PRE and LR_DECODE x4,
  EPOCHS_DECODE 600) better than the default on pendigits 20/40/60nan; E102 ([39](39-large-batch-confirmation.md))
  checks it on fresh seeds. letter and electricity are the only other tables where batch 1024
  still leaves an epoch enough steps (14 and 31 at five folds; spambase 3, kr-vs-kp 2). Does the
  gain hold there? If so, an ADR would be a rule for the large tables rather than a pendigits case.
- **The user's go:** "sim, agende" (2026-10-31, about 00:20 GMT-3), on the proposal to schedule it for
  the night without intervention, with a cap.
- **No code change:** `large_batch.py --study e103` runs M and P600 over each variant's resolved
  configuration (letter_20nan keeps ADR 0018's encoder).

## Design (fixed before the first cell)

- **Cells:** letter and electricity at 20, 40, 60 and 80nan × seeds 42, 7, 13 × arms M and P600,
  five folds, MLflow off. 48 cells.
- **Running:** `STUDY=e103 run_large_batch_queue.sh`, started on each machine when its E102 queues
  end: seed 42 on gorgona8 (two queues), seeds 7 and 13 on medusa1 (four), both arms of a variant
  and seed in one queue, letter's pairs before electricity's. Outputs in
  `trident-2026-10-30-large-batch-e103/`.
- **The cap:** at 2026-10-31 09:00 GMT-3 a STOP file is written on both machines, so no cell
  starts after it; cells running then finish. A variant missing a seed is reported missing and its
  table's reading is partial.

## Measures and rule, fixed now (`large_batch.py --study e103 --tally`)

- **Per variant:** P600 − M on the induced `impute_score`, relative to M, 15 fold pairs, the
  project's rule, 1% SESOI.
- **Primary, per table (letter, electricity) over its 4 variants:** supported if better on 2 or
  more and worse on none; refuted if better on none; mixed otherwise.
- **What follows:** no default changes from this study; with E101 and E102, the user decides on an
  ADR.

## Outcome

(open)
