# The large batch where it helped: pendigits and letter at 20 and 40nan, on fresh seeds (E104; pre-registered 2026-10-31 10:14 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E101–E103 ([38](38-large-batch.md), [39](39-large-batch-confirmation.md),
  [40](40-large-batch-large-tables.md)) found P600 (BATCH 1024, LR_PRE and LR_DECODE x4,
  EPOCHS_DECODE 600) better than the default on pendigits and letter at 20 and 40nan (−1.5% to
  −5.8%), fading or a little worse at 60–80nan, and nothing on electricity. That scope, "pendigits
  and letter at 20–40nan", was chosen after seeing those results, so it is confirmed here on seeds
  none of them used before any ADR.
- **The user's go:** "Isso, rode aa duas em paralelo agora" (2026-10-31, after 09:58 and
  before 10:14 GMT-3), on this study and E105 in parallel.
- **No code change:** `large_batch.py --study e104`.

## Design (fixed before the first cell)

- **Cells:** pendigits and letter at 20 and 40nan × fresh seeds 1357, 2468, 3579 × arms M and
  P600, five folds, MLflow off. 24 cells.
- **Running:** `STUDY=e104 run_large_batch_queue.sh`; seed 1357 on gorgona8 (two queues), 2468 and
  3579 on medusa1 (two queues, beside E105), both arms of a variant and seed in one queue.

## Measures and rule, fixed now (`large_batch.py --study e104 --tally`)

- **Per variant:** P600 − M on the induced `impute_score`, relative, 15 fold pairs, the project's
  rule, 1% SESOI.
- **Replicated** if better on 3 or more of the 4 and worse on none; **not replicated** if better on
  1 or fewer or worse on any; **partial** otherwise.
- **What follows:** replicated → an ADR proposal for the user: P600 as the configuration of these
  four variants. No default changes without the user.

## Outcome

(open)
