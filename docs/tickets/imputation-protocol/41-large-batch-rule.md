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

**Written 2026-10-31 12:58 GMT-3. Replicated: P600 is better than the default on 4 of 4, every seed below zero.**
pendigits_20nan −5.9%, pendigits_40nan −4.1%, letter_20nan −3.3%, letter_40nan −1.5% on the induced
error, near E101–E103's sizes (−5.4 to −5.8%, −4.2 to −4.3%, −2.9%, −1.5%).

- **Complete:** all 24 cells, 10:14–12:57 GMT-3, no failure. The end chain's early rsync errors
  (medusa1's P600 directory did not exist yet) are harmless: the tally ran on the full copy.
- **What follows, as fixed above:** an ADR proposal for the user: P600 as the configuration of
  these four variants. No default has changed.

### The tally, unedited (`large_batch.py --study e104 --tally`, 2026-10-31 12:58 GMT-3)

```
## P600 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_20nan: -5.93% [-6.59%, -5.26%] seeds -6.36%, -5.74%, -5.68% -> better; masked -5.51%
- pendigits_40nan: -4.10% [-4.59%, -3.61%] seeds -3.96%, -3.74%, -4.61% -> better; masked -3.30%
- letter_20nan: -3.25% [-3.57%, -2.93%] seeds -3.35%, -3.25%, -3.14% -> better; masked -2.80%
- letter_40nan: -1.54% [-1.75%, -1.32%] seeds -1.69%, -1.36%, -1.55% -> better; masked -1.31%

Primary (P600 − M), all four: better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 4 -> replicated
```
