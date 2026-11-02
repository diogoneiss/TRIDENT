# The large batch on pendigits_60nan, on fresh seeds (E109; pre-registered 2026-11-01 22:16 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** P600 (BATCH 1024, LR_PRE and LR_DECODE x4, EPOCHS_DECODE 600) was better than the default on
  pendigits_60nan in E101 (−1.19%) and E102 (−1.18%), but the variant was outside E104's confirmed
  scope and so outside ADR 0036 (its considered options).
- **The user's go:** "rode o pendigits lá" (2026-11-01, after 21:36 GMT-3 and before this ticket), on
  medusa1 while gorgona8 runs E108.
- **No code change** beyond `large_batch.py --study e109`.

## Design (fixed before the first cell)

- **Cells:** pendigits_60nan × fresh seeds 2121, 3232, 4343 (never used before) × arms M and P600,
  five folds, MLflow off, all on medusa1 (three queues). Outputs in
  `trident-2026-10-30-large-batch-e109/`.

## Measures and rule, fixed now (`large_batch.py --study e109 --tally`)

- P600 − M on the induced `impute_score`, relative, 15 fold pairs, the project's rule, 1% SESOI.
- **Supported** if better; **not supported** otherwise.
- **What follows:** supported → adding pendigits_60nan to ADR 0036 is the user's decision.

## Outcome

**Written 2026-11-01 22:42 GMT-3. Supported: P600 is better on pendigits_60nan, −1.34% [−1.66%, −1.02%]**, every
seed below zero (−0.99%, −1.49%, −1.54%), the third time at this size (E101 −1.19%, E102 −1.18%).

- **Complete:** all 6 cells on medusa1, 22:16–22:37 GMT-3, no failure.
- **What follows, as fixed above:** adding pendigits_60nan to ADR 0036 is the user's decision.

### The tally, unedited (`large_batch.py --study e109 --tally`, 2026-11-01 22:42 GMT-3)

```
## P600 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_60nan: -1.34% [-1.66%, -1.02%] seeds -0.99%, -1.49%, -1.54% -> better; masked -0.96%

Primary (E109, P600 − M on pendigits_60nan): better -> supported
```
