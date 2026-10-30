# pendigits at batch 1024 with the learning rates x4, and a longer decode (E101, T11; pre-registered 2026-10-30 20:31 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** T11. E28's Q4 (BATCH 1024, LR_PRE and LR_DECODE x4) gave pendigits its best score then
  (0.323), and E27 saw its validation loss still falling near epoch 450. Defaults have moved since
  (ADR 0013–0015, 0018, 0027). Does the large batch still help pendigits, and does a longer decode
  add to it?
- **The user's go:** "sim, rode" (2026-10-30, about 18:45 GMT-3).
- **No code change:** the arms are `hyperparams_override` on each variant's resolved
  configuration, so pendigits_20nan keeps ADR 0018's encoder. Checked: the override changes BATCH,
  LR_PRE, LR_DECODE and (P600) EPOCHS_DECODE only.

## Design (fixed before the first cell)

- **Cells:** pendigits at 20, 40, 60 and 80nan × seeds 42, 7, 13 × three arms, five folds, MLflow
  off:
  - **M:** today's default;
  - **P:** BATCH 1024, LR_PRE and LR_DECODE x4;
  - **P600:** P with EPOCHS_DECODE 600.
- **Running:** `large_batch.py` and `run_large_batch_queue.sh`; seed 42 on gorgona8, 7 and 13 on
  medusa1 (ADR 0035), every arm of a variant and seed in one queue. Outputs in
  `trident-2026-10-30-large-batch/` on each machine.

## Measures and rule, fixed now (`large_batch.py --tally`)

- **Per variant:** arm − M on the default output's induced `impute_score`, relative to M, 15 fold
  pairs, the project's rule, 1% SESOI.
- **Primary, P600 − M over the 4 variants:** supported if better on 2 or more and worse on none;
  refuted if better on none; mixed otherwise.
- **Descriptive:** P − M (the batch alone), and the masked score.
- **What follows:** no default changes from this study. Supported: a fresh-seed confirmation, then
  an ADR for the user on a per-table batch for pendigits. Refuted: T11 closes.

## Outcome

(open)
