# pendigits at batch 1024 with the learning rates x4, and a longer decode (E101, T11; pre-registered 2026-10-30 20:31 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** T11. E28's Q4 (BATCH 1024, LR_PRE and LR_DECODE x4) gave pendigits its best score then
  (0.323), and E27 saw its validation loss still falling near epoch 450. Defaults have moved since
  (ADR 0013–0015, 0018, 0027). Does the large batch still help pendigits, and does a longer decode
  add to it?
- **The user's go:** "sim, rode" (2026-10-30, between 18:41 and 20:30 GMT-3).
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

**Written 2026-10-30 23:13 GMT-3. Supported: P600 is better than M on 3 of 4 variants and worse on none** —
pendigits_20nan −5.4%, 40nan −4.3%, 60nan −1.2% on the induced error, equivalent at 80nan (+0.2%
[−0.1%, +0.4%]). Batch 1024 with the learning rates x4 still helps pendigits on today's defaults, and
the 600-epoch decode adds to it.

- **Complete:** all 36 cells, 20:31–23:12 GMT-3, no failure; seed 42 on gorgona8, 7 and 13 on
  medusa1, copied and tallied by the end chain.
- **Descriptive:** P alone (450 epochs) is better at 20/40/60nan (−3.4%, −2.5%, −0.9%) and worse at
  80nan (+0.4%); the masked score moves with the induced one in every row.
- **What follows, as fixed above:** a fresh-seed confirmation, then an ADR for the user on a
  per-table batch for pendigits. No default changed.

### The tally, unedited (`large_batch.py --tally`, 2026-10-30 23:13 GMT-3)

```
## P − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_20nan: -3.39% [-3.91%, -2.87%] seeds -3.74%, -3.47%, -2.96% -> better; masked -3.35%
- pendigits_40nan: -2.53% [-2.88%, -2.18%] seeds -2.72%, -2.52%, -2.34% -> better; masked -2.14%
- pendigits_60nan: -0.88% [-1.09%, -0.67%] seeds -1.03%, -0.89%, -0.72% -> better; masked -0.85%
- pendigits_80nan: +0.37% [+0.19%, +0.54%] seeds +0.22%, +0.46%, +0.42% -> worse; masked +0.21%
## P600 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_20nan: -5.37% [-5.92%, -4.83%] seeds -5.17%, -5.15%, -5.80% -> better; masked -5.36%
- pendigits_40nan: -4.27% [-4.69%, -3.84%] seeds -4.52%, -4.35%, -3.94% -> better; masked -3.47%
- pendigits_60nan: -1.19% [-1.56%, -0.82%] seeds -1.46%, -0.93%, -1.18% -> better; masked -1.03%
- pendigits_80nan: +0.16% [-0.08%, +0.41%] seeds -0.11%, +0.12%, +0.47% -> equivalent; masked +0.12%

Primary (P600 − M): better 3 / no difference 0 / equivalent 1 / worse 0 (fold-pair t, seed agreement) of 4 -> supported
```
