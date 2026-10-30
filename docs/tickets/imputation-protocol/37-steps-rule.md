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

**Written 2026-10-30 18:41 GMT-3. Refuted: S8 is better on none of the 16 variants and worse on 4** (credit-g
40/60nan, biodeg 20/60nan), equivalent on credit-g_80nan, no verdict on 11. More steps an epoch at
a smaller batch, with the learning rates unchanged, do not help the small tables; where they move,
they hurt (+1.2% to +5.3% on the induced error). T14 closes; BATCH stays one value for every table.

- **Complete:** all 96 cells, 17:31–18:41 GMT-3, no failure; seed 42 on gorgona8, seeds 7 and 13 on
  medusa1, copied and tallied by the end chain.
- **Descriptive:** the masked score moves the same way (S8 higher on 14 of 16).
- **Not tested:** a smaller batch with a scaled-down learning rate; this arm kept LR_PRE and
  LR_DECODE, so each epoch takes 3–4 times as many steps at the same rate.

### The tally, unedited (`steps_rule.py --tally`, 2026-10-30 18:41 GMT-3)

```
## S8 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- kc2_20nan: +1.41% [-3.91%, +6.73%] seeds -1.38%, -2.54%, +8.15% -> none; masked -0.68%
- kc2_40nan: -2.17% [-8.21%, +3.86%] seeds -4.43%, -0.09%, -2.01% -> none; masked -0.53%
- kc2_60nan: +3.23% [-0.92%, +7.39%] seeds +2.06%, +6.10%, +1.53% -> none; masked +3.42%
- kc2_80nan: +0.95% [-2.28%, +4.18%] seeds -1.46%, +4.49%, -0.18% -> none; masked +1.21%
- credit-g_20nan: +1.15% [-0.49%, +2.79%] seeds +0.71%, +3.16%, -0.43% -> none; masked +1.53%
- credit-g_40nan: +1.99% [+0.88%, +3.09%] seeds +1.85%, +1.90%, +2.21% -> worse; masked +3.97%
- credit-g_60nan: +1.17% [+0.32%, +2.03%] seeds +0.52%, +1.35%, +1.65% -> worse; masked +2.38%
- credit-g_80nan: +0.22% [-0.31%, +0.74%] seeds -0.13%, +0.25%, +0.54% -> equivalent; masked +0.39%
- vehicle_20nan: +0.68% [-1.36%, +2.71%] seeds -0.02%, +2.93%, -0.88% -> none; masked +0.81%
- vehicle_40nan: +0.57% [-2.10%, +3.25%] seeds -0.29%, +1.01%, +1.00% -> none; masked +3.01%
- vehicle_60nan: +0.25% [-1.41%, +1.91%] seeds +0.39%, +0.26%, +0.10% -> none; masked +0.99%
- vehicle_80nan: -0.56% [-1.36%, +0.24%] seeds -0.24%, -0.43%, -1.01% -> none; masked +0.23%
- biodeg_20nan: +5.32% [+2.43%, +8.20%] seeds +4.19%, +6.88%, +4.87% -> worse; masked +2.78%
- biodeg_40nan: +2.31% [-0.03%, +4.64%] seeds +4.01%, +3.79%, -0.88% -> none; masked +4.39%
- biodeg_60nan: +1.85% [+0.40%, +3.31%] seeds +2.34%, +1.65%, +1.58% -> worse; masked +1.92%
- biodeg_80nan: +0.37% [-0.28%, +1.03%] seeds +0.92%, +0.26%, -0.06% -> none; masked +0.82%

Primary: better 0 / no difference 11 / equivalent 1 / worse 4 (fold-pair t, seed agreement) of 16 -> refuted
```
