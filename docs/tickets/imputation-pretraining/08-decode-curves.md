# D1: the decode curves, pre-trained against not, from the store (E49; pre-registered 2026-10-19)

This ticket was written and committed before any curve was read.

- **Source:** T28's report
  ([2026-10-19-t28-why-pretraining-adds-little](../../reports/2026-10-19-t28-why-pretraining-adds-little.md),
  H3 and its test D1). The user asked for D1 on 2026-10-19.
- **What it is:** a read of runs already in the store. No GPU, no new run and no store write.
  Script: `scripts/experiments/decode_curves_report.py`.

## Question

Does pre-training give the decode stage a head start that the decode stage then erases (H3,
washout)? Or no head start at all (back to H1 or H2)? Or a lasting gain (real transfer)?

## Data, fixed now

- **The curve:** each parent's `cv/decode/val_loss/mean`, one value per decode epoch.
  - It is the cross-fold mean of the λ-weighted value loss on the fixed validation mask.
  - That mask is drawn from its own stream per seed and fold, so every arm of a seed is scored on
    the same cells. The training masks differ between arms (F-13-2).
- **The runs:** parents with `is_mirror=false` from three studies:
  - E24, `ablation=pretraining-2026-09-29`;
  - E25, `experiment=pretrain-objective-2026-09-30`;
  - E26, `experiment=pretrain-decode-budget-2026-09-30`.
- **The pairs:** by variant and seed, with the same decode schedule.
  - **150 decode epochs:** A (the shipped embedding objective), N (normalised) and V (value
    target), each against B (no pre-training).
  - **450 decode epochs:** M (normalised, E26) against C (no pre-training).
  - A pre-trained arm is paired with the no-pre-training arm of its own study where it has one.
    Otherwise E24's is used: pendigits_20nan's B and C exist only there, and so does A on
    pendigits.
  - **Variants:** credit-g_20nan, credit-g_80nan, kr-vs-kp_40nan and pendigits_20nan. Seeds
    42/7/13, or the two that ran.

## Measures, fixed now

Per pair, with the relative gap g(e) = (L_ref(e) − L_arm(e)) / L_ref(e), positive when the
pre-trained arm is ahead:

- the head start: g at decode epochs 0, 10 and 50;
- the catch-up epoch: the first epoch at which the reference's loss is at or below the arm's;
- the gap at the minimum: (min L_ref − min L_arm) / min L_ref.

## Reading, fixed now

Per variant and arm, over its seeds:

- **real transfer:** the gap at the minimum is at least +1% on every seed;
- **washout:** not real transfer, and the head start at epoch 0 is at least +1% on every seed;
- **nothing usable:** the head start at epoch 0 is below +1% on every seed;
- **mixed:** otherwise.

The 1% mirrors ADR 0020's smallest effect of interest.

**What each reading leads to:**

- **Washout:** the levers in T28's report (§5) come before T13: a frozen-encoder start (SMART's
  unfreeze epoch), or a lower encoder learning rate in the decode stage.
- **Nothing usable:** D2 (collapse and probes) becomes the next step.
- **Real transfer:** D3 (frozen encoder against random) checks whether it is in the encoder.

**Limit, stated in advance:** this is the decode loss, not `impute_score`. It is a diagnostic of
H3, not a claim about scores.

## Outcome (2026-10-19, 19:39 GMT-3)

The script read 48 pairs (16 variant-and-arm groups, 2 or 3 seeds each). The printout is
`scripts/experiments/decode_curves_report.py`'s, unedited.

| variant | arm | seed | reference | g(0) | g(10) | g(50) | catch-up epoch | gap at the minima |
|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | A (E25) | 7 | B (E25) | +2.4% | +1.8% | -0.8% | 3 | -0.1% |
| `credit-g_20nan` | A (E25) | 13 | B (E25) | +2.5% | +0.7% | +2.8% | 3 | +2.6% |
| `credit-g_20nan` | A (E25) | 42 | B (E25) | +2.6% | -3.2% | +1.2% | 5 | +1.3% |
| `credit-g_20nan` | M (E26) | 7 | C (E25) | +2.9% | +7.0% | +2.1% | 34 | +1.9% |
| `credit-g_20nan` | M (E26) | 13 | C (E25) | +2.4% | +0.5% | -4.9% | 4 | -1.0% |
| `credit-g_20nan` | M (E26) | 42 | C (E25) | +0.1% | -0.6% | -0.1% | 1 | +1.3% |
| `credit-g_20nan` | N (E25) | 7 | B (E25) | +4.3% | +3.8% | -1.9% | 12 | -1.5% |
| `credit-g_20nan` | N (E25) | 13 | B (E25) | +0.8% | +1.3% | +4.4% | 13 | +3.6% |
| `credit-g_20nan` | N (E25) | 42 | B (E25) | +2.7% | +2.5% | +1.7% | 1 | +2.8% |
| `credit-g_20nan` | V (E25) | 7 | B (E25) | +4.3% | +7.3% | -0.0% | 14 | +1.8% |
| `credit-g_20nan` | V (E25) | 13 | B (E25) | +4.5% | +1.5% | +1.2% | 28 | +0.9% |
| `credit-g_20nan` | V (E25) | 42 | B (E25) | +4.7% | +1.5% | +3.4% | 13 | +2.0% |
| `credit-g_80nan` | A (E25) | 7 | B (E25) | -0.2% | -0.6% | +7.9% | 0 | +0.4% |
| `credit-g_80nan` | A (E25) | 13 | B (E25) | -2.5% | +1.5% | +4.4% | 0 | +1.8% |
| `credit-g_80nan` | A (E25) | 42 | B (E25) | +2.8% | +1.3% | +2.1% | never | +1.2% |
| `credit-g_80nan` | M (E26) | 7 | C (E25) | +3.0% | +1.9% | -4.4% | 7 | +0.4% |
| `credit-g_80nan` | M (E26) | 13 | C (E25) | -4.2% | +0.9% | +0.3% | 0 | -0.2% |
| `credit-g_80nan` | M (E26) | 42 | C (E25) | -0.8% | -0.4% | -1.3% | 0 | +2.6% |
| `credit-g_80nan` | N (E25) | 7 | B (E25) | -0.7% | +1.0% | +1.0% | 0 | +0.8% |
| `credit-g_80nan` | N (E25) | 13 | B (E25) | -4.4% | +0.7% | +2.4% | 0 | +1.0% |
| `credit-g_80nan` | N (E25) | 42 | B (E25) | +2.0% | +2.2% | +0.1% | 4 | +0.2% |
| `credit-g_80nan` | V (E25) | 7 | B (E25) | -2.0% | -2.1% | -4.6% | 0 | -2.8% |
| `credit-g_80nan` | V (E25) | 13 | B (E25) | -3.2% | +1.5% | -4.2% | 0 | +0.3% |
| `credit-g_80nan` | V (E25) | 42 | B (E25) | +1.6% | -2.4% | -9.6% | 3 | -0.7% |
| `kr-vs-kp_40nan` | A (E25) | 7 | B (E25) | +2.9% | -4.0% | +3.0% | 2 | +2.5% |
| `kr-vs-kp_40nan` | A (E24) | 13 | B (E24) | +4.3% | -2.1% | +0.9% | 1 | +2.3% |
| `kr-vs-kp_40nan` | A (E25) | 42 | B (E25) | +4.3% | -4.0% | +0.6% | 2 | +1.2% |
| `kr-vs-kp_40nan` | M (E26) | 7 | C (E25) | +15.6% | +3.7% | -0.4% | 24 | +3.4% |
| `kr-vs-kp_40nan` | M (E26) | 13 | C (E25) | +18.0% | +4.5% | -0.3% | 22 | +2.9% |
| `kr-vs-kp_40nan` | M (E26) | 42 | C (E25) | +13.4% | +2.3% | +0.6% | 16 | +1.2% |
| `kr-vs-kp_40nan` | N (E25) | 7 | B (E25) | +12.6% | +2.3% | +3.3% | 19 | +2.7% |
| `kr-vs-kp_40nan` | N (E25) | 13 | B (E25) | +14.9% | +4.2% | +0.0% | 18 | -0.8% |
| `kr-vs-kp_40nan` | N (E25) | 42 | B (E25) | +15.5% | +0.3% | +1.7% | 13 | +1.5% |
| `kr-vs-kp_40nan` | V (E25) | 7 | B (E25) | +11.6% | +5.1% | -1.3% | 19 | -1.7% |
| `kr-vs-kp_40nan` | V (E25) | 13 | B (E25) | +14.2% | +5.9% | -2.8% | 18 | -1.5% |
| `kr-vs-kp_40nan` | V (E25) | 42 | B (E25) | +15.4% | +1.6% | -3.3% | 15 | +0.1% |
| `pendigits_20nan` | A (E24) | 7 | B (E24) | -11.1% | -8.5% | -8.4% | 0 | -8.3% |
| `pendigits_20nan` | A (E24) | 13 | B (E24) | -19.9% | -4.7% | -8.0% | 0 | -8.8% |
| `pendigits_20nan` | A (E24) | 42 | B (E24) | -12.7% | -5.7% | -11.0% | 0 | -11.8% |
| `pendigits_20nan` | M (E26) | 7 | C (E24) | +56.9% | +23.4% | +8.7% | 196 | +1.6% |
| `pendigits_20nan` | M (E26) | 13 | C (E24) | +58.2% | +26.6% | +7.2% | 120 | -3.0% |
| `pendigits_20nan` | M (E26) | 42 | C (E24) | +57.7% | +26.0% | +8.8% | 185 | -0.1% |
| `pendigits_20nan` | N (E25) | 7 | B (E24) | +56.0% | +23.5% | +8.0% | never | +5.0% |
| `pendigits_20nan` | N (E25) | 13 | B (E24) | +57.4% | +26.0% | +8.5% | never | +4.8% |
| `pendigits_20nan` | N (E25) | 42 | B (E24) | +59.3% | +26.5% | +8.8% | never | +4.7% |
| `pendigits_20nan` | V (E25) | 7 | B (E24) | +54.7% | +22.8% | +12.1% | never | +9.6% |
| `pendigits_20nan` | V (E25) | 13 | B (E24) | +54.3% | +28.4% | +12.5% | never | +11.4% |
| `pendigits_20nan` | V (E25) | 42 | B (E24) | +56.6% | +25.8% | +9.4% | never | +6.6% |

Reading per variant and arm (fixed in the ticket; threshold ±1%):
- `credit-g_20nan` A vs B (3 seeds): washout; head start +2.5% at epoch 0, +1.1% at 50; gap at the minima +1.2%; caught up by epoch 3 (median)
- `credit-g_20nan` M vs C (3 seeds): mixed; head start +1.8% at epoch 0, -0.9% at 50; gap at the minima +0.7%; caught up by epoch 4 (median)
- `credit-g_20nan` N vs B (3 seeds): mixed; head start +2.6% at epoch 0, +1.4% at 50; gap at the minima +1.7%; caught up by epoch 12 (median)
- `credit-g_20nan` V vs B (3 seeds): washout; head start +4.5% at epoch 0, +1.5% at 50; gap at the minima +1.5%; caught up by epoch 14 (median)
- `credit-g_80nan` A vs B (3 seeds): mixed; head start +0.0% at epoch 0, +4.8% at 50; gap at the minima +1.1%; caught up by epoch 0 (median)
- `credit-g_80nan` M vs C (3 seeds): mixed; head start -0.6% at epoch 0, -1.8% at 50; gap at the minima +0.9%; caught up by epoch 0 (median)
- `credit-g_80nan` N vs B (3 seeds): mixed; head start -1.0% at epoch 0, +1.2% at 50; gap at the minima +0.7%; caught up by epoch 0 (median)
- `credit-g_80nan` V vs B (3 seeds): mixed; head start -1.2% at epoch 0, -6.1% at 50; gap at the minima -1.1%; caught up by epoch 0 (median)
- `kr-vs-kp_40nan` A vs B (3 seeds): real transfer; head start +3.8% at epoch 0, +1.5% at 50; gap at the minima +2.0%; caught up by epoch 2 (median)
- `kr-vs-kp_40nan` M vs C (3 seeds): real transfer; head start +15.7% at epoch 0, -0.0% at 50; gap at the minima +2.5%; caught up by epoch 22 (median)
- `kr-vs-kp_40nan` N vs B (3 seeds): washout; head start +14.3% at epoch 0, +1.6% at 50; gap at the minima +1.1%; caught up by epoch 18 (median)
- `kr-vs-kp_40nan` V vs B (3 seeds): washout; head start +13.7% at epoch 0, -2.5% at 50; gap at the minima -1.0%; caught up by epoch 18 (median)
- `pendigits_20nan` A vs B (3 seeds): nothing usable; head start -14.6% at epoch 0, -9.1% at 50; gap at the minima -9.7%; caught up by epoch 0 (median)
- `pendigits_20nan` M vs C (3 seeds): washout; head start +57.6% at epoch 0, +8.2% at 50; gap at the minima -0.5%; caught up by epoch 185 (median)
- `pendigits_20nan` N vs B (3 seeds): real transfer; head start +57.6% at epoch 0, +8.4% at 50; gap at the minima +4.8%; caught up by epoch never (median)
- `pendigits_20nan` V vs B (3 seeds): real transfer; head start +55.2% at epoch 0, +11.3% at 50; gap at the minima +9.2%; caught up by epoch never (median)

**What it says, by the rule fixed in advance:**

- **H3 (washout) holds where pre-training matters most, on pendigits_20nan.**
  - The normalised objective gives the decode stage a large head start: +58% at epoch 0, +8% at
    epoch 50.
  - At 150 decode epochs it never closes, so N and V read "real transfer" (+4.8% and +9.2% at the
    minima).
  - At 450 decode epochs, today's default, the arm without pre-training catches up around epoch
    185 (median). M − C reads "washout": −0.5% at the minima.
  - This matches E26 (M ≈ C on pendigits). The long decode stage erases what pre-training gave.
- **kr-vs-kp_40nan keeps a lasting gain even with the long decode.**
  - M against C reads "real transfer": +15.7% at epoch 0 and +2.5% at the minima.
  - A against B is also real transfer; N and V at 150 epochs read washout.
  - This matches E25 and E26, where kr-vs-kp is the exception.
- **credit-g shows no clear effect.** At 20nan and 80nan the gaps are ±1–4% and change sign
  between seeds ("mixed"); A and V at 20nan read washout.
- **The shipped embedding objective (A) hurts pendigits from the start:** −15% at epoch 0, −10%
  at the minima ("nothing usable"). That is E04's collapse showing in the decode curve.
- **The catch-up epoch is noisy.** Curves that cross early can still part again (kr-vs-kp's A
  crosses at epoch 2 and ends +2.0% ahead). The minima and the head start carry the reading.

**What it leads to (fixed in advance):**

- **Washout, on pendigits:** the levers in T28's report (§5) come before T13: a frozen-encoder
  start (SMART's unfreeze epoch) or a lower encoder learning rate in the decode stage.
  - A cheap first arm is pendigits_20nan alone, M against M with the encoder frozen for the first
    N decode epochs.
- **Real transfer, on kr-vs-kp:** D3 (a frozen encoder against a random one) would show whether the
  gain lives in the encoder.
- **Limit:** this is the decode loss, not `impute_score`. Nothing here changes a default.

