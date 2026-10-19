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
