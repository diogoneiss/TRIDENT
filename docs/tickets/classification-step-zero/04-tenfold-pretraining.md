# Does pre-training help classification with ten folds and five seeds? (E87; pre-registered 2026-10-28)

This ticket was written and committed before the study's first cell.

- **Why (T41):** E71 found no detectable pre-training gain on classification (B − A from −0.005 to
  +0.004 macro-F1, no verdict at any level). The paper's Table 4 reports about +1.7 points. E71 ran
  three folds and two complete seeds, so it could not tell "no gain" from "too little power".
  This study is T41's old-sweep half. The paper's-parameters half still waits on the paper's
  hyperparameters.
- **The user's go:** "aprovo! rode os 2 em paralelo" (2026-10-28, about 16:50 GMT-3), and on scope
  "9 tabelas de uma vez", one read at the end.
- **Code:** no change to `src/`. `scripts/experiments/cls_tenfold.py` has `train`, `--todo` and
  `--tally`, and reuses E71's `sweep_run`, `cell_dir` and `fold_metrics`.
  `run_cls_tenfold_queue.sh` runs eight GPU queues.

## Design (fixed before the first cell)

- **Arms**, as E71's A and B. Both take the old sweep's `hyperparameters.json` of the variant as
  `hyperparams_override`, with `cosine_legacy`:
  - **A:** as is (`EPOCHS_PRE 300`, `EPOCH_FINE 150`);
  - **B:** `EPOCHS_PRE 0`, no pre-training (the paper's Table 4 arm).
- **Cells:** E71's 9 tables (biodeg, credit-g, electricity, kc2, kr-vs-kp, letter, pendigits,
  spambase, vehicle) × 5 levels (00 to 80nan) × 2 arms × seeds 42, 7, 13, 101, 202, with
  `--cv_folds 10`. That is 450 cells.
  - The cells are dealt seed by seed, the largest tables first.
- **Running:** all on medusa1, so every A/B pair shares a machine (E86). Eight queues, four per
  RTX 5090; another user's job also holds 7–9 GB on each GPU.
  - MLflow tag `experiment=cls-tenfold-2026-10-28`, logged through the leader's server; state in
    `/scratch/diogoneiss/trident-2026-10-28-cls-tenfold/`.
  - **Cost:** about 180 GPU-hours at gorgona8's speed (E71's times × 4.5 for ten folds and the
    larger training splits). On medusa1 that is about 16 h of wall time if the eight queues scale
    cleanly, and more likely 20–24 h.
- **Smoke:** one cell runs first (kc2_00nan s42 B), and the queues start only after it lands in
  the leader's store with `host=medusa`. It counts.

## Measures and rule, fixed now (`cls_tenfold.py --tally`)

- **Per level:** B − A on test macro-F1, per fold. The pairs are pooled over the 9 tables, with one
  array per seed: 450 fold pairs a level. The project's rule applies (95% t-interval, seed
  agreement), with SESOI 0.01:
  - **A higher** (interval below 0, every seed below 0): pre-training helps at that level;
  - **B higher** (above 0): pre-training hurts;
  - **equivalent** (inside ±0.01): no gain of the paper's size;
  - **no verdict** otherwise.
- **The call:**
  - **supported** (the gain replicates) if A is higher at 3 or more of the 5 levels and B is
    higher at none;
  - **refuted** if A is higher at 1 level or none;
  - **mixed** otherwise.
- **Read only with every cell done.** The tally refuses a partial read. A cut stays partial, with no
  call.
- **Descriptive:** each table's mean B − A per level; A's mean macro-F1; the comparison with E71's
  three-fold numbers (run on gorgona8, so descriptive only, by E86).

## Outcome

**Done 2026-10-29, 03:37 GMT-3: refuted, no pre-training gain.** All 450 cells ran on medusa1,
2026-10-28 17:06 to 2026-10-29 03:36 GMT-3, with none failed: 449 in the queues plus the smoke cell.
The tally ran on medusa1 (`cls_tenfold.py --tally`, also in the state directory as `tally.txt`).
Its printout, unedited:

```
done 449 failed 0
- 00nan (A's mean 0.8625): B − A +0.0006 [-0.0017, +0.0028] seeds +0.0012, -0.0011, -0.0014, +0.0024, +0.0017, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0025; credit-g +0.0063; electricity +0.0043; kc2 -0.0094; kr-vs-kp -0.0008; letter +0.0091; pendigits -0.0002; spambase +0.0001; vehicle -0.0019
- 20nan (A's mean 0.8043): B − A +0.0022 [-0.0006, +0.0049] seeds +0.0029, +0.0091, -0.0041, -0.0024, +0.0053, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg +0.0049; credit-g +0.0076; electricity -0.0050; kc2 +0.0072; kr-vs-kp -0.0012; letter +0.0046; pendigits -0.0004; spambase +0.0027; vehicle -0.0009
- 40nan (A's mean 0.7515): B − A -0.0010 [-0.0039, +0.0020] seeds -0.0029, +0.0028, -0.0040, -0.0023, +0.0016, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0037; credit-g -0.0008; electricity -0.0021; kc2 -0.0039; kr-vs-kp -0.0039; letter -0.0049; pendigits -0.0019; spambase -0.0007; vehicle +0.0133
- 60nan (A's mean 0.6720): B − A -0.0035 [-0.0070, +0.0000] seeds -0.0025, -0.0022, -0.0014, -0.0079, -0.0035, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0051; credit-g -0.0052; electricity -0.0016; kc2 +0.0024; kr-vs-kp +0.0003; letter -0.0070; pendigits +0.0007; spambase -0.0037; vehicle -0.0122
- 80nan (A's mean 0.5592): B − A -0.0013 [-0.0057, +0.0030] seeds +0.0021, -0.0052, -0.0037, -0.0031, +0.0032, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0054; credit-g +0.0158; electricity +0.0014; kc2 -0.0089; kr-vs-kp +0.0027; letter -0.0040; pendigits +0.0047; spambase -0.0098; vehicle -0.0087

Primary: A higher at no level, B higher at no level of 5 -> refuted: no pre-training gain (fold-pair t, seed agreement, SESOI 0.01)
```

- **Primary: refuted.** A is higher at no level, and B is higher at no level.
  - At every level the 95% interval of B − A lies inside ±0.01, so the call is equivalent: no
    pre-training gain of the paper's size (about +0.017) at any rate.
  - The means run from −0.0035 (60nan) to +0.0022 (20nan).
- **Closest to a gain: 60nan,** at −0.0035 [−0.0070, +0.0000] with all five seeds negative. Its
  upper bound touches zero without excluding it, so by the rule it is no verdict on direction. The
  gain it would allow is a third of a point.
- **By table** the means stay within ±0.016. The largest are credit-g +0.016 at 80nan (B higher)
  and vehicle at 40nan (+0.013) and 60nan (−0.012), with no pattern across rates.
- **Against E71** (three folds on gorgona8, descriptive only by E86): E71's B − A ran from −0.005
  to +0.004 with no verdict. Ten folds and five seeds narrow that to equivalence at every level.
- **What follows:**
  - Under the fork's old sweep configuration, pre-training adds nothing detectable to
    classification, and a gain of the paper's size is excluded at every rate.
  - T41's other half (the paper's hyperparameters and split) remains the only open route to the
    Table 4 gain, and it waits on those parameters.
