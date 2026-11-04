# C15 Stage 1: gate G1 on the positive controls, and A1 against A2 on the development set (E124; pre-registered 2026-11-04)

This ticket was written and committed before the stage's first cell (00:25 GMT-3).

- **Why:** the plan's Stage 1 ([plan](../../plans/2026-11-03-c15-mnar-prediction.md)). G1 checks that the whole
  pipeline sees informative missingness where it is the only signal (F5); the development set gives A1 − A2
  estimates on every family before confirmation (Stage 3) decides H2 on fresh draws of P.
- **The user's go:** "comece agora! ... pode desenvolver e executar nas duas maquinas" (2026-11-04, 00:15 GMT-3),
  within the approved Stages 0–3. P is fixed by E123 ([02](02-g2-out-of-fold.md)).
- **Code:** `C15_ARM` (`src/training/c15_inputs.py`; `src/embedder.py`, `pretraining.py`, `finetuning.py`,
  `runner.py`), off by default, unit-tested (`tests/unit/test_c15_inputs.py`): A1 the shipped `[NULL]`; A2
  missForest's fill (fitted on the fold's training rows' features) plus a gap indicator read by each numerical
  column's value MLP (1 → 2 inputs), pre-training holes equal to A1's draw for draw and never on a gap; A4:k the
  k-th `IterativeImputer(sample_posterior=True)` draw, no indicator. With the arm on, every fold writes per-row
  validation and test probabilities, unweighted log loss and the kept state (`c15_readout_fold_<k>.pt`). Cells:
  `scripts/experiments/c15_run.py`, `c15_queue.sh`. A 3-epoch smoke of A1, A2 and A4:0 on kc2's control ran
  end to end on gorgona8; no arm comparison was read from it.

## Design (fixed before the first cell)

- **Configuration:** each variant's resolved classification configuration (no promoted files: 300/150 epochs,
  batch 256, `cosine_legacy`), plus `PRETRAIN_OBJECTIVE embedding_normalized` on every arm (the user's choice),
  the shipped checkpoint rule; `--cv_folds 5`, MLflow off.
- **Stage 1a, short mode (seed 42), F5's three controls:** A1, A2 and A4:0–4 (five MICE draws, A4's prediction
  the mean of the five test probabilities). 21 cells.
- **Stage 1b, decision mode (seeds 42/7/13), the development set:** A1 and A2 on F0 (7: biodeg, kc2, vehicle at
  20/60nan, spambase_60nan), F1 (their 7 self-masking siblings), F2 (biodeg's dial, 3) and F3 (6 outcome
  variants). 138 cells. The margins are fixed (the user), so seeds 7 and 13 run at once.
- **Machines (ADR 0035, every arm of a (variant, seed) pair on one machine):** gorgona8 takes seed 42 of the
  development set (46 cells); medusa1 takes seeds 7 and 13 and F5 (113 cells). Data md5 equal on both
  (2026-11-04 00:40 GMT-3).

## Rules, fixed now

- **G1 (on F5, short mode, 5 cells a variant), passes if both hold on at least 2 of the 3 controls:**
  1. A2 beats A4 on test log loss: the per-cell relative difference (LL_A2 − LL_A4) / LL_A4 has a 95% t interval,
     Nadeau–Bengio corrected (1/5 + 0.286), below 0;
  2. A1's direct contrast on the noise column, C (logit P(y = 1) at `[NULL]` minus the logit of the mean over
     K = 20 MICE draws of the noise cell given the row, other gaps `[NULL]`, averaged over the column's test gap
     cells, re-scored through each fold's kept A1 model), is positive (the sign of the odds-3 tilt), its corrected
     interval above 0. The scorer is committed before G1 is read.
  If G1 fails, C15 stops before Stage 2, as the protocol fixes.
- **Stage 1b is descriptive:** per variant, A1 − A2 on test macro-F1 and the relative log-loss difference over 15
  cells, Nadeau–Bengio corrected and plain intervals, with seed agreement, and the H2 per-variant call against the
  fixed margins (−0.01, +2%). Nothing is decided on it; H2 is decided in Stage 3 on fresh draws of P.
- **Read only with every cell of a block done.** GBDT stays internal.

## Amendment 2026-11-04 01:10 GMT-3 (machines only; the user's "utilize as gpus o melhor possivel")

medusa1 finished 102 of its 113 cells by 01:08, gorgona8 4 of its 46 (spambase cells take about 17 minutes there). To
keep both GPUs busy, whole (variant, seed 42) pairs move to medusa1 with every arm of every stage (ADR 0035's pairing
holds): gorgona8 keeps seed 42 of the spambase and kc2 development variants, medusa1 takes the rest of seed 42.
Only spambase's seed-42 pairs had started on gorgona8 (A1, then A2), and they stay there. gorgona8's queues stopped
between cells on their STOP file and were relaunched on the new deal. Same cells, seeds and rules; tickets 04 and 06
follow the same rule.

## G1's reading (2026-11-04 01:16 GMT-3): fails, 1 of 3; by the rule, C15 stops before Stage 2

All 21 short-mode cells on F5 done on medusa1 (00:25–01:08 GMT-3, none failed). The unedited
`c15_report.py --g1` (the contrast with one MICE fit per fold and 20 posterior draws, as committed in ec7a1d5):

```
## G1 (F5, seed 42, 5 folds): (1) A2 − A4 relative test log loss below 0; (2) A1's noise-column contrast above 0; Nadeau–Bengio corrected 95% t intervals; passes if both hold on >= 2 of 3
- biodeg_noise3p0_s2060_60nan: A2 − A4 -0.0041 [-0.1805, +0.1723] (plain -0.1173); A1 contrast +1.778 [+1.072, +2.483] logit -> fail
- kc2_noise3p0_s2060_60nan: A2 − A4 +0.1544 [-0.3787, +0.6875] (plain -0.1876); A1 contrast +3.800 [-0.530, +8.129] logit -> fail
- spambase_noise3p0_s2060_60nan: A2 − A4 -0.3278 [-0.3688, -0.2869] (plain -0.3541); A1 contrast +1.747 [+1.238, +2.257] logit -> pass

Primary (G1): 1 of 3 -> fails
```

- **Condition 2 (A1 reads the informative gap):** positive on all three controls (+1.78, +3.80, +1.75 logit), its
  corrected interval above 0 on biodeg and spambase; kc2's reaches −0.53.
- **Condition 1 (A2 beats A4 on log loss):** only on spambase (−0.33 [−0.37, −0.29]). On biodeg and kc2 the corrected
  intervals over 5 cells span about ±0.2 and ±0.5; the plain intervals' lower ends sit below 0 too.
- **What follows, as fixed above:** C15 stops before Stage 2; tickets 04, 05 and 06 (E125–E127) are withdrawn unrun.
  Stage 1b (A1 − A2 on the development set) finishes as pre-registered and is read descriptively. Whether a G1 with more
  power (the decision mode's 15 cells on F5) is worth a new pre-registration is the user's decision; this reading
  stays recorded as a failure, and the reading is post hoc: condition 1's arms differ in the imputer as well as the
  indicator (the protocol's Open decision 14a), and five cells under the Nadeau–Bengio factor (0.2 + 0.286) leave
  biodeg's and kc2's intervals wide.

## Outcome (2026-11-04 01:34 GMT-3): G1 failed (above); Stage 1b, descriptive

All 159 cells done (2026-11-04 00:25–01:31 GMT-3), none failed in its own right. The three `FAILED` lines in
medusa1's progress.log are mine: two cells (A2_spambase_probit_60nan_s13, A1_biodeg_probit1p0_60nan_s7) started twice
after the rebalancing, because the relaunched queues dealt cells an old queue was still running; both copies were
killed, the cells' directories removed, and each cell rerun alone (`queue single` lines). The unedited
`c15_report.py --tally`:

```
## Stage 1b (descriptive): A1 − A2 per variant, 15 cells; macro-F1 difference and relative log loss (LL_A1 − LL_A2) / LL_A2, Nadeau–Bengio corrected (plain); H2's per-variant call at −0.01 / +2%
- biodeg_20nan: F1 -0.0164 [-0.0513, +0.0186] (plain [-0.0316, -0.0012]); LL +0.0847 [-0.0500, +0.2194] -> inconclusive
- biodeg_60nan: F1 -0.0324 [-0.0769, +0.0121] (plain [-0.0517, -0.0131]); LL +0.1623 [+0.0539, +0.2707] -> inferior
- kc2_20nan: F1 +0.0138 [-0.0438, +0.0714] (plain [-0.0113, +0.0389]); LL -0.0071 [-0.1870, +0.1729] -> inconclusive
- kc2_60nan: F1 -0.0291 [-0.0955, +0.0373] (plain [-0.0580, -0.0003]); LL +0.0842 [-0.1239, +0.2923] -> inconclusive
- vehicle_20nan: F1 -0.0290 [-0.0647, +0.0068] (plain [-0.0445, -0.0134]); LL +0.1398 [-0.0545, +0.3341] -> inconclusive
- vehicle_60nan: F1 -0.0930 [-0.1583, -0.0277] (plain [-0.1214, -0.0646]); LL +0.2617 [+0.1434, +0.3801] -> inferior
- spambase_60nan: F1 +0.0002 [-0.0126, +0.0131] (plain [-0.0053, +0.0058]); LL +0.0002 [-0.0715, +0.0720] -> inconclusive
- biodeg_probit_20nan: F1 -0.0286 [-0.0748, +0.0175] (plain [-0.0487, -0.0086]); LL +0.1160 [-0.0964, +0.3285] -> inconclusive
- biodeg_probit_60nan: F1 -0.0508 [-0.1048, +0.0033] (plain [-0.0743, -0.0272]); LL +0.1455 [+0.0018, +0.2892] -> inconclusive
- kc2_probit_20nan: F1 +0.0011 [-0.0313, +0.0336] (plain [-0.0130, +0.0153]); LL -0.0249 [-0.1915, +0.1418] -> inconclusive
- kc2_probit_60nan: F1 -0.0335 [-0.1118, +0.0449] (plain [-0.0676, +0.0006]); LL +0.1538 [-0.3553, +0.6629] -> inconclusive
- vehicle_probit_20nan: F1 -0.0533 [-0.0859, -0.0208] (plain [-0.0675, -0.0392]); LL +0.3012 [+0.0244, +0.5780] -> inferior
- vehicle_probit_60nan: F1 -0.0618 [-0.1164, -0.0071] (plain [-0.0855, -0.0380]); LL +0.2455 [+0.0819, +0.4092] -> inferior
- spambase_probit_60nan: F1 -0.0014 [-0.0156, +0.0127] (plain [-0.0076, +0.0047]); LL +0.0254 [-0.0242, +0.0750] -> inconclusive
- biodeg_probit0p25_60nan: F1 -0.0431 [-0.0834, -0.0027] (plain [-0.0606, -0.0255]); LL +0.1412 [+0.0270, +0.2554] -> inferior
- biodeg_probit1p0_60nan: F1 -0.0267 [-0.0625, +0.0092] (plain [-0.0423, -0.0111]); LL +0.1414 [+0.0159, +0.2669] -> inconclusive
- biodeg_probit2p0_60nan: F1 -0.0144 [-0.0436, +0.0147] (plain [-0.0271, -0.0017]); LL +0.0301 [-0.1040, +0.1642] -> inconclusive
- biodeg_outcome1p5_s2060_60nan: F1 +0.0140 [-0.0258, +0.0538] (plain [-0.0033, +0.0313]); LL -0.0470 [-0.2508, +0.1568] -> inconclusive
- biodeg_outcome3p0_s2060_60nan: F1 -0.0028 [-0.0132, +0.0077] (plain [-0.0073, +0.0018]); LL +1.2728 [-1.9594, +4.5049] -> inconclusive
- kc2_outcome1p5_s2060_60nan: F1 -0.0008 [-0.0422, +0.0405] (plain [-0.0188, +0.0171]); LL +0.0976 [-0.5223, +0.7175] -> inconclusive
- kc2_outcome3p0_s2060_60nan: F1 +0.0094 [-0.0226, +0.0413] (plain [-0.0045, +0.0233]); LL +2.5226 [-8.7781, +13.8232] -> inconclusive
- spambase_outcome1p5_s2060_60nan: F1 -0.0009 [-0.0100, +0.0082] (plain [-0.0048, +0.0030]); LL +0.0085 [-0.0760, +0.0930] -> inconclusive
- spambase_outcome3p0_s2060_60nan: F1 +0.0014 [-0.0009, +0.0038] (plain [+0.0004, +0.0025]); LL -0.5550 [-1.3609, +0.2510] -> inconclusive
```

- **Nothing here decides H2** (confirmation does, and it did not run).
- **A1 against A2 on development:** A1 is *inferior* on 5 of 23 variants (vehicle at 60nan MCAR and both self-masking
  levels, biodeg_60nan MCAR, biodeg's dial at 0.25), *non-inferior* or *superior* on none, *inconclusive* on 18. Its
  macro-F1 is lower on 17 of 23 by point estimate. The deficit shows under MCAR too, so it reads as the fill-plus-
  indicator input beating `[NULL]` on these tables in general, not as informative missingness.
- **The relative log loss is unstable on the outcome-dependent variants** (+1.27, +2.52, −0.55 at odds 3): where the
  indicators nearly separate the classes, A2's log loss approaches 0 and the ratio explodes. Any later read of H2 on
  such variants needs the absolute log-loss difference beside the ratio.

