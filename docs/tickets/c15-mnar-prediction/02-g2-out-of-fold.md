# C15 Stage 0, second gate: G2 read out of fold over every training and validation row (E123; pre-registered 2026-11-03)

This ticket was written and committed before the gate's first cell (23:56 GMT-3).

- **Why:** E122 ([01](01-stage0.md)) left G2 invalid: two of the three positive controls (biodeg, kc2) had means above
  0.02 but intervals below 0. Each cell read only the validation rows (about 106 on biodeg, 52 on kc2), so the gate
  lacked power. This reads the same cells on about 8 times the rows.
- **A deviation, decided after E122's result was seen:** the user chose it ("opção 1", 2026-11-03) among three
  options (a gate with more power, a stronger control, accepting the stop). E122 stays recorded as invalid.
- **Binding now:** if E123's controls fail too, C15 stops at Stage 0 for good; no third form of G2 is tried.
- **Code:** `c15_stage0.py --g2 --oof` and `--tally --oof` (`g2_oof_cell`). Smoke-tested on one non-control cell
  (kc2_probit_20nan, seed 42, fold 1) only. No change to `src/`. CPU only, on gorgona8.

## Design (fixed before the first cell)

- **The same** 19 candidates, 15 cells (seeds 42/7/13 × folds 1–5 of the fork's `build_folds` at `--cv_folds 5`),
  GBDT, indicators and probe as E122.
- **Per cell:** the rows are the fold's training and validation rows; its test rows stay unread. missForest is fitted
  once on those rows' features (never the label) and fills them. The GBDT with and without the indicators, and the
  indicators-only probe, are each read on 5 inner folds (StratifiedKFold, the cell's seed); their held-out
  probabilities are pooled over all the rows, and the cell's reading is (LL_without − LL_with) / LL_without on them.
- **What differs from E122:** the reading uses out-of-fold predictions on every training and validation row instead
  of the validation rows alone, and the missForest fit sees those rows' features without their labels.

## Rule, fixed now (unchanged from E122)

- **Informative:** the mean drop over the 15 cells is at least 0.02, with its 95% t interval above 0.
- **Validity:** all three F5 controls must be informative; if one is not, G2 is invalid and C15 stops (binding, above).
- **P:** the F1 variants and the odds-3 F3 variants that are informative; F2 and odds-1.5 F3 are read descriptively.
- **What follows:** P goes to the user with Stage 1's ticket. **Read only with every cell done.** GBDT stays internal.

## Outcome (2026-11-04 00:12 GMT-3): G2 valid; P holds 5 variants

All 285 cells done (2026-11-03 23:59 to 2026-11-04 00:11 GMT-3, gorgona8, none failed). The unedited
`c15_stage0.py --tally --oof`:

```
## G2: relative drop in out-of-fold log loss over the training and validation rows (E123) from the indicators, 15 cells (seeds (42, 7, 13) x 5 folds); informative if the mean is >= 0.02 with the 95% t interval above 0
- biodeg_probit_20nan: -0.0109 [-0.0226, +0.0008] seeds -0.0128, -0.0033, -0.0168; probe -0.0154 -> not informative
- biodeg_probit_60nan: +0.0275 [+0.0171, +0.0379] seeds +0.0298, +0.0214, +0.0312; probe +0.0295 -> informative
- kc2_probit_20nan: -0.0196 [-0.0462, +0.0070] seeds -0.0040, -0.0249, -0.0300; probe -0.0309 -> not informative
- kc2_probit_60nan: -0.0402 [-0.0646, -0.0159] seeds -0.0585, -0.0398, -0.0224; probe +0.0449 -> not informative
- vehicle_probit_20nan: -0.0132 [-0.0273, +0.0009] seeds +0.0021, -0.0136, -0.0279; probe -0.0212 -> not informative
- vehicle_probit_60nan: +0.0301 [+0.0100, +0.0503] seeds +0.0236, +0.0436, +0.0232; probe -0.0004 -> informative
- spambase_probit_60nan: +0.0184 [+0.0119, +0.0248] seeds +0.0213, +0.0116, +0.0223; probe +0.0351 -> not informative
- biodeg_probit0p25_60nan: +0.0178 [+0.0021, +0.0335] seeds +0.0334, +0.0063, +0.0136; probe -0.0081 -> not informative
- biodeg_probit1p0_60nan: +0.0350 [+0.0150, +0.0550] seeds +0.0133, +0.0516, +0.0401; probe +0.1324 -> informative
- biodeg_probit2p0_60nan: +0.1358 [+0.1120, +0.1596] seeds +0.1375, +0.1356, +0.1342; probe +0.2385 -> informative
- biodeg_outcome1p5_s2060_60nan: +0.2337 [+0.2152, +0.2521] seeds +0.2402, +0.2372, +0.2236; probe +0.3915 -> informative
- biodeg_outcome3p0_s2060_60nan: +0.8460 [+0.8346, +0.8573] seeds +0.8487, +0.8368, +0.8524; probe +0.9539 -> informative
- kc2_outcome1p5_s2060_60nan: +0.1743 [+0.1358, +0.2127] seeds +0.1559, +0.1895, +0.1775; probe +0.2353 -> informative
- kc2_outcome3p0_s2060_60nan: +0.7628 [+0.7468, +0.7788] seeds +0.7479, +0.7733, +0.7671; probe +0.8766 -> informative
- spambase_outcome1p5_s2060_60nan: +0.2489 [+0.2380, +0.2597] seeds +0.2501, +0.2473, +0.2493; probe +0.5693 -> informative
- spambase_outcome3p0_s2060_60nan: +0.7944 [+0.7326, +0.8563] seeds +0.8368, +0.7356, +0.8108; probe +0.9924 -> informative
- biodeg_noise3p0_s2060_60nan: +0.0432 [+0.0259, +0.0604] seeds +0.0337, +0.0495, +0.0463; probe +0.0447 -> informative
- kc2_noise3p0_s2060_60nan: +0.0548 [+0.0068, +0.1028] seeds +0.0432, +0.1033, +0.0180; probe +0.0534 -> informative
- spambase_noise3p0_s2060_60nan: +0.0851 [+0.0791, +0.0912] seeds +0.0857, +0.0897, +0.0801; probe +0.1027 -> informative

Primary (G2): valid (F5 3 of 3); P = 5 variants: biodeg_probit_60nan, vehicle_probit_60nan, biodeg_outcome3p0_s2060_60nan, kc2_outcome3p0_s2060_60nan, spambase_outcome3p0_s2060_60nan
- F2 (dial, descriptive): biodeg_probit0p25_60nan False, biodeg_probit1p0_60nan True, biodeg_probit2p0_60nan True
```

- **Validity: passes.** All three positive controls clear the rule (biodeg +0.043, kc2 +0.055, spambase +0.085).
- **P (5):** biodeg_probit_60nan and vehicle_probit_60nan (self-masking), and the three odds-3 outcome-dependent
  variants (biodeg, kc2, spambase). The other five self-masking variants do not clear it; at 20nan the indicators
  never help, and on kc2_probit_60nan they hurt the GBDT (−0.040) though the indicators-only probe gains (+0.045).
- **The dial (F2), descriptive:** slopes 1.0 and 2.0 clear it, 0.25 does not (+0.018).
- **What follows:** Stage 1 (the arms' code behind flags, then gate G1 on F5) is the user's to start, with P fixed
  as above. vehicle has four classes, so H2.2's sign test does not apply to vehicle_probit_60nan.

