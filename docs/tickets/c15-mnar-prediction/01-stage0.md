# C15 Stage 0: the driving columns and gate G2, which variants carry informative missingness? (E122; pre-registered 2026-11-03)

This ticket was written and committed before the stage's first cell (23:29 GMT-3).

- **Why:** C15 ([plan](../../plans/2026-11-03-c15-mnar-prediction.md)) tests whether `[NULL]` tokens predict
  as well as missForest + indicator under informative missingness. Under weak informativeness non-inferiority is
  met trivially, so G2 first picks the variants where the gap carries signal: the primary set P. If the
  positive controls fail, the pipeline cannot see informative missingness and the study stops here.
- **The user's go:** "prefiro começar pelo c15" (2026-11-03), and the four decisions recorded in the plan.
- **Code:** `datasets/mnar_family.py` (seeded generators, unit-tested in `tests/unit/test_mnar_family.py`)
  and `scripts/experiments/c15_stage0.py`. No change to `src/`. CPU only, on gorgona8.

## Data (generated before the first cell; the data directory is gitignored, the scripts are the record)

- **F1, self-masking (exist):** biodeg, kc2, vehicle `_probit_{20,60}nan` and spambase_probit_60nan (E45/E46).
- **F2, the tilt dial (exist):** biodeg_probit{0p25,1p0,2p0}_60nan (E46).
- **F3, outcome-dependent (new):** `mnar_family.py outcome --odds 1.5|3 --seed 2060` on biodeg, kc2 and spambase at
  60nan: every feature column loses the MCAR count, class 1 (`LabelEncoder` order: biodeg "2", kc2 "yes",
  spambase "1") `odds` times as readily. Realised missing rate by class, class 0 / class 1: odds 1.5,
  0.55/0.69, 0.57/0.72, 0.54/0.69; odds 3, 0.47/0.85, 0.52/0.90, 0.45/0.83.
- **F5, the positive control (new):** `mnar_family.py noise --odds 3 --seed 2060`: the MCAR `<base>_60nan` file of
  biodeg, kc2 and spambase plus one N(0, 1) column, `c15_noise`, whose 60% gaps follow odds 3 on the label.
- The 9 new files' md5 digest (of the sorted per-file sums): `cf005b09…`.

## Design (fixed before the first cell)

- **Driving columns** (`--drivers`), per table (biodeg, kc2, vehicle, spambase) on `_00nan`: a
  HistGradientBoosting classifier at fixed defaults (`random_state 0`), 5 folds (StratifiedKFold, seed 42); a
  column drives if dropping it raises held-out log loss with the 95% t interval over the 5 folds above 0. Sign:
  the coefficient of an L2 logistic regression on the standardised table; sign-test column: monotone partial
  dependence (|Spearman| ≥ 0.8, 20-point grid). Sign and PD on the binary tables only. The table is frozen
  before any arm runs and uses no arm output.
- **G2** (`--g2`), 19 candidates (F1 7, F2 3, F3 6, F5 3) × 15 cells (seeds 42/7/13 × folds 1–5 of the fork's
  `build_folds` at `--cv_folds 5`, the decision mode's folds): missForest (`src/training/missforest.py`) fitted
  on the training rows' features only fills training and validation rows; the same GBDT is fitted with and
  without one 0/1 indicator per gapped column. Reading per cell: (LL_without − LL_with) / LL_without on the
  validation rows. Beside it, an indicators-only logistic probe against the training class frequencies.
  285 cells.

## Rule, fixed now (`c15_stage0.py --tally`)

- **Informative:** the mean drop over the 15 cells is at least 0.02, with its 95% t interval above 0.
- **Validity:** all three F5 controls must be informative; if one is not, G2 is invalid and C15 stops.
- **P:** the F1 variants and the odds-3 F3 variants that are informative. F2 and odds-1.5 F3 are read
  descriptively (the dial).
- **What follows:** P goes to the user with Stage 1's ticket; if P is empty, C15 stops at H2 (nothing to test).
- **Read only with every cell done.** GBDT is internal: the gate's numbers never enter a report, page or the thesis.

## Outcome (2026-11-03 23:45 GMT-3): G2 invalid, two of the three positive controls fail; by the rule, C15 stops here

All 285 cells done (23:31–23:45 GMT-3, gorgona8, none failed). The unedited `c15_stage0.py --tally`:

```
## G2: relative drop in validation log loss from the indicators, 15 cells (seeds (42, 7, 13) x 5 folds); informative if the mean is >= 0.02 with the 95% t interval above 0
- biodeg_probit_20nan: -0.0402 [-0.0670, -0.0135] seeds -0.0563, -0.0346, -0.0297; probe -0.0158 -> not informative
- biodeg_probit_60nan: +0.0454 [+0.0015, +0.0892] seeds +0.0018, +0.0802, +0.0541; probe +0.0191 -> informative
- kc2_probit_20nan: -0.0059 [-0.0833, +0.0715] seeds -0.0008, +0.0022, -0.0191; probe +0.0121 -> not informative
- kc2_probit_60nan: +0.0208 [-0.0539, +0.0956] seeds -0.0429, +0.1044, +0.0010; probe +0.0827 -> not informative
- vehicle_probit_20nan: -0.0091 [-0.0546, +0.0365] seeds -0.0132, -0.0114, -0.0025; probe -0.0101 -> not informative
- vehicle_probit_60nan: +0.0175 [-0.0110, +0.0460] seeds -0.0053, +0.0245, +0.0332; probe +0.0135 -> not informative
- spambase_probit_60nan: +0.0176 [+0.0016, +0.0336] seeds +0.0199, -0.0029, +0.0358; probe +0.0351 -> not informative
- biodeg_probit0p25_60nan: +0.0391 [+0.0044, +0.0739] seeds +0.0744, +0.0431, -0.0001; probe +0.0011 -> informative
- biodeg_probit1p0_60nan: +0.0448 [+0.0012, +0.0885] seeds +0.0450, +0.0459, +0.0436; probe +0.1125 -> informative
- biodeg_probit2p0_60nan: +0.1450 [+0.0987, +0.1913] seeds +0.0914, +0.1721, +0.1715; probe +0.2060 -> informative
- biodeg_outcome1p5_s2060_60nan: +0.2257 [+0.1793, +0.2722] seeds +0.2035, +0.2028, +0.2709; probe +0.4325 -> informative
- biodeg_outcome3p0_s2060_60nan: +0.8473 [+0.8013, +0.8933] seeds +0.8218, +0.8709, +0.8492; probe +0.9573 -> informative
- kc2_outcome1p5_s2060_60nan: +0.1839 [+0.0790, +0.2889] seeds +0.2845, +0.0914, +0.1759; probe +0.2176 -> informative
- kc2_outcome3p0_s2060_60nan: +0.7059 [+0.5939, +0.8179] seeds +0.7982, +0.6620, +0.6574; probe +0.8866 -> informative
- spambase_outcome1p5_s2060_60nan: +0.2543 [+0.2256, +0.2830] seeds +0.2930, +0.2177, +0.2521; probe +0.5812 -> informative
- spambase_outcome3p0_s2060_60nan: +0.7360 [+0.6623, +0.8098] seeds +0.7674, +0.7712, +0.6694; probe +0.9929 -> informative
- biodeg_noise3p0_s2060_60nan: +0.0375 [-0.0218, +0.0969] seeds +0.0215, +0.0542, +0.0369; probe +0.0683 -> not informative
- kc2_noise3p0_s2060_60nan: +0.1022 [-0.0197, +0.2240] seeds +0.2250, +0.1052, -0.0236; probe +0.1277 -> not informative
- spambase_noise3p0_s2060_60nan: +0.0663 [+0.0346, +0.0980] seeds +0.1069, +0.0427, +0.0493; probe +0.0879 -> informative

Primary (G2): invalid, positive controls failing: biodeg_noise3p0_s2060_60nan, kc2_noise3p0_s2060_60nan -> stop
```

- **Validity: fails.** spambase's control passes (+0.066 [+0.035, +0.098]); biodeg's (+0.038 [−0.022, +0.097]) and
  kc2's (+0.102 [−0.020, +0.224]) have means above 0.02 and every seed but one positive, but intervals that reach
  below 0. The control is strong per column (the noise column's gap rate, class 0 / class 1: biodeg 0.48/0.85, kc2
  0.53/0.89), so the failure reads as a lack of power, not of signal: validation holds about 106 rows on biodeg
  and 52 on kc2, and the indicator is one column of 42 and 22. That reading is post hoc and changes nothing.
- **Descriptive, not P (G2 is invalid):** of the self-masking family (F1), only biodeg_probit_60nan clears the rule
  (+0.045 [+0.002, +0.089]); the other six do not, and at 20nan the indicators never help. Every outcome-dependent
  variant (F3) and every rung of biodeg's dial (F2) clears it, F3 by +0.18 to +0.85.
- **What follows, as fixed above:** C15 stops at Stage 0. Any continuation (a G2 with more power, a stronger
  control) is a new, dated pre-registration for the user, and this stage stays recorded as invalid.
- **The driving-column table** crashed on spambase in the first pass (`partial_dependence` refuses integer
  columns); it was fixed by casting the complete table's features to float (only that table reads them) and rerun.
  Rerun complete at 23:46 GMT-3 (`drivers.json` in the stage's state directory): driving columns, biodeg 2 of 41,
  kc2 1 of 21, vehicle 1 of 18, spambase 3 of 57; sign-test columns 1, 1, 0 and 1. Leave-one-column-out finds
  few drivers on tables with correlated columns; the table is frozen as computed and is used only from Stage 2.

