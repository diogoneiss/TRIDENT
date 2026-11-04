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
