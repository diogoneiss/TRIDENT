# Stage 1: the CPU bar on the sweep's folds (E97; pre-registered 2026-10-29)

This ticket was written and committed before the study's first variant. Eight variants ran first
as a smoke check, in a scratch directory outside the study, to time the script and confirm that it
handles electricity_80nan's all-missing rows. Only their per-arm means were printed, and they do
not count.

- **Why:** step zero's plan ([2026-10-23](../../plans/2026-10-23-classification-step-zero.md),
  Stage 1) needs a non-deep bar for classification under missingness before Stage 4 tunes
  anything. It also answers how far untuned GBDTs with native NaN, and impute-then-learn, sit from
  TRIDENT on the same folds.
- **The user's go:** the plan's settled decisions (2026-10-29, 21:20 GMT-3). Stage 1 needs no
  decision beyond the plan, and the bar is LightGBM and HistGradientBoosting (XGBoost is out).
- **Code:** no change to `src/`. `scripts/experiments/cls_stage1_bar.py` (`run`, `--tally`); its
  docstring fixes the arms and settings.

## Design (fixed before the first variant)

- **Folds:** the sweep's 3 folds, `build_folds(frame, 'class', 3, 42)`, on all 45 variants (9 tables
  × 00–80nan).
- **Arms (20 a fold), all untuned, `class_weight='balanced'`, `random_state` 42:**
  - **native NaN:** `native/hgb` and `native/lgbm`, with categories as training-split vocabulary
    codes and NaN for gaps or unseen values;
  - **impute, then learn:** `mode_lgbm`, `mean_mode` and `missforest`, each with and without
    `isna()` indicators, each followed by HGB, LightGBM and RandomForest.
  - The boosters stop early on the fold's validation split: at most 100 rounds, patience 10, with
    the validation loss weighted by the training split's balanced class weights.
- **Running:** gorgona8's CPU only, so every GBDT number comes from one machine. Six parallel
  processes of four threads each, the largest tables first; about 30 min. Nothing is written to the
  store.

## Reading, fixed now (`cls_stage1_bar.py --tally`)

- **Descriptive only:** one seed and three folds a variant, as the plan says. No call is made.
- **Per level:** each arm's paired (arm − TRIDENT) test macro-F1, with a 95% t-interval over the
  pooled fold pairs.
  - **The headline TRIDENT reference is E71's arm A at seed 42:** the sweep's configuration rerun
    at HEAD on gorgona8, on the same folds.
  - The old sweep's own folds are printed beside it as the plan names them. Stage 0 found they do
    not reproduce at HEAD, so they are descriptive only.
- **Also printed:** each arm's mean macro-F1 by level, and the indicator contrast (with − without)
  as an MCAR sanity control, not evidence of informative missingness.
- **What it feeds:** the choice of the GBDT bar and of the impute-then-learn arms that Stage 4
  tunes. That choice is made in Stage 4's pre-registration.

## Outcome

(open)
