# ADR 0025's panel on K against TabPFN, confirmation set (E56; pre-registered 2026-10-21)

This ticket was written and committed before the first cell.

- **Why:** E55 ([10](10-tabpfn.md)) makes a new claim about `impute_score`. On E44's 15 confirmation
  variants, K against TabPFN v2 reads better 6, no difference 4, worse 5, with TabPFN ahead at
  80nan.
  - ADR 0025 decision 3: a claim about `impute_score` stands as stated only where the panel's
    per-kind ratios agree with it, and every disagreement is listed.
  - This is the panel check behind E55's claim. It also informs the user's decision on TabPFN for
    the strong tier.
- **The user's go:** "comece a executar experimentos que não precisam da minha supervisão direta"
  (2026-10-20, about 23:10 GMT-3), with the machine until 08:30.
- **Code:** two options with their old behaviour as the default:
  - `metric_panel.py --fitters tabpfn` refits TabPFN v2 alone;
  - `metric_panel_report.py --opponent tabpfn` reads the model against one named method.
  - E44's panel reads as before through the default path, with the same disagreement list.

## Design (fixed before the first cell)

- **Cells:** E44's 15 confirmation variants × seeds 707/808/909, 45 pairs.
  - **The model's fills:** the headline K's, from the runs' fold ledgers in
    `/scratch2/diogoneiss/trident-2026-10-17-confirmation/`.
  - **TabPFN's fills:** refitted on the same training folds as in E55, asked for exactly the
    ledger's induced cells.
- **Run:** two GPU lanes under `uv run --extra tabpfn`, CSVs in
  `/scratch2/diogoneiss/trident-2026-10-21-tabpfn-panel/csv`. Nothing is written to the store.
  - **The deadline:** no pair starts after 07:30 GMT-3. A variant without all three seeds is
    reported as incomplete.

## Reading, fixed now (ADR 0025)

`metric_panel_report.py <csv> --opponent tabpfn`, per variant, model − TabPFN over 15 fold pairs
with the verdict rule:

- **Where they agree:** K − TabPFN's `impute_score` call stands where the per-kind ratios agree
  (`rmse_ratio` and `mae_ratio` for numbers, `err_ratio` and `balacc_error_ratio` for categories).
- **Disagreements:** every opposite verdict is listed by name.
- **Distributional metrics** (dispersion, Wasserstein, category TV, absolute bias) are reported as
  caveats, not reversals (ADR 0025 decision 4).
- **No decision follows here.** It feeds E55's proposal, which is the user's.

## Outcome (2026-10-21, 04:54 GMT-3)

- **Run:** 45 of 45 pairs, 03:53 to 04:53 GMT-3, two GPU lanes on fb68ec9.
- **Integrity:**
  - the panel's model `impute_score` equals the runs' K fold by fold (largest difference 0);
  - TabPFN's equals E55's to 3.6e-4. Here the induced cells are asked alone, while in E55 they
    shared the masked call's `predict`, which is the grouping effect E55 records.

**The printout** (`scripts/experiments/metric_panel_report.py <csv> --opponent tabpfn`, unedited;
model − TabPFN; lower is better; dispersion as the distance from 1):

```text
- `biodeg_40nan` (opponent tabpfn): impute_score -0.041 better, rmse_ratio -0.041 better, mae_ratio +0.089 worse, colnorm_ratio +0.058 worse, abs_bias -0.010 better, dispersion -0.175 better, wasserstein +0.017 worse, impute_score_other -0.036 better, impute_score_with_context -0.041 better
- `biodeg_80nan` (opponent tabpfn): impute_score +0.052 worse, rmse_ratio +0.052 worse, mae_ratio +0.172 worse, colnorm_ratio +0.094 worse, abs_bias -0.019 better, dispersion -0.211 better, wasserstein -0.034 none, impute_score_other +0.053 worse, impute_score_with_context +0.052 worse
- `electricity_40nan` (opponent tabpfn): impute_score +0.030 none, rmse_ratio +0.039 none, mae_ratio +0.021 worse, colnorm_ratio +0.086 none, err_ratio -0.033 better, balacc_error_ratio -0.033 better, abs_bias -0.001 none, dispersion -0.050 none, wasserstein -0.013 better, tv -0.022 none, impute_score_other +0.030 none, impute_score_with_context +0.030 none
- `electricity_60nan` (opponent tabpfn): impute_score +0.004 none, rmse_ratio +0.007 none, mae_ratio +0.015 worse, colnorm_ratio +0.003 none, err_ratio -0.016 better, balacc_error_ratio -0.016 better, abs_bias -0.001 none, dispersion -0.057 better, wasserstein -0.019 better, tv -0.064 better, impute_score_other +0.004 none, impute_score_with_context +0.005 none
- `electricity_80nan` (opponent tabpfn): impute_score +0.010 worse, rmse_ratio +0.013 worse, mae_ratio +0.015 worse, colnorm_ratio +0.043 none, err_ratio -0.005 better, balacc_error_ratio -0.005 better, abs_bias +0.002 none, dispersion -0.070 better, wasserstein -0.017 better, tv +0.027 none, impute_score_other +0.010 worse, impute_score_with_context +0.014 worse
- `kc2_40nan` (opponent tabpfn): impute_score -0.002 none, rmse_ratio -0.002 none, mae_ratio +0.091 worse, colnorm_ratio +0.052 worse, abs_bias +0.004 none, dispersion -0.072 better, wasserstein +0.025 worse, impute_score_other -0.017 none, impute_score_with_context -0.002 none
- `kc2_80nan` (opponent tabpfn): impute_score -0.058 better, rmse_ratio -0.058 better, mae_ratio +0.023 none, colnorm_ratio -0.038 none, abs_bias -0.011 none, dispersion -0.191 better, wasserstein -0.006 none, impute_score_other -0.077 better, impute_score_with_context -0.061 better
- `letter_40nan` (opponent tabpfn): impute_score -0.034 better, rmse_ratio -0.034 better, mae_ratio -0.012 better, colnorm_ratio -0.035 better, abs_bias -0.031 better, dispersion -0.091 better, wasserstein -0.057 better, impute_score_other -0.034 better, impute_score_with_context -0.034 better
- `letter_60nan` (opponent tabpfn): impute_score -0.030 better, rmse_ratio -0.030 better, mae_ratio -0.019 better, colnorm_ratio -0.030 better, abs_bias -0.039 better, dispersion -0.141 better, wasserstein -0.105 better, impute_score_other -0.026 better, impute_score_with_context -0.030 better
- `letter_80nan` (opponent tabpfn): impute_score +0.002 worse, rmse_ratio +0.002 worse, mae_ratio +0.012 worse, colnorm_ratio +0.002 worse, abs_bias -0.025 better, dispersion -0.069 better, wasserstein -0.067 better, impute_score_other +0.003 worse, impute_score_with_context +0.002 worse
- `pendigits_40nan` (opponent tabpfn): impute_score -0.011 better, rmse_ratio -0.011 better, mae_ratio +0.016 worse, colnorm_ratio -0.009 better, abs_bias -0.006 better, dispersion -0.044 better, wasserstein -0.014 better, impute_score_other -0.011 better, impute_score_with_context -0.011 better
- `pendigits_60nan` (opponent tabpfn): impute_score -0.018 better, rmse_ratio -0.018 better, mae_ratio -0.009 better, colnorm_ratio -0.018 better, abs_bias -0.007 better, dispersion -0.069 better, wasserstein -0.046 better, impute_score_other -0.018 better, impute_score_with_context -0.018 better
- `pendigits_80nan` (opponent tabpfn): impute_score +0.016 worse, rmse_ratio +0.016 worse, mae_ratio +0.024 worse, colnorm_ratio +0.016 worse, abs_bias +0.007 worse, dispersion -0.028 better, wasserstein -0.018 better, impute_score_other +0.016 worse, impute_score_with_context +0.017 worse
- `vehicle_40nan` (opponent tabpfn): impute_score -0.000 none, rmse_ratio -0.000 none, mae_ratio +0.020 worse, colnorm_ratio +0.001 none, abs_bias +0.009 worse, dispersion -0.046 better, wasserstein -0.007 better, impute_score_other -0.000 none, impute_score_with_context -0.000 none
- `vehicle_80nan` (opponent tabpfn): impute_score +0.053 worse, rmse_ratio +0.053 worse, mae_ratio +0.068 worse, colnorm_ratio +0.056 worse, abs_bias +0.014 worse, dispersion -0.102 better, wasserstein -0.062 better, impute_score_other +0.053 worse, impute_score_with_context +0.054 worse

Disagreements between impute_score and a panel metric (opposite verdicts, fold-pair t, seed agreement):
- biodeg_40nan: impute_score better, mae_ratio worse
- biodeg_40nan: impute_score better, colnorm_ratio worse
- biodeg_40nan: impute_score better, wasserstein worse
- biodeg_80nan: impute_score worse, abs_bias better
- biodeg_80nan: impute_score worse, dispersion better
- electricity_80nan: impute_score worse, err_ratio better
- electricity_80nan: impute_score worse, balacc_error_ratio better
- electricity_80nan: impute_score worse, dispersion better
- electricity_80nan: impute_score worse, wasserstein better
- letter_80nan: impute_score worse, abs_bias better
- letter_80nan: impute_score worse, dispersion better
- letter_80nan: impute_score worse, wasserstein better
- pendigits_40nan: impute_score better, mae_ratio worse
- pendigits_80nan: impute_score worse, dispersion better
- pendigits_80nan: impute_score worse, wasserstein better
- vehicle_80nan: impute_score worse, dispersion better
- vehicle_80nan: impute_score worse, wasserstein better
```

**Reading, by ADR 0025's rule:**

- **The RMSE ratio agrees with `impute_score` on all 15 variants.** E55's calls stand as stated for
  point accuracy in squared error.
- **Per-kind disagreements, listed by name:**
  - **biodeg_40nan and pendigits_40nan:** K is better in `impute_score` and RMSE, worse in MAE
    (+0.089 and +0.016). biodeg_40nan is also worse in the column-normalised ratio. These two wins
    hold in RMSE, not in MAE.
  - **electricity_80nan:** K is worse in `impute_score` (numbers), but better on its categorical
    column (`day`: error and balanced error −0.005).
- **The distribution caveat runs the other way from E44's.**
  - K's fills keep the spread better than TabPFN's on 14 of 15 variants (dispersion; no
    difference on electricity_40nan). Wasserstein is better on 11.
  - So TabPFN's 80nan wins (letter, pendigits, vehicle, biodeg, electricity) are point-accuracy
    wins with more shrunk fills. On those five, K is better in dispersion, and in Wasserstein on
    four of them.
  - ADR 0025 decision 4 reads this as a caveat, not a reversal.

**What it adds to E55's proposal (the user's decision):** TabPFN's lead over K at 80nan holds in
RMSE and MAE. It comes with fills shrunk further toward the centre than K's. If TabPFN joins the
strong tier, the thesis's 80nan statement would read: TabPFN fills more accurately per cell, and K
keeps the imputed distribution closer to the truth.

