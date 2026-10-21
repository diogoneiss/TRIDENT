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
