# TabPFN v2 joins the strong tier as an offline member

## Status

**Accepted on 2026-10-21** by the user ("1 ok"), on E55's pre-registered proposal. Written at
09:00 GMT-3. It amends ADR [0021](0021-two-tiers-of-baseline-imputers.md) decision 2 (the strong
tier's members). Decisions 1, 3, 4 and 5 of ADR 0021 stand.

## Context

- **E55** ([imputation-protocol/10](../tickets/imputation-protocol/10-tabpfn.md)) ran TabPFN v2,
  column-wise and untuned, on E48's 108 (variant, seed) pairs, with
  `src/training/tabpfn_baseline.py` behind the optional extra `tabpfn`.
  - It beat the strong tier's best (missForest, condexp, SoftImpute, ICE) on 11 of 15 confirmation
    variants and was worse on none.
  - That clears the threshold of 5 that E48's rule fixed for proposing a new member.
- **E56** ([imputation-protocol/11](../tickets/imputation-protocol/11-tabpfn-panel.md)) checked K
  against TabPFN on ADR 0025's panel.
  - The RMSE ratio agrees with every call.
  - MAE reverses K's wins on biodeg_40nan and pendigits_40nan.
  - K's fills keep the spread better on 14 of 15 variants. TabPFN's 80nan wins are point-accuracy
    wins with more shrunk fills.
- **The runs cannot log it live.**
  - It needs an optional dependency, which Windows and the default environment lack.
  - It costs GPU time per fold, 2–53 s.
  - It is not a baseline-cache family. Running it live would change the training path's
    dependencies.

## Decision

1. **The strong tier is missForest, condexp, SoftImpute, ICE and TabPFN v2.** TabPFN is pinned as
   in E55: `tabpfn==9.2.0`, `ModelVersion.V2`, 10,000-row context, 8 estimators.
2. **TabPFN is an offline member.**
   - It is read from a study's offline pass (`scripts/experiments/tabpfn_run.py`), merged into the
     fold tables by the readers.
   - The runs keep logging the four live members. `baseline/best_strong` and
     `gap_to_best_strong_baseline` in the store mean the live four, and nothing is backfilled.
3. **The readers** (`report_common`):
   - **`strong`** is the best of the live four and, where a study's frame has it, TabPFN
     (`OFFLINE_STRONG`).
   - **`strong_live`** is the live four alone. A study without TabPFN's pass reads the live four
     and says so.
   - TabPFN joins the tuned strong tier (E42's tuned missForest, E41's arms, TabPFN untuned) and the
     best of all.
   - The thesis table merges E55's CSVs (`thesis_table.py --tabpfn`; `none` reads the tier as
     before this ADR).
4. **TabPFN − strong reads against `strong_live`,** as E55 pre-registered.
5. **The distribution caveat travels with every 80nan claim against the strong tier** (E56, ADR
   0025 decision 4). TabPFN's fills there are more accurate per cell and more shrunk than K's.

## Considered options

- **Leave TabPFN as an external baseline,** like MaCoDE. Rejected: E48's rule, fixed before E55,
  gives the threshold, and TabPFN cleared it by a wide margin (11 against 5).
- **Score it live in every run.** Rejected for now: it needs an optional dependency and GPU time in
  the training path, and it would miss every cache entry.

## Consequences

**The development thesis table, seeds 101/202/303** (`scripts/experiments/thesis_table.py`):

- Before this ADR (`--tabpfn none`):

```text
- K vs bar on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
- K vs tuned on 21 variants: better 13 / no difference 5 / worse 3 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0273)
- K vs missforest on 21 variants: better 15 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
- K vs strong on 21 variants: better 13 / no difference 7 / worse 1 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
- K vs tuned_strong on 21 variants: better 13 / no difference 7 / worse 1 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0117)
- K vs best_of_all on 21 variants: better 13 / no difference 6 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0391)
- K vs per_type on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
- Holm across opponents: missforest 0.0273, strong 0.0273, tuned_strong 0.0586, bar 0.0781, per_type 0.0781, tuned 0.0781, best_of_all 0.0781
- D4, a win needs K better than both the bar and the tuned bar: 13 wins of 21; worse than either on 3 (kr-vs-kp_60nan, credit-g_80nan, electricity_20nan)
```

- After it:

```text
- K vs bar on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
- K vs tuned on 21 variants: better 13 / no difference 5 / worse 3 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0273)
- K vs missforest on 21 variants: better 15 / no difference 6 / worse 0 (fold-pair t, seed agreement); tables below zero 9 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0039)
- K vs strong on 21 variants: better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement); tables below zero 5 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.9102)
- K vs tuned_strong on 21 variants: better 6 / no difference 8 / worse 7 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.8203)
- K vs best_of_all on 21 variants: better 5 / no difference 8 / worse 8 (fold-pair t, seed agreement); tables below zero 4 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.5703)
- K vs per_type on 21 variants: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement); tables below zero 8 of 9 (two-sided Wilcoxon on per-table mean relative gaps, p = 0.0195)
- Holm across opponents: missforest 0.0273, bar 0.1172, per_type 0.1172, tuned 0.1172, best_of_all 1.0000, tuned_strong 1.0000, strong 1.0000
- D4, a win needs K better than both the bar and the tuned bar: 13 wins of 21; worse than either on 3 (kr-vs-kp_60nan, credit-g_80nan, electricity_20nan)
```

**The confirmation set, E44's seeds 707/808/909** (`scripts/experiments/tabpfn_report.py`, K − the
live four and K − the strong tier with TabPFN):

```text
## Development set, seeds 101/202/303 (labelled variants: none on this set)
  - K − strong_live: better 13 / no difference 7 / worse 1 (fold-pair t, seed agreement)
  - K − strong (ADR 0028): better 5 / no difference 9 / worse 7 (fold-pair t, seed agreement)
## Confirmation set (E44), seeds 707/808/909
  - K − strong_live: better 12 / no difference 1 / worse 2 (fold-pair t, seed agreement)
  - K − strong (ADR 0028): better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
```

- **K against the strong tier goes from mostly better to mixed.**
  - Development: better 13, no difference 7, worse 1 becomes better 5, no difference 9, worse 7
    (Wilcoxon over tables, p = 0.91).
  - Confirmation: 12, 1, 2 becomes 6, 4, 5.
  - The bar, the tuned bar, missForest and D4's 13 of 21 are unchanged.
- **The thesis's statement against the strong tier changes:**
  - K leads it at 20–60nan on spambase, letter, pendigits and biodeg.
  - TabPFN leads at 80nan and on kr-vs-kp and credit-g from 40nan.
  - K keeps the imputed distribution closer to the truth (E56).
- **Every earlier ticket's K − strong count** was the live four's. It stays as written, and is now
  read as `strong_live`.
- **Seed sets without a TabPFN pass** (42/7/13, 404/505/606, 1001/2002/3003, the MNAR family) read
  `strong_live` until one runs. Each pass is offline, so it needs no change to the runs.
