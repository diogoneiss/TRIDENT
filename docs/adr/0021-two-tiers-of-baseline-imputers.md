# Two tiers of baseline imputers, every one logged, each tier with its own best

## Status

The design was accepted on 2026-10-16, as the user's own proposal, in reply to option C of the
[plan](../plans/2026-10-17-experiment-improvements.md) (§2b): "fazer isso duas vezes, uma com
baselines fracos (existente) e outra com os novos fortes". It extends ADRs 0007, 0016 and 0017.

**Implemented on 2026-10-17, at the user's request** ("faça a implementação"), on the branch
`feat/adr-0021-strong-tier`. **Landed on 2026-10-17 at about 18:20 GMT-3** (merge f21ee33), after
E42's lanes and E46 had ended, with nothing importing the checkout; the merged tree passes 394 unit
tests and both integration fixtures. Every imputation run from f21ee33 on logs the strong tier
live.

**Tests and checks:**

- the unit suite (392 tests, mypy included) and both integration fixtures pass;
- a live kc2_60nan cell at seed 404 equals E40's on all 47 shared columns;
- its linear arms equal E41's offline CSV within 4.4e-16.

**Full dry run, 2026-10-17, 22:52 to 23:17 GMT-3,** on the user's go. It ran on a copy of the store
taken through SQLite's backup API at 22:50, in four lanes split by variant, and filled the live
baseline cache.

- **Planned:** 1,177 runs; 1,176 plans.
- **Refused: 1,** `impute_electricity_20nan_20260925_044917`. Its logged mean/mode `acc_cat` is
  0.000, against 0.145 recomputed, so its cells are not reproduced and it is left alone.
- **Against E41's offline CSVs:**
  - 774 runs have an E41 cell, and their arms agree within 4.4e-16.
  - 4 differ, by 0.025 to 0.095. They are 2-fold runs (`null-path-recheck-2026-09-30`), compared
    with E41's 5-fold cells of the same seed: other folds, not an error.
- **The strong tier's best on the induced population:** missForest 663, condexp 277, ICE 159,
  SoftImpute 77.
- **Applied on the user's go, 2026-10-18, 00:03 to 02:41 GMT-3.**
  - **The backup first:** a backup through SQLite's API at 23:57 GMT-3 on 2026-10-17
    (`/scratch2/diogoneiss/trident-2026-10-17-store-backup-before-apply/mlflow.db`,
    `integrity_check` ok).
  - **Written:** 1,176 run trees, re-mirrored. Refused: the same 1 electricity_20nan run as the
    dry run.
  - **Checked afterwards:**
    - a second dry run finds 0 runs pending;
    - 1,176 source parents and 1,176 mirror parents carry `strong_tier_backfilled=true` and
      `baseline/best_strong`.
  - **Log:** `/scratch2/diogoneiss/trident-2026-10-18-store-apply/`.

## Context

**What runs log today:**

- ADR 0007's bar, the best of mean_mode, knn5, knn10, hgb and lgbm since ADR 0017;
- missForest as an external baseline, with a gap of its own.

**What E41 measured** ([imputation-protocol/01](../tickets/imputation-protocol/01-linear-baselines.md)),
offline on the decoder's cells, with three classical arms:

- condexp beats the bar on 2–4 variants per seed set;
- SoftImpute and ICE beat it on at most 2;
- an arm beats the model only on credit-g_80nan, where the model is above the mean/mode fill.

Putting them into the bar would change what the bar means; leaving them unlogged would hide them.

## Decision

1. **The weak tier is today's bar, unchanged.** `baseline/best`, the tag
   `best_baseline/<population>` and `gap_to_best_baseline*` keep their names and values on every
   run.
2. **The strong tier is missForest plus condexp, SoftImpute and ICE.**
   - Each run logs `impute/<population>/baseline/best_strong/*`, the tag
     `best_strong_baseline/<population>` and `gap_to_best_strong_baseline` with its `_pct`. The
     best is chosen once per run on the CV mean, as ADR 0007 decision 8 chooses the weak one.
   - `gap_to_missforest` stays.
   - Stronger baselines join the strong tier when they are cheap enough to run live.
3. **The arms are a fourth cache family, `linear`** (`src/training/linear_baselines.py`). A new
   family leaves the existing entries valid.
4. **The best of all is derived by readers** (`report_common`) and not logged.
5. **A backfill writes the strong tier onto every imputation parent,** in the ADR 0017 pattern. It
   must reproduce E41's per-fold CSVs to the last digit, and it holds the store for about 2 h, so
   it runs in a morning gap between studies after a backup through SQLite's API.

## Considered options

- **The arms in the bar.** Rejected: it retags `baseline/best` on about 1,000 runs and changes what
  "the bar" means mid-thesis.
- **Each arm external, like missForest, with no tier.** Rejected: no single opponent would read
  "the strongest classical imputer".
- **Offline only.** Rejected by the user: everything should be logged.

## Consequences

- **On today's default (101/202/303),** the model is better than the strong tier on 13 variants,
  no different on 7 and worse on 1 (credit-g_80nan), against 16 / 3 / 2 for the weak tier (E41).
- **Each run carries a few more keys** for the arms' metrics and the strong tier's best and gap.
- **The tuned bar (E42) is not a tier.** It stays offline while it costs about 12× a fit.
