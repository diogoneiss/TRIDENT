# A validation-only tuning budget for the baselines, co-reported offline

## Status

Accepted. The plan of 2026-10-17 proposed it, and the user accepted every recommendation on
2026-10-16 at about 22:45 GMT-3: options B of this ADR, D4 and D8. It is written after E42 (ticket
[imputation-protocol/02](../tickets/imputation-protocol/02-baseline-budget.md)), whose two passes
ended on 2026-10-18.

It amends ADR [0007](0007-baseline-imputers-scored-on-the-decoders-cells.md) decision 5 (the
baselines' settings are fixed, with no key and no flag): those fixed settings stay what every run
logs, and a declared, tuned version now exists beside them, offline. ADR
[0020](0020-one-reading-of-imputation-comparisons.md) decision 4 already reads its result.

## Context

The model's configuration was tuned on these variants, and the baselines ran at their library
defaults. The review of 2026-10-16 named this threat 1: the bar the model beats may be weak because
nobody tuned it. One hand-picked hgb recipe moved the bar on a probe, but that recipe was chosen on
the test split, so the size of the effect was unknown.

E42 gave every weak-tier family and missForest the same declared budget and the same selection
signal as the model's checkpoint:

- **Grids:**
  - knn: k ∈ {5, 10, 20, 50};
  - hgb and lgbm: 12 settings each;
  - missForest: 4 settings.
  - At 60% missing and above, each setting is also read shrunk toward the training mean by
    α ∈ {0, 0.25, 0.5}.
- **Selection:** per fold, on the validation rows' own gaps, the signal K uses. Every setting is
  fitted on the training fold only, with no refit on train plus validation, and only the chosen
  setting is scored on the test split.
- **Masked-cell selection** is the sensitivity reading.

**What E42 found:**

- **The budget lowers the bar on most variants:** on 17 of 21 (101/202/303) and 9 of 12
  (404/505/606), typically by 1–3%, at most 5.7%.
- **It moves few of K's verdicts:** better / no difference / worse is 13/5/3 and 7/2/1/2 against the
  tuned bar, against 16/3/2 and 10/1/1 against the untuned one. The changes are kr-vs-kp at 20, 60
  and 80nan, and spambase_80nan.
- **A tuned missForest** beats the live one only at 60–80nan. It moves 2 of K's 48 verdicts against
  the strong tier.
- **On E44's confirmation set,** K beats both bars on 9 of 15 variants.

## Decision

1. **The tuned baselines are computed offline, never live.**
   - The driver is `scripts/experiments/baseline_budget_run.py`, with the imputers in
     `src/training/tuned_baselines.py`.
   - It writes per-fold CSVs beside a study, never the store.
   - A live tuned baseline would cost about 12 times a fit, and it would need new cache families
     (option C, rejected).
2. **The budget is the one E42 declared:** the grids, the selection on the validation gaps, no
   refit, and the shrinkage at 60% missing and above.
   - Changing it is a new study, not an edit.
   - Ties keep the family's current setting.
3. **The tuned bar is the best tuned weak-tier family** (with mean/mode), chosen per run on its CV
   mean, as ADR 0007 decision 8 chooses the bar. Its CSV column is
   `impute/<population>/baseline/tuned_bar/impute_score`. The tuned missForest joins the strong
   tier only as a reported opponent ("tuned_strong").
4. **The opponent of record (D4, ADR 0020 decision 4):** a thesis win counts only where K beats both
   the untuned bar (what the store logs) and the tuned bar.
   - The development table (`scripts/experiments/thesis_table.py`) and E44's primary read it that
     way.
   - It gives 13 of 21, 7 of 16 and 7 of 12 on the development seed sets, and 9 of 15 on the
     confirmation set.
5. **Every new study that makes a claim against the bar** runs the driver on its own folds. E44's
   pass took about 4 hours on 4 CPU lanes.

## Consequences

- **The bar in the store is unchanged.** `baseline/best` and `gap_to_best_baseline` stay the untuned
  bar; the tuned numbers live in the studies' CSV directories, named in their tickets.
- **The tuned bar is an oracle on natural data, as K is.** It is selected on the validation gaps'
  truth. The masked-cell selection is its deployable twin, and K beats that bar on 14 of 21 and 9
  of 12.
- **A larger grid could lower the bar further.** The grids are a declared budget, not a search for
  the best possible imputer.
- **The cost:**
  - E42's weak tier took 55.8 process-hours at 4 threads for 207 (variant, seed) files.
  - Its missForest pass took about a day on 4 lanes.
