# Read every imputation comparison one way: verdict tallies per opponent, the test named

## Status

Accepted on 2026-10-16. The user agreed with the recommendations D2, D3 and D4 of the
[plan of 2026-10-17](../plans/2026-10-17-experiment-improvements.md), which turns the 2026-10-16
review into work. It names how ADR 0007's bar and ADR 0017's counts are read. No training code and
no default change.

## Context

The 2026-10-16 review (wiki `projects/trident/findings-2026-10-16-baselines-methodology`, finding
1 and threat 5) found the headline "the model is below the bar on 19 of 21 variants" fragile, for
three reasons.

**It counts seed means.** The project's own verdict rule (15 fold pairs, a 95% t-interval, seeds
agreeing in sign) gives different counts, and those depend on the opponent.

**Three bar definitions coexist:**

- the store's per-run minimum;
- ADR 0017's minimum of pooled means;
- a minimum over a four-name tuple without lgbm, used by 14 report scripts. E40's report was one
  of them.

**E34 and E35 store H, not K.** They were `decode_checkpoint=loss` runs, so their headline and gap
keys hold H; a store-native query reads H where the tables meant K.

## Decision

1. **One reading module,** `scripts/experiments/report_common.py`, which every new report uses. It
   has seam tests (`tests/unit/test_report_common.py`).
   - **The model's readout comes from the run's `decode_checkpoint` tag.** K is the headline of an
     `induced` run and the `induced_checkpoint` family of a `loss` run. H and M are read the same
     way. The module refuses rather than guess.
   - **The bar (the weak tier) is the best baseline that is neither outside the bar nor of another
     class,** chosen per run on its CV mean (ADR 0007 decision 8). "Outside the bar" is
     `summary.OUTSIDE_THE_BAR`; the other classes are ADR 0021's strong tier, E42's tuned
     families and diagnostic arms. Every baseline name must belong to a class, so a new arm cannot
     slip into the bar unread.
   - **The other opponents** are missForest, the strong tier (ADR 0021), the best of all, the
     per-type bar (the best baseline per kind of cell, recombined with `impute_score`'s cell
     weights) and, where computed, the tuned bar (E42).
2. **Per variant: the verdict rule, plus equivalence.** A call without a verdict is "equivalent"
   when its interval lies inside ±1% of the opponent's mean, the smallest effect of interest (D3).
   Counts are tallies (better / no difference / equivalent / worse), always printed with the test
   named beside them.
3. **Method claims on the development set are read across tables** (D2).
   - The statistic is the per-table mean relative gap, tested with a two-sided Wilcoxon
     signed-rank test over the tables, with Holm across opponents.
   - A seed random-effects interval is the sensitivity reading.
   - Per-variant tallies are descriptive.
4. **A win counts only where the model beats both the untuned bar and the tuned bar** (D4), once
   E42's tuned bar exists for those cells.
5. **Development and confirmation are labelled.**
   - **Development:** the 21 variants and the seed sets 42/7/13, 101/202/303 and 404/505/606.
   - **Confirmation:** E44's 15 variants on 707/808/909, read once.
   - Five variants on 42/7/13 ran a configuration that is no longer the default (the files ADR
     0019 removed, the encoder before ADR 0018). They are labelled and never pooled.
   - A seed set is retired once it has decided something.

## Consequences

**The thesis table** (`scripts/experiments/thesis_table.py`,
`docs/reports/2026-10-17-thesis-table-development.md`) reads today's default on 101/202/303 as:

| opponent | verdict tally | tables below zero (two-sided Wilcoxon) |
|---|---|---|
| the bar | better 16 / none 3 / worse 2 | 8 of 9, p = 0.020 |
| missForest | 15 / 6 / 0 | 9 of 9, p = 0.004 |
| the strong tier | 13 / 7 / 1 | 9 of 9, p = 0.004 |

These match the review's independent rebuild. "19 of 21 by means" is no longer quoted.

- **ADR 0017's Outcome table** stays as written. Its counts are seed means of the E35 cells, and
  that is now named beside them.
- **E40's Outcome** ("best baseline" column; "below the best baseline: 11 of 12") read a bar of
  four. No decision rested on it; the helper's reading supersedes it.
- **Old report scripts are not rewritten.** They reproduce their pre-registered readings, and new
  studies use the helper.
