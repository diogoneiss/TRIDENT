# A cross-fit side pick (E168, T54; pre-registered 2026-11-09 00:57 GMT-3)

This ticket was written and committed before the rule was run on any table.

- **Why:** E165's pick ([61](61-hybrid-side-pick.md)) chooses TabPFN or missForest on the same validation cells that
  choose each hybrid's weights, so a side that overfits those cells is favoured. Its two losses to H were on small
  tables (spambase_probit_60nan, kc2_marf_60nan), where that bias is largest.
- **The user's go:** "C: escolha cross-fit" (2026-11-09 00:40 GMT-3).
- **The rule (`missforest_hybrid.both_sides`, `crossfit`):** per fold k and side s, each of the 5 validation groups is
  filled with the `table` weights chosen on the other 4 groups; the side whose filled validation cells score lower
  (one `impute_score` over all groups) is taken, a tie keeping TabPFN. The test fills are that side's `table` hybrid,
  weights chosen on all 5 groups, as in the pick. Only the side choice differs from E165's.

## Readings (fixed now)

- **Development, descriptive:** on E164/E165's 38 variants (seeds 4545/5656/6767, CPU on medusa1, minutes):
  `uv run --python 3.11 python -B scripts/experiments/missforest_generality.py --tally e168`. It prints E165's rule
  applied to `crossfit`, and `crossfit` − `pick`. It is read after this ticket is committed and before E167's
  cells, and changes nothing below.
- **Primary, on E167's 12 fresh variants** ([62](62-hybrid-fresh-seeds.md)), E165's rule on `crossfit`:
  **supported** if worse than H on none, worse than `table_tabpfn` on at most 1 and better on at least 2; **refuted** if
  worse than `table_tabpfn` on 3 or more, or better on none; **mixed** otherwise. Secondary: `crossfit` − `pick`.
  Read in E167's printout.
- Fold-pair verdict, 1% SESOI. **What it leads to:** whether a side-picking rule is worth the user's ADR call.
