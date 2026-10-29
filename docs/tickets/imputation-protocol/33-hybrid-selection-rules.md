# Can fewer or better-fed choices close the deployable hybrid's gap to TabPFN? (E89; pre-registered 2026-10-28)

This ticket was written and committed before the study's first lane.

- **Why:** E82 ([31](31-deployable-hybrid.md)) chose the H + TabPFN hybrid's weights per column on
  hidden observed validation cells, with no complete table. It beat H on 10 of 10 and KC on 7, but
  lost to TabPFN on 5, and the selection was the loss:
  - thin cells: vehicle and biodeg at 80nan have a median of 18–21 selection cells a column a fold;
  - categorical picks: kr-vs-kp and credit-g.
  - E82's ticket named the candidates: fewer parameters to choose (one weight per table, or
    shrinkage toward the equal blend), and more selection cells.
- **The user's go:** "continue o e82 então" (2026-10-28, about 20:50 GMT-3). The user wants the
  method's viability in a real deployment (T39).
- **This is a development read, by construction.** The rules below were written after reading
  E82's failure on seeds 1212/2323/3434, and E89 scores them on the same seeds and the same H runs.
  - **If it comes out supported,** the Outcome says "supported (development)", nothing changes in
    ADR 0031, and the next step is a confirmation on fresh seeds in E82's exact shape: 30 new H
    cells with the in-place selection read, then the blend, at about E82's cost. That needs the
    user's go.
  - **If it comes out refuted or mixed,** the deployable route stays where E82 left it.
- **Code:** no change to `src/`. `scripts/experiments/hybrid_selection_rules.py` (`cells`,
  `--tally`) reuses E82's script; `run_hybrid_selection_rules_queue.sh` runs two GPU lanes.
- **Smoke:** vehicle_80nan s1212 ran before this commit, twice (the second after the table rule's
  scale moved to the training split's sd). Only the reproduction of E82 was read: H identical;
  TabPFN, the equal blend and E82's deployable within 1.1e-9 of E82's scores, fold by fold. No new
  rule was scored. The cell counts as one of the 30.

## Design (fixed before the first lane)

- **Cells:** E82's 30 (its 10 hard variants × seeds 1212, 2323, 3434), on E82's own H runs and
  selection files. No training.
  - `cells` refits TabPFN v2 on each training fold, as E82's `blend` did. It writes the test split's
    induced cells and the validation split's hidden observed cells, both with H's and TabPFN's
    fills, so every rule is scored offline.
  - Two GPU lanes on gorgona8, the largest cells first, so the six electricity cells (~18 min each)
    are split between the lanes. About 2 h. Nothing is written to the store.
- **Four rules.** None reads the complete table. The selection truths are observed values, and
  each fold's fills come from that fold's own model on its own validation rows.
  - **`column`:** E82's deployable rule (E69's grid per column on fold k's validation cells). It
    reproduces E82 and is the reference.
  - **`table`:** one numerical weight per table (the grid, with each column's residuals scaled by
    its training split's sd) and one categorical pick per table, on fold k's validation cells.
    - **Expected:** it helps the categorical tables (kr-vs-kp, credit-g), where per-column picks
      are noisy, and costs on electricity, where per-column weights have the cells to work.
  - **`shrunk` (the primary):** each numerical column's weight is pulled toward the equal blend,
    `(n·w + N0·0.5) / (n + N0)`, with n the column's selection cells. A categorical column keeps its
    own pick when it has at least N0 cells, and takes the table's pick below that.
    - **N0 = 50, fixed now, and no other value is tried.** With it the thin tables keep about 25–30%
      of their own weight and electricity about 95%.
    - **Expected:** it targets vehicle and biodeg at 80nan, and is close to `column` on electricity
      and on kr-vs-kp (median 191 cells). So it is not expected to move kr-vs-kp or credit-g much.
  - **`pooled`:** the per-column rule on the validation cells of all five folds. It is the offline
    stand-in for E82's "more selection cells" candidate. H's fills on the test split's observed
    cells were not saved by E82, and they would need a re-read.
    - **Expected:** it helps where cells are thin, by five times the count.

## Measures and rule, fixed now (`hybrid_selection_rules.py --tally`)

- **Per variant, for each rule,** rule − H, rule − TabPFN and rule − `column` on `impute_score`
  over 15 fold pairs. The rule is the project's, with a SESOI of 1% of the opponent's mean, as E82.
- **The call, per rule, in E82's form:**
  - **supported** if better than H on 8 or more of 10, worse on none, and worse than TabPFN on 2
    or fewer;
  - **refuted** if better than H on 5 or fewer, or worse on 2 or more, or worse than TabPFN on 5
    or more;
  - **mixed** otherwise.
- **The primary is `shrunk`.** `table` and `pooled` are secondary and read with the same rule.
  `column` is the reproduction check: it must give E82's result.
- **Reproduction, printed first:** the largest fold difference from E82's TabPFN and deployable
  scores.

## Outcome

(open)
