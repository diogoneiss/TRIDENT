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

**Done 2026-10-28, 22:44 GMT-3: supported (development).** The 30 cells ran 21:01–22:42 GMT-3
on gorgona8 in two lanes, with none failed. The smoke cell is one of them.

The tally, unedited (`hybrid_selection_rules.py --tally`; the rule's name prints twice on the
secondary lines, a cosmetic slip in the print):

```
- reproduction: largest fold difference from E82's TabPFN and deployable scores 2.28e-05

- `electricity_40nan` (H 0.7441, TabPFN 0.7435): column − tabpfn -0.0144 [-0.0191, -0.0097] better; table − tabpfn -0.0122 [-0.0172, -0.0072] better; shrunk − model -0.0150 [-0.0296, -0.0005] better; shrunk − tabpfn -0.0145 [-0.0192, -0.0098] better; shrunk − column -0.0001 [-0.0001, -0.0000] better; pooled − tabpfn -0.0146 [-0.0190, -0.0102] better
- `electricity_80nan` (H 0.8995, TabPFN 0.8904): column − tabpfn -0.0009 [-0.0014, -0.0004] better; table − tabpfn -0.0012 [-0.0017, -0.0007] better; shrunk − model -0.0100 [-0.0132, -0.0069] better; shrunk − tabpfn -0.0010 [-0.0015, -0.0004] better; shrunk − column -0.0001 [-0.0001, -0.0000] better; pooled − tabpfn -0.0013 [-0.0018, -0.0009] better
- `vehicle_80nan` (H 0.8178, TabPFN 0.7569): column − tabpfn +0.0048 [+0.0005, +0.0090] worse; table − tabpfn -0.0017 [-0.0040, +0.0007] equivalent; shrunk − model -0.0577 [-0.0643, -0.0510] better; shrunk − tabpfn +0.0032 [-0.0006, +0.0071] equivalent; shrunk − column -0.0015 [-0.0035, +0.0004] equivalent; pooled − tabpfn -0.0008 [-0.0030, +0.0014] equivalent
- `biodeg_80nan` (H 0.9777, TabPFN 0.9195): column − tabpfn +0.0054 [+0.0028, +0.0080] worse; table − tabpfn -0.0005 [-0.0130, +0.0120] none; shrunk − model -0.0598 [-0.0666, -0.0531] better; shrunk − tabpfn -0.0016 [-0.0060, +0.0027] equivalent; shrunk − column -0.0070 [-0.0117, -0.0024] better; pooled − tabpfn -0.0009 [-0.0041, +0.0022] equivalent
- `kr-vs-kp_40nan` (H 0.7247, TabPFN 0.7139): column − tabpfn +0.0017 [-0.0042, +0.0077] none; table − tabpfn +0.0024 [-0.0006, +0.0054] equivalent; shrunk − model -0.0090 [-0.0162, -0.0018] better; shrunk − tabpfn +0.0017 [-0.0042, +0.0077] none; shrunk − column +0.0000 [+0.0000, +0.0000] equivalent; pooled − tabpfn +0.0057 [-0.0002, +0.0115] none
- `kr-vs-kp_60nan` (H 0.8742, TabPFN 0.8261): column − tabpfn +0.0153 [+0.0114, +0.0192] worse; table − tabpfn +0.0039 [-0.0045, +0.0124] none; shrunk − model -0.0328 [-0.0393, -0.0262] better; shrunk − tabpfn +0.0153 [+0.0114, +0.0192] worse; shrunk − column +0.0000 [+0.0000, +0.0000] equivalent; pooled − tabpfn +0.0071 [+0.0040, +0.0101] worse
- `kr-vs-kp_80nan` (H 1.0098, TabPFN 0.9430): column − tabpfn +0.0221 [+0.0120, +0.0321] worse; table − tabpfn +0.0000 [+0.0000, +0.0000] equivalent; shrunk − model -0.0446 [-0.0553, -0.0339] better; shrunk − tabpfn +0.0222 [+0.0122, +0.0322] worse; shrunk − column +0.0002 [-0.0001, +0.0005] equivalent; pooled − tabpfn +0.0104 [+0.0050, +0.0158] worse
- `credit-g_40nan` (H 0.9274, TabPFN 0.9042): column − tabpfn +0.0021 [-0.0064, +0.0106] none; table − tabpfn +0.0013 [-0.0034, +0.0059] equivalent; shrunk − model -0.0218 [-0.0271, -0.0164] better; shrunk − tabpfn +0.0014 [-0.0072, +0.0100] none; shrunk − column -0.0007 [-0.0010, -0.0004] better; pooled − tabpfn -0.0001 [-0.0062, +0.0060] equivalent
- `credit-g_60nan` (H 0.9982, TabPFN 0.9524): column − tabpfn +0.0140 [+0.0060, +0.0220] worse; table − tabpfn -0.0004 [-0.0041, +0.0032] equivalent; shrunk − model -0.0466 [-0.0612, -0.0319] better; shrunk − tabpfn -0.0008 [-0.0046, +0.0030] equivalent; shrunk − column -0.0148 [-0.0233, -0.0063] better; pooled − tabpfn +0.0043 [+0.0001, +0.0085] worse
- `credit-g_80nan` (H 1.0198, TabPFN 1.0001): column − tabpfn +0.0004 [-0.0025, +0.0034] equivalent; table − tabpfn +0.0002 [-0.0032, +0.0035] equivalent; shrunk − model -0.0205 [-0.0305, -0.0106] better; shrunk − tabpfn -0.0008 [-0.0046, +0.0029] equivalent; shrunk − column -0.0013 [-0.0061, +0.0035] equivalent; pooled − tabpfn +0.0000 [-0.0032, +0.0033] equivalent

Secondary `column`: better than H on 10 of 10, worse on none, worse than TabPFN on 5 (['vehicle_80nan', 'biodeg_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_60nan']) -> refuted (fold-pair t, seed agreement)
Secondary `table`: better than H on 10 of 10, worse on none, worse than TabPFN on 0 (none) -> supported; against column: better 4 / no difference 2 / equivalent 4 / worse 0 (fold-pair t, seed agreement) (fold-pair t, seed agreement)
Primary `shrunk`: better than H on 10 of 10, worse on none, worse than TabPFN on 2 (['kr-vs-kp_60nan', 'kr-vs-kp_80nan']) -> supported; against column: better 5 / no difference 0 / equivalent 5 / worse 0 (fold-pair t, seed agreement) (fold-pair t, seed agreement)
Secondary `pooled`: better than H on 9 of 10, worse on none, worse than TabPFN on 3 (['kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'credit-g_60nan']) -> mixed; against column: better 5 / no difference 1 / equivalent 4 / worse 0 (fold-pair t, seed agreement) (fold-pair t, seed agreement)
```

- **Reproduction:** `column` gives E82's result again: better than H on 10, worse than TabPFN on 5,
  refuted. The largest fold difference from E82's scores is 2.3e-5. TabPFN's refits are not
  bit-identical, and the difference is far below any verdict.
- **Primary, `shrunk`: supported (development).**
  - Better than H on 10 of 10, worse on none.
  - Worse than TabPFN on 2 (kr-vs-kp at 60 and 80nan), the rule's limit.
  - Against E82's rule: better on 5, equivalent on 5, worse on none.
  - As expected, it closes vehicle and biodeg at 80nan, where it is now equivalent to TabPFN. It
    is unchanged on kr-vs-kp, where every categorical column has enough cells to keep its own pick.
    It also closes credit-g at 60nan.
- **Secondary, `table`: supported, and the strongest of the four.**
  - Better than H on 10 of 10, and worse than TabPFN on none.
  - It beats TabPFN on both electricity variants and is equivalent or without verdict on the
    other 8.
  - One categorical pick per table is what reaches TabPFN on kr-vs-kp.
- **Secondary, `pooled`: mixed.** Worse than TabPFN on 3 (kr-vs-kp at 60 and 80nan, credit-g at
  60nan).
- **What follows, as fixed before the run:**
  - This is a development read (rules written after E82, scored on its seeds), so ADR 0031 does
    not change.
  - The next step is a confirmation on fresh seeds in E82's exact shape, with the user's go.
  - That study must pre-register one rule. `table` did better than the primary here, so the choice
    between `table` and `shrunk` is made, and written down, before its first cell.
