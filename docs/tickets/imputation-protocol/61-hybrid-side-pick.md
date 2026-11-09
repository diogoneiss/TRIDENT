# A deployable rule that picks the hybrid's other side, TabPFN or missForest (E165, T54; pre-registered 2026-11-08 21:14 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E162 left this open: the missForest side was chosen with the complete table, which a deployment does not
  have. This rule picks the side from the same hidden validation cells that choose the hybrid's weights.
- **The user's go:** "faça 1 2 e 3" (2026-11-08), on "2. Regra que escolhe o outro lado do híbrido, TabPFN ou
  missForest, sem a tabela completa".
- **Disclosure:** the same as E164's ([60](60-missforest-hybrid-generality.md)). Nothing of E164 has been read; both
  are pre-registered before either's first cell.

## Design

- **Cells:** E164's 114 pairs; no cell of its own.
- **The rule (`pick`, `missforest_hybrid.both_sides`), per fold:**
  - fit the `table` rule twice on fold k's validation cells, once with TabPFN and once with missForest as the
    other side;
  - score each hybrid on those same validation cells, with the fold's mean/mode denominators;
  - take the side that scores lower, keeping TabPFN, ADR 0031 decision 6's side, on a tie.
- **Stated:** the side is chosen on the cells that choose the weights, one more in-sample choice on top of the
  `table` rule's. A within-group cross-fit is the upgrade if the result reads optimistic. It is not built now.

## Measures and rule (fixed now)

- **Verdict:** as E164.
- **Primary,** `pick` on the 38 variants, against `table_tabpfn`, the accepted rule, and against H:
  - **supported** if it is worse than H on none, worse than `table_tabpfn` on at most 1, and better than
    `table_tabpfn` on at least 2;
  - **refuted** if it is worse than `table_tabpfn` on 3 or more, or better on none;
  - **mixed** otherwise.
- **Secondary:** `pick` − `table_missforest`, and how often it took missForest.
- **Read:** once, after E164's read, with all 114 pairs in, `missforest_generality.py --tally e165` on medusa1.
- **What it leads to:** whether ADR 0031 decision 6 could name a side rule is the user's call.
