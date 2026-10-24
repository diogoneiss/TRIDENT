# Per-column weights for the K + TabPFN blend, cross-fitted: do they reach past TabPFN where the equal blend could not? (E69; pre-registered 2026-10-23)

This ticket was written and committed before the study's first cell. Two smoke pairs ran before
it, written to a scratch directory and not counted (see "Launch").

- **Why:** this is T37, accepted by the user on 2026-10-23 ("Aprovo T36 e T37").
  - E60 found an equal-weight blend of K's and TabPFN's fills beat both on 10 of 15 confirmation
    variants ([15](15-tabpfn-blend.md)), and E66 found it at the rule's floor on E40's set, 3 of 9
    ([21](21-tabpfn-blend-replication.md)).
  - Where one method dominates, an equal weight lands near the better one at best. On kr-vs-kp
    and credit-g, where TabPFN leads from 40nan and E61 put 71–100% of K's gap in the categorical
    cells, the equal blend keeps the model's category and cannot reach TabPFN.
  - A weight per column is the untested lever.
- **The user's go:** "vamos com t37" (2026-10-23, about 22:50 GMT-3). It runs ahead of T35 by that
  go; TASKS had T37 after T33–T35.
- **Route (a) of T37, offline:** `scripts/experiments/blend_weights.py`. It reuses E60's fills
  (`tabpfn_blend.py`'s ledger reader and TabPFN refit) and writes nothing to the store or to `src/`.
  Route (b), weights from the validation rows' own gaps, needs fills the ledgers do not hold and is
  not run.
- **Descriptive, like E60:** no output or default changes on any reading. A hybrid output would
  need its own pre-registered rule on a fresh seed set.

## Design (fixed before the first cell)

- **Cells, 21 variant sets, 63 pairs:**
  - **confirmation:** E44's 15 confirmation variants at 707/808/909, E60's set;
  - **e40:** E40's kr-vs-kp and credit-g at 40, 60 and 80nan, at 404/505/606, E61's set (a
    subset of E66's).
- **The fills, E60's:**
  - **the model's:** K, from the run's fold ledger, asserted equal to the run's K fold by fold;
  - **TabPFN's:** v2 refitted on the same training fold and asked for the ledger's induced cells.
    The smoke pairs reproduced E60's and E66's TabPFN fold scores to 1e-16.
- **The weighted blend, cross-fitted.** All five folds' fills are collected first. For fold k and
  each column, the weight comes from the other four folds' induced cells of that column:
  - **numerical:** the model's weight w from {0, 0.25, 0.5, 0.75, 1}, the blend being
    w·model + (1 − w)·TabPFN, that minimises the summed squared error in the ledger's z units;
  - **categorical:** the model's or TabPFN's category, whichever is right on more of those cells;
  - **ties, and a column with no cells on the other folds:** 0.5 for a numerical column and the
    model for a categorical one, which is E60's equal blend.
- **Why this is the grid's best and why it does not leak:**
  - `score_cells` pools numerical cells into one RMSE and categorical ones into one accuracy, so
    squared errors and correct counts add across columns. A per-column choice is the grid's best
    on the other folds.
  - Fold k's truth never enters fold k's weights; the other folds' cells were scored by their own
    models. Because 0.5 is in the grid, the weighted blend can lose to the equal one only on the
    held-out fold.
- **Outputs, in `/scratch2/diogoneiss/trident-2026-10-23-blend-weights/`:**
  - `csv/`: per fold, `impute_score` for the model, TabPFN, the equal blend and the weighted blend;
  - `weights/`: the chosen weight per fold and column;
  - `cells/`: both fills per induced cell, so the reading can be redone offline.
- **Running:** two GPU lanes, large tables first, resumable, about 2 h (E60 and E66's per-pair
  times). Threads capped at 4, beside E68's four CPU lanes.

## Measures and reading, fixed now (`blend_weights.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI of the
opponent's value:

- **Per variant:** weighted − equal, weighted − TabPFN, weighted − model, and the equal blend's
  own equal − TabPFN and equal − model as this run recomputes them.
- **Primary, weighted − TabPFN on the 10 hard variants,** the ones where E60's or E66's equal
  blend was not better than TabPFN: electricity_40nan (no difference), electricity_80nan,
  vehicle_80nan and biodeg_80nan (equivalent), and kr-vs-kp and credit-g at 40/60/80nan (worse).
  - **supported** (per-column weights reach past TabPFN where the equal blend could not): better on
    4 or more;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **Secondary:**
  - weighted − equal over all 21: do the weights add anything at all;
  - how many variants the weighted and the equal blend each beat both the model and TabPFN on.
- **Descriptive (`--weights`):** TabPFN's mean weight per variant and kind, the spread of chosen
  weights, and how often a column's modal choice holds across its 15 folds. Which columns prefer
  TabPFN, and how stable that is.
- **What each reading leads to (no default changes):**
  - **supported:** a per-column hybrid output is worth a pre-registered test on a fresh seed set,
    for the user to decide.
  - **refuted:** E60's complementarity is the whole of it; per-column weights do not overturn
    TabPFN's lead where it holds.
  - **mixed:** reported, and the user chooses.

## Launch

- **Smoke pairs (not counted), to a scratch directory:** kc2_80nan at 707 and kr-vs-kp_40nan at
  404, run after the script and its rule were written and before this ticket. Both matched the
  run's K on every fold and reproduced E60's and E66's TabPFN and equal-blend fold scores (largest
  difference 1.1e-16). Their fold scores were seen; neither is a hard variant's only seed, and the
  rule above was in the script before they ran. Their pairs run again in the study.
