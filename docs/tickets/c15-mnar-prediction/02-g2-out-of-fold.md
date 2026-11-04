# C15 Stage 0, second gate: G2 read out of fold over every training and validation row (E123; pre-registered 2026-11-03)

This ticket was written and committed before the gate's first cell (23:56 GMT-3).

- **Why:** E122 ([01](01-stage0.md)) left G2 invalid: two of the three positive controls (biodeg, kc2) had means above
  0.02 but intervals below 0. Each cell read only the validation rows (about 106 on biodeg, 52 on kc2), so the gate
  lacked power. This reads the same cells on about 8 times the rows.
- **A deviation, decided after E122's result was seen:** the user chose it ("opção 1", 2026-11-03) among three
  options (a gate with more power, a stronger control, accepting the stop). E122 stays recorded as invalid.
- **Binding now:** if E123's controls fail too, C15 stops at Stage 0 for good; no third form of G2 is tried.
- **Code:** `c15_stage0.py --g2 --oof` and `--tally --oof` (`g2_oof_cell`). Smoke-tested on one non-control cell
  (kc2_probit_20nan, seed 42, fold 1) only. No change to `src/`. CPU only, on gorgona8.

## Design (fixed before the first cell)

- **The same** 19 candidates, 15 cells (seeds 42/7/13 × folds 1–5 of the fork's `build_folds` at `--cv_folds 5`),
  GBDT, indicators and probe as E122.
- **Per cell:** the rows are the fold's training and validation rows; its test rows stay unread. missForest is fitted
  once on those rows' features (never the label) and fills them. The GBDT with and without the indicators, and the
  indicators-only probe, are each read on 5 inner folds (StratifiedKFold, the cell's seed); their held-out
  probabilities are pooled over all the rows, and the cell's reading is (LL_without − LL_with) / LL_without on them.
- **What differs from E122:** the reading uses out-of-fold predictions on every training and validation row instead
  of the validation rows alone, and the missForest fit sees those rows' features without their labels.

## Rule, fixed now (unchanged from E122)

- **Informative:** the mean drop over the 15 cells is at least 0.02, with its 95% t interval above 0.
- **Validity:** all three F5 controls must be informative; if one is not, G2 is invalid and C15 stops (binding, above).
- **P:** the F1 variants and the odds-3 F3 variants that are informative; F2 and odds-1.5 F3 are read descriptively.
- **What follows:** P goes to the user with Stage 1's ticket. **Read only with every cell done.** GBDT stays internal.
