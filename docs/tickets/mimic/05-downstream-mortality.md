# Fills and gap patterns judged by a mortality classifier (E150; pre-registered 2026-11-05 as E152)

Written and committed before any table exists; blocked on [01](01-external-datasets-root.md) and
[02](02-tables.md). The third way: no ground truth for the cells at all, only the outcome.

- **Why:** the natural gaps cannot be scored cell by cell, but what is done with them can be judged by a
  classifier's score on an outcome nobody imputes. On this cohort the pattern itself tells: deaths are 19.9% where a
  lactate was measured and 7.9% where it was not.
- **Hypotheses:** (i) a classifier fed the decoder's fills of the natural gaps is better than one fed the mean/mode
  fill; (ii) a classifier fed the mean/mode fill with one indicator per column is better than one without them.

## Design and rule (fixed before the first cell)

- **Tables:** `mimic-cardio_natural` and `mimic-sofa_natural`; label: death in hospital (16.5% by the probe).
- **Downstream classifier, one and fixed:** scikit-learn's logistic regression (L2, `C = 1`, `class_weight =
  "balanced"`, `max_iter = 1000`) on standardised numerical columns and one-hot categories.
- **Inputs, four:**
  - `N`: mean/mode fill.
  - `NI`: `N` and one indicator for each column with gaps.
  - `T`: the decoder's fills of the natural gaps, from [E148](03-masked-cells-natural-gaps.md)'s run of the same
    seed. Fold k reads fold k's export alone: that fold's decoder (checkpoint by `masked`, as E148) fills the gaps
    of its training, validation and test rows, and the export carries the fold's partition. A row's fill from
    another fold's decoder would come from a model trained on this fold's test rows. The export of ticket 01 lands
    before E148's first cell, so both studies read the same runs.
  - `TI`: `T` and the indicators.
- **Leakage:** the mean/mode fill, the scaler and the decoder are fitted on the fold's training rows (the decoder
  also selects its epoch on the fold's validation rows); the label never enters an imputer.
- **Cells:** 2 tables × seeds 42/7/13 × five folds × four inputs, on the CPU; no new training.
- **Primary, four tests as one family:** per table, `T − N` and `NI − N` in macro-F1 on the test rows, paired per
  (seed, fold), 15 replicates, plain 95% t interval, Holm over the four. Each contrast **holds** with Holm p < 0.05
  and a mean above 0.
- **Beside it, descriptive:** `TI − NI` (do the fills add once the pattern is given); AUROC and AUPRC for every
  input; as levels only, TRIDENT's own classifier on the natural table (`--task classification`, gaps as `[NULL]`)
  and TabPFN reading the gaps natively (ADR 0032's tier).
- **What this cannot say:** a better classifier does not make the fills closer to the unmeasured values, and a
  linear classifier is one reader among many.
- **What follows:** no default change and no ADR either way.
- **Read only with all cells done.**

## Implementation (2026-11-06, before any cell)

`scripts/experiments/mimic_mortality.py` reads E148's state directory and fixes what the design left open:

- The classifier is fitted on the fold's training rows and scored on its test rows. The validation rows, on which
  the decoder chose its epoch, are left out of both.
- The numerical columns are standardised by the training rows after the fill; the categories are one-hot; an
  indicator is 0 or 1, unscaled, for each column with a gap anywhere in the table.
- The reader refuses to run unless all six of E148's cells are done, and checks that the fills cover every gap of
  the table once and no measured cell.
- Checked on the toy smoke's exports (two folds, a decoder of 40 epochs): the four inputs are built and scored.
