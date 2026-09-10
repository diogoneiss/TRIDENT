# 03. What is the imputation ground truth, and on which cells is error scored?

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: none

## Question

"Comparing with the original dataset with masked values" admits two ground truths, and
the choice shapes the loss, the metrics, the preview and the CLI:

- **(a) Self-masked observed cells.** Apply the dynamic masking of `preprocess_table`
  to observed cells (exactly as pretraining does) and score the decoder at those
  `[MASK]` positions. Works on every variant, including `_00nan`; the training signal
  and the evaluation signal are the same kind of cell.
- **(b) Induced-missing cells.** For a `_20nan`..`_80nan` variant, score the decoder at
  the cells that are NaN in the variant against the row-aligned `_00nan` sibling (verified
  aligned; see the map's facts). This is the true imputation benchmark, but those cells
  are `[NULL]` tokens to the model, never `[MASK]`, so it evaluates the `[NULL]` path
  rather than the `[MASK]` path, and it does not exist for `_00nan`.
- **Both**: (a) as the training/validation signal and fold-ranking metric, (b) as an
  extra test-only metric whenever a `_00nan` sibling exists.

Sub-questions that settle with it:

- Which split(s) are scored: test only, or also validation for checkpoint selection?
- For (a): is the evaluation mask drawn once with a fixed seed (reproducible numbers) or
  re-rolled like training masks? At the training `PROB_MASCARA`, or at a separate
  evaluation mask rate?
- For (b): does the model see the variant's `[NULL]` at those cells (imputing missing
  data) or a `[MASK]` substituted in (treating them like self-masked cells)?
- Does the answer change the glossary? Propose the canonical term *imputation ground
  truth* in `CONTEXT.md`.

Recommended starting answer: **both**, with (a) on validation (fixed-seed mask at
`PROB_MASCARA`) for checkpoint selection and on test as the fold-ranking signal, and (b)
on test as `test/induced_*` metrics logged only when the sibling exists. State the
assumption explicitly in the ADR.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

Decisions:

1. **Scored cells: both populations.** *Self-masked cells* (observed test-fold cells
   hidden by `preprocess_table`) are scored on every variant. *Induced-missing cells*
   (NaN in the `_XXnan` variant, observed in `_00nan`) are scored whenever the
   row-aligned `_00nan` sibling exists and are the headline number there. Cells that are
   NaN in `_00nan` itself are never scored. The two populations are logged under
   separate metric names (naming is ticket 05).
2. **Token path.** Induced-missing cells are presented to the model as `[MASK]` at
   evaluation, so the decoder sees the token it was trained on. A `[NULL]`-path
   diagnostic is ticket 12.
3. **Training never reads the sibling.** The training signal is self-masking on the
   observed cells of the training fold only; the `_00nan` sibling is read for test-fold
   scoring and nothing else, so the missingness ladder keeps its meaning.
4. **Evaluation mask.** The same `preprocess_table` mechanism (null-density scaling, at
   least one cell per row), drawn once per fold from a seed derived from the run seed and
   the fold ordinal, at a separate *evaluation mask rate* hyperparameter defaulting to
   0.2, the literature standard (name and JSON key are ticket 08). Validation and test
   each get one fixed draw.
5. **Splits.** Validation fold for checkpoint selection and per-epoch curves; test fold
   for the reported metrics; never the training fold.
6. **Glossary.** *Self-masked cell*, *induced-missing cell* and *imputation ground truth*
   are now defined in `CONTEXT.md`.

Stated assumptions (not grilled; each has one sensible answer, object if wrong):

- The sibling is found by naming convention,
  `datasets/processed_datasets/<base>/<base>_00nan.csv`. No sibling means the
  induced-missing metrics are skipped, not failed; on `_00nan` itself the population is
  empty by definition.
- Before scoring induced-missing cells the code asserts alignment (same shape and
  columns, every observed cell equal) and fails loudly otherwise.
- Ground-truth values enter the model's space through the variant's own preprocessing:
  numerical targets scaled with the scaler fit on the variant (StandardScaler ignores
  NaN when fitting), categorical targets encoded with the variant's LabelEncoder. A
  ground-truth category absent from the variant's vocabulary (possible at 60-80%) can
  never be predicted; it counts as wrong and the number of such cells is logged.
- All hidden cells of a row are hidden at once, for both populations; the imputer
  faces the whole row.
- Real datasets with native NaNs and no sibling get self-masked metrics only.

Consequences for other tickets:

- Ticket 05 must define metrics for both populations and a fold-ranking rule that
  exists on `_00nan`, where only self-masked cells are available.
- Ticket 07's preview covers both populations.
- Ticket 08 gains the evaluation mask rate hyperparameter.
- Graduated from the fog: ticket 11 (multi-rate evaluation) and ticket 12
  (`[NULL]`-path diagnostic).

## Comments

- 2026-09-10 (charting session): both research notes bear on this ticket. Ticket 01's
  note recommends scoring the cells NaN in `_XXnan` but observed in `_00nan`, test fold
  only, and adds a 20% MCAR mask only for `_00nan`; it points out that the injected
  NaNs are exactly the literature's "artificially masked observed cells". Ticket 02's
  note recommends feeding cells to be imputed as `[MASK]`, not `[NULL]`, at inference.
  Read both before grilling.
