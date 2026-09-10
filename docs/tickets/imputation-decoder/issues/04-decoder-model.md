# 04. The decoder model

Status: done (`cbf991c`, `10ff5fb`, this commit) on `feat/imputation-task`
Blocked by: none
Plan task: 4. ADR 0004 decisions 3 and 4. Wayfinder tickets 02, 04.

## Goal

`TridentDecoder` in `src/models.py`: per-column heads on the pretrained encoder's output
that reconstruct actual cell values, with an output space that can never emit a special
token, batched numerical heads, and the count-averaged CE + `lambda_num` * MSE loss.

## Seams under test

- `TridentDecoder(embedder, transformer, lambda_num)` construction: head shapes and the
  `valid_ids` / `local_of` buffers per categorical column.
- `TridentDecoder.forward(masked: EncodedTable, targets: EncodedTable) -> (loss, metrics)`.
- `TridentDecoder.predict(encoded: EncodedTable) -> (cat_logits_per_column, num_values)`.

## Acceptance criteria

- [x] On a tiny frame with one categorical column containing a NaN (vocabulary holds
      `[MASK]`, `[NULL]`, `"nan"`) and two numerical columns: the categorical head has
      `V - 3` outputs; `valid_ids` excludes exactly those three ids; `local_of` maps them
      to `-1`.
- [x] Numerical heads mirror the embedder's MLP (`Linear(d, hidden) -> ReLU ->
      Linear(hidden, 1)`) and are applied with stacked weights, while per-column
      parameters still appear in `state_dict`.
- [x] The loss on a hand-built batch equals the manual `L_cat + lambda_num * L_num` with
      per-type count averaging (independent literal).
- [x] Targets encoded from the processed frame contain no NaN.
- [x] `predict` never returns a special id; `argmax` is over real categories only.
- [x] No initialisation sweep is applied by the decoder itself.

## Constraints

- Excluded ids are looked up with `label_encoders[col].transform([...])`; they are not
  fixed positions (classes are sorted by `np.unique`).
- Embedder and transformer are the pretrained instances, not copies.

## Comments

- 2026-09-10, slice 0 (added on the user's instruction): **missing values are now
  normalised to one representation for both tasks.** Exploration found that the
  placeholder category depends on how the missing value was written: `np.nan` renders as
  `"nan"`, `None` as `"None"`, `NaT` as `"NaT"`. Left alone, a frame built in code would
  disagree with the same frame read from a CSV, and the decoder's excluded set would have
  to guess at pandas sentinels rather than name one.

  `as_category_strings` in `src/embedder.py` collapses every missing sentinel to `"nan"`
  before stringifying, and both the vocabulary construction and `encode` use it. The
  decoder's exclusion set is therefore exactly `{"[MASK]", "[NULL]", "nan"}`, looked up
  through the column's `LabelEncoder` (ids are not fixed positions).

  Proven a no-op on real data rather than assumed: across every categorical column of
  every dataset variant, the old and new stringification agree cell for cell.

  | Check | Result |
  |---|---|
  | Columns compared | 250 |
  | Cells compared | 866,840 |
  | Columns differing | 0 |

  Unit suite 79 green; `tests/integration/test_vehicle_regression.py` passes unedited. A
  real 2-fold classification run of `credit-g_20nan` (the mixed dataset that actually has
  missing values, 12+12 epochs, 5.3s) trains and scores normally: accuracy 0.6590,
  f1_macro 0.5819.

- 2026-09-10, slices A-F, all red-green in `tests/unit/test_decoder_model.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | A | An imputed category is always a real category |
  | B | Only hidden cells teach the decoder; nothing hidden costs nothing |
  | C | Each kind of cell is averaged over its own count, then weighted by `lambda_num` |
  | D | A numerical cell is imputed with one number per column |
  | E | Columns never share head parameters |
  | F | Attaching the decoder leaves the pretrained encoder untouched |

  Two tests were checked for discrimination rather than assumed to bite. A head over the
  whole vocabulary imputes `nan` for every row of slice A's frame, so slice A fails
  without the exclusion. Applying pre-training's Xavier sweep to the decoder changes the
  encoder's weights, so slice F fails if the hazard is ever reintroduced.

  Slice C's expected value is independent of any weight: with silenced heads every
  category scores alike, so the surprise is `ln(2)` for a two-category column and each
  numerical error is the true value squared, giving `ln(2) + 3.0 * mean(1, 9)`.

  **Real training run**, 40 epochs on `credit-g_20nan` fold 1 (13 categorical + 7
  numerical columns, 400 train rows, 6.1s on CUDA), scored on 860 categorical and 488
  numerical held-out hidden cells:

  | | mean/mode baseline | untrained | decoder, 40 epochs |
  |---|---|---|---|
  | categorical accuracy | 0.587 | 0.263 | 0.588 |
  | numerical RMSE (z) | 0.963 | 1.001 | 0.932 |

  `impute_score` 0.987, so barely better than mean/mode at this budget. The decoder
  learns (accuracy more than doubles from untrained, RMSE falls monotonically) but 40
  epochs with no pre-training is not enough to beat the naive baseline meaningfully. That
  is expected here and is the reason the real stage (ticket 06) attaches the decoder to a
  *pretrained* encoder rather than a fresh one.

  **Deferred to review, not a gap in behaviour**: the numerical heads are applied per
  column rather than with stacked weights. The embedder batches its input MLPs to cut
  kernel launches, and the decoder should mirror that, but it is an optimisation with no
  behavioural difference, so it belongs to `code-review` rather than the red-green loop.
