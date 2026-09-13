# 05 - The embedder, [MASK] and [NULL] semantics

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/embedder.py` (whole file, 1-303), `src/utils.py:18-132` (`preprocess_table`,
`split_numeric_and_special`), `src/transformer.py` (whole file), `src/models.py:1-240`
(`TridentPretrainer`, `TridentDecoder`), `src/training/decoding.py:32-346`,
`src/training/pretraining.py:25-80`, `src/training/finetuning.py:72-125`,
`src/training/data.py:44-160`, `src/training/imputation_metrics.py`,
`tests/unit/test_embedder.py`, `tests/unit/test_preprocess_table.py`,
`tests/unit/test_training_decoding.py`, `tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`,
`docs/adr/0004-imputation-decoder-task.md:150-200`. Branch diff via
`git diff main...HEAD -- src/embedder.py src/transformer.py src/utils.py src/training/data.py`.

**Method:** graphify query to orient, then the source. Six `uv run --python 3.10 python`
snippets against the real `datasets/processed_datasets/credit-g/*` tables and the real
classes: a token-order/frame-order comparison; a direct call of `decoding._score_induced_missing`
on `credit-g_20nan` with a real `PreparedDataset` and its `_00nan` sibling; an 8-epoch
in-process decode run comparing the branch's null-path metrics against the same population
scored with an aligned mask; a dtype round-trip sweep of `as_category_strings` across every
dtype `read_csv` can produce plus four it cannot; a `StandardScaler`-with-NaN check; and
positional-embedding / key-collision equivalence checks. I did not run `main.py`, did not
run pytest, and did not re-verify the `baddbmm` rewrite (cleared by a prior review). I could
not check what the 2026-09-10 null-path runs actually logged, only what the code computes
today and what the ADR records.

## Findings

### F-05-1 - `_select` hands the decoder a frame-ordered mask where a token-ordered one is required, so the `[NULL]`-path diagnostic scores cells that were never hidden

- **Kind:** bug
- **Severity:** high
- **Where:** `src/training/decoding.py:262`
- **Evidence:**

  `TabularEmbedder.encode` fixes the token order at `src/embedder.py:189-192`:

  ```python
  columns = list(self.categorical_columns) + list(self.numerical_columns)
  masked_array = np.zeros((n_rows, len(columns)), dtype=bool)
  ```

  Every consumer reads `masked_positions` in that categorical-then-numerical order:
  `TridentDecoder.forward` (`src/models.py:165`, `mask[:, index]` then
  `mask[:, offset + index]`), `TridentDecoder.predict`, and `_score_population`
  (`src/training/decoding.py:288, 307`).

  `_score_induced_missing` overwrites that field with a matrix in **DataFrame column
  order**:

  ```python
  encoded = _select(encoded, torch.tensor(test_frame.isna().to_numpy(), device=device))
  ```

  `test_frame` is `dataset.frame.drop(columns=[label])` sliced by fold, i.e. CSV header
  order. On `credit-g` the two orders differ in 19 of 20 positions (`checking_status` is
  the only one that lands on itself; `isna()` column 1 is `duration` but is read as
  `credit_history`, column 13 is `other_payment_plans` but is read as `duration`, and so
  on). Across the repo's tables:

  ```
  credit-g     cat= 13 num=  7  misaligned positions: 19/20
  electricity  cat=  1 num=  7  misaligned positions:  2/8
  kr-vs-kp     cat= 36 num=  0  misaligned positions:  0/36
  ```

  (all-numerical tables - biodeg, kc2, letter, pendigits, spambase, vehicle - are aligned
  trivially, as is all-categorical `kr-vs-kp`.)

  Run against the real `credit-g_20nan` / `credit-g_00nan` pair, a real `PreparedDataset`,
  and a real `TridentDecoder`, calling `decoding._score_induced_missing` directly for both
  values of `as_mask`:

  ```
  induced                scored  406 cells | actually missing in the variant: 406 (100.0%)
  induced_null_token     scored  406 cells | actually missing in the variant: 89 (21.9%)
     e.g. wrongly scored: [(3, 'credit_history'), (12, 'credit_history'), (14, 'credit_history')]
  ```

  78% of the cells the `[NULL]`-path diagnostic scores are cells whose true value is
  sitting in the model's own input token. 21.9% is chance overlap at a 20% missingness
  rate, not correctness.

  It is invisible to every test. `tests/unit/test_training_decoding.py` builds its frame as
  `colour, shape, size, weight, class` - categoricals first - so the two orders coincide;
  its one assertion on this path is
  `int(through_null.sum()) == int(through_mask.sum())` (line 246), a **count**, which a
  column permutation preserves exactly. The pinned fixture
  `tests/fixtures/credit-g_20nan_imputation_regression.json` contains no `null_token` key
  and the integration test never passes `score_null_path`.

- **Consequence:** `--score_null_path` on any mixed-type table - `credit-g_*` and
  `electricity_*`, the only two in the repo - produces `impute/induced/null_token/acc_cat`,
  `.../rmse_num_z`, `.../macro_f1_cat` and `.../impute_score` for a population that is
  mostly *observed* cells read back out of the model. It also writes those rows into
  `scored_cells` under `population = "induced_null_token"`, so the cell ledger artifact and
  its preview name (row, column) pairs that were never missing.

  This is not a cosmetic diagnostic. ADR 0004 decision 11 pre-registers it as a decision
  gate - "decision 5's `[MASK]` substitution is overturned only if the null path beats it
  on `impute_score` across a majority of folds" - and the ADR then records the outcome:

  | dataset | folds | `[MASK]` score | `[NULL]` score |
  |---|---|---|---|
  | `credit-g_20nan` | 2 | 1.008, 1.004 | 1.155, 1.180 |
  | `credit-g_60nan` | 2 | 1.022, 1.121 | 1.227, 1.422 |
  | `kr-vs-kp_20nan` | 2 | 0.984, 0.969 | 1.920, 1.533 |
  | `kr-vs-kp_60nan` | 2 | 1.012, 0.980 | 1.986, 1.640 |

  The four `credit-g` `[NULL]` numbers in that table are not the quantity the column names.
  The four `kr-vs-kp` ones are sound (that table is all-categorical, so the orders coincide).

  Direction of the error, measured on an 8-epoch decode of `credit-g_20nan` (no
  pre-training, so treat the magnitudes as indicative):

  ```
  as reported by the branch:  null_token acc_cat 0.5387   rmse_num_z 0.9923
  with an aligned mask:                  acc_cat 0.5333   rmse_num_z 1.0322
  ```

  The misalignment **flatters** the `[NULL]` path, which is the conservative direction for
  the recorded verdict: `[MASK]` won anyway, and the untouched `kr-vs-kp` rows carry a 2x
  margin, so the *conclusion* is probably safe. The *evidence* is not. Anyone who re-runs
  this gate on a mixed-type table at a different configuration is reading noise, and the
  ADR presents it as measured fact. (High rather than critical for exactly that reason: the
  flag is off by default, nothing here touches the ranking metric or the pinned fixture, and
  the decision it supported is likely right anyway - but a pre-registered gate reported on
  the wrong cells is still a recorded result that is not what it says it is.)

- **Direction:** `_select` should take the positions in the embedder's own order, not the
  frame's - e.g. `test_frame[list(cat_columns) + list(num_columns)].isna()`, or better, have
  `EncodedTable` expose the order so callers cannot guess it. The unit test needs a frame
  whose categoricals are *not* first, and an assertion on cell identity rather than cell
  count (every scored `(row, column)` must be `NaN` in the variant). The four `credit-g`
  cells of the ADR 0004 verdict table should be re-measured or struck.

### F-05-2 - `encode` has a third representation of a missing cell - literal `NaN` and the `'nan'` category - and pre-training feeds exactly that to its clean targets

- **Kind:** bug
- **Severity:** medium
- **Where:** `src/utils.py:124-126`, `src/embedder.py:17-27`, `src/training/pretraining.py:66-67`
- **Evidence:** The pipeline's contract is that a missing cell is `[NULL]` and a
  deliberately hidden one is `[MASK]`, and `preprocess_table` is what establishes it. But
  `encode` accepts a frame that has not been through `preprocess_table`, and then a missing
  cell becomes neither token:

  - numerical: `split_numeric_and_special` takes the non-object fast path
    (`src/utils.py:124-126`) because a float column with `NaN` is still `float64`, so
    `numeric_array[index] = col_data.astype(np.float64)` keeps the `NaN` and both
    `mask_flags` and `null_flags` stay `False`. The comment there - "No special tokens can
    be present in a purely numeric column" - is true and beside the point; the case it does
    not consider is the missing one.
  - categorical: `as_category_strings` maps `NaN` to the string `"nan"`, which
    `TabularEmbedder.__init__` put into the vocabulary as an ordinary category with its own
    embedding row.

  `pretraining.py:66-67` calls `model.embedder.encode(train_frame, device)` on the **raw**
  frame to build the clean reconstruction targets. Measured on a 50-row `credit-g_20nan`
  slice:

  ```
  raw-frame encode: NaN in num_values = 62 | null_flags True = 0 | mask_flags True = 0
  actual missing numerical cells in slice = 62
  embedding output NaN count = 496 of 8400          # 62 cells x 8 dims
  categorical 'checking_status' missing cells encode to: {'nan'}
  ```

  The only thing keeping those 496 `NaN` floats out of `mse_loss` is the invariant in
  `preprocess_table` that a mask never lands on an already-null cell
  (`src/utils.py:61, 70`), which `TridentPretrainer.forward` never states and does not
  check. Classification is safe by a different route: `finetuning.py:80-88` runs
  `preprocess_table(..., fine_tunning=True)` before encoding.

- **Consequence:** Two live inconsistencies and one live trap.

  Inconsistency 1: within a single imputation run the *same* missing cell has two clean-target
  representations - `'nan'` / `NaN` in pre-training (`pretraining.py:66`) and `[NULL]` in
  decode (`decoding.py:75-76`, via `_clean`). The two stages do not agree on what the cell is.

  Inconsistency 2: the `'nan'` embedding row is allocated per categorical column on every
  `_XXnan` variant, is only ever touched inside `torch.no_grad()` (the target path), never
  appears in any model *input*, and so is never trained. `models.py:12` then has to list it
  in `NON_CATEGORY_TOKENS` to keep the decoder head from emitting it. That is a workaround
  for a representation that should not exist.

  The trap: any future caller that encodes a raw frame and then selects a null position -
  a diagnostic, a new eval population, a notebook - gets `NaN` into a loss or a metric with
  no error. The branch came within one line of doing exactly that: `_score_induced_missing`
  builds the `[NULL]` view with `test_frame.mask(test_frame.isna(), "[NULL]")`, and had it
  passed the raw frame instead, `rmse_num_z` for that population would have been `NaN`.

- **Direction:** Make the invariant explicit rather than incidental. `split_numeric_and_special`
  should treat `NaN` on the fast path as `[NULL]` (set `null_flags`, zero the value) or
  raise; `as_category_strings` should map missing to `"[NULL]"` rather than `"nan"`, which
  would also retire the `"nan"` entry from `NON_CATEGORY_TOKENS`. Either way `encode` then
  has one answer for "this cell is absent", and `pretraining.py` can drop its dependence on
  the never-mask-a-null invariant. Note this moves the pre-training targets and so is a
  protected behaviour change under AGENTS.md - it needs the `vehicle_00nan` fixture checked
  (it has no missing cells, so it should be byte-identical) and a documented note for the
  `_XXnan` variants, which will move.

### F-05-3 - the induced population is scored on a row shape the decoder never trained on: every gap becomes `[MASK]` and no `[NULL]` survives

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/decoding.py:243`
- **Evidence:** `hidden = test_frame.mask(test_frame.isna(), "[MASK]")` replaces *every*
  missing cell in the row, so the scored row carries no `[NULL]` at all. Training draws its
  view from `preprocess_table`, which writes `[NULL]` at the gaps and then hides a fraction
  of the *observed* cells. Measured on the full `credit-g_20nan` feature frame:

  ```
  DECODE TRAIN view  p_base=0.5 : [MASK]=0.331  [NULL]=0.200  rows with 0 NULL=0.011
  MASKED eval view   rate=0.2   : [MASK]=0.137  [NULL]=0.200  rows with 0 NULL=0.011
  INDUCED scoring view          : [MASK]=0.200  [NULL]=0.000  rows with 0 NULL=1.000
  ```

  98.9% of the rows the decoder trains on contain at least one `[NULL]`; 100% of the rows
  it is scored on for `impute/induced/*` contain none. The `[MASK]` density differs too
  (0.200 vs 0.331 in training, 0.137 in the population it is compared against).

  This is not the ordinary "evaluate on held-out cells" situation - the *held-out cells* are
  legitimately unseen. What is off-distribution here is the row's missingness pattern, which
  the model consumes as an input feature (`[NULL]` has its own trained embedding per column,
  verified: `n_mask != n_null` and the `[MASK]`/`[NULL]` vocabulary rows are distinct) and
  which the induced view erases.

- **Consequence:** `impute/induced/impute_score` is the metric ADR 0004 names as the
  benchmark on a `_XXnan` variant ("`cv/test/impute/induced/impute_score/mean` with its
  `ci95_lower`/`ci95_upper`"), and it is measured under an input distribution the checkpoint
  was selected under 1.1% of the time. It is reported beside `impute/masked/impute_score`
  in MLflow, in the CV summary, and in the pinned fixture, as if the two were on one axis;
  they are two different questions asked of one checkpoint. A reader comparing
  `impute/masked` (1.3996) with `impute/induced` (1.3402) in
  `tests/fixtures/credit-g_20nan_imputation_regression.json` will read that as "the model
  does better on real gaps", when part of the difference is that the two populations show
  the model different rows.

  The effect grows with the ladder: at `_60nan` the training view carries `[NULL]` in 60%
  of cells and the induced view in none, so the gap between the two input distributions is
  widest exactly where the imputation question is most interesting.

- **Direction:** Either train the decoder on rows shaped like the scoring view (mask *all*
  gaps in some fraction of training rows, so the pattern is in-distribution), or score the
  induced population one gap at a time with the row's other gaps left as `[NULL]` - which is
  also what a deployed imputer would see. Whichever is chosen, the comparability caveat
  between `impute/masked` and `impute/induced` belongs in ADR 0004 decision 6 and in the
  README's metric section, because both are currently presented as one family.

### F-05-4 - `rmse_num_z`'s unit is fit on the whole table, evaluation rows included

- **Kind:** methodology
- **Severity:** low
- **Where:** `src/training/data.py:69-71`
- **Evidence:** `scaler.fit_transform(frame[numerical_columns])` runs once over the whole
  loaded table, before `build_folds`. The numerical imputation metrics are then reported in
  that scaler's z-space (`rmse_num_z`, `mae_num_z`), and `_score_induced_missing:253` pushes
  the complete sibling's ground truth through the *same* fitted scaler. Verified on
  `credit-g_20nan`: `StandardScaler` ignores `NaN` in `fit` (`sc.mean_` equals
  `np.nanmean`, computed over the 800 observed rows of each 1000-row column) and preserves
  `NaN` through `transform`, so the fit itself is correct - but it consumed every fold's
  observed values, including the test fold's.

  The whole-frame fit is pre-existing (`git show main:src/training/data.py` has the same
  line) and AGENTS.md protects the scaling order, so this is not a regression. What is new
  is what the scaler now defines: on `main` it only scaled model *inputs*; on this branch it
  also fixes the **units of the reported error** and the units of the sibling's truth.

- **Consequence:** `impute/masked/rmse_num_z` and `impute/induced/rmse_num_z` - both pinned
  in the regression fixture and both logged to MLflow - are expressed in a unit derived in
  part from the rows they score. The effect is small (one column's mean/std shifted by one
  fold's contribution) and `impute_score` is immune because it divides by a baseline in the
  same unit, but the two `rmse_num_z` families are not clean held-out numbers and should not
  be quoted as such. Separately, that unit is the *variant's own observed* distribution, so
  `rmse_num_z` is not comparable across the `_00nan`/`_20nan`/.../`_80nan` ladder either -
  the ADR already restricts ladder comparisons to `impute_score`, but the README's metric
  paragraph lists `rmse_num_z` first and does not repeat the caveat.
- **Direction:** Cheapest honest fix is documentation: state in ADR 0004 decision 6 and the
  README that `rmse_num_z` is a within-run diagnostic in a leaked unit and that only
  `impute_score` is quotable across folds and variants. The principled fix - fit the scaler
  per fold on the training rows - is a protected training-behaviour change and would move
  every published classification result, so it needs its own flag-and-tag treatment.

## Checked and cleared

- **Token ordering after the rewrite.** The old `forward` built `torch.cat(cat_list, dim=1)`
  then `.view(B, n_tokens, d)`, which lays tokens out column-by-column in
  categorical-then-numerical order; the new `torch.stack(..., dim=1)` produces the identical
  order. `encode`, `forward`, `TridentPretrainer.forward:61`, `TridentDecoder.forward:165`,
  `TridentDecoder.predict:220-231` and `_score_population:287-322` all agree on it.
  `decoding.py:262` is the only place that does not (F-05-1).
- **Positional embedding after the token-ordering change.** `pos_embedding_layer` is sized
  `n_tokens + 1` and `seq_len` is always exactly `n_tokens + 1`, so
  `weight[:seq_len].unsqueeze(0)` uses the whole table. Checked numerically against the old
  `self.pos_embedding_layer(torch.arange(seq).unsqueeze(0).expand(B, seq))`:
  `torch.equal(old, new) -> True`, 4 of 4 rows used. `[CLS]` still occupies position 0.
- **The ModuleDict key-collision guard is complete.** It fires on a within-kind collision in
  both directions (verified: `"a-b"`/`"a b"` raises for categorical, `"n-1"`/`"n 1"` for
  numerical). Cross-kind collisions cannot share parameters because `embedding_layers` and
  `mlp_layers` are separate `ModuleDict`s and `declared_column_types` makes the two lists
  disjoint by construction. The `_mask`/`_null` suffixes in the single `special_embeddings`
  `ParameterDict` cannot collide either: columns `v`, `v_mask`, `v_null` produce
  `['v_mask', 'v_null', 'v_mask_mask', 'v_mask_null', 'v_null_mask', 'v_null_null']`, all
  distinct. The collision it was guarding is the `re.sub` in the old `forward`, which
  recomputed the key per batch and would have silently routed two punctuated column names
  through one embedding table.
- **`[MASK]` and `[NULL]` are genuinely different tokens.** Categorical: two distinct
  vocabulary ids with distinct `nn.Embedding` rows (verified not equal). Numerical: two
  distinct `nn.Parameter`s per column (verified not equal). Both appear in the model's
  *input* view (`preprocess_table` writes both), so both receive gradient; the
  `torch.where(mask, ..., torch.where(null, ...))` precedence is safe because
  `preprocess_table` never masks an already-null cell (`src/utils.py:61, 70`, pinned by
  `test_an_already_missing_cell_is_never_chosen_as_the_row_guarantee`).
- **`as_category_strings` round-trips every dtype `read_csv` can produce.** The vocabulary
  is fitted on the raw frame's native dtype and `encode` sees the object-upcast frame
  `preprocess_table` returns; I swept both sides. `int64` (including `9007199254740993`),
  `float64` (including `0.1+0.2`, `1/3`, `1e16`, `1e-5`, `1e25`), `bool`, and `object`
  strings (including `""` and values with leading/trailing spaces) all produce byte-identical
  strings before and after the upcast, so no unseen label appears. Two dtypes break
  `preprocess_table` outright - pandas `category` (`TypeError: Cannot setitem on a
  Categorical with a new category ([MASK])`) and nullable `Int64` (`Invalid value '[MASK]'
  for dtype 'Int64'`) - but `pd.read_csv` never produces either with this repo's defaults, so
  neither is reachable. The one real dtype hazard is the cross-*file* one the prior review
  already found (integer-coded categorical, `int64` in `_00nan` and `float64` in `_20nan`);
  I found no second class beyond it.
- **`StandardScaler` and `NaN`.** `fit` ignores missing values rather than poisoning
  `mean_`/`scale_` (`sc.mean_` matches `np.nanmean` exactly, 800 of 1000 rows per column on
  `credit-g_20nan`) and `transform` preserves them, so a gap neither shifts the mean nor
  silently becomes zero. The `raw_numerical` copy taken at `data.py:70` really is
  pre-scaling, so `actual_original` is the number the CSV holds.
- **Fine-tuning never reaches the `NaN` encode path; pre-training does, for both tasks.**
  `finetuning.py:80-88` runs all three frames through
  `preprocess_table(..., fine_tunning=True)` before `model.embedder.encode`, so the
  classifier's own inputs and the decode stage's `_clean` targets are safe. Pre-training is
  shared by both tasks and does encode the raw frame, so F-05-2 is present in every
  `_XXnan` run of either task - confined to the pre-training target tensor, where the
  never-mask-a-null invariant keeps it out of the loss.
- **The `baddbmm` rewrite.** Not re-verified; cleared by a prior review.

## Open questions

- **Were the ADR 0004 verdict runs affected the way today's code is?** The table is dated
  2026-09-10 and `_select` has carried the frame-ordered mask since `345574b` ("feat(training): the decode stage", 2026-09-10) introduced it, so the
  `credit-g` rows almost certainly were - but I could not find the MLflow runs behind the
  table. The `impute/induced/null_token/n_cat_cells` and `n_num_cells` of those runs would
  settle it: under the bug, on `credit-g` they split the 406 cells by the *token-order*
  column types, not the frame-order ones, so they differ from the aligned counts.
- **How large does F-05-1's bias get with a properly pre-trained decoder?** My 8-epoch,
  no-pre-training measurement showed the misalignment flattering the `[NULL]` path by ~0.005
  accuracy and ~0.04 z-RMSE. The mechanism (scoring a cell whose value is in its own input
  token) should get much stronger as the decoder learns to copy a visible token, which would
  widen the gap. One paired run at the ADR's actual configuration, with and without the
  reordering, would answer it.
- **Is `electricity` reachable for the induced populations at all?** The prior review
  reproduced a crash at `decoding.py:264` for that table's integer-coded `day`. If it is
  unreachable, F-05-1 has only one live dataset today (`credit-g`) - but fixing that crash
  makes `electricity` the second, so the two need fixing together.
