# 05 - The embedder, [MASK] and [NULL] semantics: verdicts

_Adversarial verification of `05-embedder-tokens.md`. 2026-09-11._

**Method:** `graphify query "embedder encode token order masked_positions decode induced
null token"` to orient, then the source end to end: `src/embedder.py` (`as_category_strings`,
`EncodedTable`, `TabularEmbedder.__init__`/`encode`/`forward`), `src/utils.py:18-132`,
`src/models.py` (`TridentPretrainer.forward`, `TridentDecoder.__init__`/`forward`/`predict`),
`src/training/decoding.py` (whole file), `src/training/pretraining.py:17-110`,
`src/training/data.py:44-160`, `src/training/imputation_metrics.py`,
`src/training/artifacts.py:224-375`, `src/training/config.py` (the `--score_null_path`
gate), `tests/unit/test_training_decoding.py:20-250`, `docs/adr/0004-imputation-decoder-task.md:100-205`.
Three `uv run --python 3.10 python` scripts against the real tables and the real classes
(no pytest, no `main.py` run):

1. `_score_induced_missing` called directly on a real `prepare_dataset("credit-g_20nan")`
   + `load_complete_sibling` + `build_folds(cv_folds=2, seed=42)` + a real
   `TridentDecoder`, for both values of `as_mask`, checking each returned `(row, column)`
   against `test_frame.isna()`:

   ```
   FRAME column order : ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', ...]
   TOKEN column order : ['checking_status', 'credit_history', 'purpose', 'savings_status', ..., 'duration', ...]
   misaligned positions: 19/20
   induced                scored  1976 cells | actually missing: 1976 (100.0%)
   induced_null_token     scored  1976 cells | actually missing: 478 (24.2%)
      wrongly scored e.g.: [(9, 'credit_history'), (10, 'credit_history'), (14, 'credit_history')]
   total actual NaN cells in test_frame: 1976
   ```

2. A raw-frame `encode` census plus a `[MASK]`/`[NULL]` census of the three views:

   ```
   raw-frame encode: NaN in num_values = 62 | null_flags True = 0 | mask_flags True = 0
   actual missing numerical cells in slice = 62
   embedding output NaN count = 496 of 8400
   categorical 'checking_status' missing cells encode to: {'nan'}
   'nan' in vocabulary of checking_status : True
   cleaned-frame encode: NaN in num_values = 0 | null_flags True = 62
   cleaned categorical missing cells encode to: {'[NULL]'}
   overlap of [MASK] positions with NaN positions: 0

   DECODE TRAIN p_base=0.5      [MASK]=0.328  [NULL]=0.200  rows with 0 NULL=0.011
   MASKED eval rate=0.2         [MASK]=0.136  [NULL]=0.200  rows with 0 NULL=0.011
   INDUCED scoring view         [MASK]=0.200  [NULL]=0.000  rows with 0 NULL=1.000
   INDUCED null-path view       [MASK]=0.000  [NULL]=0.200  rows with 0 NULL=0.011
   ```

3. Reachability of the second mixed-type table, and the preview artifact's rendering of
   the null-token population (see *What this critique missed*):

   ```
   electricity_20nan RAISED: ValueError y contains previously unseen labels: '2'
   | model saw | [NULL]->[MASK] | [MASK] |     # left cell's population is induced_null_token
   ```

## Verdicts

### F-05-1 - `_score_induced_missing` overrides `masked_positions` with a DataFrame-ordered `isna()` matrix, while every reader of that field indexes it categorical-then-numerical, so the `[NULL]` diagnostic scores the wrong cells

- **Verdict:** CONFIRMED
- **Severity after review:** high
- **Basis:** Reproduced exactly, at full strength. `encode` builds the field in token
  order (`src/embedder.py:189-192`, `columns = list(self.categorical_columns) +
  list(self.numerical_columns)`); `TridentDecoder.forward` (`src/models.py:170, 185`),
  `TridentDecoder.predict` and `_score_population` (`src/training/decoding.py:288, 307`)
  all read `mask[:, index]` / `mask[:, offset + index]` in that order; `decoding.py:262`
  writes

  ```python
  encoded = _select(encoded, torch.tensor(test_frame.isna().to_numpy(), device=device))
  ```

  and `test_frame` is `dataset.frame.drop(columns=[label])` sliced by fold, i.e. CSV
  header order. `declared_column_types` (`src/training/data.py:20-42`) builds
  `categorical_columns` from `datasets/categorical_columns/<base>.txt` and everything else
  as numerical, so the two orders coincide only when the declared categoricals happen to
  be the leading columns of the CSV. On `credit-g` they do not: 19 of 20 positions are
  misaligned, and my run of the real `_score_induced_missing` on the real `_20nan` /
  `_00nan` pair scored 1976 cells of which only 478 (24.2%) were actually missing --
  chance overlap at a 20% missingness rate. The `as_mask=True` path, which does not touch
  `masked_positions`, scored 1976 of 1976 (100.0%). The shapes match, so nothing raises.

  The direction of the bias is not just measured, it is mechanical: the sibling is
  asserted cell-for-cell equal to the variant at every observed cell
  (`_assert_row_aligned`, `data.py:110-128`), so for a wrongly selected cell the "truth"
  the model is scored against is the very value sitting in its own input token at that
  position. The null path is scored on a copy task at ~76% of its cells, which flatters it.

  Invisible to the tests, as claimed. `tests/unit/test_training_decoding.py:20-21` fixes
  `CATEGORICAL = ["colour", "shape"]`, `NUMERICAL = ["size", "weight"]` and builds the
  frame in that same order, so frame order == token order in the only test that exercises
  the path; its single assertion,
  `assert int(through_null.sum()) == int(through_mask.sum())` (line 246), is a count,
  which a column permutation preserves exactly.

  Severity kept at high rather than cut: the flag is off by default
  (`config.py:133, 245`) and the metric never ranks a fold, but ADR 0004 decision 11
  pre-registers this as a decision gate and then records its outcome as measured fact,
  and 4 of the 8 folds in that table are `credit-g`. Not raised to critical: nothing here
  touches the ranking metric, the pinned fixture, or classification, and the recorded
  conclusion is carried by the `kr-vs-kp` rows, which are all-categorical and therefore
  aligned (verified: `kr-vs-kp` has 36 categorical and 0 numerical columns).

- **Correction:** One scope correction to the critic's blast radius. `electricity` is the
  other mixed-type table and is misaligned 2/8, but the induced population is not
  reachable there today at all: `decoding.py:264` raises
  `ValueError: y contains previously unseen labels: '2'` before any scoring, because
  `day` is `int64` in `_00nan` and `float64` in `_20nan`, so the sibling stringifies to
  `'2'` where the variant's vocabulary holds `'2.0'`. I reproduced that. So `credit-g_*`
  is the only live dataset for this bug today -- which strengthens rather than weakens the
  finding, since `credit-g` is the protected imputation fixture dataset and the one in the
  ADR table. Also worth recording for whoever fixes it: every *other* consumer addresses
  columns by name (`encode` transforms `df[col]`, `_score_population` walks
  `embedder.categorical_columns`), so `_select` at line 262 really is the only positional
  assumption and the fix is local.

### F-05-2 - `encode` on a frame that never went through `preprocess_table` represents a missing cell as raw `NaN` (numerical) or the `'nan'` category, and pre-training's clean targets are built that way

- **Verdict:** CONFIRMED
- **Severity after review:** low
- **Basis:** Every mechanical claim reproduces. `split_numeric_and_special`'s non-object
  branch (`src/utils.py:124-126`) is `numeric_array[index] = col_data.astype(np.float64)`
  with no missing-value handling, and `as_category_strings` (`src/embedder.py:27`) is
  `column.where(column.notna(), np.nan).astype(str)`, which yields the literal string
  `'nan'`; `TabularEmbedder.__init__` then fits the LabelEncoder over those strings, so
  `'nan'` is an ordinary class (verified: `'nan' in classes_ -> True`).
  `pretraining.py:66-67` does call `model.embedder.encode(train_frame, device)` on the raw
  frame. On a 50-row `credit-g_20nan` slice that produced 62 `NaN` in `num_values` with
  `null_flags` and `mask_flags` all `False`, 496 `NaN` in the embedding output, and
  `{'nan'}` for the missing cells of `checking_status`. The same slice through
  `preprocess_table` gives 0 `NaN`, 62 `null_flags`, and `{'[NULL]'}`.

- **Correction:** Severity cut from medium to low, because the "bug" produces no wrong
  number anywhere today and I checked both escape routes rather than assuming. (a) The
  `NaN` cannot reach the loss: `TridentPretrainer.forward` selects
  `emb_target[:, 1:, :][mask_tensor]` where `mask_tensor` comes from the *masked* view, and
  `preprocess_table` zeroes `dynamic_mask` at null positions (`utils.py:62`) -- measured
  overlap of `[MASK]` positions with `NaN` positions on a real slice: **0**. (b) The `NaN`
  cannot spread to other cells: the target is `self.embedder(original)` only, never the
  transformer, and `embedder.forward` is strictly per-column (per-column MLP, per-column
  embedding, `torch.stack(..., dim=1)`, plus a positional add), so a `NaN` stays inside its
  own token's `d` values -- consistent with the measured 496 = 62 x 8. The whole target
  path is under `torch.no_grad()`, so no gradient is poisoned either. What survives is a
  genuine but latent defect: an invariant that `TridentPretrainer.forward` depends on and
  neither states nor checks, and an untrained vocabulary entry that `models.py:12` has to
  blacklist. The critic's own framing ("the trap") is the accurate one; the finding should
  be reported as robustness/latent, not as a live miscomputation, and the fix is a
  protected training-behaviour change (it moves every `_XXnan` pre-training target) with no
  numerical payoff today.

### F-05-3 - the induced population is scored on rows where every gap has become `[MASK]` and no `[NULL]` survives, a missingness pattern the decoder essentially never trained on

- **Verdict:** SOUND
- **Severity after review:** medium
- **Basis:** Premise verified in both halves. `decoding.py:243` is
  `hidden = test_frame.mask(test_frame.isna(), "[MASK]" if as_mask else "[NULL]")`, which
  replaces *every* gap in the row, not only the one being scored; and the training view
  comes from `preprocess_table`, which writes `[NULL]` at gaps and then hides a fraction of
  the *observed* cells (`utils.py:47, 58-62`). My census of the full `credit-g_20nan`
  feature frame reproduces the critic's numbers within seeding noise: decode-train view
  `[MASK]=0.328 [NULL]=0.200`, 1.1% of rows `[NULL]`-free; masked eval view
  `[MASK]=0.136 [NULL]=0.200`, 1.1% `[NULL]`-free; induced scoring view
  `[MASK]=0.200 [NULL]=0.000`, **100%** of rows `[NULL]`-free. The reasoning holds because
  `[NULL]` is an input feature with its own trained parameters per column -- a distinct
  vocabulary id and embedding row for categoricals, a distinct
  `special_embeddings[f"{key}_null"]` `nn.Parameter` for numericals
  (`embedder.py:147-148`) -- so erasing it changes what the encoder is given, not merely
  which cells are held out. The consequence follows: ADR 0004 decision 6 names
  `cv/test/impute/induced/impute_score/mean` as the benchmark on a `_XXnan` variant, and
  the checkpoint is selected on the validation *masked* view, which carries `[NULL]` at the
  variant's own rate.

- **Correction:** One thing to keep honest when this is written up, and it does not change
  the verdict: the gap between `impute/masked` and `impute/induced` is over-determined.
  Besides the input-shape difference the critic names, the two populations differ in which
  cells they score (MCAR gaps from the generator vs. cells the evaluation mask chose) and
  in density (0.200 vs 0.136). So "part of the difference is that the two populations show
  the model different rows" is right, but the fixture's 1.3996 vs 1.3402 cannot be
  attributed to the `[NULL]` erasure specifically without a paired measurement. The
  comparability caveat is the finding; an effect size is not yet in evidence. Severity held
  at medium: it is the headline benchmark's meaning, and the cheapest fix is a documented
  caveat rather than a code change.

### F-05-4 - the `StandardScaler` that defines `rmse_num_z`'s unit is fitted on the whole table, including the rows the metric scores

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** Premise verified. `src/training/data.py:69-71`:

  ```python
  raw_numerical = frame[numerical_columns].copy()
  frame[numerical_columns] = scaler.fit_transform(frame[numerical_columns])
  ```

  runs once on the full loaded table, before `build_folds` is ever called, and
  `decoding.py:251-255` pushes the complete sibling's ground truth through that same
  fitted scaler. The reported numerical errors are therefore in that scaler's z-space
  (`imputation_metrics._error_metrics` computes `rmse_num_z` / `mae_num_z` straight from
  `imputed - actual`), so the unit is partly derived from the rows being scored. The
  "`impute_score` is immune" half is also correct and I checked it rather than took it:
  `score_cells` builds `naive` from `mean_mode_baselines`, which are means of the *scaled*
  training fold, and `_ratio(rmse_num_z, naive_rmse)` divides two quantities in the same
  z-space, so any common rescaling cancels. The cross-variant point stands too -- each
  `_XXnan` variant fits its own scaler on its own observed values, so `rmse_num_z` is not
  on a common axis across the ladder.

- **Correction:** The critic *understates* the magnitude, not overstates it. "The effect is
  small (one column's mean/std shifted by one fold's contribution)" assumes a small test
  fold; at the `--cv_folds 2` setting the ADR's own verdict runs used, the test fold is
  half the table, so the z-unit is roughly half-defined by the rows it scores. That does
  not raise the severity -- it is a unit normalisation, not a target leak, `actual` still
  comes from the sibling and `impute_score` still divides it out -- but the write-up should
  not soften it with "one fold's contribution". Severity stays low; this is a reporting
  caveat on a pre-existing, AGENTS.md-protected scaling order.

## What this critique missed

Three things in this dimension, all reproduced:

1. **The preview artifact tells a reader the opposite of what the `[NULL]` path did.**
   `src/training/artifacts.py:359` renders the `model saw` row as
   `"[NULL]->[MASK]" if population.startswith("induced") else "[MASK]"`. The null-path
   population is named `induced_null_token` (`decoding.py:244`), which also starts with
   `"induced"`, so every cell of the one population whose entire purpose is that the model
   saw `[NULL]` **and it was not converted** is labelled `[NULL]->[MASK]`. Verified by
   calling `_render_preview` directly on a two-row ledger:
   `| model saw | [NULL]->[MASK] | [MASK] |`, where the left cell's population is
   `induced_null_token`. Related: `write_imputation_preview` draws its sample from
   `scored_cells["row"].unique()` over the concatenated frame, so with `--score_null_path`
   the same `(row, column)` appears twice in a preview block, both times with that label.
   Low severity, but it is exactly the artifact a human would use to sanity-check F-05-1,
   and it would have hidden it. Fix is `population == "induced"`.

2. **The categorical vocabulary has the same whole-table fit F-05-4 objects to in the
   scaler, and on this branch it became load-bearing in a new way.**
   `pretraining.py:29-35` builds `TabularEmbedder(df=feature_frame, ...)` where
   `feature_frame` is every row of the variant, test fold included, and
   `TabularEmbedder.__init__:104-112` fits each column's LabelEncoder on it. On `main` that
   only sized embedding tables. On this branch `TridentDecoder.__init__:118-131` derives
   each categorical head's *output space* (`valid_ids_<key>`) from that same vocabulary, so
   the set of answers the decoder is allowed to give for a column is fixed in part by the
   test fold's values. F-05-4 makes exactly this argument about the numerical unit and
   stops at the scaler; the categorical half is the same argument and is unstated.

3. **The `electricity` induced crash is reproducible and belongs in this file, not in
   "Open questions".** `decoding.py:264` raises
   `ValueError: y contains previously unseen labels: '2'` on `electricity_20nan` before any
   cell is scored, because the embedder's vocabulary is fitted on the variant (where
   `day` is `float64` and stringifies to `'2.0'`) while the truth comes from the sibling
   (`int64`, `'2'`). It is the direct consequence of item 2 plus `as_category_strings`'s
   dtype dependence, i.e. it is an `[MASK]`/`[NULL]`/vocabulary-semantics defect and not
   only the cross-file dtype hazard a prior review logged. It also settles the critic's own
   open question: today `credit-g_*` is the only table on which F-05-1 can fire, and fixing
   this crash makes `electricity_*` the second, so the two must be fixed together.

One thing the critique got right that deserves to survive into any fix: `_select` is the
**only** place in the decode path that addresses columns positionally. Every other
consumer -- `encode`, `split_numeric_and_special`, `_score_population`,
`TridentDecoder.forward`/`predict` -- reaches a column by name or by walking
`embedder.categorical_columns` / `numerical_columns`. Exposing the order on `EncodedTable`,
as the critic suggests, would make that the only reasonable way to write it.
