# 04 - Ground truth and the evaluation protocol

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/data.py:45-149` (`prepare_dataset`, `load_complete_sibling`,
`_assert_row_aligned`, `evaluation_mask`), `src/training/decoding.py:32-346` (both scored
populations, `_score_induced_missing`, `_select`, `_score_population`),
`src/utils.py:18-82` (`preprocess_table`, the masking helper both populations depend on),
`src/training/imputation_metrics.py:31-137` (`score_cells`, `mean_mode_baselines`),
`src/training/runner.py:22-46`, `src/training/pretraining.py:24-36`,
`src/embedder.py:104-200`, `src/models.py:100-225`, `datasets/generate_splits.py:26-49`,
`docs/adr/0004-imputation-decoder-task.md` decisions 5, 6, 10, 11 and "How to compare",
`README.md:266-330`, `tests/unit/test_training_data.py:103-175`.

**Method:** graphify query to orient, then read every file above at source. Ran nine
`uv run --python 3.10 python -c` probes against the real functions and the real CSVs:
dtype and NaN audits of all nine `_00nan` tables; embedder-order vs CSV-order comparison
for all nine bases; a `TabularEmbedder` + `_select` round trip on a synthetic mixed frame;
`load_complete_sibling` on `electricity_20nan` and `credit-g_20nan`; realised
`preprocess_table` mask rates across the credit-g and electricity ladders at three
nominal rates; the forced-mask floor's share of scored cells; `evaluation_mask`
validation-vs-test stream overlap; a `StandardScaler` NaN and scale-invariance check; a
StratifiedKFold vocabulary-coverage sweep. **Not checked:** no training run (the allowed
isolated form gives no way to shorten 150 decode epochs on CPU, and direct function probes
are stronger evidence); no claim about GPU-specific behaviour; no MLflow store inspection,
so I take the ADR's decision-11 verdict table at face value as a record of what was run.

**Vocabulary note.** The review brief calls the variant's own gaps "natural-missing" and
the evaluation mask "induced". This file follows the codebase instead: `induced` is the
population of the variant's own gaps scored against the `_00nan` sibling, and `masked`
(self-masked) is the population `preprocess_table` hides for scoring. Read F-04-2 as being
about the brief's "induced mask", not about the brief's "natural-missing" cells.

## What an `impute_score` from this protocol can and cannot support

**Can support.** For a `_XXnan` variant, `impute/induced/impute_score` is a score over a
**fixed** cell set — the gaps `datasets/generate_splits.py` injected at a fixed count per
feature column with `rng.choice(..., replace=False)`, independent of every cell's value
(verified MCAR by construction; the label column is excluded). That set does not move with
`--seed` or `--cv_folds`, so two runs on the same variant score the same cells, and the
`_00nan` sibling genuinely holds each one's true value (all nine `_00nan` tables have zero
NaN). It supports the claim "on this corpus, at this injected rate, the decoder beats /
matches / loses to per-column mean-and-mode imputation" — and, because `_ratio` divides by
that baseline, the claim survives the global scaler (verified: the numerical term is exactly
scale-invariant).

**Cannot support.** (a) Anything comparative across the `_20nan.._80nan` ladder for
`impute/masked/*` — the ADR says that number "is comparable across the ladder"
(`docs/adr/0004-imputation-decoder-task.md:222`) and it is not; the realised rate is
non-monotone across credit-g (0.205, 0.167, 0.143, 0.152, 0.251) and at `_80nan` 81% of the
scored cells are the "at least one mask per row" floor rather than the nominal draw
(F-04-2). (b) A *paired* comparison of `impute/masked/*` between two runs unless they share `--seed`
and `--cv_folds` — the cell set is a seed x fold x rate dependent sample, so across seeds
you are comparing two noisy estimates of a quantity that itself changes character with the
variant (F-04-2), not two measurements of the same thing. `impute/induced/*` has no such
restriction: across all folds every induced cell is scored exactly once, so the union is
the same fixed set at any seed or fold count. (c) Any absolute reading of `rmse_num_z` /
`mae_num_z` — those are z-units of a sigma fit on all rows of the variant including the test
fold, and re-estimated from a different observed subset at every rung of the ladder
(F-04-7). (d) "The decoder is better at higher missingness" from `impute/induced/*` — at
`electricity_80nan` 20.9% of induced cells sit in rows with **no** observed cell at all, so a
growing slice of the headline population is bounded at the marginal baseline and drags the
ratio toward 1.0 independently of model quality (F-04-3). (e) That feeding induced gaps as
`[MASK]` rather than `[NULL]` is the right choice on a **mixed** table: the only evidence for
that is ADR decision 11's pre-registered comparison, and on credit-g it was measured through
a transposed mask (F-04-1). That question matters more than the ADR allows, because per
token the `[MASK]` substitution is in distribution but per *row* it is not: during decode
training a `_60nan` row is ~60% `[NULL]` plus a few `[MASK]`, while at induced scoring the
same row is 60% `[MASK]` and 0% `[NULL]`.

## Findings

### F-04-1 - `--score_null_path` scores the wrong cells on any table whose CSV interleaves categorical and numerical columns

- **Kind:** bug
- **Severity:** high
- **Where:** `src/training/decoding.py:262` (`_select(encoded, torch.tensor(test_frame.isna().to_numpy(), ...))`)
- **Evidence:** `EncodedTable.masked_positions` is built in embedder order —
  `src/embedder.py:189`, `columns = list(self.categorical_columns) + list(self.numerical_columns)`
  — while `test_frame.isna().to_numpy()` is in CSV feature order. The two orders differ
  wherever the declared categorical list is not a CSV-order prefix:

  ```
  credit-g   n_feat 20 n_cat 13  ORDER MATCHES CSV: False
     csv  : ['checking_status', 'duration', 'credit_history', 'purpose', ...]
     embed: ['checking_status', 'credit_history', 'purpose', 'savings_status', ...]
  electricity n_feat 8 n_cat 1   ORDER MATCHES CSV: False
  kr-vs-kp / vehicle / kc2 / biodeg / spambase / letter / pendigits: True
  ```

  Round trip through the real `TabularEmbedder` and the real `_select` on a four-column
  mixed frame whose only gaps are `n1` at row 0 and `c2` at row 1:

  ```
  embedder column order: ['c1', 'c2', 'n1', 'n2']
  frame column order   : ['c1', 'n1', 'c2', 'n2']
  masked_positions (correct, from encode):   after _select (what --score_null_path uses):
  [[0 0 1 0]                                 [[0 1 0 0]
   [0 1 0 0]                                  [0 0 1 0]
   ...                                         ...
  ```

  The two marked cells are exactly transposed. On the real `credit-g_20nan` feature frame:

  ```
  missing cells total      : 4000
  cells _select marks      : 4000
  overlap (truly missing)  : 944 => 23.6%
  selected cells that are OBSERVED in the variant: 76.4%
  ```

- **Consequence:** With `--score_null_path` on a mixed table, `impute/induced/null_token/*`
  is computed over a population that is 76% cells the model **can already read** in its
  input (it sees their real value, not `[NULL]`) and only 24% actual gaps. The counts match
  — every column carries the same injected count — so nothing looks wrong in the ledger or
  in `n_num_cells`/`n_cat_cells`. The primary path is untouched: `_score_population` and the
  `masked`/`induced` populations index `masked_positions` produced by `encode` itself, and
  `_already_missing` is a scalar count, so `impute/induced/*` and `impute/masked/*` are
  correct. The flag is off by default and all-numerical datasets plus all-categorical
  `kr-vs-kp` are unaffected (their orders match). The damage is to the record: ADR decision
  11's pre-registered criterion requires the null path to win "on at least two datasets",
  and the verdict table at `docs/adr/0004-imputation-decoder-task.md:176-179` reports four
  credit-g rows measured through this path. `_select` was introduced with the decode stage
  in `345574b` (2026-09-10 19:10) and the verdict was committed in `74229fd` the same
  evening (19:46), so the credit-g numbers were measured with this code in place — the two
  later commits touching `decoding.py` (`24703c9`, `9a38b90`) leave `_select` unchanged.
  Only the `kr-vs-kp` rows were validly measured,
  so decision 5's `[MASK]` substitution currently rests on one dataset — the all-categorical
  one — not two, and specifically not on the mixed table the substitution is hardest to
  justify for.
- **Direction:** Build the selection tensor in embedder order rather than frame order —
  reindex with `test_frame[list(embedder.categorical_columns) + list(embedder.numerical_columns)]`
  before `.isna()`, or have `_select` take column names and place them itself. Then re-run
  decision 11's credit-g arm before the ADR's verdict is treated as settled. A unit test on
  a frame whose categorical columns are not a CSV-order prefix would have caught this;
  `tests/unit/test_training_decoding.py` has no such case.

### F-04-2 - the self-masked population's sampling design changes character across the ladder, and the ADR asserts it is comparable anyway

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/utils.py:55` and `src/utils.py:64-74`, reached from `src/training/data.py:146`; claim at `docs/adr/0004-imputation-decoder-task.md:220-222`
- **Evidence:** `evaluation_mask` delegates to `preprocess_table`, which does two things
  besides the nominal Bernoulli draw. It deflates the rate by the row's null density
  (`p_dynamic = p_base * (1 - prop_nulls)`, `src/utils.py:55`) and then **forces exactly one
  mask into every row that drew none** (`src/utils.py:64-74`). Realised share of *eligible*
  (observed) cells, measured by running the real `preprocess_table` on the real variants
  (seed 7):

  ```
  nominal 0.2   credit-g_00nan 0.2049   _20nan 0.1672   _40nan 0.1426   _60nan 0.1515   _80nan 0.2512
  nominal 0.2   electricity_00nan 0.2210   _20nan 0.2182   _60nan 0.3236
  nominal 0.1   electricity_00nan 0.1535   _60nan 0.3116
  nominal 0.05  credit-g_00nan 0.0660   _60nan 0.1269   _80nan 0.2475
  nominal 0.05  electricity_00nan 0.1327  _60nan 0.3082   (6.2x the nominal rate)
  ```

  The floor's share of the scored cells, with fully-missing rows excluded as
  `preprocess_table` excludes them:

  ```
  credit-g_00nan p=0.2 :  0.2%      credit-g_80nan p=0.2 : 81.4% (818 of 1005)
  credit-g_60nan p=0.2 : 44.3%      electricity_60nan p=0.2: 70.6% (33118 of 46926)
  credit-g_00nan p=0.05: 26.9%      electricity_00nan p=0.05: 62.7%
  ```

- **Consequence:** Two distinct problems. (1) `docs/adr/0004-imputation-decoder-task.md:222`
  tells a reader that "on any variant, `cv/test/impute/masked/impute_score/mean` is
  comparable across the ladder." It is not. At `_00nan` the number describes ~20% of cells
  drawn near-uniformly; at `_80nan` it describes a population that is 81% "one uniformly
  chosen observed cell per row" — a different estimand, on a different share of eligible
  cells, and the share is non-monotone (0.205 -> 0.143 -> 0.251), so not even a monotone
  difficulty ordering can be read off it. Anyone plotting `impute_score` against the ladder
  to argue "TRIDENT degrades gracefully with missingness" is plotting an artefact of the
  masking helper. (2) The multi-rate diagnostic of ADR decision 10 is mislabeled at the
  segment key: `--eval_mask_rates_extra 0.05` on `electricity_60nan` writes
  `impute/masked/rate_5/...` for a population actually at 30.8%, and `realised_rate` is
  emitted only for the primary rate (`src/training/decoding.py:154-157`), never inside the
  `for extra in ...` loop at `:162-172`, so the discrepancy is invisible from MLflow. The
  narrow tables are worst: on eight-feature electricity even `_00nan` at nominal 0.05 lands
  at 13.3%.

  Note the design is still value-independent at every step — Bernoulli draw, deflation, and
  the uniform `np.random.choice` over the row's observed columns all ignore cell contents.
  The problem is not that the mask is informative; it is that its *intensity* tracks row
  completeness, so the scored cells are drawn preferentially from rows with more surviving
  context, and at high missingness the floor takes over entirely.
- **Direction:** Either (a) make the evaluation mask its own draw rather than reusing the
  training corruption helper — a flat Bernoulli over observed cells with no deflation and no
  per-row floor, which is the conventional meaning of a masking rate; or (b) keep `preprocess_table` and stop claiming ladder comparability: emit
  `realised_rate` per rate segment (including every `rate_N`), and state in the ADR and
  README that `impute/masked/*` is a paired within-variant number only. Either way, the
  induced population — a fixed cell set — is the one that carries cross-ladder meaning, and
  the ADR should say so exclusively.

### F-04-3 - a growing share of the headline induced population has no observed cell to condition on, bounding `impute_score` toward 1.0

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/decoding.py:243` (`hidden = test_frame.mask(test_frame.isna(), "[MASK]")`), scored at `:174-182`
- **Evidence:** `datasets/generate_splits.py:43-48` injects the gaps **independently per
  column**, so at high rates whole rows vanish. Counted on the shipped tables:

  ```
  electricity_80nan: 7593/45312 rows fully missing -> 60744/290000 induced cells = 20.9%
  electricity_60nan:  790/45312 rows fully missing ->  6320/217496 induced cells =  2.9%
  credit-g_80nan   :   10/1000  rows fully missing ->   200/16000  induced cells =  1.2%
  ```

  For such a row `_score_induced_missing` builds an all-`[MASK]` input, so the transformer
  sees nothing but the column-position embeddings. The decoder's best possible answer is the
  column's unconditional mode / mean — which is precisely the `mean_mode_baselines`
  comparator (`src/training/imputation_metrics.py:120-137`), so each of those cells
  contributes a per-cell ratio of about 1.0 to `impute_score`.
- **Consequence:** `impute/induced/impute_score` is a mixture of "cells the model could
  genuinely reason about" and "cells where the ceiling is parity with the baseline", and the
  mixing weight rises with the variant's missingness and falls with the table's width. On
  `electricity_80nan` a fifth of the headline number is pinned at ~1.0 by construction. A
  reader comparing `electricity_80nan` against `electricity_20nan` — exactly the comparison
  ADR decision 10 says the dataset ladder already provides ("Induced-missing cells need no
  sweep: the dataset ladder already is one") — will read regression to the baseline as model
  degradation. Nothing in the metrics distinguishes the two; `n_num_cells`/`n_cat_cells`
  count the context-free cells alongside the rest.
- **Direction:** Either report the induced score split by the row's observed-cell count
  (even two buckets, "row had context" / "row had none", would do), or log the share of
  scored cells from context-free rows as a companion metric beside `realised_rate`. The
  per-column artifact already exists as a home for the breakdown; this is a per-row
  stratification, so it needs its own key.

### F-04-4 - the row-alignment guard passes on the one sibling incompatibility present in the shipped corpus

- **Kind:** design
- **Severity:** medium
- **Where:** `src/training/data.py:112-129` (`_assert_row_aligned`), called from `:108`
- **Evidence:** The guard compares values only, through `to_numpy()` on a mixed-dtype frame,
  which yields an object array where `1.0 == 1` is `True`:

  ```python
  observed = variant.notna().to_numpy()
  if not (variant.to_numpy()[observed] == complete.to_numpy()[observed]).all():
  ```

  Run against the real corpus (known finding 1 reports the crash this later causes; this is
  about the guard that exists to prevent it):

  ```
  electricity_20nan dtype mismatches: {'day': ('float64', 'int64')} | row-align guard: PASSED
  credit-g_20nan    dtype mismatches: {}                            | row-align guard: PASSED
  ```

  The docstring says the guard exists because "scoring against the wrong row is worse than
  not scoring at all", and `tests/unit/test_training_data.py:117` covers exactly one failure
  mode: a changed observed value. There is no check for the two properties
  `_score_induced_missing` actually needs — that the sibling stringifies into the variant's
  vocabulary (`src/embedder.py:177`, `label_encoders[col].transform(as_category_strings(...))`,
  fit from the *variant*), and that the sibling is non-NaN wherever the variant has a gap
  (asserted only in a test at `tests/unit/test_training_data.py:114`, never in code).
- **Consequence:** The guard gives false assurance. It certifies `electricity_20nan` as
  aligned and hands `run_training` a sibling that cannot be encoded. Worse for the protocol:
  if a future `_00nan` ever carried a NaN at a position the variant is also missing,
  `_score_induced_missing` would mask it, `_clean` would turn it into `[NULL]`,
  `encoder.inverse_transform` would record `actual = "[NULL]"` for a categorical cell (a
  value `TridentDecoder`'s head is structurally unable to emit, `src/models.py:120-127`) and
  `truth.num_values` would record `0.0` — the z-space mean — for a numerical one. Both are
  silently scored. ADR decision 5 states "Cells NaN in `_00nan` itself are never scored";
  today that holds only because all nine shipped `_00nan` tables happen to have zero NaN,
  not because anything enforces it.
- **Direction:** Extend the guard to the properties scoring depends on: per-column dtype
  equality (or a canonicalisation both sides go through before comparison), a refusal when
  the sibling is NaN where the variant is NaN, and a check that the sibling's categorical
  values are a subset of the variant's observed categories. That last one is cheap and turns
  known finding 1's crash into a message that names the column.

### F-04-5 - the README's `EVAL_MASK_RATE` explanation is wrong in direction and points at a metric with a different denominator

- **Kind:** methodology
- **Severity:** low
- **Where:** `README.md:272`
- **Evidence:** The README says the helper "scales it down by each row's null density and
  never hides an already-missing cell, so asking for `0.2` hides about 20% of a `_00nan`
  variant but about 5% of an `_80nan` one. Each run logs the realised share as
  `impute/masked/realised_rate`." Measured on `credit-g_80nan` at nominal 0.2: masked cells
  are 5.0% of *all* cells and **25.1%** of *eligible* (observed) cells.
  `src/training/decoding.py:154-157` divides by eligible cells
  (`masked_positions.numel() - _already_missing(test_frame)`), so the metric the sentence
  names reads 0.251, not 0.05.
- **Consequence:** Two errors in three sentences. The mechanism is incomplete — it omits the
  per-row floor (`src/utils.py:64-74`), which is what actually dominates at high missingness
  and *raises* the realised share rather than lowering it, so the stated direction is wrong
  wherever the floor bites. And the quoted figure is off by 5x from the metric it directs the
  reader to. A reader tuning `EVAL_MASK_RATE` against `realised_rate` will conclude the
  helper is broken, or will accept a number five times the one they expected.
- **Direction:** State the denominator explicitly (share of observed cells), describe both
  mechanisms including the floor, and replace the illustrative numbers with measured ones.
  The realised-rate table in F-04-2 is enough to fill it in.

### F-04-6 - the validation and test evaluation masks are the same draw, not two independent ones

- **Kind:** methodology
- **Severity:** low
- **Where:** `src/training/data.py:143-148` (`evaluation_mask`), called at `src/training/decoding.py:78` and `:138`
- **Evidence:** `evaluation_mask` resets the global stream with
  `np.random.seed((seed * 1_000_003 + fold) % (2**32))` on every call, and both the
  validation call and the test call for a fold pass the same `(seed, fold_ordinal)`.
  `preprocess_table` then draws `np.random.rand(*data.shape)` row-major from that same start,
  so row *i* of the validation frame and row *i* of the test frame receive the identical
  vector of uniforms. Measured on two disjoint slices of `credit-g_00nan` at rate 0.2, seed
  42, fold 1:

  ```
  validation mask shape (90, 20) test mask shape (200, 20)
  identical on the shared row prefix: False
  per-cell agreement on prefix: 0.9989
  ```

  (the 0.11% divergence is not in the Bernoulli draw: `np.random.rand(90, 20)` and
  `np.random.rand(200, 20)` consume different amounts of the stream, so the forced-mask loop
  that follows starts from different RNG states in the two calls.) The same property makes the extra-rate masks nested inside the
  primary one: 226 of 285 rate-0.05 cells are also rate-0.20 cells.
- **Consequence:** The checkpoint that `src/training/decoding.py:125-130` restores is
  selected on a mask whose per-column composition is, for the first `n_validation` rows,
  literally the test mask's composition. The rows differ, so there is no cell-level leak, but
  the two questions are not independent draws: a chance column imbalance in the uniform
  stream is shared between the selection criterion and part of the thing being scored, which
  biases test error optimistically by an unquantified but non-zero amount. ADR decision 5
  says "validation uses the same draw for checkpoint selection", so this may be intended —
  but "the same seed" and "positionally the same mask" are different properties, and only the
  first is what the sentence reads as.
- **Direction:** Mix the split's identity into the derived seed (`... + fold * 3 + split_id`
  or similar) so validation, test and each extra rate draw from disjoint streams. It costs
  one argument and makes the independence claim true; the nesting across rates is arguably a
  feature and could be kept deliberately if it is documented as such.

### F-04-7 - `rmse_num_z` and `mae_num_z` are in units of a sigma fit on the test rows, and the unit changes at every rung of the ladder

- **Kind:** methodology
- **Severity:** low
- **Where:** `src/training/data.py:65-71` (`prepare_dataset` fits `StandardScaler` on the whole frame before `build_folds` runs at `src/training/runner.py:46`)
- **Evidence:** `scaler.fit_transform(frame[numerical_columns])` runs once over every row of
  the variant; folds are constructed afterwards. `StandardScaler` is NaN-aware, so `mean_`
  and `scale_` are the variant's nan-statistics over all rows — the test fold included, and
  for the `masked` population the scored cells' own true values included. Confirmed:

  ```
  StandardScaler on a frame with NaN: mean_ uses nanmean? True | NaNs preserved: True
  ```

  The `impute_score` ratio is unaffected — I checked that the train-fold-mean baseline in
  scaled space equals the raw train-fold-mean imputer divided by the global sigma, so sigma
  cancels exactly:

  ```
  train-fold-mean baseline rmse: scaled-space=0.995906  raw/sigma_global=0.995906
  => baseline identical up to the global sigma: True
  ```

- **Consequence:** What does not cancel is the absolute error. `rmse_num_z` and `mae_num_z`
  are logged per population to MLflow and pinned in
  `tests/fixtures/credit-g_20nan_imputation_regression.json`, and their unit is a standard
  deviation estimated with the test rows' own values. Worse for cross-ladder reading: on a
  `_XXnan` variant that sigma is estimated from the surviving observed cells only, so
  `credit-g_20nan` and `credit-g_80nan` report `rmse_num_z` in measurably different units
  (80% vs 20% of the rows entering the estimate) — a difference that has nothing to do with
  imputation quality. The split-and-scale order is protected by `AGENTS.md`, so this is a
  documentation obligation, not a change request; ADR decision 8 notes the scaler is retained
  "which does not move the split-and-scale order" but nowhere says what that order costs the
  imputation numbers.
- **Direction:** Do not touch the order. State in ADR 0004 decision 6 and README's metric
  table that `rmse_num_z`/`mae_num_z` are diagnostic only, that the ratio metrics are the
  comparable ones, and that the z-unit is variant-specific. If an absolute error in original
  units is ever wanted, `dataset.raw_numerical` and `actual_original` already carry the
  material to compute one.

## Checked and cleared

- **The induced gaps really are MCAR, and the sibling really is their truth.**
  `datasets/generate_splits.py:43-48` draws `n_inject = round(n_rows * pct)` row indices per
  feature column with `rng.choice(non_nan_idx, size=n_here, replace=False)` — a fixed count
  per column, independent of every cell's value, label column excluded. No value-dependent
  step anywhere. All nine `_00nan` tables have zero NaN (biodeg, credit-g, electricity, kc2,
  kr-vs-kp, letter, pendigits, spambase, vehicle), so every induced gap has a real value in
  the sibling. The ADR's MCAR claim is correct as stated.
- **Row alignment between variant and sibling is positional and correct.**
  `complete_sibling.iloc[fold.test_indices]` (`src/training/decoding.py:246`) indexes a frame
  read by `pd.read_csv` with a default RangeIndex, against positional indices produced by
  scikit-learn splitters over `dataset.frame`, which is never reordered between
  `prepare_dataset` and `build_folds`. `truth[[column for column in test_frame.columns]]`
  then puts the sibling into the variant's column order before
  `dataset.scaler.transform(truth[list(dataset.numerical_columns)])`, which is the order the
  scaler was fit in. No off-by-one, no index misalignment. (The *guard* is weak — F-04-4 —
  but the alignment itself holds on this corpus.)
- **The induced mask cannot land on an already-missing cell, and the self-mask cannot
  either.** The two populations are disjoint by construction: `_score_induced_missing` masks
  exactly `test_frame.isna()`, while `preprocess_table` zeroes `dynamic_mask[null_values]`
  (`src/utils.py:62`) and the forced pick draws from `np.where(~null_values[i])[0]`
  (`src/utils.py:71`), guarded against an empty set at `:72`. No cell is scored twice within
  a run, and no cell is ever scored against `[NULL]`.
- **The two populations are never pooled into one number.** `score_cells` is called
  separately per population in `src/training/decoding.py:150`, `:179` and `:194` before the
  `pd.concat` at `:178`/`:190`, and `runner._per_column_scores` groups by `population` before
  scoring (`src/training/runner.py:29-32`). The concatenated `cells` frame feeds only the
  ledger and preview artifacts. Even with `--score_null_path` on — which puts the same gap in
  the table twice — no metric double-counts.
- **The `impute_score` numerical baseline is scale-invariant, so the global scaler does not
  contaminate the ratio.** Verified numerically (output quoted in F-04-7): the training-fold
  mean of globally-z-scored values is exactly the raw training-fold mean expressed in global
  sigma, so `_ratio(rmse_model, rmse_naive)` is unchanged by the fit-on-everything scaler.
  `mean_mode_baselines`'s docstring claim — "learned from the training fold alone so that
  nothing about the scored cells leaks into their own baseline" — holds for the numerical
  half. The categorical half uses raw category strings from the training fold and never
  touches the scaler.
- **The decoder's categorical output space is not a meaningful vocabulary leak.** The
  embedder fits its `LabelEncoder`s on the full feature frame (`src/training/pretraining.py:30`
  passes `df=feature_frame`), so the head can in principle emit a category absent from the
  training fold. Swept all 5 folds of `credit-g_20nan` and `kr-vs-kp_20nan` at seed 42: zero
  such categories on credit-g, exactly one across all folds on kr-vs-kp (`spcop='t'`, fold 4).
  Real but negligible on this corpus.
- **Two runs of the same fold do ask the same question, and evaluation does not disturb the
  training draw sequence.** `evaluation_mask` saves and restores `np.random.get_state()`
  (`src/training/data.py:143-148`); `tests/unit/test_training_data.py:141` and `:158` pin
  both. The determinism claim in ADR decision 5 is correct; the independence claim is where
  it slips (F-04-6).
- **`impute/masked/realised_rate` is computed correctly for the primary rate.**
  `masked_positions.numel()` is the total cell count, `_already_missing` the NaN count, and
  `len(cells)` the masked count — all order-independent, so the metric is right even on the
  datasets where F-04-1's ordering problem exists. It is the one instrument that makes F-04-2
  detectable at the primary rate; it simply is not emitted for the extra rates.
- **`actual_original` takes its raw value from the right frame for each population.**
  `raw_variant` (`dataset.raw_numerical`, captured before scaling at `src/training/data.py:70`)
  for `masked` cells, which are observed in the variant; `raw_truth` (the sibling before
  `dataset.scaler.transform`) for `induced` cells, which are not. No population reads a raw
  value from a frame that does not hold it.

## Open questions

- **How large is the F-04-6 optimistic bias in practice?** Settleable by running one fold
  twice — once with the current shared stream, once with a split-distinct seed — and
  comparing `impute/masked/impute_score`. Needs a real training run, which I did not do.
- **Is the per-row masking floor load-bearing for pre-training, or only inherited?**
  `preprocess_table` is shared between training corruption and evaluation. If the floor
  exists so that no pre-training row contributes a zero loss, then giving evaluation its own
  draw (F-04-2, option a) costs nothing; if some published result depends on it, the split
  needs the flag-and-tag treatment `AGENTS.md` mandates. The git history of `src/utils.py`
  before this branch would say which.
- **Should an induced cell in a fully-missing row be scored at all?** F-04-3 argues the
  mixture is misleading; whether the right answer is stratified reporting or exclusion is a
  protocol decision, not a code one, and the literature the ADR cites (protocol survey on
  `research/imputation-eval-protocol`, which I could not read from this branch) may already
  have a convention.
