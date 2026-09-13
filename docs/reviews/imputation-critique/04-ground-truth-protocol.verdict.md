# 04 - Ground truth and the evaluation protocol: verdicts

_Adversarial verification of `04-ground-truth-protocol.md`. 2026-09-11._

**Method:** Oriented with `graphify query "imputation evaluation mask, scored populations,
induced vs masked, score_null_path"`, then read at source: `src/training/decoding.py`
(whole file), `src/training/data.py:1-160`, `src/utils.py:1-80`, `src/embedder.py:17-27` and
`:100-200`, `src/models.py:95-230`, `src/training/imputation_metrics.py:31-140`,
`src/training/runner.py:20-60`, `datasets/generate_splits.py:25-48`,
`docs/adr/0004-imputation-decoder-task.md:150-240`, `README.md:264-300`,
`tests/unit/test_training_data.py:100-180`, `opt.py` (LAMBDA_NUM lines).

Then ran five `uv run --python 3.10 python` probe scripts against the real functions and the
real CSVs. No training run: the allowed isolated form cannot shorten 150 decode epochs on
CPU, and direct probes of `preprocess_table`, `TabularEmbedder.encode`, `_select` and
`_assert_row_aligned` settle every claim more tightly than a 2-fold run would. No pytest
(instructed).

Probe outputs, verbatim:

```
# chk_order.py - embedder order vs CSV order, all nine bases
biodeg       n_feat= 41 n_cat=  0 MATCHES=True
credit-g     n_feat= 20 n_cat= 13 MATCHES=False
   csv  : ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status']
   embed: ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status']
electricity  n_feat=  8 n_cat=  1 MATCHES=False
   csv  : ['date', 'day', 'period', ...]     embed: ['day', 'date', 'period', ...]
kc2 / kr-vs-kp / letter / pendigits / spambase / vehicle : MATCHES=True

# chk_f1.py - real TabularEmbedder + real _select
A) credit-g_20nan
   missing cells total : 4000
   cells _select marks : 4000
   overlap             : 944 => 23.6%
   selected-but-OBSERVED: 76.4%
B) embedder col order: ['c1', 'c2', 'n1', 'n2']   frame col order: ['c1', 'n1', 'c2', 'n2']
   masked_positions from encode ([MASK] path, CORRECT):   after _select ([NULL] path, USED):
   [[0 0 1 0]                                             [[0 1 0 0]
    [0 1 0 0]                                              [0 0 1 0]

# chk_f2.py - realised evaluation-mask rate (share of OBSERVED cells), seed 7
credit-g     p=0.2 : 00nan:0.2049(floor 0.2%)  20nan:0.1672(floor 2.1%)  40nan:0.1426(floor 12.7%)  60nan:0.1515(floor 44.3%)  80nan:0.2512(floor 81.4%)
credit-g     p=0.05: 00nan:0.0660(floor 26.9%) 20nan:0.0717(floor 45.2%) 40nan:0.0867(floor 65.9%)  60nan:0.1269(floor 83.6%)  80nan:0.2475(floor 95.7%)
electricity  p=0.2 : 00nan:0.2210(floor 9.4%)  20nan:0.2182(floor 24.2%) 60nan:0.3236(floor 70.6%)
electricity  p=0.05: 00nan:0.1327(floor 62.7%) 20nan:0.1609(floor 74.4%) 60nan:0.3082(floor 92.2%)

# chk_f346.py
electricity_80nan  rows 7593/45312 fully missing -> 60744/290000 induced cells = 20.9%
electricity_60nan  rows  790/45312 fully missing ->  6320/217496 induced cells =  2.9%
electricity_20nan  rows    0/45312 fully missing ->      0/72496 induced cells =  0.0%
credit-g_80nan     rows   10/1000  fully missing ->    200/16000 induced cells =  1.2%
credit-g_60nan / credit-g_20nan                  ->                              0.0%
electricity_20nan  dtype mismatches: {'day': ('float64', 'int64')} | guard: PASSED
credit-g_20nan     dtype mismatches: {}                            | guard: PASSED
all nine _00nan tables: NaN count 0

# chk_f57.py
credit-g_80nan p=0.2: masked 1005 | 5.0% of ALL cells | 25.1% of OBSERVED cells (== realised_rate)
credit-g_00nan p=0.2: 20.5% of all cells
StandardScaler mean_=nanmean: True | NaNs preserved: True
baseline rmse scaled-space=1.070101  raw/sigma=1.070101  equal=True
credit-g duration sigma: 00nan 12.0528 (n=1000) | 20nan 11.8835 (800) | 60nan 12.2502 (400) | 80nan 12.1850 (200)

# chk_f46b.py - validation vs test mask coupling, and sibling encodability
credit-g_00nan  per-cell agreement on shared prefix: 0.9989  (both-masked 373 of val-masked 374)
credit-g_20nan  per-cell agreement on shared prefix: 0.9144  (both-masked 164 of val-masked 249)
credit-g_60nan  per-cell agreement on shared prefix: 0.9094  (both-masked  33 of val-masked 114)
credit-g_80nan  per-cell agreement on shared prefix: 0.9033  (both-masked   3 of val-masked  90)
variant 'day' vocabulary: ['1.0','2.0',...,'7.0','[MASK]','[NULL]','nan']
sibling 'day' stringified: ['1','2','3','4','5']
sibling encodes: RAISES -> ValueError y contains previously unseen labels: '2'
```

## Verdicts

### F-04-1 - `--score_null_path` marks the selection tensor in CSV column order while the model's mask is in categorical-then-numerical order, so the diagnostic scores transposed cells

- **Verdict:** CONFIRMED
- **Severity after review:** high
- **Basis:** Unambiguous at source and reproduced end to end. `src/embedder.py:189` builds
  the mask as `columns = list(self.categorical_columns) + list(self.numerical_columns)`;
  `src/training/decoding.py:262` overwrites it with
  `_select(encoded, torch.tensor(test_frame.isna().to_numpy(), device=device))`, which is in
  `test_frame.columns` order. `_score_population` then reads `mask[:, index]` with `index`
  enumerating `embedder.categorical_columns` (`decoding.py:287`) and `offset + index` for
  numerical (`:306`), so mask column *i* is interpreted as embedder column *i* regardless of
  which CSV column supplied it.

  The round trip through the real `TabularEmbedder` and the real `_select` shows the two
  marked cells exactly transposed (probe B above). On the real `credit-g_20nan` feature
  frame, `_select` marks 4000 cells of which only 944 (23.6%) are actual gaps; 76.4% are
  cells observed in the variant. The critic's order table reproduces exactly across all nine
  bases: only `credit-g` and `electricity` differ, the seven all-numerical / all-categorical
  bases match.

  The provenance claim holds too:

  ```
  git log -S"_select" -- src/training/decoding.py
  345574b 2026-09-10 19:10 feat(training): the decode stage      <- only commit, never changed since
  git log -S"kr-vs-kp_60nan" -- docs/adr/0004-imputation-decoder-task.md
  74229fd 2026-09-10 19:46 feat: imputation regression fixture, docs, and the token-path verdict
  ```

  So the four `credit-g` rows in the decision-11 verdict table
  (`docs/adr/0004-imputation-decoder-task.md:176-179`) were measured 36 minutes after the
  only implementation of `_select` landed, with this code in place. `kr-vs-kp` is
  all-categorical (36/36), so its rows are valid; the pre-registered criterion's "at least
  two datasets" therefore currently rests on one.

  Blast radius confirmed limited: `_score_population` for the `masked` and `induced`
  populations indexes `masked_positions` produced by `encode` itself, and `_already_missing`
  is an order-independent scalar, so no primary metric moves.
- **Correction:** One phrase in the critic's consequence needs tightening before it is
  quoted. They describe the transposed cells as ones "the model **can already read** in its
  input", implying the null-path score is flattered. The head weights are trained (one
  `nn.Linear` per column, shared across rows), but `TridentDecoder.forward`
  (`src/models.py:161-190`) computes loss **only** at `mask[:, index]` positions, so the head
  has never been optimised against a context vector taken from an *observed* position. The
  credit-g null-path figures (1.155, 1.180, 1.227, 1.422) are therefore a trained readout on
  an out-of-distribution input: the direction of the error is indeterminate, not
  systematically optimistic. That strengthens the finding — the numbers are not biased, they
  are meaningless — rather than weakening it, but "can already read" overstates what is
  known.
- **Severity note:** `--score_null_path` is off by default and `experiment_imputation.ps1`
  does not pass it (grepped: the flag appears only in ADR/ticket prose, never in a launcher).
  High is still right: the defect is silent (cell counts match, `n_num_cells`/`n_cat_cells`
  look correct, nothing raises), it lands on `credit-g`, the dataset the imputation
  regression fixture is built on, and its output was written into a pre-registered decision
  record that now reads as settled.

### F-04-2 - the self-masked population's realised rate is non-monotone and floor-dominated across the ladder, contradicting the ADR's comparability claim

- **Verdict:** SOUND
- **Severity after review:** high
- **Basis:** Premise verified line by line and numerically. `src/utils.py:55`,
  `p_dynamic = p_base * (1 - prop_nulls.values[:, None])`, deflates the rate by the row's
  null density; `src/utils.py:64-74` then forces one mask into every row that drew none,
  skipping only fully-null rows (`if len(non_null_indices) > 0`). Every realised-rate and
  floor-share figure the critic reported reproduces to four decimals (chk_f2.py above),
  including the non-monotone credit-g sequence 0.2049 / 0.1672 / 0.1426 / 0.1515 / 0.2512 and
  the 6.2x mislabel on `electricity_60nan` at nominal 0.05 (0.3082).

  The instrumentation gap is real at source: `metrics["impute/masked/realised_rate"]` is set
  once at `src/training/decoding.py:155-157`, outside the `for extra in
  hyperparameters.eval_mask_rates_extra` loop at `:162-172`, which emits only
  `f"impute/masked/rate_{round(extra * 100)}"`-prefixed score keys. Nothing in MLflow reveals
  that `rate_5` on `electricity_60nan` names a 30.8% population.

  The ADR claim is quoted accurately: `docs/adr/0004-imputation-decoder-task.md:222` reads
  "on any variant, `cv/test/impute/masked/impute_score/mean` is comparable across the
  ladder."

  Reasoning holds. At `_00nan` the estimand is "a near-uniform 20% sample of cells"; at
  `_80nan` it is "one uniformly chosen observed cell from each of 81% of rows" — a different
  population on a differently sized share of eligible cells, in a non-monotone sequence. A
  ladder plot of `impute_score` would show a shape driven by the masking helper. The critic
  is also correct to note the design stays value-independent throughout, so this is not a
  leakage claim; the problem is that the sampling *intensity* tracks row completeness.
- **Correction:** None. Severity kept at high against the usual deflation: this is not a
  miscomputed number but a written protocol claim that would produce an artefact if acted
  on, and the mislabeled segment keys are invisible from the tracking store.

### F-04-3 - a growing share of the induced population sits in rows with no observed cell, bounding those cells' contribution at baseline parity

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** Premise true. `datasets/generate_splits.py:39-47` injects
  `n_inject = int(round(n_rows * pct))` gaps **per feature column** independently, so whole
  rows can vanish; `src/training/decoding.py:243`,
  `hidden = test_frame.mask(test_frame.isna(), "[MASK]")`, masks them unconditionally. Counts
  reproduce exactly: `electricity_80nan` 7593/45312 rows fully missing = 60744/290000 induced
  cells = 20.9%; `electricity_60nan` 2.9%; `credit-g_80nan` 1.2%. For such a row the
  transformer input carries only column-identity, mask and positional embeddings, so the
  per-column head emits a constant; the optimal constant is the column's unconditional
  mean/mode, which is what `mean_mode_baselines` (`src/training/imputation_metrics.py:120-137`)
  supplies as the denominator in `_ratio`. Nothing in `_error_metrics` distinguishes these
  cells — `n_num_cells`/`n_cat_cells` pool them with the rest.
- **Correction:** Two adjustments. (1) "contributes a per-cell ratio of about 1.0" understates
  the direction. A decode-training row on an `_80nan` variant is ~80% `[NULL]` plus roughly
  one `[MASK]`, so an all-`[MASK]` row is a shape the model essentially never saw; its output
  is an extrapolation and is more likely worse than the marginal than equal to it. These
  cells are bounded **below** by ~1.0 in expectation, not pinned at it. (2) Severity cut from
  medium to low on reach: the effect is 0.0% on every variant of the corpus except
  `electricity_60nan` (2.9%), `credit-g_80nan` (1.2%) and `electricity_80nan` (20.9%) — one
  rung of one ladder carries essentially all of it. The ratio also remains internally fair
  (the baseline is scored on the same cells), so the number is not wrong; what is at risk is
  the causal attribution decision 10 invites when it says "the dataset ladder already is one
  [sweep]". That is a reporting/stratification gap, not a measurement error.

### F-04-4 - `_assert_row_aligned` checks only value equality through an object-dtype comparison, and certifies the corpus's one sibling incompatibility

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** Premise verified. The guard (`src/training/data.py:112-129`) is exactly
  `observed = variant.notna().to_numpy()` then
  `(variant.to_numpy()[observed] == complete.to_numpy()[observed]).all()`. Run against the
  shipped corpus it passes on `electricity_20nan` despite `day` being `float64` in the
  variant and `int64` in the sibling, and the downstream failure is real, not hypothetical —
  reproducing the embedder's own path (`as_category_strings` at `src/embedder.py:17-27`, fit
  on the variant per `src/embedder.py:104-114`) gives a variant vocabulary of
  `['1.0'..'7.0','[MASK]','[NULL]','nan']` and the sibling's `'1'..'7'`, so
  `LabelEncoder.transform` raises `ValueError: y contains previously unseen labels: '2'`.

  The second half of the premise also holds: nothing in code checks that the sibling is
  non-NaN where the variant has a gap (only `tests/unit/test_training_data.py:113`
  asserts it), and `inject_nans` returns `df.copy()` for `pct <= 0`, so an `_00nan` built
  from a raw table containing NaN would inherit it and `_clean` would hand
  `_score_population` a `[NULL]` `actual` for a categorical cell (unemittable by the head,
  `src/models.py:118-127`) or `0.0` for a numerical one — both silently scored. ADR decision 5's
  "Cells NaN in `_00nan` itself are never scored" is indeed a property of this corpus
  (all nine `_00nan` tables: NaN count 0), not an enforced invariant.
- **Correction:** Severity cut from medium to low for two reasons. First, the guard does
  satisfy its own stated purpose — for row *misalignment* the value comparison catches the
  failure, and the object-dtype `1.0 == 1` looseness matters only for dtype drift, which is
  not what the docstring promises. The finding reframes the guard's job rather than showing
  it fails at it. Second, the concrete consequence (the `electricity_20nan` crash) is the
  critic's own "known finding 1"; counting it again here double-counts severity, and the
  remaining consequence — a NaN-bearing `_00nan` silently scored — has zero occurrences in
  the corpus and would require regenerating a variant from a NaN-bearing source. The design
  point (cheap checks that would turn a stack trace into a named column) stands on its own at
  low.

### F-04-5 - the README's `EVAL_MASK_RATE` paragraph omits the per-row floor and quotes a figure against a different denominator than the metric it names

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** Premise verified. `README.md:272` states the helper "scales it down by each
  row's null density and never hides an already-missing cell, so asking for `0.2` hides about
  20% of a `_00nan` variant but about 5% of an `_80nan` one. Each run logs the realised share
  as `impute/masked/realised_rate`." Measured on `credit-g_80nan` at nominal 0.2: 1005 masked
  cells = **5.0% of all cells** and **25.1% of observed cells**.
  `src/training/decoding.py:154-157` divides by observed cells
  (`masked_positions.numel() - _already_missing(test_frame)`), so the named metric reads
  0.251. A reader who tunes `EVAL_MASK_RATE` against `realised_rate` sees a number 5x the one
  the paragraph promised. The omitted floor (`src/utils.py:64-74`) supplies 81.4% of the
  scored cells at that setting.
- **Correction:** The headline "wrong in direction" needs qualifying. Under the paragraph's
  own implied denominator — share of *all* cells — the prose is arithmetically correct:
  20.5% on `_00nan` versus 5.0% on `_80nan`, and the direction is down as stated. The
  direction only inverts (0.2049 -> 0.2512) once you adopt the denominator of the metric the
  sentence points at. So the defect is a denominator mismatch between prose and metric plus
  an incomplete mechanism, not a false statement about how the helper behaves. Severity stays
  low: documentation, with a fix the F-04-2 table already supplies.

### F-04-6 - validation and test evaluation masks are drawn from one stream reseeded identically, so the two questions are not independent

- **Verdict:** UNSOUND
- **Severity after review:** low
- **Basis:** The premise is true and I reproduced the headline figure. `evaluation_mask`
  (`src/training/data.py:132-148`) does `np.random.seed((seed * 1_000_003 + fold) % (2**32))`
  on every call, and `src/training/decoding.py:78` and `:138` pass the same
  `(seed, fold_ordinal)`; `preprocess_table` then draws `np.random.rand(*data.shape)`
  row-major from that identical start. On `credit-g_00nan`, seed 42, fold 1, rate 0.2, the
  per-cell agreement on the shared prefix is 0.9989 and 373 of 374 validation-masked cells
  are also test-masked at the same position. The nesting across rates also reproduces: 226 of
  285 `rate_5` cells are `rate_20` cells.

  What does not survive is the generality. The coupling exists only because `p_dynamic` is
  row-constant when there are no nulls. On the gapped variants — where the imputation work
  actually lives — the per-row deflation and `dynamic_mask[null_values] = False` break it:

  ```
  credit-g_20nan  both-masked 164 of val-masked 249  (66%)
  credit-g_60nan  both-masked  33 of val-masked 114  (29%)
  credit-g_80nan  both-masked   3 of val-masked  90  ( 3%)
  ```

  The critic measured only `_00nan` and stated "the validation mask is 99.89% positionally
  identical to the top block of the test mask" as a property of the protocol.
- **Correction:** Premise true, consequence overstated — hence UNSOUND rather than SOUND. The
  claimed harm ("biases test error optimistically by an unquantified but non-zero amount") is
  conceded to be unquantified, and the mechanism it rests on is second-order even at its
  strongest: the rows are disjoint so there is no cell-level leak, and what is shared is
  column composition, whose sampling difference between two independent masks is already
  small at 200x20. On `_20nan` and above the shared composition is mostly gone. What survives
  as a genuine, cheap-to-fix point: (a) the ADR's "validation uses the same draw" is
  ambiguous between "same seed" and "positionally the same mask", and on `_00nan` variants
  the second reading is the true one; (b) the extra rates are nested inside the primary rate
  rather than being independent difficulties, which is the more concretely checkable of the
  two and is not documented anywhere. Severity low; the fix (mix a split id into the derived
  seed) is one argument, but no result on this branch is materially in question because of
  it.

### F-04-7 - `rmse_num_z` / `mae_num_z` are in units of a sigma fit over all rows including the test fold, re-estimated at each rung

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** Premise verified at source and numerically.
  `frame[numerical_columns] = scaler.fit_transform(frame[numerical_columns])` runs at
  `src/training/data.py:71` inside `prepare_dataset`, and `build_folds` is called afterwards
  at `src/training/runner.py:46` — folds do not exist when the scaler is fit. `StandardScaler`
  is NaN-aware here (`mean_` matches `np.nanmean`, NaNs preserved through `transform`), so
  the statistics come from every observed row of the variant, test fold included. The
  scale-invariance check also reproduces: the train-fold-mean baseline RMSE in scaled space
  equals the raw train-fold-mean RMSE divided by the global sigma to six decimals
  (1.070101 both ways), so sigma cancels in `_ratio` and `impute_score` is unaffected.
  `rmse_num_z` and `mae_num_z` are emitted per population in
  `src/training/imputation_metrics.py:80-83` and are pinned in
  `tests/fixtures/credit-g_20nan_imputation_regression.json`, so the absolute numbers do
  carry the variant-specific unit.
- **Correction:** The "measurably different units" across the ladder is true but far smaller
  than the framing suggests. Measured on credit-g's `duration`: sigma = 12.0528 (n=1000),
  11.8835 (800), 12.2502 (400), 12.1850 (200) — about a 3% spread across a 5x change in the
  estimating sample, not a difference a reader would notice against imputation error. The
  substantive half of the finding is the first half: the unit is fit with the scored rows'
  own values, so `rmse_num_z` is not an honest held-out absolute error. Severity low is right
  — it is a documentation obligation (the order is protected by `AGENTS.md`, the ratio metric
  is unaffected, and the critic correctly scopes the fix to prose).

## What this critique missed

Two protocol problems squarely in this dimension, both verified at source, neither touched by
any of the seven findings.

**1. The Optuna search objective is computed on the very mask its checkpoint was selected on.**
`src/training/decoding.py:204-206` scores the search objective with `hidden_validation` —
not a fresh draw, but the identical `EncodedTable` built at `:77-82` and evaluated every
epoch at `:114`, with the checkpoint kept at `:125-127` as the argmin over all
`decode_epochs` (default 150) of the loss on exactly those cells. So
`validation/impute/masked/impute_score`, the objective ADR 0005:103 names, is a
max-over-150-epochs statistic on the selector's own question. ADR 0005's claim is that this
avoids ranking trials on the test split, which it does — but it substitutes a different bias:
every trial's reported objective is optimistically selected, and the amount of that
optimism grows with `EPOCHS_DECODE`, which is itself a sampled parameter (`opt.py:107`
region). A trial that draws more epochs gets more selection bites at the same fixed mask than
one that draws fewer, so the search is partly ranking trials on how many chances they had. An
honest objective needs either a third mask or a nested split. F-04-6 examines
validation-versus-test coupling and misses this, which is the same class of problem one level
in.

**2. The checkpoint is selected on a criterion the fold is not ranked on, and the criterion
moves between trials.** `decoding.py:114` selects on `model(hidden_validation,
clean_validation)`, which `TridentDecoder.forward` returns as
`cross_entropy_mean + lambda_num * mse_mean` (`src/models.py:193-207`). The fold is then
ranked on `impute_score`, which weights the two kinds by *cell count*
(`imputation_metrics.py:50-58`) and contains no `lambda_num` at all. `LAMBDA_NUM` is a
sampled hyper-parameter (`opt.py:107` and `:143`, `suggest_float('LAMBDA_NUM', 0.1, 10.0,
log=True)`), so on a mixed table like credit-g a trial drawing lambda 0.1 selects the epoch
best at categories while a trial drawing 10.0 selects the epoch best at numbers — and both
are then compared on one lambda-free ratio. That is a confound between a searched parameter
and the selection rule, not just a mismatch of units, and on `credit-g_20nan` (13 categorical
/ 7 numerical) the two criteria genuinely disagree about which epoch is best.

Smaller, worth a line: the critique's "Checked and cleared" entry on population disjointness
is correct for a run, but with `--score_null_path` on, the ledger written to the artifacts
contains the `induced_null_token` rows carrying F-04-1's transposed `row`/`column` pairs, so
the per-cell preview and the ledger CSV — not only the MLflow metric — record cells that were
never the ones scored.
