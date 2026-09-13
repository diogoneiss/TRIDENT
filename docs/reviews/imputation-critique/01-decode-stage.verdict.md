# 01 - The decode stage: verdicts

_Adversarial verification of `01-decode-stage.md`. 2026-09-11._

**Method:** Oriented with `graphify query "how does the decode stage select its checkpoint
and score imputation cells"`, then read the source rather than the graph:
`src/training/decoding.py` (all 346 lines), `src/utils.py:18-82` (`preprocess_table`),
`src/training/data.py:20-241` (`declared_column_types`, `prepare_dataset`,
`load_complete_sibling`, `evaluation_mask`, `build_folds`),
`src/training/imputation_metrics.py` (whole file), `src/models.py:95-250`
(`TridentDecoder.forward` / `.predict`), `src/embedder.py:17-27, 100-200`
(`as_category_strings`, vocabulary construction, `encode`), `src/training/types.py:30-195`,
`src/training/config.py:186-275`, `src/training/runner.py:22-70`, `opt.py:85-150`,
`docs/adr/0004-imputation-decoder-task.md`,
`docs/adr/0005-reduced-optuna-search-for-imputation.md`.

Five probe scripts plus one real training run, all `uv run --python 3.10`, scripts under
`C:\Users\DIOGON~1\AppData\Local\Temp\claude\C--Users-Diogo-Neiss-Documents-Mestrado-TRIDENT\f48d5c06-b3ce-4977-b950-7a95c1d2053f\scratchpad\verify-01\`:

- `p1.py` - `preprocess_table` across nominal rates on a complete 500x10 frame; a
  synthetic 2000x20 MCAR ladder through `evaluation_mask`; the per-row null/mask
  correlation; the validation-vs-test stream test; the prefix-rounding collision.
- `p2.py` - the same stream test on *gappy* frames; nesting of the extra-rate ladder;
  the low-end collapse on a narrow (4-feature) table.
- `p3.py` - the effect size of the shared stream: `P(cell masked in test | same cell
  masked in validation)` against the test masking base rate, across the ladder.
- `p4.py` - a sibling-vs-variant categorical-vocabulary sweep over every shipped
  `_XXnan` table.
- `p5.py` - **the shipped ladders through the real path**: `prepare_dataset` ->
  `build_folds(frame, label, 5, 42)` -> fold 1 test frame with the label dropped ->
  `evaluation_mask(., 0.2, 42, 1)` -> `masked / (rows*features - nan)`, which is exactly
  what `decoding.py:154-157` computes. Plus the erased-category test.

One real training run, in the sanctioned isolated form:

```
uv run --python 3.10 python main.py --dataset_name kc2_20nan --task imputation \
  --cv_folds 2 --disable_mlflow --metrics_dir .../scratchpad/verify-01/run
```

```
impute/masked/n_num_cells:       759.0000 +/- 15.5563
impute/masked/n_cat_cells:         0.0000 +/-  0.0000
impute/masked/rmse_num_z:          0.4245 +/-  0.1690
impute/masked/mae_num_z:           0.1820 +/-  0.0255
impute/masked/impute_score:        0.5319 +/-  0.0705
impute/masked/realised_rate:       0.1729 +/-  0.0034   <- configured EVAL_MASK_RATE = 0.2
impute/induced/n_num_cells:     1092.0000 +/-  2.8284
impute/induced/n_cat_cells:        0.0000 +/-  0.0000
impute/induced/rmse_num_z:         0.7034 +/-  0.0622
impute/induced/mae_num_z:          0.2156 +/-  0.0175
impute/induced/impute_score:       0.6113 +/-  0.0864
```

No pytest. No writes outside this file. (The sanctioned command form carries no
`--output_dir`, so the run also left `results/kc2_20nan/20260911_111752/` - a new
untracked directory; nothing pre-existing was touched.)

## Verdicts

### F-01-1 - The realised eval-mask rate is neither the configured rate nor comparable across the ladder, and `decoding.py:152-153` names a mechanism that is already netted out

- **Verdict:** SOUND
- **Severity after review:** high
- **Basis:** the premise is arithmetic, and the consequence reproduces on the shipped
  tables rather than only on synthetic ones.

  `evaluation_mask` (`data.py:137-148`) is a seeded wrapper around `preprocess_table`, the
  *training* corruption helper. That helper down-weights per row, then floors:

  ```python
  # src/utils.py:55
  p_dynamic = p_base * (1 - prop_nulls.values[:, None])
  # src/utils.py:62
  dynamic_mask[null_values] = False
  # src/utils.py:65-74
  no_mask_rows = ~dynamic_mask.any(axis=1)
  for i in np.where(no_mask_rows)[0]:
      ...
      dynamic_mask[i, random_index] = True
  ```

  For null fraction `q` over `C` columns, expected masks per row are
  `p_base*(1-q)*C*(1-q)`, and `decoding.py:154` divides by the `C*(1-q)` *observed* cells,
  so the ratio the metric reports is `p_base*(1-q)` - plus what the per-row floor adds
  back, which is `1/(C*(1-q))` per floored row and therefore grows without bound as `q`
  rises. That is the whole finding: a falling term and a rising term, crossing somewhere in
  the middle.

  **Shipped `credit-g` ladder** (`p5.py`, fold 1 of 5, seed 42, `EVAL_MASK_RATE = 0.2`,
  200 test rows x 20 feature columns), `realised_rate` computed exactly as the code does:

  | variant | `_00nan` | `_20nan` | `_40nan` | `_60nan` | `_80nan` |
  |---|---|---|---|---|---|
  | realised | 0.2085 | 0.1685 | 0.1548 | 0.1572 | **0.2500** |

  **Shipped `electricity` ladder** (8 feature columns, so the floor alone forces >= 0.125):

  | variant | `_00nan` | `_20nan` | `_40nan` | `_60nan` | `_80nan` |
  |---|---|---|---|---|---|
  | realised | 0.2230 | 0.2198 | 0.2437 | 0.3246 | **0.5242** |

  On `electricity` the exam at `_80nan` hides **2.4x** the share of surviving cells it hides
  at `_00nan`, monotonically upward - the opposite of what the comment claims. On
  `credit-g` it falls to 0.155 and then reverses to 0.250. Both are labelled
  `EVAL_MASK_RATE = 0.2`.

  The real run agrees: `kc2_20nan` (21 feature columns, ~20% missing) reported
  `impute/masked/realised_rate = 0.1729` for a configured `0.2`.

  Floor behaviour at the low end (`p1.py`, complete 500x10 frame, realised share of all
  cells): nominal 0.00 -> 0.1000, 0.05 -> 0.1074, 0.10 -> 0.1348, 0.20 -> 0.2140. Below
  about 0.15 the floor, not the request, sets the exam.

  The comment being corrected:

  > `# Nominal is what was asked for; realised is what the masking helper actually hid,`
  > `# which falls as missingness rises because it never hides an already-missing cell.`

  Both halves are wrong about this metric. Excluding already-missing cells is netted out by
  the denominator at `:154`, and the quantity does not monotonically fall.

  The consequence lands on a claim the ADR does make, verbatim at
  `docs/adr/0004-imputation-decoder-task.md:221-222`: *"on any variant,
  `cv/test/impute/masked/impute_score/mean` is comparable across the ladder."* It is not.
  `impute_score` is a ratio of model error to a *context-free* mean/mode baseline
  (`imputation_metrics.py:44-57`), so hiding a larger share of a row removes context from
  the model and none from the baseline: exam difficulty moves the score directly. A
  degradation reported down the `credit-g` ladder mixes model decay with an exam that gets
  easier and then abruptly harder; down the `electricity` ladder it mixes it with an exam
  that gets steadily harder.

- **Correction:** two of the critic's framings need trimming.
  1. *"the sign is backwards"* overstates a comment that is wrong in two separate ways
     rather than inverted: it names a netted-out mechanism, and it asserts monotonicity
     where there is none. The direction of the error is not even constant across datasets -
     down-then-up on `credit-g`, straight up on `electricity`.
  2. The codebase is inconsistent, not uniformly wrong. `types.py:118-120` already
     documents the real mechanism - *"the masking helper scales it down by each row's null
     density: asking for 0.2 hides about 20% of a complete variant but about 5% of an
     80%-missing one"* - though that comment misses the floor and so predicts 5% where the
     shipped `credit-g_80nan` delivers 25% and `electricity_80nan` 52%. Both comments need
     the same fix; only one of them is even directionally right.

  Severity stays **high**: this is the number ADR 0004 nominates for the cross-ladder
  comparison, and the confound is a factor of about 1.6 on `credit-g` and 2.4 on
  `electricity`.

### F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet reported as a pair

- **Verdict:** SOUND
- **Severity after review:** medium
- **Basis:** both mechanisms are real and reproduce on shipped data.

  *Row composition.* The masked population inherits `p_base*(1-prop_nulls)`, so the gappier
  a row is the less of it is scored. `_score_induced_missing` is the exact mirror:

  ```python
  # src/training/decoding.py:243
  hidden = test_frame.mask(test_frame.isna(), "[MASK]" if as_mask else "[NULL]")
  ```

  Every NaN becomes a `[MASK]`, and `embedder.encode` marks every `[MASK]` as a scored
  position (`embedder.py:188-190`), so a row contributes exactly its null count. On the
  shipped `credit-g` ladder (`p5.py`, fold 1, rate 0.2), `corr(nulls_in_row,
  masks_in_row)` is **-0.359 / -0.406 / -0.336 / -0.345** at `_20/_40/_60/_80nan`. The two
  populations also change size in opposite directions as missingness rises: masked
  541 -> 374 -> 249 -> 203 cells while induced goes 789 -> 1584 -> 2416 -> 3188.

  *Context tokens.* For the masked population the row keeps its natural gaps as `[NULL]`
  (`preprocess_table` writes `[NULL]` before masking). For the induced population every gap
  in the row is converted to `[MASK]`, so the model is shown a gappy row with **zero**
  `[NULL]` tokens and zero null flags - a token configuration that never occurs in decode
  training, where a gappy row always carries its gaps as `[NULL]`.

  The real run emits both into one namespace from one checkpoint, with nothing recorded
  about the differing conditions: `impute/masked/impute_score = 0.5319` against
  `impute/induced/impute_score = 0.6113`, over 759 and 1092 cells respectively. ADR 0005
  presents them as a unit at `:161-162` and `:191-192`: *"Headline
  `cv/test/impute/induced/impute_score/mean` ... companion
  `cv/test/impute/masked/impute_score/mean`."* A reader comparing 0.53 to 0.61 cannot tell
  "the model does worse on real gaps" from "real gaps sit in the rows the other population
  deliberately under-samples".

- **Correction:** one sub-claim is wrong and one framing is the critic's own.
  1. *"asked to fill ~10 cells at once ... a configuration that occurs in neither decode
     training nor pre-training"* - the **cell count** is in distribution. Decode training
     corrupts at `mask_probability`, default **0.5** (`types.py:103`, used at
     `decoding.py:92-94`), so a `_40nan` row in training is shown with about 0.5*0.6*20 = 6
     cells masked against about 8 at induced scoring. What is genuinely out of distribution
     is the *absence of `[NULL]`*, not the number of hidden cells. The finding survives on
     the narrower claim; the argument as written overreaches.
  2. Neither ADR says the two are "one measurement at two difficulties". ADR 0004:221-222
     and ADR 0005:161-162 say headline plus companion. The non-attributability is a real
     hazard for anyone who reads the pair as a difficulty ladder, which is the natural
     reading, but it is a hazard the ADRs create by juxtaposition rather than a claim they
     assert.

  Severity **medium**, not high: this changes how a reported pair must be *interpreted*,
  and is fixed by logging the per-row mask-count distributions for both populations and
  stating the caveat. Unlike F-01-1 it does not make any single reported number wrong.

### F-01-3 - Validation and test masks come from one RNG stream, so the checkpoint is selected under the configurations the test score is measured under

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** the premise is exactly as described, and the coupling is stronger than
  "shared seed" suggests.

  ```python
  # src/training/data.py:145
  np.random.seed((seed * 1_000_003 + fold) % (2**32))
  ```

  Nothing encodes the split or the rate, and `preprocess_table`'s first act is
  `np.random.rand(*data.shape)`, filled row-major - so the validation frame's `Nv*C`
  uniforms are literally the leading block of the test frame's `Nt*C` draw. Validation row
  *i* and test row *i* (different data rows) are compared against the same uniforms.

  `p3.py`, validation 200 rows / test 500 rows / 20 columns, same `(seed=42, fold=1)`,
  reporting `P(cell masked in test | same cell masked in validation)` against the test
  masking base rate:

  | null fraction | 0.0 | 0.2 | 0.4 | 0.6 |
  |---|---|---|---|---|
  | base rate | 0.208 | 0.136 | 0.090 | 0.062 |
  | conditional | **0.999** | **0.705** | **0.490** | 0.236 |

  On a complete variant the coupling is essentially total; at the `_20nan` rung - the rung
  the regression fixture and the kc2 run both use - a validation-masked cell is still about
  5x more likely than chance to be masked at the same position in test. `p1.py`
  corroborates at the row level: 187/200 rows share an identical masked-column set with no
  nulls, cellwise agreement 0.9892. `p2.py` shows it decaying with gaps (47/200 identical
  row patterns at 20% nulls, 27/200 at 40%), because per-row `p_dynamic` and the
  null-zeroing then differ.

  The checkpoint is chosen at `decoding.py:125-127` on the validation loss under that
  design, and the test score measured at `:145-147` under the same design for the leading
  test rows, so the selection set and the measurement set are not independent.

  The critic's secondary observation also holds: because the stream is shared across rates,
  the `rate_*` populations are near-nested rather than independent draws (`p2.py`: 535/682
  cells masked at 0.05 are also masked at 0.20; 1006/1062 at 0.10).

- **Correction:** severity cut from medium to **low**. The premise is airtight and the
  mechanism real, but nothing in the critique bounds the bias, and three things damp it
  hard: the rows are different rows, so only "which columns are hidden together" is shared;
  the decoder trains on masks re-rolled every epoch across all columns
  (`decoding.py:91-96`), leaving little room for a checkpoint to specialise to a column
  subset; and only `Nv/Nt` of the test rows fall in the overlapping index range
  (`validation_ratio = 0.1/(1-1/cv_folds)` at `data.py:196` gives `Nv/Nt` of 0.2 at
  `cv_folds=2` and 0.5 at `cv_folds=5`). The critic says as much - *"the bias is bounded by
  the across-configuration variance and is small"* - and then files it at medium. Low is
  the honest grade for an unquantified, small, one-line-to-fix bias. It should still be
  fixed: mixing the split identity and the rate into the seed derivation costs nothing and
  removes the question.

### F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** the premise is true as read. Selection is on the decode validation loss
  (`decoding.py:114`, `:125-127`), which combines the two kinds with fixed coefficients:

  ```python
  # src/models.py:200-208
  categorical_mean = categorical_loss / categorical_cells
  loss = loss + categorical_mean
  ...
  numerical_mean = numerical_loss / numerical_cells
  loss = loss + self.lambda_num * numerical_mean
  ```

  while the reported and fold-ranking `impute_score` weights each kind by its share of
  scored cells (`imputation_metrics.py:46-57`). `lambda_num` defaults to 1.0
  (`types.py:117`) and is sampled only when the table is mixed (`opt.py:106-107`,
  `opt.py:142-143`), so a default `--task imputation` run never touches it. Confirmed
  column mixes: `datasets/categorical_columns/electricity.txt` declares only `day`, giving
  1 categorical / 7 numerical features; `credit-g.txt` declares 13, giving 13/7. The two
  quantities do disagree in a systematic, dataset-dependent direction, and on `electricity`
  the disagreement is at its most extreme.

- **Correction:** the quantification must not survive into the report. The critic writes
  that selection gives the categorical column **"50%"** of the criterion against
  `impute_score`'s **"12.5%"**, *"a 4x overweight"*. No such share exists: the two terms
  are a mean cross-entropy in nats and a mean squared error in z-units, added with equal
  coefficients. Equal coefficients on incommensurable quantities do not make equal
  influence, and the influence is not a percentage of anything - it depends on the columns'
  entropies and variances and moves during training. The defensible claim is the
  directional one: **the checkpoint criterion is not cell-count weighted and the reported
  score is**, so on an unbalanced table the argmin epoch is not the epoch that minimises
  what gets reported.

  Severity cut from medium to **low**. Every early-stopping surrogate differs from the
  metric it serves; that alone is not a defect. To earn medium this needs evidence that the
  selected epoch actually moves - a per-epoch validation `impute_score` trace against the
  loss trace on `electricity_20nan` - which the critique does not have and neither do I.
  The suggested fix is still the right one and is nearly free: the machinery to score the
  validation split already exists at `decoding.py:202-211`, gated behind
  `score_search_objective`.

### F-01-5 - The `rate_*` family collides when two configured rates round to the same percent and overwrites silently; the low end of the ladder is degenerate

- **Verdict:** CONFIRMED
- **Severity after review:** low
- **Basis:** both halves reproduce; they are two different defects with different weight.

  *The collision.* `decoding.py:169` builds the key by rounding to whole percent, and
  `:170-172` writes it with `metrics.update`, which overwrites without complaint:

  ```python
  prefix = f"impute/masked/rate_{round(extra * 100)}"
  ```

  `p1.py`: `0.115 -> rate_12` and `0.124 -> rate_12` collide; so do `0.0` and `0.004`.
  Banker's rounding widens it - `0.115` and `0.125` both give `rate_12`. The run would
  report one population under a key that claims to be both. Nothing validates the tuple at
  configuration time (`types.py:171-176` only coerces to float).

  *The low-end collapse.* `p2.py`, a 4-feature 50-row frame at 20% missing, 162 eligible
  cells:

  ```
  nominal=0.0   key=rate_0    scored=50   realised/observed=0.3086
  nominal=0.05  key=rate_5    scored=50   realised/observed=0.3086
  nominal=0.1   key=rate_10   scored=51   realised/observed=0.3148
  nominal=0.2   (primary)     scored=55   realised/observed=0.3395
  nominal=1.0   key=rate_100  scored=146  realised/observed=0.9012
  ```

  `rate_0` and `rate_5` are the same exam - one forced cell per row - and any difference in
  their `impute_score` is which column the floor's `np.random.choice` happened to pick. At
  the top, nominal 1.0 is unreachable on any gappy table: `p1.py` gives 0.808 of observed
  cells at 20% missing and 0.626 at 40%, because `p_dynamic = 1.0*(1-q) < 1`. So a plot of
  `impute_score` against `rate_*` on a narrow table shows a flat low end and a ceiling that
  are both artefacts of the mask helper.

- **Correction:** the two halves deserve different treatment, and the third note is not a
  finding at all.
  1. The **collision is latent**, and I checked rather than assumed.
     `eval_mask_rates_extra` defaults to `()` (`types.py:125`), ADR 0004 decision 10 says
     the feature is off by default, and `grep -rn "EVAL_MASK" datasets/hiperparams/`
     returns exactly four lines - `credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan` and
     `kr-vs-kp_40nan`, each setting `"EVAL_MASK_RATE": 0.2` and none setting
     `EVAL_MASK_RATES_EXTRA`. Those four `.imputation.json` files are the entire shipped
     imputation config surface. So the whole `rate_*` family is dormant today: a real
     silent overwrite, with a trivial fix and no current victim. **Low.**
  2. The **low-end collapse and the ceiling are F-01-1's root cause restated** in the
     extra-rate family, not independent evidence. The critic says so in the Direction; the
     report should not count them twice.
  3. *"`eval_mask_rate` / `eval_mask_rates_extra` have no CLI flag"* - verified
     (`config.py:186-275` defines none), but this is CLI ergonomics, which the review
     explicitly excludes. It belongs in the ADR as a documentation note, not in a finding.

## What this critique missed

**The critic's open question 3 is answerable from the shipped tables alone, and the answer
is yes - one `kr-vs-kp` fold raises.** The critique lists *"Can a category be erased
entirely at high missingness?"* as unsettled and needing runs. It needs no run; a
vocabulary sweep over every shipped variant (`p4.py`, `p5.py`) settles it:

```
kr-vs-kp_00nan['spcop'] == 't' in 1 row(s): [2891]
the same row in kr-vs-kp_40nan: [nan]          (the generator cut it)
variant-fitted vocabulary for 'spcop': ['[MASK]', '[NULL]', 'f', 'nan']
transform of the sibling column RAISES: ValueError: y contains previously unseen labels: 't'
row 2891 is in the TEST split of fold 4/5 (seed 42)
```

The vocabulary is fitted on the variant (`embedder.py:109-114`, via `pretraining.py:29`)
and the sibling is transformed against it at `embedder.py:177` - reached from
`decoding.py:264`, `embedder.encode(_clean(truth))` inside `_score_induced_missing`. So
`kr-vs-kp_40nan` at `--cv_folds 5 --seed 42` completes folds 1-3 and then raises in fold 4,
*after* pre-training and decode training have run. The blast radius is one variant at one
fold geometry, which is why I would file it low-to-medium rather than high - but it is a
crash inside the shipped configuration space, and the critique filed it as unknown. (The
mechanism belongs to critique 05's area, and `electricity`'s `int64`/`float64` sibling
dtype mismatch - which my sweep also reproduces on all four `electricity` variants -
belongs to critique 04's. I claim only the settling of the open question, not those
findings.)

**F-01-3's more consequential twin is filed under "Checked and cleared".** The critique
notes, as a caveat rather than a finding, that `validation/impute/masked/impute_score` is
computed on the very cells the checkpoint was argmin'd over (`decoding.py:204-206` reuses
the `hidden_validation` / `clean_validation` encoded once at `:76-82`), and that the `full`
profile samples `EPOCHS_DECODE` (`opt.py:134`) so longer trials get more draws at the same
fixed validation mask. That is a selection-optimism mechanism acting on the quantity that
*chooses the hyperparameters*, with a magnitude that grows with the epoch range - strictly
more consequential than F-01-3's effect on one reported number, and not damped by
"different rows" the way F-01-3 is. It belongs in the search critique as a finding, not in
the cleared list.

**Nothing in the critique checks what happens when a fold's validation loss never
improves.** `best_state` starts at `None` (`decoding.py:87`) with `best_validation_loss` at
`float("inf")`, and `:129-130` guards the `None`. The guard is unreachable in practice
because the first epoch always improves on `inf` - except that a `nan` validation loss
compares false against `inf`, would leave `best_state` at `None` for every epoch, and the
stage would then silently score the final epoch's weights instead of a selected
checkpoint, with no warning anywhere in the metrics. It is a narrow path (it needs a `nan`
loss) and I did not make it fire, so it is a note rather than a finding - but the silent
degradation to "last epoch" is the part worth a one-line warning.

**Two cheap correctness checks the critique passed over, both clean.** `_per_column_scores`
(`runner.py:22-33`) groups by `population` before scoring, so the masked, induced and
null-token cells concatenated at `decoding.py:178, 190` are never pooled into one
per-column number. And `mean_mode_baselines`' categorical mode is stringified the same way
the vocabulary is (`str(...)` against `as_category_strings`' `astype(str)`), so an
integer-coded categorical column such as `electricity`'s `day` does not silently get a
baseline that can never be correct - which would have deflated every `impute_score` on
that column and looked like model skill.
