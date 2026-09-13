# 01 - The decode stage

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/decoding.py` (all 346 lines); `src/training/runner.py:22-150`; the
decode stage's direct dependencies where they decide what the stage measures —
`src/training/data.py:43-148` (`prepare_dataset`, `load_complete_sibling`,
`evaluation_mask`, `build_folds`), `src/utils.py:18-82` (`preprocess_table`,
`split_numeric_and_special`), `src/embedder.py:169-200` (`TabularEmbedder.encode`),
`src/models.py:95-250` (`TridentDecoder.forward` / `.predict`),
`src/training/imputation_metrics.py` (whole file, as the consumer of `scored_cells`),
`src/training/pretraining.py` (to trace what the decoder inherits),
`src/training/types.py:34-175` (`TaskSpec`, `Hyperparameters` defaults),
`opt.py:85-150, 205-290` and `src/training/summary.py:242-257` (to trace where the
decode stage's numbers end up). Read `docs/adr/0004-imputation-decoder-task.md` and
`docs/adr/0005-reduced-optuna-search-for-imputation.md` for the claims the code makes.

**Method:** Oriented with `graphify query "how does the decode stage select its checkpoint
and score imputation cells"`, then read the source. Four empirical probes, all
`uv run --python 3.10 python`, all against the real functions:

1. `preprocess_table` at nominal rates 0.0 … 1.0 on a 500x10 complete frame.
2. `evaluation_mask` at the default `EVAL_MASK_RATE=0.2` across a synthetic 2000x20 MCAR
   missingness ladder (0 / 20 / 40 / 60 / 80 %), plus the within-variant correlation
   between a row's null count and its mask count.
3. `evaluation_mask` on a 200-row and a 500-row frame with the same `(seed, fold)`, to
   test whether the validation and test masks are independent draws.
4. The real `train_and_evaluate_decoder` end to end (3 decode epochs, 200-row toy
   `_20nan` frame with a complete sibling, `eval_mask_rates_extra=(0.0, 0.05, 0.10, 1.0)`)
   — script at
   `C:\Users\DIOGON~1\AppData\Local\Temp\claude\C--Users-Diogo-Neiss-Documents-Mestrado-TRIDENT\f48d5c06-b3ce-4977-b950-7a95c1d2053f\scratchpad\decode-critic\probe.py`.

**Not checked:** whether the shipped `_XXnan` variants are actually MCAR (the mean/mode
baseline's unbiasedness for the *induced* population rests on it); the magnitude of
F-01-3's bias on a real dataset, which needs paired runs with and without a split-aware
seed. No pytest, no full training run, no writes outside this file.

**Headline.** On the question I was asked to answer first — is the reported `impute_score`
optimistic, did the model or the schedule see something it will not have at inference? —
the answer is **almost no, with one narrow exception**. Train / validation / test indices
are disjoint; the complete sibling is touched only after `best_state` is restored; the
checkpoint is selected on validation. The exception is F-01-3: the validation mask and the
test mask are drawn from the same RNG stream, so the epoch is chosen under the very
masked-column configurations the test score is then measured under.

The larger problems are not leakage. They are that **the number `impute/masked/*` reports
is not the experiment the configuration describes**, and that the two populations the ADRs
compare are drawn from systematically different rows.

## Findings

### F-01-1 - The masked population's realised mask rate is neither the configured rate nor comparable across the missingness ladder, and the code's stated explanation of the gap has the sign backwards

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/decoding.py:152` (the comment), `:154-157` (`realised_rate`),
  via `src/training/data.py:146` → `src/utils.py:55` and `:65-74`

- **Evidence:** `evaluation_mask` delegates to `preprocess_table`, the *training*-time
  corruption helper, which does two things no evaluation mask should do:

  ```python
  # src/utils.py:55
  p_dynamic = p_base * (1 - prop_nulls.values[:, None])   # down-weight gappy rows
  # src/utils.py:65-74
  no_mask_rows = ~dynamic_mask.any(axis=1)
  for i in np.where(no_mask_rows)[0]:                      # floor: >= 1 mask per row
      ...
      dynamic_mask[i, random_index] = True
  ```

  Probe 1, complete 500x10 frame, realised fraction of cells masked:

  | nominal | 0.00 | 0.01 | 0.05 | 0.10 | 0.20 | 0.50 | 0.90 | 1.00 |
  |---|---|---|---|---|---|---|---|---|
  | realised | 0.1000 | 0.1000 | 0.1074 | 0.1348 | 0.2140 | 0.5052 | 0.9016 | 1.0000 |

  At nominal 0 the helper still hides one cell per row. Below ~0.15 the floor, not the
  requested rate, sets the exam.

  Probe 2, `evaluation_mask(df, 0.2, 42, 1)` on a 2000x20 MCAR ladder — realised masked
  rate as a fraction of *observed* cells:

  | variant | `_00nan` | `_20nan` | `_40nan` | `_60nan` | `_80nan` |
  |---|---|---|---|---|---|
  | realised | 0.2031 | 0.1677 | 0.1450 | 0.1522 | **0.2549** |

  It is **non-monotonic**: it falls to 0.145 and then turns back up past the complete
  table's own rate, because at 80 % missing `p_dynamic = 0.2 * 0.2 = 0.04` produces almost
  nothing and the per-row floor supplies one mask out of ~4 observed cells.

  Probe 4, the real stage on a 4-column toy `_20nan`: `impute/masked/realised_rate` came
  back **0.371** for a configured `eval_mask_rate` of **0.200**.

  The code's own account of this, at `decoding.py:152-153`, is:

  > *"realised is what the masking helper actually hid, which falls as missingness rises
  > because it never hides an already-missing cell."*

  The stated mechanism cannot be the cause — the denominator at `:154` already subtracts
  `_already_missing(test_frame)`, so excluding null cells is netted out. The real cause is
  `p_base * (1 - prop_nulls)`, and the direction is not "falls": it falls, then rises, and
  at nominal 0.2 on a narrow table it is nearly double the nominal.

- **Consequence:** ADR 0004's comparison protocol says *"on any variant,
  `cv/test/impute/masked/impute_score/mean` is comparable across the ladder"*
  (`docs/adr/0004-imputation-decoder-task.md:221-222`). It is not. A `credit-g_40nan` run
  is graded on a ~14.5 % exam and a `credit-g_80nan` run on a ~25.5 % one, both labelled
  `EVAL_MASK_RATE = 0.2`. Any degradation reported down the ladder is a mixture of the
  model getting worse and the exam changing difficulty in a direction that reverses
  partway. The effect is worst on narrow tables: `electricity` has 8 feature columns, so
  the floor alone guarantees ≥ 0.125 of all cells, and `credit-g` has 20, so it guarantees
  ≥ 0.05. This is the number the branch's headline comparison is built on.

- **Direction:** evaluation must not reuse the training corruption helper. Draw the eval
  mask directly: sample `rate` of the *observed* cells of the split uniformly, with no
  `(1 - prop_nulls)` down-weighting and no per-row floor (the floor exists so a
  pre-training batch is never loss-free; a scoring pass has no such need — a row with
  nothing hidden simply contributes no cells). Keep `realised_rate` as a check, and fix
  its comment; once the draw is direct, realised should equal nominal to sampling noise,
  and a persistent gap becomes a real alarm rather than expected behaviour.

### F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet are reported side by side as the same measurement at different difficulties

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/decoding.py:137-147` (masked) vs `:243` and `:174-182` (induced)

- **Evidence:** two independent biases push the populations apart.

  *Which rows get scored.* The masked population inherits `p_base * (1 - prop_nulls)`, so
  the gappier a row is the less of it is scored. The induced population is the exact
  mirror: `_score_induced_missing` scores *every* NaN in the row
  (`hidden = test_frame.mask(test_frame.isna(), "[MASK]")`, `decoding.py:243`), so a row's
  contribution is its null count. Probe 2 on one `_40nan` variant, `eval_mask_rate=0.2`:

  ```
  corr(nulls_in_row, masks_in_row) = -0.4153
  rows with  0-4  nulls: n=96    masks/row = 2.77   induced cells/row =  3.56
  rows with  5-8  nulls: n=1084  masks/row = 1.96   induced cells/row =  6.83
  rows with  9-12 nulls: n=763   masks/row = 1.34   induced cells/row =  9.93
  rows with 13-20 nulls: n=57    masks/row = 1.07   induced cells/row = 13.51
  ```

  The masked population is concentrated in rows with rich surviving context; the induced
  population is concentrated in rows with little.

  *What the model sees around the scored cell.* For the masked population the row still
  carries its natural gaps as `[NULL]` (`preprocess_table` writes `[NULL]` before
  masking). For the induced population every gap in the row becomes `[MASK]`, so the model
  is never shown a `[NULL]` at all — on a `_40nan` row it is asked to fill ~10 cells at
  once with no null tokens present, a configuration that occurs in neither decode training
  nor pre-training.

  Probe 4 shows both being emitted from one checkpoint into one metric namespace:
  `impute/masked/impute_score = 1.034` against `impute/induced/impute_score = 1.065`, with
  no recorded difference in the per-row conditions that produced them.

- **Consequence:** the gap between `impute/masked/impute_score` and
  `impute/induced/impute_score` — which ADR 0004 and ADR 0005 both present as the headline
  pair, induced being the benchmark and masked the cross-ladder companion — is not
  attributable. It confounds (a) the population type, (b) how many cells are hidden per
  row, (c) how much context each row had left, and (d) whether `[NULL]` was present. A
  reader concluding "the model does worse on real gaps than on induced masks" cannot tell
  that apart from "the real gaps happen to be in the rows we deliberately under-sampled on
  the other population."

- **Direction:** score the two on matched conditions or stop presenting them as a pair.
  The cheap fix that makes them comparable: for the masked population, hide *per row* the
  same number of cells that row is naturally missing (rather than a global rate), and
  present the natural gaps as `[MASK]` in both. Failing that, log the per-row mask-count
  distribution for both populations alongside the scores, and state in the ADR that the
  two are not a difficulty ladder over one population.

### F-01-3 - The validation and test masks are drawn from one RNG stream, so the checkpoint is selected under the same masked-column configurations the test score is measured under

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/data.py:145`, consumed at `src/training/decoding.py:77-82`
  (validation, which selects the checkpoint at `:125-127`) and `:138` (test)

- **Evidence:** the seed is derived from the run seed and the fold only —

  ```python
  # src/training/data.py:145
  np.random.seed((seed * 1_000_003 + fold) % (2**32))
  ```

  — and nothing identifies *which split* or *which rate* is being masked.
  `preprocess_table` then draws `np.random.rand(*data.shape)` as its first act, row-major.
  So the validation frame's `(Nv x C)` uniforms are literally the first `Nv*C` values of
  the test frame's `(Nt x C)` draw: validation row *i* and test row *i* receive the same
  uniforms, hence the same masked-column set.

  Probe 3, 200-row validation frame and 500-row test frame, same `(seed=42, fold=1)`, no
  nulls:

  ```
  rows 0..199 whose masked-column set is IDENTICAL between validation and test: 187 / 200
  cellwise agreement over those 200x12 cells: 0.9892
  ```

  The 13 divergent rows are the ones the per-row floor had to fix, where the
  `np.random.choice` loop consumes the stream at different offsets.

  `evaluation_mask`'s docstring claims the draw comes "from a stream of its own, derived
  from the run's seed and the fold". It does — but the *same* stream serves every split
  and every rate.

- **Consequence:** this is the one path by which the reported test `impute_score` can be
  optimistic. The checkpoint is the epoch minimising loss on validation under mask design
  *P*; the test score is then measured under *P* again for the first `min(Nv, Nt)` rows.
  With the default fold geometry that is every validation row's pattern reappearing in
  the test split (validation is ~10 % of train+val, test is `1/cv_folds`, so typically
  20-60 % of test rows carry a configuration the selection set also used). Because
  difficulty genuinely varies with *which* columns are hidden together (hiding
  `credit_amount` alone is not hiding it together with `duration` and `age`), the chosen
  epoch is the one best at exactly those configurations rather than at an independent
  sample of them. The bias is bounded by the across-configuration variance and is small,
  but it is real, it is in the flattering direction, and it is free to remove.

  Secondarily, the same shared stream makes every `eval_mask_rates_extra` population a
  near-nested subset of the primary one rather than an independent draw — benign for
  comparability, but undocumented and not what the `rate_*` labels suggest.

- **Direction:** mix the split identity and the rate into the seed, e.g.
  `evaluation_mask(frame, rate, seed, fold, purpose)` hashing `("validation"|"test", rate)`
  into the derivation. One line, no behaviour change for any other caller, and it makes the
  selection set and the measurement set independent.

### F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently, so the epoch chosen is not the epoch that best serves the reported score

- **Kind:** design
- **Severity:** medium
- **Where:** `src/training/decoding.py:125-127` (selection on the decode validation loss),
  against `src/models.py:203-208` and `src/training/imputation_metrics.py:52-56`

- **Evidence:** the loss the checkpoint is selected on averages each kind over its *own*
  cells and then adds them with a fixed 1:λ ratio:

  ```python
  # src/models.py:203-208
  categorical_mean = categorical_loss / categorical_cells
  loss = loss + categorical_mean
  ...
  numerical_mean = numerical_loss / numerical_cells
  loss = loss + self.lambda_num * numerical_mean
  ```

  — the comment above it is explicit that this is deliberate ("a table dominated by one
  kind cannot drown the other's term"). The metric that ranks the fold does the opposite:

  ```python
  # src/training/imputation_metrics.py:52-56
  score += (len(numerical) / total) * _ratio(metrics["rmse_num_z"], naive_rmse)
  score += (len(categorical) / total) * _ratio(1.0 - metrics["acc_cat"], naive_error)
  ```

  With `lambda_num` at its default 1.0 (`types.py:117`) and `electricity`'s declared mix of
  1 categorical and 7 numerical feature columns
  (`datasets/categorical_columns/electricity.txt`), checkpoint selection gives the single
  categorical column **50 %** of the criterion while `impute_score` gives it **12.5 %** —
  a 4x overweight. `credit-g` (13 categorical, 7 numerical) is milder but inverted:
  selection 50/50, ranking 65/35.

- **Consequence:** on any table with an unbalanced column mix, the epoch restored at
  `decoding.py:129-130` is chosen to serve a weighting the reported
  `impute/masked/impute_score` and `impute/induced/impute_score` do not use. On
  `electricity` the stage will keep an epoch that improved the one categorical column at
  the expense of the seven numerical ones, then report a score that is 87.5 % numerical.
  `lambda_num` can compensate, but it is only searched when the table is mixed
  (`opt.py:106-107`, `opt.py:142-143`), and a default `--task imputation` run — which is
  what the imputation launcher does — never touches it.

- **Direction:** either select the checkpoint on the same quantity that ranks the fold
  (score the validation split's `impute_score` each epoch and take its argmin — the
  machinery already exists at `decoding.py:204-209`, it is just gated behind
  `score_search_objective` and run once at the end), or cell-weight the two loss terms so
  the criterion matches. Do not fix it by making `lambda_num` mandatory: that hides the
  mismatch behind a tuned constant.

### F-01-5 - The extra-rate diagnostic ladder collapses at the low end and silently overwrites itself when two rates round to the same percent

- **Kind:** bug
- **Severity:** low
- **Where:** `src/training/decoding.py:162-172`

- **Evidence:** the metric prefix is built by rounding the rate to whole percent:

  ```python
  # src/training/decoding.py:169
  prefix = f"impute/masked/rate_{round(extra * 100)}"
  ```

  Two configured rates that round to the same integer (`0.115` and `0.124` → `rate_12`;
  `0.0` and `0.004` → `rate_0`) produce the same keys, and the second `metrics.update(...)`
  at `:170-172` overwrites the first with no warning — the run reports one population and
  claims it is both.

  Independently, the floor from F-01-1 makes the low end of the ladder degenerate. Probe 4,
  real stage, 50 test rows x 4 feature columns, `eval_mask_rates_extra=(0.0, 0.05, 0.10, 1.0)`:

  ```
  nominal   0%  key=impute/masked/rate_0/     scored cells= 50   realised=0.2500
  nominal   5%  key=impute/masked/rate_5/     scored cells= 50   realised=0.2500
  nominal  10%  key=impute/masked/rate_10/    scored cells= 54   realised=0.2700
  nominal 100%  key=impute/masked/rate_100/   scored cells=138   realised=0.6900
  primary  20%  key=impute/masked/            scored cells= 59   realised=0.2950
  ```

  `rate_0` and `rate_5` score exactly the same number of cells (one per row); they differ
  only in which column the forced draw picked, and their `impute_score` values (1.092 vs
  1.042) are that noise, not a difficulty effect. At the top end `rate_100` reaches only
  138 of the 159 eligible cells (87 %), because `p_dynamic = 1.0 * (1 - prop_nulls) < 1`
  means the nominal rate is unreachable on any variant that has gaps.

- **Consequence:** `--task imputation` with extra rates configured produces a
  `impute/masked/rate_*` family whose labels do not describe the exam. Anyone plotting
  `impute_score` against `rate_*` on a narrow table reads a flat low end and a ceiling at
  ~0.87 as model behaviour when both are artefacts of the mask helper. The key collision is
  latent today only because the shipped configs use whole-percent rates; nothing rejects a
  pair that collides.

- **Direction:** F-01-1's direct draw removes the low-end collapse and the ceiling on its
  own. Separately, reject duplicate prefixes at configuration-resolution time (or key the
  family by the rate's exact repr) so two configured rates can never share a key. Note also
  that `eval_mask_rate` / `eval_mask_rates_extra` have no CLI flag at all
  (`src/training/config.py:186-275` has no `--eval_mask_rate`), so these are reachable only
  through the per-dataset hyperparameter JSON — worth stating in the ADR, since the fixed
  0.2 is what every default run and every Optuna trial is graded on.

## Checked and cleared

- **No test-split information reaches decode training or checkpoint selection.** Traced
  every consumer of `fold.test_indices` inside the stage: `decoding.py:133` (the test
  frame), `:141` (the raw variant for display only), `:246` (the sibling slice). All three
  are first touched at line 133, after the epoch loop has ended and `best_state` has been
  restored at `:129-130`. `build_folds` (`data.py:151-241`) partitions with an outer
  `KFold`/`StratifiedKFold` and then splits train/validation *inside* the non-test
  indices, so the three index sets are disjoint by construction.

- **No complete-sibling information reaches training.** `complete_sibling` enters
  `train_and_evaluate_decoder` as a parameter and is referenced only at `:174-196`, inside
  the post-training scoring block. `runner.py:43-45` reads it once per run and passes it
  straight through. It never touches the embedder, the optimizer, the baselines, or the
  validation loss.

- **`impute_score` is invariant to the transductive `StandardScaler`.** `prepare_dataset`
  (`data.py:70-72`) fits the scaler on the whole frame, test rows included, before folds
  exist — a genuine transductive shortcut. But `impute_score` is a ratio of two RMSEs
  measured in the *same* z-space (`imputation_metrics.py:55-59`), so both an affine shift
  and a scale change cancel exactly. Only the absolute `rmse_num_z` / `mae_num_z` carry the
  contamination, and the effect of including 20 % more rows in a mean and a variance is
  below the noise the fixture tolerances already allow. Same for
  `_score_induced_missing`'s `dataset.scaler.transform(truth[...])` at `:253-255`: it puts
  the sibling in the variant's own space, which is the right choice given the model
  predicts in that space.

- **A scored cell's truth is always a real value; the `-1` sentinel is unreachable.**
  `TridentDecoder` maps `[MASK]`/`[NULL]` vocabulary ids to `-1` in `local_of`
  (`models.py:132-133`), which would be an out-of-range cross-entropy target. The masked
  population cannot produce one: `preprocess_table` zeroes the dynamic mask at null
  positions (`utils.py:62`), so `masked_positions` never selects a cell whose clean
  target is `[NULL]`. The induced population cannot either: its truth is the complete
  sibling, which has no gaps by definition.

- **The decoder cannot see a hidden cell's value.** For numerical columns the embedder
  computes the MLP output for every cell and then discards it wherever the mask or null
  flag is set (`embedder.py`, the nested `torch.where`), and
  `split_numeric_and_special` writes `0.0` into the value slot at every special token
  (`utils.py:123`). For categorical columns the hidden cell carries the `[MASK]`
  vocabulary id. There is no residual path from the true value into the encoder input.

- **The search objective is computed on the validation split, not the test split, and the
  key survives to `opt.py`.** `decoding.py:202-211` scores `hidden_validation` /
  `clean_validation` and prefixes the family with `validation/`;
  `TaskSpec.search_objective` is `validation/impute/masked/impute_score`
  (`types.py:66-72`) and `opt.py:266` reads exactly that key. It reaches there in both
  evaluation modes: single-split returns `results[0].metrics` (`runner.py:192`), which is
  `{**metrics, **validation_metrics}` from `decoding.py:219`; cross-validation goes through
  `compute_cv_summary`, which means *every* column except `fold`/`dataset`
  (`summary.py:247-248`) and so carries the `validation/` key through unrenamed. ADR 0005's
  "the test split never chooses hyperparameters" holds.

  Caveat worth recording rather than a finding: that objective is `impute_score` evaluated
  on the very cells the checkpoint was selected on (both use `hidden_validation` /
  `clean_validation`, encoded once at `:76-82`). Selection is by loss argmin and the
  objective is a different statistic, so this is not a direct max-over-epochs, but it is
  still an in-sample-of-selection estimate. It is uniform across trials in the `reduced`
  profile, which does not sample `EPOCHS_DECODE` (`opt.py:94-108`). In the `full` profile,
  which does (`opt.py:134`), trials with more decode epochs get more selection
  opportunities on the same fixed validation mask and are favoured by that alone.

- **`best_state` restoration is correct.** `:127` clones detached tensors each time the
  validation loss improves, `:129-130` reloads before any scoring. `model.eval()` is set at
  `:112` for the per-epoch validation forward and again at `:131`, so dropout is off for
  every number that is reported.

- **`realised_rate`'s denominator is right, even though its explanation is not.**
  `hidden_test.masked_positions.numel()` is `n_rows * (n_categorical + n_numerical)`, and
  `declared_column_types` (`data.py:20-42`) assigns every non-label feature to exactly one
  of the two lists, so that product is the full feature-cell count and subtracting
  `_already_missing(test_frame)` leaves the eligible cells. The arithmetic is sound; F-01-1
  is about what the numerator turned out to be.

## Open questions

- **Are the `_XXnan` variants MCAR?** The induced population's baseline is the train fold's
  mean/mode over *observed* cells (`imputation_metrics.py:120-140`), while the induced
  cells' true values come from the *unobserved* population. Under MCAR those are the same
  distribution and `impute_score`'s denominator is honest; under MAR or MNAR the baseline
  is biased and every induced `impute_score` shifts with it. Settled by reading the
  generator that produced the variants, or by comparing, on a `_20nan` variant, the
  complete sibling's mean over the removed cells against its mean over the kept ones.

- **How large is F-01-3 in practice?** Settled by two runs of `credit-g_20nan` at the same
  seed, one with the current `evaluation_mask` and one with the split mixed into the seed
  derivation, comparing `cv/test/impute/masked/impute_score/mean`. I expect a small
  positive shift when the streams are separated; I cannot claim the size.

- **Can a category be erased entirely at high missingness?** The decoder's answer space and
  the sibling's encoding both come from `LabelEncoder`s fitted on the *variant*
  (`embedder.py:109-114`), so a rare category whose every occurrence was removed in an
  `_80nan` variant is absent from the vocabulary while present in the `_00nan` sibling —
  `encoder.transform` at `embedder.py:177` would raise on `_clean(truth)`. This is a
  different trigger from the known dtype mismatch, and I could not confirm it fires on the
  shipped tables without running the `_80nan` variants. Settled by, for each base dataset,
  comparing the set of distinct values per declared categorical column between
  `<base>_80nan.csv` and `<base>_00nan.csv`.
