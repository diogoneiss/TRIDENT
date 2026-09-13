# 09 - Test adequacy for the imputation task

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `tests/unit/test_training_decoding.py` (1-288), `tests/unit/test_imputation_metrics.py` (1-192),
`tests/unit/test_imputation_artifacts.py` (1-248), `tests/unit/test_decoder_model.py` (1-166),
`tests/unit/test_training_tasks.py` (1-154), `tests/unit/test_training_data.py` (branch diff),
`tests/integration/test_credit_g_imputation_regression.py` (1-61),
`tests/fixtures/credit-g_20nan_imputation_regression.json`, read against
`src/training/decoding.py` (1-346), `src/training/imputation_metrics.py` (1-137),
`src/training/data.py` (1-229), `src/training/summary.py` (1-73), `src/training/runner.py` (1-192),
`src/training/artifacts.py` (280-430), `src/models.py` (94-241), `src/embedder.py` (17-200),
`src/utils.py` (18-80), `src/training/types.py` (30-140), `opt.py` (200-240).

**Method:** graphify orientation, then the source. Nine `uv run --python 3.10 python` snippets driving the
*real* functions (`_score_induced_missing`, `train_and_evaluate_decoder`, `evaluation_mask`,
`score_cells`) over fabricated and over real `datasets/processed_datasets/credit-g` frames. No pytest
(per instructions). **I deliberately did not spend the one permitted training run**: the project defaults
are 300 pre-training and 150 decode epochs at `DIM=128`, so a CPU run of `main.py` would be long, and the
`--output_dir` default writes into the repo's `results/`, which I am told to preserve. Everything below is
reproduced through in-process calls instead. What I could not check: whether the pinned fixture numbers
reproduce on CPU (the fixture records `device: cuda`), and the magnitude of F-09-1 on a fully-trained
credit-g decoder.

The summary: the unit suite is unusually thoughtful — nearly every test names a real risk in its docstring —
but it has one systematic blind spot that hides a live bug (every fabricated frame in every decode test
happens to be ordered categorical-columns-first, which no real mixed dataset is), and it has no test
anywhere that the decoder ever beats the naive baseline, so the whole path could be wired backwards and
stay green.

## Findings

### F-09-1 - The `[NULL]`-path diagnostic scores a different set of cells than it claims, and the only test that guards it asserts a permutation-invariant count

- **Kind:** bug
- **Severity:** high
- **Where:** `src/training/decoding.py:262`, guarded only by `tests/unit/test_training_decoding.py:233`
- **Evidence:**
  `EncodedTable.masked_positions` is built in **embedder order** — `src/embedder.py:189`:
  `columns = list(self.categorical_columns) + list(self.numerical_columns)`. `_score_population`
  reads it that way (`mask[:, index]` for categorical, `mask[:, offset + index]` for numerical).
  But the null path overwrites it with a boolean matrix in **frame order**:

  ```python
  # decoding.py:259-262
  if not as_mask:
      encoded = _select(encoded, torch.tensor(test_frame.isna().to_numpy(), device=device))
  ```

  `test_frame` is `dataset.frame.drop(columns=[label])`, i.e. CSV column order. The two orders agree
  only when every categorical column precedes every numerical one. `credit-g` interleaves them:

  ```
  frame order:    checking_status, duration, credit_history, purpose, credit_amount, savings_status, ...
  embedder order: checking_status, credit_history, purpose, savings_status, ..., duration, credit_amount, ...
  ```

  Driving the real `_score_induced_missing` on a four-feature frame ordered `size, colour, weight, shape`
  (one gap, in `colour`):

  ```
  MASK path scored cells:  row 0, column colour   (categorical)
  NULL path scored cells:  row 0, column shape    (categorical)
  counts equal? True        same cell set? False
  ```

  On the real `credit-g_20nan` table the permutation means **only 944 of the 4000 cells the null path
  marks for reconstruction (23.6%) are actually missing**; the other 76% are observed cells whose true
  value sits in the model's input un-replaced. Running the full decode stage on an interleaved,
  learnable table (60 decode epochs, CPU) gives:

  ```
  impute/induced/acc_cat            = 1.0000     impute/induced/impute_score            = 0.2996
  impute/induced/null_token/acc_cat = 0.5455     impute/induced/null_token/impute_score = 1.2238
  cells scored by both populations: 3 of 23
  ```

  The existing guard, `test_the_null_path_diagnostic_asks_the_same_cells_through_the_other_token`, asserts
  `int(through_null.sum()) == int(through_mask.sum())` — a **count**, which is invariant under exactly the
  permutation that is wrong. And `_complete_frame()` in that test is ordered `colour, shape, size, weight`:
  categorical first, so the permutation is the identity and the bug is unreachable from the fixture.
- **Consequence:** `--score_null_path` on `credit-g_*nan`, `electricity_*nan` or any future mixed table
  answers a question nobody asked. The flag exists to decide whether the decode head transfers to the
  `[NULL]` token (wayfinder issue 12); the number it produces instead grades the head on mostly-visible
  cells at rows chosen by an unrelated column's missingness. The distortion has no sign — it can flatter or
  punish — which is worse than a biased estimate, because there is no direction to correct for. The metric
  is diagnostic-only and never ranks a fold, which is the only reason this is high and not critical.
- **Direction:** build the selection in embedder order (`test_frame[list(cat) + list(num)].isna()`), or
  better, have the embedder expose the order so no caller has to know it. Then change the test's assertion
  from a count to set equality of `(row, column)` pairs, and re-order at least one fabricated frame in
  `test_training_decoding.py` so a categorical column does not come first. A single frame ordered
  `num, cat, num, cat` in the shared `_complete_frame()` helper would have caught this on the day it landed.

### F-09-2 - No test pins `realised_rate`, and the metric it would have pinned shows `EVAL_MASK_RATE` is nearly inert above 40% missingness

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/decoding.py:154-157`; no assertion in any test file, not pinned in the fixture
- **Evidence:** `impute/masked/realised_rate` is a published metric documented in `README.md:272`, and
  nothing in `tests/` mentions it (`grep -rn realised tests/` → no hits). Driving the real
  `evaluation_mask` over the real credit-g variants at the default nominal rate 0.2:

  ```
  credit-g_00nan  realised=0.2044   credit-g_20nan  realised=0.1708   credit-g_40nan  realised=0.1456
  credit-g_60nan  realised=0.1531   credit-g_80nan  realised=0.2512
  ```

  The realised rate is **not monotone** — it bottoms out at 40% missingness and then climbs back above the
  complete table's value. Sweeping the nominal rate on `credit-g_80nan`:

  ```
  nominal 0.05 -> realised 0.2480     nominal 0.10 -> realised 0.2482
  nominal 0.20 -> realised 0.2512     nominal 0.50 -> realised 0.2795
  ```

  A 10x change in `EVAL_MASK_RATE` moves the actual evaluation difficulty by 1.13x. The cause is in
  `src/utils.py:62-73`: `preprocess_table` forces at least one masked cell per row, drawn from that row's
  non-null indices. At 80% missingness a credit-g row has 4 observed cells, so the guarantee alone floors
  the realised rate near 1/4 and the probabilistic draw becomes noise on top of it.
- **Consequence:** three things a reader would get wrong. (1) `README.md:272` says asking for 0.2 hides
  "about 5% of an `_80nan` one. Each run logs the realised share as `impute/masked/realised_rate`" — but the
  logged number is 0.2512, not 0.05; the 5% figure is share-of-all-cells and the sentence attaches it to the
  wrong metric, a 5x discrepancy in the only place the code's own honesty check is documented.
  (2) `EVAL_MASK_RATES_EXTRA` — the whole multi-rate diagnostic of wayfinder issue 11 — produces three
  indistinguishable difficulties on a 60/80nan variant, so a table of `rate_10 / rate_20 / rate_50` scores
  there is three measurements of the same thing. (3) Comparing `impute_score` across missingness levels is
  confounded: the 40nan and 80nan runs are scored at *harder* realised rates than the trend suggests.
- **Direction:** assert `realised_rate` at two missingness levels in a test that reads the real variants
  (this is cheap — `evaluation_mask` needs no model, so it is a unit test, not an integration one), so the
  number is pinned and the non-monotonicity is on the record. Correct `README.md:272` regardless. Then
  decide separately whether the row guarantee belongs in evaluation masking at all — it exists so
  pre-training batches always carry a loss, which an evaluation mask does not need — but that is a
  behaviour change: it moves every masked-population number and requires regenerating
  `credit-g_20nan_imputation_regression.json`, so under AGENTS.md it needs a documented decision, not a
  quiet fix.

### F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder whose outputs are unrelated to the truth passes every test including the fixture

- **Kind:** methodology
- **Severity:** high
- **Where:** the whole decode suite; `tests/fixtures/credit-g_20nan_imputation_regression.json:29-54`
- **Evidence:** Every pinned fixture number is at or beyond parity with mean-and-mode imputation:
  `impute/masked/impute_score` 1.397 and 1.402, `impute/induced/impute_score` 1.374 and 1.306 — all above
  1.0, i.e. *worse than filling the column mean*. The pinned `rmse_num_z` values are 1.0563 and 0.9997;
  since the columns are z-scored, a decoder that emits the constant 0 scores `sqrt(mean(z^2)) ≈ 1.0` — so
  fold 1's pinned value sits 0.06 from the null predictor and fold 2's sits 0.0003 from it. The
  fixture is therefore pinning arithmetic a null predictor very nearly reproduces. Its
  docstring is honest about this ("the fixture pins determinism, not quality") — the problem is that no
  *other* test takes up the slack. The unit suite checks counts (`len(listed) == hidden_here`), key
  presence, loss-curve length, checkpoint selection, and — genuinely well — that a masked numerical cell's
  `actual_original` is the exact pre-scaling number. None of them constrains the relationship between
  `imputed` and `actual`.

  Such a test is cheap. On a synthetic table where `echo` duplicates `colour` and `y = 3x`, the real
  `train_and_evaluate_decoder` at 60 decode epochs on CPU (a few seconds) reaches:

  ```
  impute_score = 0.2971   rmse_num_z = 0.4593   acc_cat = 1.0000
  ```

  well clear of the 1.0 parity line. Two runs of that identical script gave 0.2971 and 0.3043, which is a
  warning attached to the recommendation, not noise I measured across seeds: `preprocess_table` draws from
  the *global* numpy stream, which my snippet never seeded (its `default_rng(0)` is a separate generator).
  Any such test must call `np.random.seed(...)` — or accept the decode stage's own `set_global_seed` — or it
  will be flaky in exactly the way this project cannot afford.
- **Consequence:** the class of bug this misses is exactly the class this branch is most exposed to:
  a prediction tensor paired with the wrong truth tensor, a head wired to the wrong column token, a
  selection mask off by one. Every one of those leaves `rmse_num_z ≈ 1.0` and `acc_cat ≈` the majority-class
  rate — which is what the fixture already pins — and every listed count, key and loss assertion still
  holds. F-09-1 is a live instance: it survived the suite for exactly this reason.
- **Direction:** one unit test on a learnable synthetic table asserting `impute/masked/impute_score < 0.6`
  for both kinds, plus the same for the `induced` population against a sibling. Put a floor, not an exact
  value, so it is not a second regression fixture. If the run cost is a concern it belongs in the
  integration marker, but 60 epochs on 160 rows does not need one.

### F-09-4 - The fixture's own justification for choosing `credit-g_20nan` is false, and the dataset it wrongly excluded is the one that would have caught a shipped crash

- **Kind:** methodology
- **Severity:** medium
- **Where:** `tests/integration/test_credit_g_imputation_regression.py:3-6`, `AGENTS.md`
- **Evidence:** the claim is "`credit-g_20nan` is the only variant that exercises both column types and
  both scored populations". Reading the declarations directly:

  ```
  credit-g:  13 categorical / 7 numerical  (21 columns)
  electricity: 1 categorical ('day') / 7 numerical  (9 columns)
  kr-vs-kp:  36 categorical / 0 numerical  (37 columns)
  (biodeg, kc2, letter, pendigits, spambase, vehicle: no categorical declaration at all)
  ```

  `electricity_20nan` exists, has both column types and a `_00nan` sibling, and therefore exercises both
  populations. The claim is simply wrong. It matters because the prior review's finding 1 — the induced
  population crashing on `electricity` through the integer-coded `day` column — is precisely what a second
  fixture on `electricity_20nan` would have turned into a red test. The false uniqueness claim is the
  reason no one looked.

  The coverage gaps this claim papers over: 6 of the 9 datasets are all-numerical and **none** goes through
  the decode stage in any test; `kr-vs-kp` is the only all-categorical table and is likewise untouched.
  I checked both shapes survive by driving `train_and_evaluate_decoder` directly (see Checked and cleared),
  so this is a coverage gap rather than a second crash — but it is a gap covering two thirds of the corpus.
- **Consequence:** a contributor reads the docstring and `AGENTS.md`, concludes one fixture suffices, and
  ships a change that only breaks a single-categorical-column or all-numerical table. That is exactly what
  happened once already.
- **Direction:** correct the claim to what it is — credit-g is the *richest* mixed variant, not the only
  one. Add a second, much cheaper integration fixture on `electricity_20nan` (1 categorical column, the
  integer-coded case) or at minimum a unit test that runs the decode stage over a frame whose single
  categorical column is integer-coded and whose sibling reads back at a different dtype.

### F-09-5 - The fixture pins six pooled averages and no structural invariant, so the cheapest and most decisive checks are absent

- **Kind:** methodology
- **Severity:** medium
- **Where:** `tests/integration/test_credit_g_imputation_regression.py:22-29`
- **Evidence:** `PINNED` is six pooled means. Deterministic quantities the same run already computes and
  the fixture ignores: `impute/masked/n_num_cells`, `impute/masked/n_cat_cells`,
  `impute/induced/n_num_cells`, `impute/induced/n_cat_cells`, `impute/masked/realised_rate`,
  `impute/masked/macro_f1_cat`, `impute/masked/mae_num_z`. The cell counts are integers fixed entirely by
  the seed, the fold split and the mask draw — they either match exactly or the evaluation population moved.
  `macro_f1_cat` is a metric this branch introduced specifically because accuracy flatters a
  majority-class predictor (`imputation_metrics.py:88-90`); it has one unit test and no end-to-end pin at
  all. The run also writes a preview, a cell ledger and `metrics/per_column_imputation.csv`; the integration
  test opens none of them, so the artifact path is only ever exercised on hand-built two-row frames in
  `test_imputation_artifacts.py`.
- **Consequence:** a change to `evaluation_mask`'s seed derivation, to the fold construction, or to which
  population a cell lands in shifts pooled averages by an amount that may or may not clear the 0.01
  tolerance. For `acc_cat` the fold scores roughly 880 categorical cells (500 test rows x 13 categorical
  columns x 0.8 observed x 0.17 realised), so a completely different mask draw moves it by about 0.016 on
  average — above the tolerance, but only just, and "only just" is doing work that an integer equality would
  do for free. A silently empty or doubled population is caught only if it happens to move a mean.
  Three further shapes the single fixture never reaches, all cheap to state and none of them tested
  anywhere else:
  - **Row count.** `credit-g` is 1000 rows. `electricity` is 45312, `letter` 20000, `spambase` 4601. The
    decode stage's un-batched full-split forward passes (prior review's finding 2) are only ever exercised
    at 500 test rows, which is why nothing has ever run out of memory in a test.
  - **`cv_folds=None`.** The predefined-split path under `--task imputation` is untested everywhere: the
    integration fixture uses `cv_folds=2`, and `runner.py:126-139` writes `per_column[ordinal]` keyed on
    an ordinal that the fold result records as `"single_split"`, then takes the
    `log_single_split_record` branch that no imputation test visits.
  - **Missingness pattern.** Every variant removes a uniform ~20% from *every* column, MCAR. Nothing
    exercises a column-concentrated pattern, an all-missing column in a train fold (the shape prior
    review's finding 3 is about), or a column with no gaps at all. The uniformity is also why a per-kind
    cell-count check would not have caught F-09-1: with every column equally missing, the permuted
    selection happens to preserve the categorical/numerical split exactly (2600/1400 either way).
- **Direction:** add the four `n_*_cells` counts and `realised_rate` to `PINNED` with `abs=0` for the
  counts, and add `macro_f1_cat`. Regenerating the fixture costs nothing extra since the run already
  produces them.

### F-09-6 - `test_imputation_summary_carries_the_decode_stage_timing` fabricates an event the runner never emits, certifying a path no run can take

- **Kind:** design
- **Severity:** medium
- **Where:** `tests/unit/test_training_tasks.py:100-115`
- **Evidence:** the test hand-builds `LoggedMetric("time/decode_seconds", 4.0, None)` and asserts the
  summariser surfaces it. But `stage_timing_metrics` (`src/training/summary.py:40-46`) emits exactly three
  keys — `time/pretrain_seconds`, `time/finetune_seconds`, `time/total_seconds` — unconditionally, for both
  tasks, and `runner.py:106-108` is its only caller. No imputation run has ever produced a
  `time/decode_seconds` event. The test name asserts a property of a *run* ("carries the decode stage
  timing") that is false; what it actually verifies is that `TIMING_METRIC_KEYS` contains a string.
- **Consequence:** this is the specific mechanism by which the decode-timing defect (prior review's finding
  9 — decode wall-clock filed under `time/finetune_seconds`) shipped with a green suite and a test whose
  name says it cannot have. A reader auditing "is imputation timing handled?" finds this test and stops.
  Tests that construct their input from the shape the producer *should* have, rather than from the producer,
  convert a missing feature into apparent coverage.
- **Direction:** build the record from `stage_timing_metrics(...)` itself rather than from literal
  `LoggedMetric`s, so the test fails until the producer is task-aware. The neighbouring
  `test_imputation_summary_bands_the_decode_stage_losses_instead_of_finetuning` also fabricates its events,
  but the key it fabricates (`decode/train_loss`) *is* one the decode stage really emits
  (`decoding.py:117-118`), so it is fabricating a true shape; only the timing test fabricates a key no
  producer ever writes. That is the whole distinction, and it is why this finding is about one test.

### F-09-7 - The search objective is a string literal hardcoded twice and connected nowhere, so a renamed metric family ends a study as "every trial pruned"

- **Kind:** design
- **Severity:** medium
- **Where:** `tests/unit/test_training_tasks.py:30` and `tests/unit/test_training_decoding.py:133`;
  consumer at `opt.py:266`
- **Evidence:** `opt.py:405` sets `objective.search_objective = task.search_objective` and `opt.py:266`
  does `score = metrics[self.search_objective]` — a plain lookup into the run's mean metrics. The producer
  is `decoding.py:207-210`, which builds the key as an f-string,
  `f"validation/impute/masked/{name}"`. Two tests name that key, and both do it by typing the literal:

  ```python
  # test_training_tasks.py:30
  assert task_spec("imputation").search_objective == "validation/impute/masked/impute_score"
  # test_training_decoding.py:133
  key = "validation/impute/masked/impute_score"
  ```

  They agree with each other because someone copied the string, not because anything checks that the
  stage's output key *is* the spec's objective. Nothing in the suite asserts
  `task_spec("imputation").search_objective in outcome.result.metrics`.
- **Consequence:** rename the metric family in `decoding.py` (say `validation/impute/masked/` to
  `search/impute/masked/`) and update `_TASK_SPECS` without updating it, or the reverse, and both tests
  still pass. The failure surfaces at `opt.py:266` as a `KeyError` — *after* a full trial has trained — and
  `opt.py:292` converts a failed trial into `optuna.TrialPruned`. The study then prunes every trial and dies
  in `study.best_trial`, having burned the whole budget and written nothing. The symptom a user sees is
  "the search produced no result", with no mention of a metric name anywhere.
- **Direction:** replace the literal in `test_training_decoding.py` with
  `key = task_spec("imputation").search_objective` and assert it is present in
  `outcome.result.metrics`. That single change makes the producer and the contract fail together and makes
  `test_each_task_declares_its_search_objective` redundant, which is the right way to retire it.

### F-09-8 - The "no validation key reaches the test family" guarantee is tested at the fold tracker only; the cross-validation summariser is uncovered and does pass one through

- **Kind:** design
- **Severity:** low
- **Where:** `src/training/decoding.py:198-220` (the claim), `tests/unit/test_training_decoding.py:122-144`
  (the guard), `src/training/summary.py:61-66` and `src/training/tracking.py:393` (the uncovered path)
- **Evidence:** the decode stage's comment promises "no ordinary or cross-validation run ever carries a
  `validation/` key and the cv/test summariser never sees one", and the test asserts
  `not any(event.key.startswith("test/validation/") for event in tracker_asked.metric_events)`. That guards
  the per-fold tracker. But `DecodingOutcome.result.metrics` is `{**metrics, **validation_metrics}`
  (`decoding.py:219`), `summarize_cross_validation` summarises *every* key in `record.result.metrics`
  (`summary.py:61-66`), and the tracker logs each as `cv/test/{name}` (`tracking.py:393`). Optuna sets
  `args.score_search_objective = True` for every imputation trial (`opt.py:221`) and leaves MLflow enabled
  by default (`opt.py:209`), and trials run with `cv_folds`. So each imputation trial run logs
  `cv/test/validation/impute/masked/impute_score`. No test covers the summariser with a
  search-objective-bearing record.
- **Consequence:** in MLflow, imputation trial runs carry a validation number under the `cv/test/` prefix.
  The blast radius is small and I want to be accurate about it: trials are tagged
  `mlflow_run_role = "optuna_trial"` (`opt.py:228`), and the project's comparison rule is
  `tags.run_role = parent` (CLAUDE.md, ADR 0002), so these runs are already filtered out of every
  cross-run comparison. The docstring's "only when asked" is honest about the gating; what is not true is
  the parenthetical that follows it, that the cv/test summariser never sees one. The finding that matters is
  the shape, not the damage: the branch's strongest methodological commitment — the test split never
  chooses hyperparameters — is asserted at the fold-tracker layer and unasserted at the layer above, and the
  one place it leaks is the layer nobody tested.
- **Direction:** extend the guard to `summarize_cross_validation`: feed it a record whose metrics include a
  `validation/` key and assert the summariser either excludes it or files it under its own prefix. Then make
  the code match whichever the test says.

### F-09-9 - `score_cells` has no baseline-coverage contract, and its two kinds fail in opposite directions when one is missing

- **Kind:** bug
- **Severity:** low
- **Where:** `src/training/imputation_metrics.py:48-56`; no test
- **Evidence:** driving the real function with a baseline mapping that omits one scored column:

  ```
  numerical   ({'amount': 0.0}, cells for amount and fee):  impute_score = nan
  categorical ({'grade': 'a'}, cells for grade and tier):   impute_score = 1.0
  ```

  `column.map(baselines)` yields NaN for the absent column. On the numerical side NaN propagates into
  `naive_rmse` and the whole score becomes NaN. On the categorical side `NaN != 'x'` is `True`, so the
  missing baseline is silently counted as a naive *error*, inflating the denominator and making the model
  look better than it is. No test exercises either.
- **Consequence:** today this is reachable only through prior finding 3 (an all-missing column in a train
  fold), where the numerical branch yields NaN — and a NaN `impute/masked/impute_score` flows into
  `_diagnostic_roles`, whose min/max over NaN picks an arbitrary best fold. The categorical direction is
  latent. The point for this dimension is that neither behaviour is anyone's decision: there is no test
  saying which is intended.
- **Direction:** make `score_cells` refuse a scored column it has no baseline for, and pin that with a
  two-line test. A raised error at the boundary is strictly better than either a NaN ranking metric or a
  quietly inflated one.

### F-09-10 - Tests that restate the implementation and could go

- **Kind:** design
- **Severity:** low
- **Where:** `tests/unit/test_decoder_model.py:108-122`, `tests/unit/test_training_decoding.py:112-119`,
  `tests/unit/test_training_tasks.py:48-57`
- **Evidence:**
  - `test_a_numerical_cell_is_imputed_with_one_number_per_column` zeroes every numerical head and asserts
    the output is all zeros. That is an assertion that `nn.Linear` with zero weight and zero bias returns
    zero. `test_columns_never_share_head_parameters` (lines 125-140) does the same zeroing and additionally
    checks the *other* column is untouched, which is the risk worth guarding; it strictly subsumes this one.
  - `test_the_decode_stage_reports_a_loss_for_every_epoch_it_trained` asserts
    `len(outcome.train_losses) == 3` against a loop that appends once per epoch. The tracker-key half of the
    same test is worth keeping; the length half restates `for epoch in range(decode_epochs)`.
  - `test_classification_ranks_folds_by_macro_f1_maximised` and
    `test_imputation_ranks_folds_by_impute_score_minimised` read two tuples back out of `_TASK_SPECS`.
    The consumer risk is covered by `test_imputation_summary_marks_the_lowest_score_as_the_best_fold` and
    `test_imputation_summary_rejects_folds_missing_the_impute_score`, which exercise the same constants
    through the summariser. `test_unknown_task_name_is_rejected` is not in this group — it guards real
    behaviour.
- **Consequence:** none directly; the cost is that a suite where a third of the task-contract file restates
  constants trains a reader to skim, which is part of why F-09-1's vacuous count assertion and F-09-6's
  fabricated event read as coverage.
- **Direction:** delete `test_a_numerical_cell_is_imputed_with_one_number_per_column`; drop the length
  assertions from the loss-curve test and keep the tracker-key one; collapse the two ranking-literal tests
  into the summariser assertions that already exercise them.

## Checked and cleared

- **All-numerical and all-categorical tables survive the decode stage end to end.** I feared
  `TridentDecoder.predict` would fail stacking an empty head list for the 6 all-numerical datasets.
  `_stack(…, rows, dtype, device)` handles the empty case. Driving the real `train_and_evaluate_decoder`
  with a sibling on a 3-numerical/0-categorical table and a 0-numerical/3-categorical table:
  both complete and emit the right key sets (`rmse_num_z`/`mae_num_z` only, and `acc_cat`/`macro_f1_cat`
  only, respectively), with `impute_score` present in both. This is untested territory covering two thirds
  of the corpus (F-09-4), but it is not broken.
- **The masked population's `actual` is never a fabricated value.** I suspected
  `evaluation_mask` could hide an already-missing cell, whose clean encoding would give
  `num_values = 0.0` and score the model against a truth that does not exist. `preprocess_table`
  (`src/utils.py:62-73`) sets `dynamic_mask[null_values] = False` *and* draws the per-row guarantee from
  `np.where(~null_values[i])[0]` only, so a `[MASK]` never lands on a gap.
  `tests/unit/test_preprocess_table.py:59` pins the second half of that.
- **The masked path's own column indexing is correct.** Only `_select` in the null path is wrong.
  `_score_population` and `TridentDecoder.forward`/`predict` all index `masked_positions` and the
  transformer context in embedder order, which is the order `encode` builds.
- **`test_the_truth_beside_each_guess_is_the_number_the_dataset_actually_holds` is real coverage, not
  ceremony.** Its expectations are computed independently in the test (`dataset.raw_numerical.iloc[...]`,
  `_complete_frame().iloc[...]`), so a swapped clean-truth encoding or a wrong `fold.test_indices` slice
  would fail its final `rescaled ≈ actual_original` assertion. It is the one test in the decode suite that
  constrains a value rather than a count — which is why F-09-3 matters only for the categorical half and
  the induced population.
- **`evaluation_mask`'s isolation of the global RNG is properly tested.**
  `test_drawing_an_evaluation_mask_leaves_the_training_draws_alone` restores and compares the global stream
  directly; this is the assertion that protects every published classification number from the new task, and
  it is written correctly.
- **The imputation metric unit tests are well-chosen.** `test_the_baseline_comes_from_training_not_from_the_cells_being_scored`
  constructs a case where reading the baseline off the scored cells would give exactly parity while the
  correct baseline gives 0.5 — a real leak, caught by a number that could not arise by accident.
  `test_a_column_the_naive_imputer_never_gets_wrong_still_ranks` pins both sides of the divide-by-zero
  guard. I have no criticism of that file beyond F-09-9.

## Open questions

- **Does the credit-g fixture reproduce on CPU?** `environment.device` is `"cuda"` and `runs: 3`;
  `observed_max_deviation` is 0.0, but nothing records whether any of the three was a CPU run. The vehicle
  fixture's test docstring says "CPU/CUDA"; the imputation one says only "deterministic". Settled by running
  the integration marker once on each device and recording the device list in the fixture.
- **What does F-09-1 do to a fully-trained credit-g decoder?** My interleaved demonstration used 60 epochs
  on 160 synthetic rows and the null-path number came out *lower* than the true one. A decoder trained long
  enough to copy visible values would likely push it the other way. Settled by one
  `--task imputation --score_null_path` run on `credit-g_20nan` at the production epoch budget, comparing
  `impute/induced/acc_cat` against `impute/induced/null_token/acc_cat` before and after the fix.
- **Has `--score_null_path` been used for any recorded result?** If a wayfinder note or an MLflow run
  concluded anything about `[NULL]` transfer from a mixed dataset, that conclusion needs withdrawing.
  Settled by `mlflow search` over runs carrying an `impute/induced/null_token/*` metric.
