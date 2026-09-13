# 09 - Test adequacy for the imputation task: verdicts

_Adversarial verification of `09-test-adequacy.md`. 2026-09-11._

**Method:** Orientation with
`graphify query "how does the decode stage select which cells to score in the null path diagnostic"`
(208 nodes, truncated; it named `_select()` at `decoding.py:268` and `_score_induced_missing()` at
`:227`, which is where I went next). Every verdict below rests on the source, not the graph:
`src/training/decoding.py` (all 346 lines), `src/training/imputation_metrics.py` (whole file),
`src/training/summary.py` (whole file), `src/training/data.py:21-50, 132-172`,
`src/training/runner.py:22-192`, `src/training/artifacts.py:79-100`, `src/training/tracking.py:378-400`,
`src/embedder.py:160-200`, `src/utils.py:40-82`, `opt.py:200-310, 400-412, 500-515`,
`tests/unit/test_training_decoding.py` (all 288 lines), `tests/unit/test_training_tasks.py` (all 154),
`tests/unit/test_decoder_model.py:100-145`, `tests/unit/test_training_runtime.py:360-560`,
`tests/integration/test_credit_g_imputation_regression.py`,
`tests/fixtures/credit-g_20nan_imputation_regression.json`, `README.md:262-278`, and the already-written
`01-decode-stage.verdict.md`, `03-imputation-metrics.verdict.md`, `05-embedder-tokens.verdict.md`.

Six `uv run --python 3.10 python` probes against the **real** functions, scripts under
`C:\Users\DIOGON~1\AppData\Local\Temp\claude\C--Users-Diogo-Neiss-Documents-Mestrado-TRIDENT\f48d5c06-b3ce-4977-b950-7a95c1d2053f\scratchpad\verify09\`:

```
$ uv run --python 3.10 python .../f1.py          # real _score_induced_missing, interleaved frame
frame order   : ['size', 'colour', 'weight', 'shape']
embedder order: ['colour', 'shape', 'size', 'weight']
test_frame gaps: [(0, 'colour')]
MASK path cells: [(0, 'colour')]
NULL path cells: [(0, 'shape')]
counts equal? True  same set? False

$ uv run --python 3.10 python .../f1b.py         # real declared_column_types on credit-g_20nan
n_cat 13 n_num 7
cells the null path marks: 4000; actually missing in the column it is read as: 944 (23.6%)

$ uv run --python 3.10 python .../f2.py          # real evaluation_mask, real credit-g ladder, nominal 0.2
00nan: realised=0.2044 share_of_all=0.2044      20nan: realised=0.1708 share_of_all=0.1366
40nan: realised=0.1456 share_of_all=0.0873      60nan: realised=0.1531 share_of_all=0.0612
80nan: realised=0.2512 share_of_all=0.0503
--- nominal sweep on credit-g_80nan ---
0.05 -> 0.2480   0.10 -> 0.2482   0.20 -> 0.2512   0.50 -> 0.2795

$ uv run --python 3.10 python .../f2b.py         # the same sweep on the 8-column electricity
electricity 80nan 0.05->0.5189 0.1->0.5205 0.2->0.5237 0.5->0.5468
electricity 60nan 0.05->0.3092 0.1->0.3123 0.2->0.3241 0.5->0.3943
credit-g    60nan 0.05->0.1274 0.1->0.1328 0.2->0.1531 0.5->0.2455

$ uv run --python 3.10 python .../f4.py .../f4b.py   # real TabularEmbedder on electricity
20nan day dtype float64 ('2.0')   00nan day dtype int64 ('2')
variant vocabulary for 'day': ['2.0','3.0','4.0','5.0','6.0','[MASK]','[NULL]','nan']
sibling encode RAISED: ValueError y contains previously unseen labels: '2'

$ uv run --python 3.10 python .../f9.py .../f9b.py   # real score_cells with an incomplete baseline
numerical, baseline missing 'fee':  impute_score = nan
categorical, full baseline       :  impute_score = 1.0
categorical, 'tier' omitted      :  impute_score = 0.5
```

**No training run was spent.** Nothing above needed one: `_score_population` emits one row per `True` in
`masked_positions`, so the in-process probes are the same numbers a run would print, and the fixture's
2-epoch configuration is not reachable through the mandated `main.py --cv_folds 2` form anyway. No pytest.
Nothing outside this file was written; no git state changed.

## Verdicts

### F-09-1 - The `[NULL]`-path diagnostic builds its cell selection in frame order and it is read in embedder order, and the only guard asserts a count

- **Verdict:** CONFIRMED
- **Severity after review:** high — **but this is the same defect as F-05-1, already CONFIRMED at high
  in `05-embedder-tokens.verdict.md`. It must be carried once, not twice.**
- **Basis:** reproduced end to end. `embedder.py:189` builds `masked_positions` as
  `columns = list(self.categorical_columns) + list(self.numerical_columns)`; `decoding.py:262` overwrites it
  with `torch.tensor(test_frame.isna().to_numpy(), device=device)`, which is `features.columns` order.
  Driving the real `_score_induced_missing` on a frame ordered `size, colour, weight, shape` with one gap at
  `(0, colour)`: the MASK path scores `(0, colour)`, the NULL path scores `(0, shape)` — counts equal, sets
  different (probe `f1.py` above). On the real `credit-g_20nan` header the permutation leaves
  **944 of the 4000 marked cells (23.6%)** actually missing in the column the selection is read as
  (probe `f1b.py`); the critic's figure reproduces to the digit. The guard,
  `test_the_null_path_diagnostic_asks_the_same_cells_through_the_other_token`
  (`test_training_decoding.py:233-245`), ends at `assert int(through_null.sum()) == int(through_mask.sum())`,
  and `_complete_frame()` (`:24-36`) is ordered `colour, shape, size, weight`, so the permutation is the
  identity in the fixture and the assertion is invariant under the exact error.
- **Correction:** none to the mechanism. Two notes for the report. First, the **novel content of this
  finding in the test-adequacy dimension** is not the bug — it is the two test-design facts: a
  permutation-invariant `int(sum) == int(sum)` assertion where set equality was available for free, and a
  shared fixture frame that is categorical-first, which is the one ordering no real mixed dataset has
  (`credit-g` and `electricity` both interleave). That is what should survive here; the defect itself
  belongs to F-05-1. Second, the critic's "impute/induced/acc_cat 1.0000 vs null_token 0.5455" figure came
  from its own 60-epoch toy and is not a statement about any shipped run — the direction of the distortion
  is genuinely unsigned, as the critic itself says.

### F-09-2 - No test pins `realised_rate`, and the number it would pin shows `EVAL_MASK_RATE` is nearly inert at high missingness

- **Verdict:** SOUND
- **Severity after review:** low (critic said high)
- **Basis:** every number reproduces. `grep -rn realised tests/` returns nothing — `realised_rate` is
  published (`decoding.py:154-157`), documented (`README.md:272`) and unasserted anywhere. The real
  `evaluation_mask` on the real credit-g ladder at nominal 0.2 gives 0.2044 / 0.1708 / 0.1456 / 0.1531 /
  0.2512, and the nominal sweep on `credit-g_80nan` gives 0.2480 → 0.2795 for a 10x change in the knob
  (1.13x). The `README.md:272` mismatch is real and is the sharpest half of this finding: the sentence
  *"asking for `0.2` hides about 20% of a `_00nan` variant but about 5% of an `_80nan` one. Each run logs
  the realised share as `impute/masked/realised_rate`"* is correct in the **all-cells** denominator
  (`share_of_all` = 0.2044 and 0.0503 above) and then names a metric computed in the **eligible-cells**
  denominator, where the same run reads 0.2512. Two denominators, one sentence.
- **Correction:** the severity drops three bands, for three reasons.

  1. **The measurement is a third telling of an already-adjudicated finding.** F-01-1 (SOUND, high) and
     F-03-1 (SOUND, medium) both establish that the realised rate is neither the configured rate nor
     comparable across the ladder, with the same probe on the same tables. The only content new to this
     dimension is *no test pins it*, plus the README line. A missing pin on a documented metric and a wrong
     doc sentence is low.
  2. **"Non-monotone in missingness" is credit-g's 20-column shape, not a property of the mechanism** —
     exactly the correction F-01-1 already made. On the 8-column `electricity` the realised rate rises
     monotonically (0.2230 → 0.5242). My own sweep (`f2b.py`) confirms the *inertness* generalises at high
     missingness (electricity_80nan: 0.5189 → 0.5468 for the same 10x nominal change) but the
     non-monotonicity does not.
  3. Consequence (2) is overstated in one direction and understated in another: three nominal rates do
     collapse to nearly one difficulty at `_80nan`, but at `credit-g_60nan` the same sweep spans
     0.1274 → 0.2455, a near-doubling. "Three measurements of the same thing" holds at 80nan, not
     "above 40%".

  The recommended fix — a unit test pinning `realised_rate` on two real variants, since `evaluation_mask`
  needs no model — is right and costs nothing, and the README line should be corrected regardless.

### F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder wired to the wrong truth passes the whole suite

- **Verdict:** SOUND
- **Severity after review:** medium (critic said high)
- **Basis:** premise verified by exhaustion. Grepping every quality-bearing name in the decode suite
  (`impute_score`, `acc_cat`, `rmse_num_z` in `test_training_decoding.py` and the integration test) returns
  only `in metrics` membership assertions, one `!=` between two splits, and the fixture's six
  `pytest.approx` pins. Not one inequality against a baseline anywhere. The fixture's pins are all on the
  wrong side of parity — `impute/masked/impute_score` 1.397 / 1.402 and `impute/induced/impute_score`
  1.374 / 1.306, and `impute_score > 1` means *worse than mean-and-mode* by construction
  (`imputation_metrics.py:52-56`). The `rmse_num_z` arithmetic holds: the columns are z-scored, so a
  constant-0 predictor scores `sqrt(mean(z^2)) ≈ 1`, and the pinned 1.0563 / 0.9997 sit 0.06 and 0.0003
  from it. The one test that constrains a *value* rather than a count —
  `test_the_truth_beside_each_guess_is_the_number_the_dataset_actually_holds` — constrains
  `actual_original` against the raw frame, never `imputed` against `actual`. The critic is right that
  F-09-1/F-05-1 is a live instance of precisely the bug class this leaves open.
- **Correction:** one framing correction and one severity cut.

  The fixture pins *regressions from a recorded run*; it never claimed to certify that the recorded run was
  right. So the exposure is to **day-one wiring errors**, not to defects introduced later — a later change
  that breaks the prediction/truth pairing *would* move the pinned means. That is a narrower claim than
  "a decoder whose predictions are unrelated to the truth passes everything", which is true only of a
  decoder that was already unrelated when the fixture was generated. Narrower, but not empty: the branch is
  new, the fixture was generated from it, and one day-one wiring error has already been found.

  Medium rather than high because a coverage gap with one demonstrated victim, whose fix the critic itself
  prices at 60 CPU epochs on 160 rows, is not in the same band as a wrong published number. The warning
  attached to the recommendation is the most useful thing in this finding and should be kept verbatim:
  `preprocess_table` draws from the **global** numpy stream (`utils.py:58, 71`), so such a test must seed it
  or it will be flaky.

### F-09-4 - The fixture's stated reason for choosing `credit-g_20nan` is false, and the variant it wrongly excluded is the one whose induced path crashes

- **Verdict:** SOUND
- **Severity after review:** medium (critic said medium — kept)
- **Basis:** both halves verified directly.

  `datasets/categorical_columns/` holds exactly three files: `credit-g.txt` (13 columns),
  `electricity.txt` (`day`), `kr-vs-kp.txt` (36 columns). `electricity_20nan.csv` exists with an
  `electricity_00nan.csv` sibling, so it has both column kinds and both scored populations. The docstring at
  `test_credit_g_imputation_regression.py:3-5` — "the only variant that exercises both column types and both
  scored populations" — is therefore false as written, and `AGENTS.md` repeats it.

  The consequence is stronger than the critic could show. I drove the real `TabularEmbedder` over
  `electricity_20nan` and its sibling:

  ```
  20nan day dtype float64 -> as_category_strings gives '2.0'
  00nan day dtype int64   -> as_category_strings gives '2'
  variant vocabulary for 'day': ['2.0','3.0','4.0','5.0','6.0','[MASK]','[NULL]','nan']
  sibling encode RAISED: ValueError y contains previously unseen labels: '2'
  ```

  `_score_induced_missing` calls `embedder.encode(_clean(truth), device)` on exactly that sibling
  (`decoding.py:265`), so `--task imputation` on `electricity_20nan` raises before it scores a single
  induced cell. A second integration fixture on `electricity_20nan` would have been red on the day the
  branch landed.
- **Correction:** none material. Two scope notes: six of the nine datasets carry no categorical declaration
  and `kr-vs-kp` carries no numerical one, so two thirds of the corpus never reaches the decode stage in any
  test — the critic verified both shapes *work* in-process, so this is coverage, not a second crash. And the
  crash itself is filed in `04-ground-truth-protocol.md`; what belongs here is that a false uniqueness claim
  in a docstring and in `AGENTS.md` is the reason nobody added the fixture that would have caught it.

### F-09-5 - The fixture pins six pooled averages and no structural invariant

- **Verdict:** SOUND
- **Severity after review:** medium (critic said medium — kept, with one sub-claim corrected)
- **Basis:** `PINNED` (`test_credit_g_imputation_regression.py:22-29`) is six pooled means and nothing else.
  Every quantity the critic names is computed by the same run and discarded: `_error_metrics`
  (`imputation_metrics.py:74-103`) emits `n_num_cells`, `n_cat_cells`, `mae_num_z` and `macro_f1_cat` for
  both populations, and `decoding.py:155` emits `realised_rate`. `macro_f1_cat` — introduced on this branch
  precisely because accuracy flatters a majority predictor — has exactly one assertion in the entire suite
  (`test_imputation_metrics.py:149`) and no end-to-end pin. The integration test opens none of the preview,
  the cell ledger or `per_column_imputation.csv`. The cell counts are integers fixed by the seed, the split
  and the mask draw, so `abs=0` equality is available for free on a run that already computes them; that is
  the finding's whole case and it stands.
- **Correction:** one of the three "shapes never reached" sub-bullets is wrong.

  **`cv_folds=None` under `--task imputation` is not untested.**
  `tests/unit/test_training_runtime.py:487` runs
  `run_training(_minimal_request(tmp_path / "b", "imputation"))` with `cv_folds=None` (the default of
  `_minimal_request`, `:410-424`), and `:537` runs the same request with `plot_losses=True`. Both take the
  `else` branch at `runner.py:176-186` — `write_tracking_provenance`, `log_prepared_dataset`,
  `log_single_split_record` — with `per_column[ordinal]` populated at `:126-139`. The decoder is stubbed and
  no imputation-specific assertion is made about that branch, so the accurate statement is **unasserted**,
  not untested. "That no imputation test visits" is false.

  The other two sub-bullets stand as stated: no imputation test exceeds 1000 rows (500 test rows at
  `cv_folds=2`), and every variant is uniform per-column MCAR. The critic's own admission that a per-kind
  count check would not have caught F-09-1 (2600/1400 either way) is correct and honest, and is a reason to
  pin the counts anyway rather than a reason not to.

### F-09-6 - `test_imputation_summary_carries_the_decode_stage_timing` fabricates a metric key no producer writes

- **Verdict:** SOUND
- **Severity after review:** low (critic said medium)
- **Basis:** premise exactly true, and I checked it the way the critic did not.
  `grep -rn decode_seconds src/ opt.py main.py train.py` returns **one** hit:
  `src/training/summary.py:35`, the `TIMING_METRIC_KEYS` constant itself. No code anywhere emits it.
  `stage_timing_metrics` (`summary.py:40-46`) returns exactly `time/pretrain_seconds`,
  `time/finetune_seconds`, `time/total_seconds`, unconditionally for both tasks, and `runner.py:107` is its
  only non-test caller. The test at `test_training_tasks.py:100-115` hand-builds
  `LoggedMetric("time/decode_seconds", 4.0, None)` and asserts the summariser surfaces it, so its name
  asserts a property of an imputation *run* that no run has. The distinction the critic draws against the
  neighbouring loss-bands test is right: `decode/train_loss` really is emitted, at `decoding.py:117-118`.
- **Correction:** the causal claim is overstated, which is the severity cut. "This is the specific mechanism
  by which the decode-timing defect shipped with a green suite" does not follow — the suite would be green
  without this test too; its absence would not have failed anything. And the test does exercise a real
  behaviour of `fold_timings_for_tracking` / `_summarize_timings`: pass-through of any key in
  `TIMING_METRIC_KEYS`. What is actually wrong is narrower and still worth fixing: a constant with no
  producer, and a test name that states something false about every imputation run. No runtime consequence;
  the mis-filed decode wall-clock is a separate finding elsewhere. Low.

### F-09-7 - The search objective is a string literal typed into two test files with no producer-to-spec cross-check

- **Verdict:** SOUND
- **Severity after review:** low (critic said medium)
- **Basis:** the wiring is as described. `opt.py:405` does `objective.search_objective = task.search_objective`;
  `opt.py:266` does `score = metrics[self.search_objective]`; the producer builds the key as an f-string,
  `f"validation/impute/masked/{name}"` (`decoding.py:207-210`). `test_training_tasks.py:30` and
  `test_training_decoding.py:133` each type `"validation/impute/masked/impute_score"` as a literal, and
  nothing in the suite asserts `task_spec("imputation").search_objective in outcome.result.metrics`. The
  failure mode reproduces by reading: `opt.py:295-302` catches every exception from `train_main` and
  `raise optuna.TrialPruned() from e`, so a `KeyError` at `:266` prunes the trial *after* it trained, and
  `opt.py:502` then reads `study.best_trial`, which raises once every trial is pruned. A whole budget spent,
  nothing written, and the word "metric" appears nowhere in the symptom.
- **Correction:** "connected to neither producer nor consumer" is not accurate, and the scenario is narrower
  than stated. `test_training_decoding.py:122-144` runs the **real** `train_and_evaluate_decoder` and
  asserts `key in asked.result.metrics` — that is a genuine producer connection; what is missing is the
  cross-check from the *spec* to the producer. Walk the critic's own scenario: rename the family in
  `decoding.py` **and** update `_TASK_SPECS` and both tests fail, because both hold the old literal. The
  edit that actually slips through is a rename on one side with that side's test updated in lockstep and the
  other side left stale — real, but a specific shape of partial edit rather than "a rename in either
  direction". The recommended one-line change (`key = task_spec("imputation").search_objective`) is still
  strictly better than the literal and costs nothing. Low.

### F-09-8 - The "no `validation/` key reaches the test family" guarantee is asserted at the fold tracker only, and the cv summariser leaks one

- **Verdict:** SOUND
- **Severity after review:** low (critic said low — kept)
- **Basis:** the leak is real and traceable line by line. `decoding.py:219` returns
  `metrics={**metrics, **validation_metrics}`; `final_metrics_for_tracking` (`summary.py:76-88`) copies
  `record.result.metrics` wholesale; `_validate_final_metrics` accepts any finite numeric key
  (`summary.py:142-144`); `summarize_cross_validation` summarises every one of them (`:63-66`); and
  `_log_summary_metrics` (`tracking.py:392-394`) logs each as `cv/test/{metric_name}`. `opt.py:221` sets
  `args.score_search_objective = self.task == 'imputation'` for every imputation trial and `opt.py:209`
  leaves MLflow on, so every imputation Optuna trial run carries
  `cv/test/validation/impute/masked/impute_score`. The existing guard asserts only
  `not any(event.key.startswith("test/validation/") ...)` over fold-tracker events
  (`test_training_decoding.py:141-143`) — the layer below the one that leaks.
- **Correction:** none. The critic's own bounding of the blast radius is accurate and should be kept: trials
  carry `run_role = "optuna_trial"` (`opt.py:228`) and the project compares `tags.run_role = parent`
  (CLAUDE.md, ADR 0002), so these values never enter a comparison. The finding is that the branch's
  strongest methodological commitment is asserted one layer below where it can break. Low is right.

### F-09-9 - `score_cells` has no baseline-coverage contract, and the two kinds fail in opposite directions

- **Verdict:** CONFIRMED for the numerical direction; the categorical direction is real but **unreachable
  from the pipeline**, and the critic's number for it came from a degenerate case
- **Severity after review:** low (critic said low — kept)
- **Basis:** the mechanism is exactly as described and both directions reproduce against the real function.

  ```
  numerical   ({'amount': 0.0}, cells for amount and fee):  impute_score = nan
  categorical, full baseline    ({'grade':'a','tier':'x'}): impute_score = 1.0
  categorical, 'tier' omitted   ({'grade':'a'}):            impute_score = 0.5
  ```

  `column.map(baselines)` yields NaN for the absent column; on the numerical side it propagates through
  `naive_rmse` into the score, and on the categorical side `NaN != 'x'` is `True`, so the absent baseline is
  counted as a naive *error*, inflating the denominator — in the probe above it halves `impute_score`, i.e.
  makes the model look twice as good. Neither is covered by any test.
- **Correction:** two, neither changing the severity.

  1. **The critic's categorical figure (1.0) came from a degenerate construction.** In the case it built,
     the full-baseline run hits `_ratio`'s zero guard (`naive_error == 0 → 1.0 + model_error`) and returns
     the same 2.0 the missing-baseline run returns by arithmetic coincidence, so that probe demonstrates
     nothing. My `f9b.py` rebuilds it with the covered column's baseline *wrong*, which is where the
     inflation actually shows (1.0 → 0.5). The direction the critic claims is right; the evidence it offered
     for it was not.
  2. **A missing baseline *key* is not reachable from the pipeline.** `mean_mode_baselines`
     (`imputation_metrics.py:132-137`) builds an entry for every declared numerical and categorical column,
     and both callers (`decoding.py:139`, `runner.py:26`) pass exactly the dataset's declared lists, which
     are the same lists the embedder's column order comes from. What *is* reachable is a present key with a
     **NaN value**: an all-missing numerical column in a train fold gives `float(series.mean()) = nan`, and
     the propagation is identical — a NaN `impute/masked/impute_score` then reaches `_diagnostic_roles`
     (`summary.py:221-235`), whose `min` over NaN picks an arbitrary best fold. The categorical direction is
     not merely "latent": an all-missing categorical train column raises `IndexError` at
     `mode().iloc[0]` *before* `score_cells` is ever called, so it cannot be reached that way at all.

  The direction — make `score_cells` refuse a scored column it has no usable baseline for, and pin it —
  is right, and should say *usable* (not NaN), which is the reachable case.

### F-09-10 - Tests that restate the implementation and could go

- **Verdict:** UNSOUND
- **Severity after review:** low (critic said low)
- **Basis:** the factual claims are true. `test_a_numerical_cell_is_imputed_with_one_number_per_column`
  (`test_decoder_model.py:108-122`) zeroes every numerical head and asserts the output is
  `[[0.0]*6, [0.0]*6]`; `test_columns_never_share_head_parameters` (`:125-140`) zeroes only `size`, asserts
  `after[0] == [0.0]*6` and additionally `after[1] == before[1]`, which does subsume it and tests the risk
  that matters. `test_the_decode_stage_reports_a_loss_for_every_epoch_it_trained` does assert
  `len(train_losses) == 3` against a loop that appends once per epoch
  (`decoding.py:100, 107, 116`).
- **Correction:** the reasoning does not carry the recommendation, on two counts.

  1. **The stated consequence is "none directly"**, followed by a causal story — that a suite restating
     constants "trains a reader to skim, which is part of why F-09-1's vacuous count assertion ... read as
     coverage" — with no evidence offered and none available. F-09-1's guard is in
     `test_training_decoding.py`, not in the task-contract file this finding is about. A finding whose own
     consequence line says "none" and whose only asserted harm is speculative should not recommend deleting
     working tests.
  2. **The two ranking-tuple tests should stay.** `test_classification_ranks_folds_by_macro_f1_maximised`
     and `test_imputation_ranks_folds_by_impute_score_minimised` pin a documented ADR 0004 contract
     (`ranking_metric`, `direction`) for two lines each, and they fail on a direct edit to `_TASK_SPECS`,
     which the summariser tests do not distinguish from a summariser bug. Cheap change-detectors on a public
     constant earn their place; "reads the tuple back out of the constant that defines it" is the shape of
     every contract test.

  What survives: the subsumed zeroed-head test can go, and the length half of the loss-curve assertion is
  weak. That is a tidy-up, not a finding, and it is at the edge of what this review was told is out of
  scope.

## What this critique missed

**A cross-validation run has two metric aggregators, and the one Optuna is scored on is not the one the
tests cover.** In CV mode `run_training` does **not** return `summarize_cross_validation`'s numbers. It
returns `compute_cv_summary(frame)` (`runner.py:183-190`) over the frame `write_metrics` built from
`result.metrics` (`artifacts.py:79-89`). Both aggregators run on the same fold results, over the same keys,
and nothing compares them. Three things follow that this critique — whose subject is what the tests do and
do not constrain — should have said:

- `opt.py:266`'s `metrics[self.search_objective]` reads a pandas `.mean()` over that frame, while every
  summariser test in the suite exercises `summarize_cross_validation`'s `_summarize_metric` instead. The two
  should agree on finite numerics, and no test says so for the imputation metric set. The equivalent
  classification path is old and shared, which is why nobody looked.
- `compute_cv_summary` averages **every non-`fold`, non-`dataset` column with no key validation at all**
  (`summary.py:246-249`), where `summarize_cross_validation` validates keys and finiteness
  (`summary.py:136-145`). Since both run, a NaN fold is still caught — so the objective path is not
  unguarded; it is simply guarded by a different function than the one it reads from, and that pairing is
  untested.
- This is also the *real* extent of F-09-8. The leaked `validation/` keys do not only reach MLflow as
  `cv/test/validation/...`; `write_metrics` spreads `result.metrics` straight into the row, so they become
  **columns of `metrics.csv`** — the deterministic artifact the regression fixture's whole design is built
  around — on every imputation Optuna trial. The critic looked at `summarize_cross_validation` and
  `tracking.py` and stopped one caller short.

**Two smaller ones.** (a) The suite has no test that an imputation run's `metrics.csv` has the columns a
reader expects — the fixture reads `result.fold_results[*].metrics` in memory and never opens the file, so
the artifact that `AGENTS.md` treats as the deterministic record is unexercised in the imputation task.
(b) `test_the_decode_stage_scores_the_validation_split_only_when_asked` asserts
`asked.result.metrics[key] != asked.result.metrics["impute/masked/impute_score"]`. The two scores do differ
for real reasons — different rows, different values, different cells — so the assertion passes honestly.
But it is an inequality of floats standing in for "a different split", and the property it actually wants —
that the validation score is computed on `hidden_validation` and `clean_validation` rather than being a
relabelled test score — is available directly and is what a reader would expect the test to say.
