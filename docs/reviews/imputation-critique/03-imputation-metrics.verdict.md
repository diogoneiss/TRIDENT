# 03 - Imputation metrics and the composite score: verdicts

_Adversarial verification of `03-imputation-metrics.md`. 2026-09-11._

**Method:** read `src/training/imputation_metrics.py` (all 137 lines), `src/training/decoding.py:32-346`,
`src/training/data.py:45-215`, `src/utils.py:18-82`, `src/training/runner.py:22-155`,
`src/training/artifacts.py:224-290`, `src/training/tracking.py:37-240`, `src/training/summary.py:123-158`,
`src/training/types.py:64-125`, `opt.py:81-150,195-290,400-420`, `imputation_studies.ps1`,
`tests/unit/test_imputation_metrics.py`, `tests/fixtures/credit-g_20nan_imputation_regression.json`,
ADR 0004 decision 6 and "How to compare", ADR 0005, and — decisively, see below —
`docs/tickets/imputation-decoder/issues/05-evaluation-data-support.md` and `06-decode-stage.md`.

Four scratchpad scripts run under `uv run --python 3.10` with `PYTHONPATH` at the repo root, all
calling the real `prepare_dataset` / `build_folds` / `evaluation_mask` / `mean_mode_baselines` /
`score_cells` on the real `credit-g_{00,20,40,60,80}nan` tables. No training run (none of the six
claims is about a trained model; a 2-fold toy run would discriminate nothing). No pytest.

`v1_ladder.py` — realised exam rate across the ladder, fold 1 of 5, `eval_mask_rate=0.2`,
using exactly `decoding.py:154-157`'s definition:

```
variant        | missing% | masked | eligible | realised | rows with exactly 1 mask
credit-g_00nan |  0.00    |   834  |   4000   |  0.2085  |  15/200
credit-g_20nan | 19.85    |   569  |   3206   |  0.1775  |  44/200
credit-g_40nan | 40.45    |   359  |   2382   |  0.1507  | 109/200
credit-g_60nan | 59.48    |   259  |   1621   |  0.1598  | 160/200
credit-g_80nan | 80.58    |   202  |    777   |  0.2600  | 194/200
```

`v2_extra_rates.py` — the same, swept over the nominal rates `eval_mask_rates_extra` would use.

`v3_composite.py` — the critic's demo A through the real `score_cells`, plus a parity probe.

`v4_denominator.py` — builds the real masked cell population on `credit-g_20nan` for all 5 folds
via the real `evaluation_mask` and `mean_mode_baselines`, then scores two model classes through
the real `score_cells` and compares the fold-to-fold spread of `rmse_num_z` against `ratio_num`.

## Verdicts

### F-03-1 - The masked exam is a different exam at each rung of the ladder, so the ADR's cross-ladder comparability claim is unsupported

- **Verdict:** SOUND
- **Severity after review:** medium (down from high)
- **Basis:** the premise reproduces. `v1_ladder.py` gives realised rates
  0.2085 / 0.1775 / 0.1507 / 0.1598 / 0.2600 — non-monotone, with the turn-up at `_80nan` caused
  by the one-mask-per-row floor at `src/utils.py:65-74` binding on 194 of 200 rows. The mechanism
  is exactly as described: `p_dynamic = p_base * (1 - prop_nulls)` (`utils.py:55`), then
  `dynamic_mask[null_values] = False` (`:62`), then the forced floor. The naive denominator cannot
  track it: `mean_mode_baselines` is context-free, so a row losing its context does not make the
  baseline worse. And the claim really is in the ADR —
  `docs/adr/0004-imputation-decoder-task.md:222`: *"on any variant,
  `cv/test/impute/masked/impute_score/mean` is comparable across the ladder"* — and is propagated
  into the live comparison protocol at
  `docs/wayfinder/imputation-optuna-reduced/issues/06-importance-and-comparison.md:35`
  ("companion `cv/test/impute/masked/impute_score/mean`, comparable across the ladder").
- **Correction:** two things, and the first is the reason severity drops.

  **This is a re-derivation of something the repo already wrote down, and the critic did not cite
  it.** `docs/tickets/imputation-decoder/issues/05-evaluation-data-support.md:65-77` carries the
  same table (20.4 / 17.1 / 14.6 / 15.3 / 25.1 per eligible cell — my numbers to within fold
  draw) and the sentence *"a self-masked score at 'rate 0.2' is not comparable across the ladder,
  and it is not the literature's 20% MCAR either"*, with the same explanation for the `_80nan`
  turn-up. `06-decode-stage.md:47-51` repeats it as a carried-forward comment. The prescribed
  mitigation — log the realised rate — was implemented (`decoding.py:155`). So the live defect is
  narrower and more specific than the critique states: not "nobody noticed", but **ADR 0004:222
  contradicts its own ticket 05, and ADR 0005's ticket 06 inherited the wrong half.** That is a
  protocol-document defect, medium, not a measurement defect, high.

  **The directional consequence is speculation.** "A model that is genuinely equally good at every
  rung will show a score that *improves* from `_20nan` to `_40nan`" assumes the hidden-cell count
  is the only thing that moved. It is not: at `_40nan` each row also offers the model 40% fewer
  observed cells as context. The two effects push opposite ways and the critic ran no model, so
  the sign is unknown. The defensible claim is "the exam and the yardstick both change, so the
  reading is uninterpretable" — which is enough, and is what ticket 05 says.

### F-03-2 - "1.0 is baseline parity" allegedly false for the composite

- **Verdict:** UNSOUND
- **Severity after review:** low (down from high)
- **Basis:** the headline claim is false in its own terms, and the real `score_cells` says so.
  `v3_composite.py`, a model whose predictions **are** the mean/mode imputer on both halves:

  ```
  model == mean/mode on BOTH halves: {'rmse_num_z': 0.977933, 'acc_cat': 0.643,
                                      'macro_f1_cat': 0.260905, 'impute_score': 1.0}
  ```

  Exactly 1.000000. The composite is a convex combination of two ratios each anchored at 1.0, so
  parity everywhere is parity in the sum. ADR 0004's gloss ("1.0 meaning no better than the naive
  baseline") is correct as written.

  The demo A numbers do reproduce (`mode-collapsed 0.7557` vs `genuinely-learned 0.4378`), but
  they show the metric doing the right thing: the collapsed model is ranked strictly worse, and
  0.756 is a truthful aggregate statement — that model really did halve the naive RMSE on half the
  scored cells and really did tie the baseline on the other half. The critique's own Consequence
  concedes the search is not fooled ("collapse costs nothing relative to the baseline, but it is
  not rewarded either"). What is actually being claimed is that *below 1.0 does not imply better
  on every half*, which is an ordinary property of every weighted average and is not the gloss the
  ADR offers.
- **Correction:** three errors. (1) The stated claim — parity is not 1.0 — is measurably wrong.
  (2) The demo uses a 50/50 cell split; on `credit-g` the critique's own `w_cat ≈ 0.65` bounds a
  categorically-collapsed model at `impute_score ≥ 0.65` even with a *perfect* numerical half, so
  0.75 is not reachable there and the "dominant half" argument cuts the other way. (3) The
  regression-fixture point is weak: `acc_cat` **is** pinned at tolerance 0.01 per population, so
  "a change that trades macro-F1 for a mode-biased accuracy would pass regression silently" is
  false for any change that moves accuracy — which is every such change. What survives, at low,
  is that `macro_f1_cat` is computed, commented on at `:88-90` as the reason accuracy is the wrong
  quantity, and then used by nothing; and the Direction (emit `ratio_num` / `ratio_cat` beside the
  composite) is a good suggestion that costs two lines. See also "What this critique missed" on
  commensurability, which is the real design question here.

### F-03-3 - The Optuna trial score is read off the same cells that selected the checkpoint

- **Verdict:** SOUND
- **Severity after review:** low (down from medium)
- **Basis:** the premise is exact. `hidden_validation` is built once (`decoding.py:77-82`), the
  per-epoch validation loss is `model(hidden_validation, clean_validation)` (`:114`), `best_state`
  is the argmin over `decode_epochs` (`:125-127`), and after `model.load_state_dict(best_state)`
  (`:129-130`) the objective is `_score_population(model, hidden_validation, clean_validation,
  "masked")` (`:204-206`) — the identical tensor. `opt.py:266` reads
  `validation/impute/masked/impute_score`; `opt.py:201-230` never sets `cv_folds` and
  `config.py:129` defaults it to `None`. Min-over-epochs on a fixed sample, then reporting a
  correlated functional on that same sample, is optimistically biased. That much is textbook.
- **Correction:** the consequence is overstated on both legs.

  **The differential-across-trials argument does not apply to the search that ships.**
  `imputation_studies.ps1:62` passes `--search_space reduced`, and `opt.py:408-411` makes
  `reduced` the default for `task == 'imputation'` even without the flag. The reduced space
  (`opt.py:94-108`) samples `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and
  `LAMBDA_NUM` — **not** `EPOCHS_DECODE`. Every trial therefore takes the minimum over the same
  150 epochs (`types.py:112`), so the "longer trials take a deeper minimum" channel, which is the
  critique's strongest argument, exists only in the `full` profile that the imputation launcher
  never uses. The `LR_DECODE → noisier curve → deeper minimum` channel remains, but the critic did
  not measure it and neither did I; it is a plausible mechanism, not an observed bias.

  **The bias never reaches a comparison.** It does reach the tracking store — `opt.py:270` logs
  `optuna/objective_value`, and `best_objective_value` with it — so the critique is right that
  those numbers are optimistic. But `score_search_objective` is set only at `opt.py:221`, and
  `decoding.py:202-211` emits the `validation/` family only under it, so no ordinary run, no
  `cv/test/*` key and no regression fixture carries it. The promotion comparison
  (`imputation_studies.ps1:83`) is two fresh `--cv_folds 5` runs scored on the **test** split with
  their own masks. The only exposure is which configuration wins the search — real, but it is a
  search-efficiency cost, not a contaminated claim. The Direction (a second, independent
  validation mask) is cheap and correct; keep it, at low.

### F-03-4 - The baseline normalisation is a per-fold scalar rescale that allegedly injects noise into the promotion CI

- **Verdict:** UNSOUND
- **Severity after review:** low (down from medium)
- **Basis:** the **premise** reproduces in full. `v4_denominator.py` on `credit-g_20nan`, 5 folds,
  real masks:

  ```
  naive_rmse per fold:    [1.032, 0.918, 1.022, 1.052, 1.080]  cv=6.0%
  naive_cat_err per fold: [0.389, 0.371, 0.429, 0.401, 0.405]  cv=5.4%
  per-column mode error:  foreign_worker 0.062 ... purpose 0.738   (12x spread)
  per-column naive RMSE:  credit_amount 0.658-1.545, residence_since 1.009-1.083
  ```

  So: the numerical denominator really is ≈1 by construction of the z-scoring, the per-column
  heterogeneity really is collapsed into one pooled number, and the denominators really do move
  ±6% per fold. Every measured claim in the Evidence block holds.

  The **consequence** does not follow, and it is the load-bearing half. "The per-fold denominators
  inject variance that `_summarize_metric` then reports as a Student-t CI, where it reads as model
  variance" asserts a sign the critic never measured. The sign depends on `Cov(numerator,
  denominator)`, and both are computed on the *same cells*, so for any model that partially
  recovers the signal the covariance is positive and the ratio **cancels** the noise. Measured
  through the real `score_cells` on the real masked populations:

  ```
  -- fold-to-fold spread (std / mean), numerical half --
  A_proportional   raw rmse cv=0.0592  ratio cv=0.0015   (ratio LOWER, 40x)
  B_fixed_noise    raw rmse cv=0.0433  ratio cv=0.0908   (ratio HIGHER, 2x)
  ```

  Model A is `pred = 0.6 * actual` — error proportional to the cell's magnitude. The normalisation
  removes essentially all fold-to-fold noise there: `ratio_num` is flat to 0.15% while the raw
  `rmse_num_z` swings 5.9%. Model B is `pred = actual + N(0, 0.6)` — error independent of the
  cell — and there the ratio does inherit the denominator's noise. Both models are stylised, and
  Model A's flatness is partly algebraic (with a baseline at ≈0 the ratio collapses to `1 - k`), so
  this is not evidence about where the real decoder sits. It is a proof that the sign can go either
  way: the effect is **indeterminate without measuring the trained model on the real folds**, which
  the critique did not do before asserting one direction and building a promotion-risk consequence
  on it.
- **Correction:** beyond the sign error, two framings are wrong. (1) *"The thing the normalisation
  was for — making the score robust and comparable — is not delivered"* attributes a purpose the
  ADR does not state. ADR 0004 decision 6 specifies pooled per-type ratios with cell-fraction
  weights and claims exactly one thing for them, an interpretable anchor at 1.0 — which is
  delivered exactly (measured `1.000000` in F-03-2). The critique's own "Checked and cleared"
  concedes the weights match the stated design, which makes the Consequence here inconsistent with
  it. (2) *"`ratio_cat` carries no information `1 - acc_cat` does not"* is true within one fold and
  false across folds, which is the only place the CV mean lives: the denominator swings
  0.371-0.429, so the ratio and the raw accuracy order the five folds differently. Demo C shows
  only that a pooled metric does not decompose per column, which is what "pooled" means.

  What survives at low: the ratio-estimator point in the last sentence of the Consequence. The CV
  statistic is a mean of five ratios with five different denominators, and a Student-t interval on
  it is neither a ratio-of-pooled-errors nor an interval on a well-behaved mean. That observation
  is untouched by anything above and the critique should have led with it instead of the noise
  claim.

### F-03-5 - `_ratio`'s zero guard allegedly inverts the ranking and guards the wrong thing

- **Verdict:** UNSOUND
- **Severity after review:** none
- **Basis:** demo B reproduces exactly, and it does not show what the critique says. Running the
  real `score_cells` with a perfect model and varying only the baseline:

  ```
  perfect model, baseline wrong on 0/200 -> impute_score 1.0000
  perfect model, baseline wrong on 1/200 -> impute_score 0.0000
  perfect model, baseline wrong on 2/200 -> impute_score 0.0000

  mode-copying model, baseline wrong on 0/200 -> impute_score 1.0000
  mode-copying model, baseline wrong on 1/200 -> impute_score 1.0000
  mode-copying model, baseline wrong on 2/200 -> impute_score 1.0000
  ```

  No ranking is inverted. When the baseline is flawless, "the perfect model" and the mode imputer
  emit **the same predictions on every cell** — they are the same predictor, and 1.0 (parity) is
  the correct score for both, which the second block confirms. When the baseline errs, the perfect
  model scores 0.0 and the mode-copier 1.0, correctly ordered. This is the documented intent
  (`_ratio` docstring, `imputation_metrics.py:108-113`: *"Matching a flawless baseline is parity
  rather than a win"*) and it is pinned red-green by
  `test_a_column_the_naive_imputer_never_gets_wrong_still_ranks` at 1.0 / 1.5.
- **Correction:** the discontinuity is real — I measured `_ratio(0.1, 0.0) = 1.1` against
  `_ratio(0.1, 1e-9) = 1.0e8` — but it is unreachable, and not for the reason the critique gives.
  The categorical denominator is `np.mean(naive != actual)`, a mean of 0/1 indicators, so it is
  either exactly 0 or at least `1/n`; `1e-9` is not in its range at any population size. The
  numerical denominator is a pooled RMSE over *every* numerical column at once, so reaching the
  near-zero region needs every numerical column near-constant on the scored cells. That leaves the
  ADR 0004:101-102 sentence ("A constant column can yield a zero baseline; that division is
  guarded") being imprecise about *where* the guard sits — which is ADR wording, explicitly
  excluded by the brief's "What is NOT a finding". Cut it.

  One sub-claim inside the Evidence is also backwards on its own terms: *"a constant column
  contributes 0 to the denominator sum while the model's errors on it enter the numerator
  unnormalised."* That is the correct behaviour, not a defect — a model that is wrong where a
  flawless baseline is right should be penalised with no offset.

### F-03-6 - The per-column artifact carries errors without their baselines

- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** premise verified line by line. `ImputationScores.per_column` is built at
  `imputation_metrics.py:59-69` from `_error_metrics` alone, which emits `n_num_cells`,
  `n_cat_cells`, `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat` (`:72-104`) — no ratio, no
  baseline — while the dataclass docstring at `:22-24` says it *"says where a poor fold went
  wrong"*. `runner.py:26-32` computes `mean_mode_baselines`, passes it to `score_cells(group,
  baselines)`, and keeps only `.per_column`, discarding `.metrics` — the sole place the baseline
  enters. With the measured per-column mode errors (0.062 on `foreign_worker`, 0.738 on `purpose`,
  `v4_denominator.py`), `acc_cat` alone genuinely cannot separate a learned column from a
  mode-collapsed one: `acc_cat = 0.93` on `foreign_worker` is a *loss* to that column's baseline
  accuracy of 0.938, while `acc_cat = 0.45` on `purpose` is a clear win over its 0.262 — and the
  artifact shows only the two numbers 0.93 and 0.45, which order them backwards. The Direction is
  one line — `score_cells` already holds
  `baselines` at `:59`.
- **Correction:** one factual trim. The critique writes as if the information were gone; it is
  recomputable, just not from this artifact. `artifacts.write_imputation_preview` (`:224-265`)
  writes the full cell ledger — `row, column, kind, population, actual, imputed` plus original
  units — for **every** fold (`runner.py:127`, suffix `_fold_{ordinal}`), so anyone with the
  results directory can rebuild per-column baselines offline. The reason low still stands rather
  than none is MLflow: `tracking.py:214-236` replays a fold's artifacts only for the
  `best_fold` / `worst_fold` diagnostic children, so `per_column_imputation.csv` on the parent —
  the baseline-free one — is the only per-column view that exists for a middling fold in the
  tracking store, which is exactly the fold this artifact is for.

## What this critique missed

**1. The `eval_mask_rates_extra` diagnostic is degenerate at high missingness, and this is a
direct, unmade consequence of F-03-1.** ADR 0004:155 sells the extra-rate sweep
(`impute/masked/rate_10/...`, `decoding.py:162-172`) as a difficulty curve. `v2_extra_rates.py`,
same fold, same seed:

```
credit-g_60nan | 0.1 |  220 | 1621 | 0.1357     credit-g_80nan | 0.1 | 198 | 777 | 0.2548
credit-g_60nan | 0.2 |  259 | 1621 | 0.1598     credit-g_80nan | 0.2 | 202 | 777 | 0.2600
credit-g_60nan | 0.3 |  302 | 1621 | 0.1863     credit-g_80nan | 0.3 | 204 | 777 | 0.2625
credit-g_60nan | 0.4 |  356 | 1621 | 0.2196     credit-g_80nan | 0.4 | 212 | 777 | 0.2728
```

On `_80nan` a 4x change in the nominal rate moves the realised exam by 7% (198 → 212 cells): the
one-mask-per-row floor has absorbed the entire sweep, and the four "difficulties" are four
readings of one exam. `_60nan` is half-absorbed (0.136 → 0.220 for a 4x ask). The critic scoped
`decoding.py:162-172` and never measured it.

**2. The mitigation ticket 05 prescribed is not applied where it is needed.** `realised_rate` is
emitted for the primary masked population only (`decoding.py:154-157`). The extra-rate populations
(`:162-172`) get no realised-rate key, so the one metric that would make point 1 visible in the
tracking store is absent from exactly the populations where the nominal rate lies most. Two lines
inside the existing loop. (The induced population is out of scope here — it scores every gap the
generator left, so it has no nominal rate to be honest about.)

**3. Commensurability is the real design question under the composite, and the critique never asks
it.** `ratio_num` is a ratio of RMSEs; `ratio_cat` is a ratio of 0/1 error rates. They are summed
with cell-count weights as though "30% less RMSE" and "30% fewer wrong categories" were the same
amount of better. They are not on the same scale and there is no argument anywhere in ADR 0004 or
the tickets that they should be added. This is a sharper version of what F-03-2 was reaching for
and it would have been the right finding: it affects every mixed table, it changes what the
headline number means, and unlike F-03-2's claim it is actually true.

**4. Nobody has checked what `impute_score` means when `w_num` or `w_cat` is itself a random
variable across folds.** The weights are `len(numerical)/total` computed on the *drawn* cells
(`imputation_metrics.py:44-56`), so they move fold to fold with the mask draw — measured
`n_num/n_cat` across the five `credit-g_20nan` folds: 199/337, 197/372, 186/359, 201/369, 169/351,
i.e. `w_num` 0.371 down to 0.325, a 14% swing. The CV mean of `impute_score` is therefore a mean
of five differently-weighted composites. It is small next to the other effects, but it is a third
source of fold-to-fold movement in the number ADR 0005 decision 6 reads a CI on, and the critique
lists only two.
