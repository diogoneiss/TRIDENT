# 03 - Imputation metrics and the composite score

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/imputation_metrics.py` (all 137 lines); every consumer of
`impute_score` — `src/training/decoding.py:32-224` (baselines, the masked/induced/extra-rate
populations, the search objective), `src/training/runner.py:22-32,146-190`,
`src/training/summary.py:40-73,123-158,221-235`, `src/training/types.py:64-73,95-125`,
`opt.py:60-150,255-300,392-510`, `imputation_studies.ps1`; the protocol these implement,
`docs/adr/0004-imputation-decoder-task.md` decision 6 and "How to compare", and
`docs/adr/0005-reduced-optuna-search-for-imputation.md` decisions 3 and 6. Also read
`src/utils.py:18-82` (`preprocess_table`), `src/training/data.py:45-148`, and
`tests/fixtures/credit-g_20nan_imputation_regression.json`.

**Method:** read the source, then exercised the real functions on the real
`credit-g_{00,20,40,60,80}nan` tables through `prepare_dataset` / `build_folds` /
`evaluation_mask` / `mean_mode_baselines` / `score_cells` (four scratchpad scripts, `e1`–`e4`,
run under `uv run --python 3.10`). Cross-checked the conclusions against the committed
regression fixture by back-solving its `impute_score` from its own `rmse_num_z` and `acc_cat`.
No training run (the one I was allowed would have measured the wrong thing — see Open
questions), no pytest.

## Findings

### F-03-1 - The masked `impute_score` is not comparable across the missingness ladder: the exam's realised mask rate is non-monotone, 0.209 → 0.155 → 0.250

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/utils.py:49-74` (via `src/training/data.py:132-148` and
  `src/training/decoding.py:137-157`); claim at `docs/adr/0004-imputation-decoder-task.md:221-222`
- **Evidence:** ADR 0004 states: *"on any variant, `cv/test/impute/masked/impute_score/mean`
  is comparable across the ladder."* The masked population is drawn by
  `evaluation_mask(test_frame, eval_mask_rate, ...)` → `preprocess_table(p_base=0.2)`, which
  does two things the nominal rate does not describe:
  - `p_dynamic = p_base * (1 - prop_nulls)` (`src/utils.py:55`) — the per-row rate is scaled
    down by that row's missingness, *and then* already-null positions are additionally
    zeroed (`:62`), so the realised rate per **eligible** cell is `p_base * (1 - prop_nulls)`,
    not `p_base`;
  - `no_mask_rows` (`:65-74`) forces at least one mask per row, which becomes the binding
    constraint once a row has few observed cells.

  Measured with the real `evaluation_mask` on `credit-g`, fold 1 test split, `eval_mask_rate=0.2`,
  using exactly the definition `decoding.py:154-157` logs as `impute/masked/realised_rate`
  (masked cells / eligible cells):

  | variant | missing | masked cells | eligible | realised rate |
  |---|---|---|---|---|
  | `_00nan` | 0.00% | 834 | 4000 | **0.2085** |
  | `_20nan` | 19.73% | 541 | 3211 | **0.1685** |
  | `_40nan` | 39.60% | 374 | 2416 | **0.1548** |
  | `_60nan` | 60.40% | 249 | 1584 | **0.1572** |
  | `_80nan` | 79.70% | 203 | 812 | **0.2500** |

  At `_80nan` there are 203 masked cells over 200 test rows: essentially every row received
  exactly the one forced mask, which is why the rate turns back up. The `Hyperparameters`
  comment (`types.py:121-124`) anticipates a monotone decline ("about 5% of an 80%-missing
  one" — true as a share of *all* cells, 203/4000 = 5.1%); as a share of what the model could
  have been asked, the exam at `_80nan` is the **hardest** point of the ladder, not the
  easiest. Meanwhile the denominator of `impute_score` does not move at all: mean/mode
  imputation ignores the row, so the naive baseline is identical at every rung
  (measured induced naive RMSE 0.95–1.02, naive mode error 0.39–0.42 across `_20nan`.. `_80nan`).
- **Consequence:** the ladder comparison ADR 0004 explicitly invites — reading
  `cv/test/impute/masked/impute_score/mean` across `credit-g_20nan` … `credit-g_80nan` as a
  degradation curve — compares four different exams while holding the yardstick fixed. A
  model that is genuinely equally good at every rung will show a score that *improves* from
  `_20nan` to `_40nan` (fewer cells hidden per row) and then jumps at `_80nan` (one forced
  mask on a row with ~3 observed cells). Blast radius is confined to *cross-variant* reads:
  the Optuna objective and the promoted-vs-default comparison (ADR 0005 decision 6) both
  stay inside one variant and are unaffected.
- **Direction:** for evaluation only, draw the mask uniformly among the *eligible* cells at
  the nominal rate and drop the one-per-row floor — `evaluation_mask` already owns a private
  RNG stream (`data.py:143-148`), so classification stays bit-identical and only the
  imputation fixture moves. Failing that, stop claiming ladder comparability for the masked
  population and report the induced one instead, which is comparable (see Checked and cleared).

### F-03-2 - "1.0 is baseline parity" is false for the composite: a model that has collapsed to mode imputation on every categorical column scores 0.75

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/imputation_metrics.py:44-57`; claims at
  `docs/adr/0004-imputation-decoder-task.md:96-100` and
  `docs/adr/0005-reduced-optuna-search-for-imputation.md:161-162`
- **Evidence:** the composite is `w_num * ratio_num + w_cat * ratio_cat` with `w` the cell
  fractions. Each half is *individually* anchored at 1.0, but the weighted sum is not: a
  model at parity on one half and better than naive on the other lands below 1.0 and is read
  as "beats the naive imputer". Run against the real `score_cells` (script `e3`, demo A):
  1000 numerical cells reconstructed at `rmse_num_z` 0.489, plus 1000 cells of a
  three-category column on which the model emits the training mode for **every** cell:

  ```
  A mode-collapsed model : {'rmse_num_z': 0.489, 'acc_cat': 0.626, 'macro_f1_cat': 0.2567, 'impute_score': 0.75}
  A genuinely-learned    : {'rmse_num_z': 0.489, 'acc_cat': 0.804, 'macro_f1_cat': 0.771,  'impute_score': 0.512}
  ```

  The collapsed model has learned literally nothing about half the table — its categorical
  predictions are bit-identical to the baseline it is being measured against — and is
  reported at 0.75, i.e. "25% better than mean/mode". `macro_f1_cat` separates the two
  cleanly (0.257 vs 0.771) and the module's own comment (`:88-90`) says exactly why accuracy
  is the wrong quantity — *"accuracy alone flatters a model that always answers with the
  common category"* — yet `macro_f1_cat` enters no decision anywhere: not the fold ranking
  (`types.py:68`), not the search objective (`types.py:71`), not promotion, not the
  comparison protocol. It is also **not** pinned by the regression fixture, which carries
  only `impute_score`, `rmse_num_z` and `acc_cat` per population
  (`tests/fixtures/credit-g_20nan_imputation_regression.json`), so a change that trades
  macro-F1 for a mode-biased accuracy would pass regression silently.
- **Consequence:** every headline number in ADR 0005's comparison protocol
  (`cv/test/impute/induced/impute_score/mean` and its companion) is reported with the gloss
  "lower is better, 1.0 is baseline parity". A reader applying that gloss to a mixed table
  will conclude a model imputes better than mean/mode when it may be doing nothing at all on
  the categorical columns. On `credit-g` (13 categorical, 7 numerical features, `w_cat` ≈
  0.65 measured) the categorical half carries most of the weight, so the failure mode is on
  the dominant half. Note the mirror image is *not* a search hazard: collapse costs nothing
  relative to the baseline, but it is not rewarded either — a trial whose categorical ratio
  slips from 0.6 to 1.0 still loses `0.4 * w_cat`.
- **Direction:** either report the two ratios beside the composite so parity per half is
  visible (`impute/<pop>/ratio_num`, `ratio_cat` are already computed, just not emitted), or
  build the categorical ratio from macro-F1 against a mode baseline's macro-F1 so collapse
  scores at 1.0 rather than being hidden by the numerical half. At minimum, pin
  `macro_f1_cat` in the regression fixture and state the parity gloss per half, not for the
  sum.

### F-03-3 - The Optuna search objective is scored on the exact cells that selected the checkpoint

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/decoding.py:77-82`, `:112-130`, `:202-211`; `types.py:71`;
  `opt.py:265`, `:405`
- **Evidence:** `hidden_validation` is drawn once at `:77-82`. Every epoch, the validation
  loss is computed on it (`:113-115`), and `best_state` is the argmin of that loss over the
  `decode_epochs` epochs (`:125-127`). After restoring that checkpoint, the search objective
  is scored on **the same tensor**:

  ```python
  validation_cells = _score_population(model, hidden_validation, clean_validation, "masked")
  ```

  `opt.py:265` then reads `metrics['validation/impute/masked/impute_score']` as the trial's
  score. `Args` in `opt.py:201-230` never sets `cv_folds`, and `config.py:129` defaults it to
  `None`, so each trial is one predefined split — a single validation mask, scored once,
  after a minimum has been taken over 150 epochs (`decode_epochs` default) of noisy
  evaluations on those same cells. The selected functional (decode loss) and the reported one
  (`impute_score`) are different but strongly correlated, so the selection optimism transfers
  partially rather than fully. ADR 0005 decision 3 describes this design as protecting the
  test split, and it does; it does not note that the validation number is itself a selected
  minimum.
- **Consequence:** `optuna/best_objective_value` and every `optuna/objective_value` are
  optimistically biased, and — worse for a search — the bias is **differential across
  trials**. In the `full` profile `EPOCHS_DECODE` is sampled 20–60 (`opt.py:134`), so trials
  that train longer take a deeper minimum and are rewarded for it. In both profiles
  `LR_DECODE` is sampled 1e-4..1e-2 (`opt.py:101`, `:137`): a high learning rate makes the
  validation loss curve noisier, which makes min-over-epochs more optimistic, so the search
  has a systematic pull toward configurations that are merely noisier. `--promote_best`
  (`imputation_studies.ps1:67`) then writes that winner to the promoted configuration file
  that the paired comparison runs read.
- **Direction:** draw a second evaluation mask on the validation split, independent of the
  one used for checkpoint selection, and score the objective on that; `evaluation_mask`
  already takes a fold ordinal, so a different offset is enough. Alternatively, select the
  checkpoint on one mask and report on the other, swapping their roles.

### F-03-4 - The baseline normalisation is a per-fold scalar rescale, not a per-column normalisation: it removes no column heterogeneity and adds sampling noise to the CI

- **Kind:** methodology
- **Severity:** medium
- **Where:** `src/training/imputation_metrics.py:44-57`; `summary.py:148-158`
- **Evidence:** both ratios divide a *pooled* model error by a *pooled* naive error, so every
  column is normalised by one number per fold.
  - **Numerical.** `dataset.frame` is already z-scored (`data.py:67-71`), so
    `mean_mode_baselines` returns the train-fold mean *of z-scores*. Measured on
    `credit-g_20nan`, 5 folds (script `e1`): the numerical baselines are
    `mean|b| = 0.016–0.024` (i.e. ≈ 0), and the pooled naive RMSE over a simulated 20% mask
    is **0.912, 0.967, 1.027, 1.010, 0.952** — ≈ 1 by construction, with a ±6% fold-to-fold
    swing that is pure cell-sampling noise. In vivo: back-solving the committed fixture's
    induced folds with `w_num = 0.35` and the independently measured `d_cat = 0.406` gives
    1.386 vs the pinned 1.374 (0.9%) and 1.333 vs 1.306 (2.1%); holding `d_num = 1.0` the
    implied `d_cat` is 0.411 / 0.418, within 3% of the measured mode error. The numerical
    denominator really is ≈ 1.0 in the real run, so `ratio_num` is `rmse_num_z` divided by a
    noisy 1.
  - **Categorical.** The per-column mode error on real data spans 18x —
    `foreign_worker` 0.038 … `purpose` 0.691 on `credit-g_20nan` fold 1 (script `e2`) — and
    all of it is collapsed to one pooled denominator, 0.402. Dividing pooled error by a
    pooled constant is a strictly monotone rescale of `acc_cat`, so `ratio_cat` carries no
    information `1 - acc_cat` does not. Script `e3`, demo C: a model that is **8x worse than
    the mode baseline** on the near-constant column and near-parity on the hard one, and a
    model that is at mode everywhere, receive the *identical* score:

    ```
    C model catastrophic on foreign_worker: {'acc_cat': 0.625, 'macro_f1_cat': 0.4558, 'impute_score': 0.8219}
    C model ~mode everywhere              : {'acc_cat': 0.625, 'macro_f1_cat': 0.365,  'impute_score': 0.8219}
    ```
  - **Squared-error mass.** Pooling RMSE weights a numerical column by its squared error, not
    its cell count, so one skewed column drives the fold-to-fold movement. Measured per-column
    naive RMSE across the five `credit-g_20nan` folds: `credit_amount` swings **0.773 → 1.344**
    while `residence_since` stays 0.941–0.982. The heavy-tailed column moves both numerator
    and denominator and dominates the ratio.
- **Consequence:** the thing the normalisation was for — making the score robust and
  comparable — is not delivered, and the per-fold denominators inject variance that
  `_summarize_metric` (`summary.py:148-158`) then reports as a Student-t CI on the mean,
  where it reads as model variance. That CI is the promotion evidence in ADR 0005 decision 6
  (`ci95_lower` / `ci95_upper` on `cv/test/impute/induced/impute_score/mean`), so a promotion
  decided on marginally non-overlapping intervals can be decided by cell-draw noise in the
  denominator. It is also a ratio estimator: the CV statistic is a mean of five ratios with
  five different denominators, not a ratio of pooled errors, and a t-interval assumes neither.
- **Direction:** normalise **per column** and aggregate the per-column ratios with explicit
  weights (cell count or equal), so a column's baseline difficulty is divided out where it
  actually varies; or compute the denominator once per dataset (over all cells, all folds)
  so it is a constant of the problem rather than a per-fold random variable. Either removes
  the denominator from the CI.

### F-03-5 - `_ratio`'s zero guard protects the pooled error, not the constant column ADR 0004 says it protects, and inverts the ranking when it does fire

- **Kind:** design
- **Severity:** low
- **Where:** `src/training/imputation_metrics.py:107-117`; claim at
  `docs/adr/0004-imputation-decoder-task.md:101-102`
- **Evidence:** ADR 0004 says *"A constant column can yield a zero baseline; that division is
  guarded."* The guard is applied to the pooled naive error of a whole kind
  (`:52`, `:56`), never per column, so a constant column never triggers it — it contributes
  0 to the denominator sum while the model's errors on it enter the numerator unnormalised.
  The guard is also discontinuous: `naive_error = 1e-9` returns `model/1e-9`, `naive_error = 0`
  returns `1 + model`. When it does fire the ordering inverts. Script `e3`, demo B — the same
  **perfect** model (every scored cell reconstructed exactly), varying only how often the
  naive baseline happens to be right:

  ```
  B perfect model, naive errs on 1/200 cells : 0.0
  B perfect model, naive errs on 0/200 cells : 1.0
  B perfect model, naive errs on 2/200 cells : 0.0
  ```

  One baseline cell moves a flawless model from 0.0 (best possible) to 1.0 (parity). On real
  populations of thousands of cells across many columns the pooled error is never exactly 0,
  so this is largely unreachable in practice; the reachable half of the finding is the
  mismatch between what the ADR claims is guarded and what the code guards.
- **Consequence:** low in practice. It matters for small populations — a
  `eval_mask_rates_extra` diagnostic at a low rate on a narrow single-type table, or a
  hand-built unit fixture — where the score would silently flip from "best possible" to
  "parity", and for the ADR's constant-column claim, which is simply not what the code does.
- **Direction:** guard where the ADR says it guards (per column, once per-column ratios
  exist per F-03-4), and make the fallback continuous — e.g. an epsilon floor on the
  denominator derived from the column's scale — rather than a branch that returns a value
  from a different region of the range.

### F-03-6 - The per-column artifact carries errors without their baselines, so the one artifact meant to localise a bad fold cannot

- **Kind:** design
- **Severity:** low
- **Where:** `src/training/imputation_metrics.py:59-69`, `:72-104`; `runner.py:22-32`
- **Evidence:** `ImputationScores.per_column` is built only from `_error_metrics`, which emits
  `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat` and cell counts — no ratio, no
  baseline. Its docstring (`:22-24`) says it *"says where a poor fold went wrong"*.
  `_per_column_scores` (`runner.py:26-32`) does compute `mean_mode_baselines` and calls
  `score_cells(group, baselines)` per population, then keeps `.per_column` and **discards**
  `.metrics`, which is where the baseline actually enters. This is distinct from known
  finding 12 (that one is about the *recomputation* being redundant); the point here is that
  the baseline is computed, used for nothing that reaches the artifact, and thrown away.
- **Consequence:** with the real per-column mode errors spanning 0.038–0.691 (F-03-4),
  `acc_cat = 0.95` on `foreign_worker` is a failure (mode gets 0.962) and `acc_cat = 0.45` on
  `purpose` is a success (mode gets 0.309) — and the artifact cannot distinguish them,
  because the reader has no per-column baseline to compare against. The numerical half is
  partially readable (z-space puts the implicit baseline near 1.0) but only by a reader who
  knows that, which is itself F-03-4's point.
- **Direction:** add the column's baseline error and its ratio to the per-column rows —
  `score_cells` already has `baselines` in hand at `:59-65`; it just does not use them in the
  long-form branch.

## Checked and cleared

- **The induced population *is* comparable across the ladder** — the opposite of ADR 0004's
  emphasis. Measured (script `e4`) on `credit-g_{20,40,60,80}nan`, fold 1, encoding the
  complete sibling with the variant's own scaler exactly as `decoding.py:251-255` does: naive
  numerical RMSE 0.998 / 1.020 / 0.963 / 0.952 and naive mode error 0.406 / 0.416 / 0.392 /
  0.405, with `w_num` 0.319–0.353. Both the exam (the generator's own gaps) and the yardstick
  are stable, so `cv/test/impute/induced/impute_score/mean` is the number that survives a
  cross-variant read. ADR 0004:221-222 reserves ladder comparability for the masked
  population and the induced one for a single variant — backwards on both counts.
- **Model and baseline live in the same space, for both populations.** The numerical baseline
  is the train-fold mean of the already-z-scored frame, and the induced ground truth is put
  through `dataset.scaler.transform` before scoring (`decoding.py:251-255`), so numerator and
  denominator are both in z units. The mean of the z-scores is an affine image of the mean in
  original units, so predicting it is exactly mean imputation — the train-fold baseline is
  **not** leaky despite the scaler being fit on all rows (`data.py:71`); the global fit
  affects the unit the error is measured in, identically for model and baseline.
- **The weights match the stated design.** `w_num = n_num / (n_num + n_cat)`,
  `w_cat = n_cat / total` (`:44-56`) are the fractions of scored cells per type, which is
  exactly what ADR 0004 decision 6 specifies. Weighting columns equally is *not* the intent,
  so it is not a defect — but combined with F-03-4 it means no level of the aggregation
  weights columns equally.
- **`_00nan` lacking the induced keys is by design, and does not break the summariser.**
  `load_complete_sibling` returns `None` when the variant is its own sibling
  (`data.py:104-105`), so the induced block never runs and no `impute/induced/*` key is
  emitted at all; the key set is therefore identical across folds and
  `_validate_final_metrics` (`summary.py:136-141`) is satisfied. ADR 0004:219-221 already
  scopes the induced headline to `_20nan`..`_80nan`.
- **The promotion protocol is a human reading two CIs, not an automated rule.**
  `imputation_studies.ps1:79-116` runs the promoted configuration and the defaults as two
  ordinary CV runs and moves the promoted file aside for the second; no threshold or
  significance test is coded. So there is no automated decision rule to be wrong — the
  exposure is that the two CIs a human compares carry the denominator noise of F-03-4.
- **`mean_mode_baselines` learns from the training fold only**, as its docstring claims — the
  frame it receives is `features.iloc[fold.train_indices]` in both call sites
  (`decoding.py:134-136`, `runner.py:24-27`), and NaNs are skipped by `.mean()` / `.mode()`,
  so no scored cell informs its own baseline.

## Open questions

- **How large is the selection optimism in F-03-3?** Settling it needs a second, independent
  validation mask scored with the same checkpoint — the gap between the two is the bias. I
  deliberately did not spend the one permitted training run on it: comparing
  `validation/impute/masked/impute_score` to `test/impute/masked/impute_score` confounds the
  selection optimism with the split difference, and the per-epoch `decode/val_loss` series is
  only logged to the (disabled) tracker.
- **Is the missingness in the `_XXnan` generator MCAR?** The whole induced-population argument
  — that the variant's scaler, fit on observed cells, puts the induced cells at z ~ N(0,1) —
  rests on it. My `e4` measurement (induced naive RMSE 0.95–1.02) is consistent with MCAR but
  does not prove it; the generator script is not in this repository. If it is MNAR, the
  induced denominator and the masked denominator are measuring different distributions and the
  two populations' scores stop being companions.
- **What does `impute_score` mean on the six all-numerical datasets?** There the composite
  reduces to `rmse_num_z / (≈1)`, i.e. the score and the raw metric are the same number up to
  noise, while on `kr-vs-kp` (all categorical) it is `(1 - acc_cat) / 0.4`-ish. ADR 0004
  claims it "degrades correctly" on both; whether a single number spanning those two regimes
  should be compared *across datasets* — which `docs/BACKLOG.md` and the study protocol both
  do informally — would be settled by running one model at fixed hyperparameters over all
  eight datasets and checking whether the ordering it produces matches the ordering of the two
  underlying error metrics.
