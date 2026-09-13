# 06 - Imputation artifacts, metric keys and MLflow identity: verdicts

_Adversarial verification of `06-artifacts-tracking.md`. 2026-09-11._

**Method:** read in full `src/training/artifacts.py`, `src/training/summary.py`,
`src/training/imputation_metrics.py`, `src/training/decoding.py:25-340`,
`src/training/runner.py:18-192`, `src/training/config.py:61-140`, `src/training/types.py:109-300`,
`src/embedder.py:161-200`, `opt.py:190-262`, `:328-346` and `:570-585`;
`git diff main...HEAD -- CLAUDE.md` and `git show main:opt.py`. Ran, all read-only on the repo:

- header scan of the 45 files in `metrics/` (5 carry an `impute` header - matches the critique);
- `results/<dataset>/*/metrics.csv` header scan for the five clobbered variants;
- sqlite3 over a **copy** of `mlflow.db` in the scratchpad: `run_role x task` census,
  `lr_scheduler x task` census on parents, the shared-metric-key set across tasks on parents,
  `cv/decode/val_loss/mean` step counts on imputation parents, and the surviving classification
  parents for each clobbered variant;
- `uv run --python 3.10 python -c ...` probes: the real `ArtifactWriter.write_imputation_preview`
  on a ledger holding both induced populations; `round(rate*100)` collisions; `score_cells` key
  sets for single-kind and empty frames plus a real `summarize_cross_validation` abort;
  `kr-vs-kp_20nan` per-column mode shares and `spcop`'s exact value counts.

**No training run** - nothing needed one. **Not checked:** decoder numerics, and whether an
MLflow artifact store copy of each `results/` tree exists (I checked the local trees only).

## Verdicts

### F-06-1 - The root `metrics/<dataset>_metrics.csv` is keyed by dataset alone, so an imputation run overwrites a classification run's fold table with a different column set
- **Verdict:** CONFIRMED (the mechanism and the five overwritten files), with the "destroyed / no
  version to recover from" half **refuted**
- **Severity after review:** medium (down from high)
- **Basis:** `artifacts.py:86` is exactly as quoted, and the header scan reproduces the five files:

  ```
  IMPUTE: metrics/credit-g_20nan_metrics.csv   metrics/credit-g_40nan_metrics.csv
  IMPUTE: metrics/kr-vs-kp_20nan_metrics.csv   metrics/kr-vs-kp_40nan_metrics.csv
  IMPUTE: metrics/spambase_20nan_metrics.csv
  ```

  Each holds one `single_split` row with a `validation/impute/...` family, so an Optuna trial is
  indeed the last writer. `git check-ignore` confirms `.gitignore:20:metrics/`.

  The finding is sharper than the critique states in one respect it did not check: this branch
  *did* apply task-keying everywhere else. `config.py:61-69`:

  ```python
  suffix = ".json" if task == DEFAULT_TASK else f".{task}.json"
  ```

  and `results_dir` is timestamped. `metrics/<dataset>_metrics.csv` is the **only** path in the
  codebase keyed by dataset alone whose column set changes with the task.
- **Correction:** three things the critic got wrong, all pushing severity down.

  1. **Nothing was destroyed.** `write_metrics` writes *two* copies and only the root one is
     clobbered. Every one of the five classification fold tables is intact on disk:

     ```
     [CLASSIF] results/credit-g_20nan/20260909_065141/metrics.csv   rows=3
     [CLASSIF] results/credit-g_40nan/20260909_085209/metrics.csv   rows=3
     [CLASSIF] results/kr-vs-kp_20nan/20260909_072520/metrics.csv   rows=3
     [CLASSIF] results/kr-vs-kp_40nan/20260909_093005/metrics.csv   rows=3
     [CLASSIF] results/spambase_20nan/20260909_080848/metrics.csv   rows=3
     ```

     And the store answers the critique's own open question in the affirmative - each of the five
     has exactly one `run_role = parent`, `task = classification` run carrying the summary:
     `credit-g_20nan` 0.6292, `credit-g_40nan` 0.6361, `kr-vs-kp_20nan` 0.8843,
     `kr-vs-kp_40nan` 0.8009, `spambase_20nan` 0.9064 on `cv/test/f1_macro/mean`, all
     `fold_count = 3`. "There is no version to recover it from" is false twice over.
  2. **Three folds, not five.** The critic describes the lost table as "five folds" and cites
     `metrics/vehicle_00nan_metrics.csv` as the surviving shape; that file has 3 data rows, and
     every classification parent in the store reports `fold_count = 3`.
  3. **No programmatic consumer reads the root file back.** Grepping every `.py`/`.ps1`/`.ipynb`
     for `metrics_dir` and `_metrics.csv` finds only writers (`artifacts.py`, `config.py`,
     `runner.py`, `opt.py` passing the flag) and tests. `compute_cv_summary` takes a frame or a
     path the caller supplies. The harm is "a human opening `metrics/` gets a different question
     answered", not a corrupted pipeline.

  What survives is real and worth fixing: a filename that is an identity claim, silently
  re-pointed at another task's units by a run that was never meant to be comparable. Medium.

### F-06-2 - The naive denominator of `impute_score` is never persisted, and the per-column artifact reports raw error with no baseline beside it
- **Verdict:** SOUND in its core, with two false sub-premises
- **Severity after review:** medium (down from high)
- **Basis:** premise confirmed. `naive_rmse` / `naive_error` are locals in `score_cells`
  (`imputation_metrics.py:48-56`) and `grep -rn baseline --include=*.py src/ opt.py main.py train.py`
  returns only call sites - `mean_mode_baselines` is never returned, logged, tagged or written.
  `_error_metrics` (which builds both the pooled metrics and every `per_column` row) emits only
  `n_num_cells`, `n_cat_cells`, `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat`. So
  `per_column_imputation.csv` is raw model error, and sorting it ascending by `acc_cat` on a
  categorical table does point at the easy columns. The mode-share probe reproduces:

  ```
  n feature cols: 36
  top: spcop 1.000  skach 0.998  hdchk 0.996  reskd 0.991  stlmt 0.986  qxmsq 0.970 ...
  bottom: simpl 0.612  cntxt 0.566  bkxbq 0.535
  cols with mode share > 0.90: 14
  ```

  The diagnosis-misdirection argument (consequence (a)) stands on its own and needs no comparison.
- **Correction:** two sub-premises are wrong, and one mitigation is unreported.

  1. **`_ratio` is never applied per column.** The critic writes that `spcop` "is constant ... so
     `_ratio` silently takes its `1.0 + model_error` branch, a number on a different scale from
     every other column's ratio". There *is* no other column's ratio: `score_cells` builds
     `per_column` from `_error_metrics` alone, and `_ratio` is called exactly twice, on the
     population-pooled numbers. The zero-baseline branch can only fire if the naive imputer's
     error over the *entire* scored population is exactly 0.
  2. **`spcop` is not constant.** `value_counts(dropna=False)` on `kr-vs-kp_00nan` gives
     `f 3195, t 1` - mode share 0.9996871, which the critic's probe rounded to 1.000. So even the
     column they picked would not trigger the branch they describe.
  3. **On single-kind tables the denominator *is* recoverable from what is already logged.** With
     one kind the weight is 1, so `naive_error = (1 - acc_cat) / impute_score`. On the logged
     `kr-vs-kp_20nan` row (`acc_cat = 0.863847`, `impute_score = 0.738351`) that is 0.184402, a
     naive accuracy of 0.8156. `kr-vs-kp`, `spambase` and `kc2` are all single-kind, so the
     cross-variant blind spot the critic describes bites only on mixed tables (`credit-g`), where
     two unknowns share one equation.

  Severity medium: the per-column artifact genuinely cannot do its stated job on categorical
  tables, and the mixed-table cross-variant gap is real; the rest is narrower than claimed.

### F-06-3 - With `--score_null_path` the same cell is in the ledger under two populations, and the preview prints it twice in one block with a row-count header
- **Verdict:** CONFIRMED
- **Severity after review:** low (down from medium)
- **Basis:** both populations select the same positions, by reading:
  `_score_induced_missing(as_mask=True)` builds `test_frame.mask(test_frame.isna(), "[MASK]")` and
  `Embedder.encode` sets `masked_array[:, index] = (df[col] == "[MASK]")` (`embedder.py:192`), so
  `masked_positions == test_frame.isna()`; `as_mask=False` sets the same tensor explicitly
  (`decoding.py:262`). Both are concatenated into `cells` (`decoding.py:178`, `:190`), which is
  what `write_imputation_preview` receives.

  Running the real writer on one row with one induced cell reproduces the critique byte for byte:

  ```
  **row 0** -- 3 cell(s) filled in

  | | duration | duration | purpose |
  |---|---|---|---|
  | actual | 24.000 | 24.000 | radio |
  | model saw | [NULL]->[MASK] | [NULL]->[MASK] | [MASK] |
  | imputed | 0.800 | -0.300 | radio |

  ledger rows: 3   distinct (row,column): 2
  ```
- **Correction:** the blast radius is much smaller than "medium".
  - **No metric is affected.** `scored = score_cells(cells, baselines)` runs at `decoding.py:150`,
    *before* the induced concat at `:178`; the induced and null-token families are each scored
    from their own frame. `_per_column_scores` (`runner.py:30`) groups by `population` first, so
    the per-column artifact is clean too.
  - **The flag has never been used.** Every `*_cells.csv` on disk holds only `masked` + `induced`,
    with `distinct (row,column) == rows` (credit-g fold 1: 3351/3351; spambase fold 1:
    43101/43101). Nothing in the working tree is affected today.
  - The ledger's `population` column already disambiguates the duplicate, which the critic
    concedes. The residual defect is the preview only: a header counting ledger rows instead of
    distinct cells, and two identically-labelled columns in one block. That is low.

### F-06-4 - `CLAUDE.md`'s comparison rule was edited on this branch and still omits `tags.task`, on the only metric families both tasks share
- **Verdict:** SOUND
- **Severity after review:** low (down from medium)
- **Basis:** every factual claim reproduces. `git diff main...HEAD -- CLAUDE.md` is a one-line
  rewrite of exactly that sentence, adding the `lr_scheduler` clause and no `tags.task`;
  `README.md:369` says "Filter `tags.run_role = parent` **and `tags.task`**". The store agrees:

  ```
  run_role x task            parents: lr_scheduler x task
    optuna_trial imputation 181    cosine_legacy classification 67
    parent       classification 87 cosine        classification 6
    parent       imputation  6     plateau       classification 6
    None         classification 12 constant      classification 4
                                   plateau       imputation      4
                                   warmup_cosine classification 4
                                   cosine        imputation      2
  ```

  and the 28 metric keys shared by both tasks on `run_role = parent` are exactly
  `cv/pretrain/{train,val}_loss/{mean,ci95_*}`, the full seven-field
  `cv/time/{pretrain,finetune,total}_seconds/*`, and `time/training_seconds`.
  `shared keys starting with cv/test: []` - the separation the critique clears is real.
- **Correction:** none to the argument; the severity is inflated. The harm is confined to timing
  aggregates and the pre-training bands (which are legitimately shared - same stage). The
  headline comparison keys cannot collide, the run names differ visibly
  (`impute_<dataset>_<ts>` vs `train_<dataset>_<ts>`), and the one genuinely misleading case -
  `cv/time/finetune_seconds` holding decode wall-clock on imputation parents - is already
  known finding 9. What is left is a one-line documentation divergence between `CLAUDE.md` and
  `README.md` with a narrow measurement consequence. Low, and cheap to fix.

### F-06-5 - No step-less `test/loss` on an imputation fold, so the decoder's objective is allegedly never scored on held-out data
- **Verdict:** UNSOUND
- **Severity after review:** low (down from medium)
- **Basis:** the narrow premise is true. `grep -rn "test/loss" src/` returns
  `finetuning.py:196` and the two `summary.py` reader lines and nothing in `decoding.py`; the
  decode stage logs only `decode/{train_loss,val_loss,learning_rate}` per epoch and the step-less
  `test/impute/...` family. No imputation parent carries `cv/test/loss` - confirmed in the key
  census. So `raw_fold_metrics.csv` has a `loss` column for classification and not for imputation.
- **Correction:** the headline and the consequence do not follow from that premise.

  1. **The objective *is* scored on held-out data, every epoch, on every fold.**
     `validation_frame = features.iloc[fold.validation_indices]` (`decoding.py:51`) is held out of
     decoder training; `validation_loss, _ = model(hidden_validation, clean_validation)` at
     `decoding.py:114` is the same composite the training loop minimises, `LAMBDA_NUM` included.
  2. **It reaches the parent for every fold, not just the two diagnostic children.** The critic
     writes "Every other fold's series is buffered and discarded". It is not:
     `_summarize_loss_bands` consumes `records` - all of them - over `task.loss_keys`, which for
     imputation is `pretrain/{train,val}_loss` + `decode/{train,val}_loss`. Verified in the store:
     both imputation CV parents carry `cv/decode/val_loss/mean` at **150 steps** with
     `fold_count = 2`. The per-epoch held-out objective curve is on the parent.
  3. **The stated consequence is therefore false.** "A trial whose loss never came down and a
     trial that fit well but scored badly against the naive baseline are indistinguishable from
     the run record" - an imputation trial logs
     `validation/impute/masked/{rmse_num_z, mae_num_z, acc_cat, macro_f1_cat, n_*_cells}` on the
     same validation split beside `impute_score`, and `rmse_num_z` / `1 - acc_cat` are close
     proxies for the two terms the composite weighs (I did not read `TridentDecoder`'s loss, so
     I claim proximity, not identity). A trial whose loss never came down is visible there.

  What genuinely survives, and is the legitimate kernel of the critic's argument:
  `decode/val_loss` is the criterion the checkpoint is *selected* on, so as a held-out estimate it
  is optimistically biased, and a test-split composite would be the unbiased one. And
  `best_validation_loss` and the selected epoch are not recorded as step-less scalars, so for
  folds that are not `best_fold`/`worst_fold` the exact checkpoint epoch is not recoverable from a
  per-fold record (the across-fold band gives the typical epoch, not that fold's). Both are low:
  the bias is bounded by a 150-epoch selection over one fixed mask, and the unbiased quality
  numbers on the test split are already logged as `test/impute/*`.

### F-06-6 - The imputation fold metric key set is data-dependent, so a disagreeing fold aborts the CV run after every fold has trained
- **Verdict:** SOUND
- **Severity after review:** low (unchanged)
- **Basis:** reproduced end to end.

  ```
  cat-only keys : ['acc_cat', 'impute_score', 'macro_f1_cat', 'n_cat_cells', 'n_num_cells']
  num-only keys : ['impute_score', 'mae_num_z', 'n_cat_cells', 'n_num_cells', 'rmse_num_z']
  empty keys    : ['n_cat_cells', 'n_num_cells']
  RAISED: All folds must have the same final metric keys.
  ```

  The ordering claim holds against the source: `write_hyperparameters` is `runner.py:156`,
  `write_metrics` `runner.py:157`, `summarize_cross_validation` `runner.py:165`. `metrics.csv`,
  `hyperparameters.json` and `per_column_imputation.csv` (written at `:152-154`) are on disk when
  the raise happens; the CV summary, manifest, provenance and every `cv/*` metric are not.
  Classification's key set is fixed by the sklearn metric functions, so the asymmetry is real.
- **Correction:** none. The critic already rated this honestly, including that they could not
  trigger it from shipped data. Low is correct.

### F-06-7 - Extra evaluation rates are keyed by a rounded integer percent, so nearby rates collide silently and a sub-0.5% rate becomes `rate_0`
- **Verdict:** CONFIRMED
- **Severity after review:** low (unchanged)
- **Basis:** `decoding.py:169` is `prefix = f"impute/masked/rate_{round(extra * 100)}"` feeding a
  `metrics.update`. Reproduced:

  ```
  0.02 -> rate_2   0.024 -> rate_2   0.025 -> rate_2   0.015 -> rate_2
  0.004 -> rate_0  0.2 -> rate_20    0.205 -> rate_20
  collision dict: {'rate_2': 0.025}      # 0.02 gone, no trace
  ```

  `_validate_final_metrics` cannot catch it: every fold collides identically. And the rate really
  is not recorded anywhere else - `EVAL_MASK_RATES_EXTRA` has no CLI flag at all (the parser at
  `config.py:235-255` exposes `--task` and `--score_null_path` and nothing for the rates), so the
  only other record would be the config file, which F-06-8 shows promotion rewrites without it.
- **Correction:** none. Low is right: it needs a hand-edited config with two rates inside one
  percent, which nothing on disk has.

### F-06-8 - `hyperparameters.json` and `complete_configuration` both omit `EVAL_MASK_RATES_EXTRA`, which `--promote_best` then erases from the task's config file
- **Verdict:** CONFIRMED
- **Severity after review:** low (unchanged), and it duplicates a finding in another critique
- **Basis:** `artifacts.py:57-64` and `config.py:97-105` both list
  `EPOCHS_DECODE, LR_DECODE, WEIGHT_DECAY_DECODE, LAMBDA_NUM, EVAL_MASK_RATE` and stop, under
  docstrings claiming "the values this run used, and only those" and "a promoted file names every
  value the task will train with". `types.py:171-176` reads the key and `decoding.py:162` consumes
  it. `opt.py:344-345`:

  ```python
  with open(path, 'w') as f:
      json.dump(complete_configuration(resolved, task), f, indent=4)
  ```

  so a hand-written extras list in the task's file is gone after the next promotion. Confirmed
  that `score_null_path` is a `TrainingRequest` field and CLI flag with no param and no tag
  (`grep -rn score_null_path` finds no `tracking.py` reference).
- **Correction:** two small ones. There are **four** promoted imputation configs on disk now
  (`credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan`, `kr-vs-kp_40nan`), not two; all four
  carry `EVAL_MASK_RATE: 0.2` and no extras, which is what the critic predicted. And this is the
  same defect as **F-10-5** in `10-config-plumbing.md`, already flagged in
  `01-decode-stage.verdict.md:318` - it should be reported once, not three times.

## What this critique missed

- **Extra rates get no `realised_rate`.** `decoding.py:154-158` computes
  `impute/masked/realised_rate` for the primary rate only; the per-rate loop at `:162-172` emits
  the `score_cells` metrics and nothing else. Realised rate is the number that says the nominal
  rate was not achieved - it falls as missingness rises, because the helper never hides an
  already-missing cell - and the populations that lack it are exactly the ones whose nominal rate
  survives only in the key F-06-7 rounds. The per-rate `n_num_cells`/`n_cat_cells` partly
  substitute, but nothing records the denominator they were drawn from.
- **`validation/impute/*` would be summarised under a `cv/test/` prefix.** `tracking.py:393` is
  `_summary_metrics(f"cv/test/{metric_name}", ...)` over every key in `FoldResult.metrics`, and
  `decoding.py:216` merges `validation_metrics` into that mapping whenever
  `score_search_objective` is set. Today only `opt.py` sets it and trials never pass `cv_folds`,
  so it cannot fire - but nothing in `_validate_final_metrics`, the summariser or the request
  validation rejects the combination, and if it ever fires a validation-split number is published
  as `cv/test/validation/impute/masked/impute_score/mean`. That is the same class of identity
  defect as F-06-1 and F-06-4 and it is one line from being reachable.
- **Cell counts are summarised as if they were measurements.** `n_num_cells` and `n_cat_cells` are
  in `FoldResult.metrics`, so the parent publishes
  `cv/test/impute/masked/n_num_cells/{mean, ci95_lower, ci95_upper, std}` - a two-sided Student-t
  interval on a count of cells. Harmless numerically, but it puts a fabricated uncertainty band
  next to the real ones in the same table a reader scans.
- **A study deletes every trial's cell ledger, including the winner's, and leaves behind only the
  clobbered root CSV.** Each trial writes a complete tree - `hyperparameters.json`, `metrics.csv`,
  `data/`, `metrics/per_column_imputation.csv` and a full `imputation_cells.csv` - under
  `results/<dataset>/temp_trials/<dataset>/<timestamp>/` (`opt.py:205-206`), and `opt.py:575-582`
  then `shutil.rmtree`s the whole `temp_trials` directory at study end. Visible on disk today:
  `results/spambase_20nan/temp_trials` holds 20 trees at 20 MB mid-study (each with a ~43k-row
  ledger), while `results/credit-g_20nan/` - whose 40-trial study finished - has none. So the
  per-cell record of the configuration the study *chose* is destroyed, and `--retrain_best` has to
  regenerate it. What does survive the cleanup is the one thing F-06-1 is about: the root
  `metrics/<dataset>_metrics.csv` row that the last trial happened to write. The search's
  artifact retention is exactly inverted.
- **The critique's own "no recovery" claim was checkable and wasn't checked.** The open question
  it leaves ("do the five clobbered files matter, or were those baselines published elsewhere?")
  is answered by two commands against `results/` and the store, both of which it already had open.
  Leaving it open let a recoverable-CSV annoyance be written up as destroyed data, which is what
  drove F-06-1's severity to high.
