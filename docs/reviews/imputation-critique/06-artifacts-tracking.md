# 06 - Artifacts, metric keys and MLflow identity

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/training/artifacts.py` (all 417 lines), `src/training/summary.py` (all 257),
`src/training/tracking.py` (all 424), `src/training/types.py` (all 383),
`tests/unit/test_imputation_artifacts.py` (all 248). Read for context, not critiqued:
`src/training/decoding.py:130-224`, `src/training/imputation_metrics.py:31-137`,
`src/training/runner.py:22-192`, `src/training/config.py:72-140`, `src/mlflow_utils.py`,
`opt.py:190-300` and `:400-575`, `src/utils.py:18-82`, `src/embedder.py:161-200`,
`README.md:369-383`, `CLAUDE.md:13-15`.

**Method:** `git diff main...HEAD` on the four source files, then the full current source of
each. Five probe scripts under the scratchpad (`probe1..5.py`), run with
`uv run --python 3.10`: (1) `write_imputation_preview` + `_render_preview` on a ledger
holding the duplicate cell `--score_null_path` produces; (2) `round(rate*100)` key
collisions, an empty `scored_cells` frame, and `summarize_cross_validation` under a
cross-fold key mismatch; (3) per-column mode share of `kr-vs-kp_20nan`; (4) reconciliation
of the logged `impute/induced/n_*_cells` against the real NaN count of `credit-g_20nan`'s
test split; (5) a **copy** of `mlflow.db` queried with sqlite3 for the tag and metric-key
census below. Also inspected `metrics/` (45 files) and `datasets/hiperparams/`. **No
training run** - nothing left needed one. **Not checked:** whether the `results/` tree on
disk matches what MLflow received (no artifact-store walk); anything about decoder numerics.

The live store independently confirms known finding 9 - six imputation parents carry
`cv/time/finetune_seconds/*` holding decode wall-clock. Not re-reported.

## Findings

### F-06-1 - The shared per-dataset metrics file is task-blind, and imputation runs have already destroyed five classification baselines in the working tree

- **Kind:** bug
- **Severity:** high
- **Where:** `src/training/artifacts.py:86`
- **Evidence:** `write_metrics` writes the run's fold table twice - once into the
  timestamped `results_dir`, once into

  ```python
  root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
  frame.to_csv(root_path, index=False)
  ```

  The name carries the dataset variant and nothing else, and the rows carry `fold`,
  `dataset` and the metric columns - no `task`, no `run_id`, no timestamp. The branch adds
  a second task that writes a *completely different column set* to that same path.

  Scanning the user's `metrics/` directory (45 files) for a header containing `impute`:

  ```
  IMPUTATION-OVERWRITTEN: metrics/credit-g_20nan_metrics.csv
  IMPUTATION-OVERWRITTEN: metrics/credit-g_40nan_metrics.csv
  IMPUTATION-OVERWRITTEN: metrics/kr-vs-kp_20nan_metrics.csv
  IMPUTATION-OVERWRITTEN: metrics/kr-vs-kp_40nan_metrics.csv
  IMPUTATION-OVERWRITTEN: metrics/spambase_20nan_metrics.csv
  ```

  `metrics/credit-g_20nan_metrics.csv` now holds exactly one row:

  ```
  fold,dataset,impute/masked/n_num_cells,...,validation/impute/masked/impute_score
  single_split,credit-g_20nan,98.0,185.0,...,0.8972680666379934
  ```

  `fold = single_split` together with a `validation/impute/...` family means this is an
  **Optuna trial** - `score_search_objective` is set only by `opt.py`, and trials never pass
  `cv_folds`. So the file that names the dataset holds one arbitrary trial of a 172-trial
  study, not the study winner and not a comparable run. The classification fold table that
  was there (the shape still visible in `metrics/vehicle_00nan_metrics.csv`:
  `fold,dataset,accuracy,f1_micro,f1_macro,...` over five folds) is gone. `metrics/` is
  gitignored, so there is no version to recover it from.
- **Consequence:** For those five variants, the file a reader reaches for as "the result for
  credit-g at 20% missing" silently answers a different question than it did before the
  branch, in different units, from a run that was never meant to be comparable. Re-running
  `--task imputation` on any of the other 40 variants loses that variant's classification
  numbers the same way, with no warning and no backup. The two halves are separable: trials
  overwriting the file predates the branch (`opt.py` passed `metrics_dir` on `main` too,
  line 113); *cross-task* overwriting, and a file whose task cannot be identified without
  inspecting its column names, are new here.
- **Direction:** The filename is the run's identity claim, so it has to carry everything
  that changes the columns: key the root file by task (and by evaluation mode, so a trial
  cannot land on a comparison run's file), or stop writing the root copy at all from a run
  whose `tracking_run_role` is not `parent`. A `task` column inside is worth adding either
  way, but it does not stop the clobber.

### F-06-2 - The denominator of the headline `impute_score` is never recorded anywhere, and the per-column artifact reports raw error with no baseline beside it

- **Kind:** design
- **Severity:** high
- **Where:** `src/training/artifacts.py:268` (`write_per_column_imputation`); the
  denominator is computed and discarded at `src/training/imputation_metrics.py:48-56`
- **Evidence:** `impute_score` is a ratio. In `score_cells`:

  ```python
  naive = numerical["column"].map(baselines).to_numpy(dtype=float)
  naive_rmse = float(np.sqrt(np.mean((naive - numerical["actual"].to_numpy(dtype=float)) ** 2)))
  score += (len(numerical) / total) * _ratio(metrics["rmse_num_z"], naive_rmse)
  ...
  naive_error = float(np.mean(naive != categorical["actual"].to_numpy()))
  score += (len(categorical) / total) * _ratio(1.0 - metrics["acc_cat"], naive_error)
  ```

  `naive_rmse` and `naive_error` are locals. `grep -rn "baseline" --include=*.py src/`
  returns only `decoding.py`, `imputation_metrics.py` and `runner.py` call sites - the
  `mean_mode_baselines` mapping is never returned past `score_cells`, never logged as a
  metric, never a param, never written to an artifact. `_error_metrics`, which produces both
  the pooled `metrics` and the per-column table, emits `n_num_cells`, `n_cat_cells`,
  `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat` and nothing about the baseline. So
  `per_column_imputation.csv`, whose docstring says "Pooled metrics rank a fold; this says
  which column a poor one struggled with", answers in raw model error.

  For numerical columns that is close to harmless: the frame is globally z-scored
  (`data.py:71`), so a train-fold-mean baseline has RMSE close to 1 and `rmse_num_z` already
  reads as roughly the ratio. **The categorical half is the problem.** On `kr-vs-kp_20nan` -
  a variant with a promoted imputation config at
  `datasets/hiperparams/kr-vs-kp/kr-vs-kp_20nan.imputation.json` - probe 3 measured the mode
  share of each of the 36 feature columns, which is the accuracy a mode imputer already
  gets:

  ```
  spcop  1.000   skach 0.998   hdchk 0.996   reskd 0.991   stlmt 0.986   qxmsq 0.970
  ...
  simpl  0.612   cntxt 0.566   bkxbq 0.535
  columns whose mode baseline already exceeds 0.90 accuracy: 14 of 36
  ```

  Sorting `per_column_imputation.csv` ascending by `acc_cat` - the only thing the artifact
  supports - points the reader at `bkxbq`, the column where naive is weakest and beating it
  is easiest, and never flags `skach`, where 0.95 accuracy is a large loss to a do-nothing
  imputer. `spcop` is constant, so `_ratio` takes its `naive_error == 0.0` branch and
  contributes `1.0 + model_error`, a number on a different scale from every other column's
  ratio, and nothing in any artifact or metric records that the branch was taken.
- **Consequence:** Two concrete failures. (a) Column-level diagnosis, the artifact's stated
  job, is systematically misdirected on categorical tables (`kr-vs-kp`, and the categorical
  half of `credit-g`). (b) In MLflow, `cv/test/impute/masked/impute_score/mean` cannot be
  decomposed. The user's own workflow compares across `tags.missingness_percent`
  (`credit-g_20nan` vs `credit-g_40nan`) and across `EVAL_MASK_RATE`; both change the scored
  cell population and therefore the denominator, so a score that moves from 0.92 to 0.87
  cannot be attributed to the model rather than to an easier or harder naive baseline, and
  nothing logged lets a reader tell. The numerator (`rmse_num_z`, `acc_cat`) is logged; the
  denominator that turns it into the ranking number is not. To be fair to the design: within
  one variant at a fixed seed and `EVAL_MASK_RATE` the train folds and the evaluation mask
  are deterministic, so the denominator is identical across runs and the ratio comparison -
  including the primary Optuna trial ranking - is sound. The gap is cross-variant,
  cross-rate, and per-column; (a) needs no comparison at all to bite.
- **Direction:** The baseline is per-column, per-fold, and already computed twice
  (`decoding.py:134` and `runner.py:26`). Carry it out of `score_cells` - a `baseline_rmse`
  / `baseline_error` entry in `ImputationScores.metrics` so it lands under
  `cv/test/impute/*/...`, and a `baseline_*` metric row plus a per-column ratio in the
  long-form per-column table. Flagging the zero-baseline branch (a `constant_column` row, or
  excluding it from the weighted sum) belongs with it.

### F-06-3 - `--score_null_path` writes every induced cell into the ledger twice, and the preview prints the column twice in one block with a wrong cell count

- **Kind:** bug
- **Severity:** medium
- **Where:** `src/training/artifacts.py:327` (`_render_preview`), fed by
  `src/training/decoding.py:190`
- **Evidence:** Distinct from known finding 4, which is about the *label* on an
  `induced_null_token` row. This is about the row existing twice.

  `_score_induced_missing(as_mask=True)` encodes `test_frame.mask(isna, "[MASK]")`, and the
  embedder sets `masked_array[:, index] = (df[col] == "[MASK]")` (`src/embedder.py:192`), so
  the selected positions are exactly `test_frame.isna()`. `as_mask=False` sets them to
  `test_frame.isna()` explicitly (`decoding.py:262`). **The two populations score the same
  cell set.** Both are concatenated into `cells` (`decoding.py:178` and `:190`), which is
  what `write_imputation_preview` receives.

  Probe 1, running the real `write_imputation_preview` on one row with one induced cell:

  ```
  **row 0** -- 3 cell(s) filled in

  | | duration | duration | purpose |
  |---|---|---|---|
  | actual | 24.000 | 24.000 | radio |
  | model saw | [NULL]->[MASK] | [NULL]->[MASK] | [MASK] |
  | imputed | 0.800 | -0.300 | radio |
  ```

  Two cells were filled in; the header says three. `duration` appears twice with two
  different imputed values and identical "model saw" text, so the reader cannot tell which
  column is the `[MASK]` path and which the `[NULL]` path. The ledger confirms both rows are
  present with the same `(row, column)`:

  ```
  0  duration  numerical  induced             1.0   0.8
  1  duration  numerical  induced_null_token  1.0  -0.3
  ```
- **Consequence:** For any run with `--score_null_path` on a `_20nan` variant, every induced
  cell (419 of them on `credit-g_20nan`'s test split) is in `imputation_fold_N_cells.csv`
  twice. The ledger is the artifact the preview's own legend points at for exact values
  ("full value is in the cell ledger beside this file"), and it is the natural place to
  recompute or audit a score. Anyone aggregating it without a `population` filter
  double-counts the induced population and blends the `[MASK]` and `[NULL]` predictions into
  one number - and the `[NULL]` path is the diagnostic the ADR says must never rank
  anything. The preview's wide-row splitter also budgets six ledger *rows* per block, so with
  the flag on a block can hold as few as three distinct cells rather than six.
- **Direction:** The preview and the ledger are two views of one table, so the population has
  to be part of a cell's identity in both: pivot the preview on `(row, column, population)`
  with the population named in the "model saw" line rather than inferred from a prefix test,
  and have the block header count distinct cells. The ledger is correct as a long-form table
  but should say so - a documented uniqueness key of `(row, column, population)`, or the
  population folded into the filename.

### F-06-4 - The repo's own MLflow filtering rule was edited by this branch and still omits `tags.task`, on the only two metric families both tasks share

- **Kind:** methodology
- **Severity:** medium
- **Where:** `CLAUDE.md:15` (the rule); `src/training/summary.py:40` and
  `src/training/tracking.py:396` (the shared keys)
- **Evidence:** `git diff main...HEAD -- CLAUDE.md` shows this branch rewrote exactly that
  sentence - to add the `lr_scheduler` clause - and left it reading:

  > Filter `tags.run_role = parent` before comparing runs ... Also compare within one
  > `tags.lr_scheduler` value ...

  No `tags.task`. `README.md:369` *was* updated ("Filter `tags.run_role = parent` and
  `tags.task`"), so the two checked-in documents now disagree, and `CLAUDE.md` is the one
  loaded first by every agent and quoted in the user's own working notes.

  Probe 5, against a copy of `mlflow.db`, shows the rule is already insufficient:

  ```
  run_role x task                          parents by lr_scheduler x task
    parent        classification  87         cosine_legacy classification 67
    parent        imputation       6         plateau       imputation      4
    optuna_trial  imputation     172         cosine        imputation      2
    <none>        classification  12         cosine        classification  6
  ```

  and the metric keys shared by both tasks on `run_role = parent` are exactly:

  ```
  cv/pretrain/train_loss/{mean,ci95_lower,ci95_upper}
  cv/pretrain/val_loss/{mean,ci95_lower,ci95_upper}
  cv/time/pretrain_seconds/*   cv/time/finetune_seconds/*   cv/time/total_seconds/*
  time/training_seconds
  ```

  No `cv/test/*` key is shared - that separation holds (see Checked and cleared). The
  overlap is entirely timing plus the shared pre-training loss bands.
- **Consequence:** `run_role = parent AND lr_scheduler = plateau` currently returns 6
  classification parents and 4 imputation parents. Any aggregate over that selection on
  `cv/time/finetune_seconds/mean`, `cv/time/total_seconds/mean` or `time/training_seconds` -
  which is what "did the schedule change training cost?" reduces to - averages 150 decode
  epochs into a fine-tuning number. The per-metric key separation that protects `cv/test/*`
  gives no protection here, and known finding 9 makes the `finetune_seconds` case actively
  mislabelled rather than merely mixed. The 12 `run_role = <none>` classification runs are a
  separate, pre-existing blind spot in the same rule.
- **Direction:** `CLAUDE.md` is the operative rule, so that is the file that has to name
  `tags.task` (and `tags.is_optuna = 'false'`, which README already documents and the rule
  also omits - 172 trials sit in the same store). Fixing known finding 9 removes the
  mislabelling but not the mixing.

### F-06-5 - An imputation fold logs no step-less test loss, so the decoder's own objective is never scored on held-out data

- **Kind:** design
- **Severity:** medium
- **Where:** `src/training/summary.py:76` (`final_metrics_for_tracking`)
- **Evidence:** `final_metrics_for_tracking` promotes a fold's step-less `test/loss` event
  into the final metric set:

  ```python
  test_loss_events = [e for e in record.metric_events if e.key == "test/loss" and e.step is None]
  if test_loss_events:
      metrics["loss"] = test_loss_events[0].value
  ```

  `train_and_evaluate_classifier` logs one (`finetuning.py:196`).
  `train_and_evaluate_decoder` logs `decode/train_loss`, `decode/val_loss`,
  `decode/learning_rate` per epoch with a step, and the `test/impute/...` family step-less -
  **no step-less `test/loss`**. So an imputation fold's `loss` key is simply absent,
  `raw_fold_metrics.csv` has no `loss` column, and no `cv/test/loss/*` reaches MLflow. The
  key census in F-06-4 is consistent: no imputation parent carries it.

  The checkpoint is chosen by `best_validation_loss` (`decoding.py:125-130`), and that number
  is likewise never recorded: it survives only as the minimum of the per-epoch
  `decode/val_loss` series, which is replayed only for the two folds that become diagnostic
  children. Every other fold's series is buffered and discarded.
- **Consequence:** `LAMBDA_NUM` is a tuned knob - the reduced Optuna profile samples it over
  `[0.1, 10.0]` on mixed tables, and the promoted `credit-g_20nan` config sits at 2.738 - and
  it rescales the exact quantity the decoder minimises. No artifact and no metric shows that
  quantity on held-out data, so a trial whose loss never came down and a trial that fit well
  but scored badly against the naive baseline are indistinguishable from the run record.
  Separately, the selected epoch is unrecoverable for any fold that is not `best_fold` or
  `worst_fold`, so "which weights produced this number" has no answer for folds 2..n-1 of a
  CV run.
- **Direction:** Have the decode stage log a step-less `test/loss` (the same composite the
  training loop minimises, evaluated on the scored test mask) so `final_metrics_for_tracking`
  keeps producing the same shaped row for both tasks, plus a step-less `decode/best_val_loss`
  and `decode/best_epoch` so the checkpoint is identified on every fold's record rather than
  only on the two replayed ones.

### F-06-6 - The imputation fold metric key set is data-dependent, and a fold that disagrees aborts the whole CV run after every fold has trained

- **Kind:** design
- **Severity:** low
- **Where:** `src/training/summary.py:141`
- **Evidence:** `_validate_final_metrics` requires `set(metrics) == expected_keys` across
  folds. For classification the key set is fixed by the metric functions. For imputation it
  is decided by the data: `_error_metrics` emits `rmse_num_z`/`mae_num_z` only
  `if len(numerical)`, `acc_cat`/`macro_f1_cat` only `if len(categorical)`, and `score_cells`
  emits `impute_score` only `if total`. So a fold whose `masked`, `induced`,
  `induced_null_token` or `rate_<pct>` population happens to contain no cell of one kind
  carries fewer keys than its siblings. Probe 2:

  ```
  RAISED: ValueError All folds must have the same final metric keys.
  ```

  The raise happens at `runner.py:165`, after every fold has trained. `metrics.csv` and
  `hyperparameters.json` are already on disk (lines 156-157); the CV summary, the manifest,
  the provenance file and every `cv/*` metric are not, and the MLflow parent closes FAILED.
- **Consequence:** Honestly low for the shipped datasets - at `EVAL_MASK_RATE = 0.2` a
  `credit-g` test fold scores hundreds of cells of each kind, and the single-type tables
  (`kr-vs-kp` all categorical, `kc2`/`spambase` all numerical) are consistently single-type
  across folds, so their key sets agree. The realistic triggers are a small extra rate in
  `EVAL_MASK_RATES_EXTRA` on a small fold, and a high-missingness variant where
  `evaluation_mask` scales the nominal rate down hard. The cost when it does fire is a whole
  cross-validation discarded at the last step.
- **Direction:** Either make the key set structural - emit every key the task's populations
  can produce, with an explicit sentinel (or `n_*_cells = 0` plus a NaN-free convention) for
  a population that had no cells of a kind - or make the summariser take the union and
  summarise each metric over the folds that carry it, recording `fold_count` per metric (the
  field already exists on `MetricSummary`).

### F-06-7 - Extra evaluation rates are keyed by a rounded integer percent, so nearby rates collide silently and a rate under 0.5% becomes `rate_0`

- **Kind:** bug
- **Severity:** low
- **Where:** `src/training/decoding.py:169` (a metric-key question, so in scope here)
- **Evidence:**

  ```python
  prefix = f"impute/masked/rate_{round(extra * 100)}"
  metrics.update({f"{prefix}/{name}": value for name, value in score_cells(at_rate, baselines).metrics.items()})
  ```

  Probe 2:

  ```
  rate 0.02  -> impute/masked/rate_2      rate 0.004 -> impute/masked/rate_0
  rate 0.024 -> impute/masked/rate_2      rate 0.2   -> impute/masked/rate_20
  rate 0.025 -> impute/masked/rate_2      rate 0.205 -> impute/masked/rate_20
  ```

  `metrics` is a dict and `update` is last-wins, so `EVAL_MASK_RATES_EXTRA = [0.02, 0.025]`
  produces one `rate_2` family holding the second rate's numbers and no trace of the first.
  `_validate_final_metrics` cannot catch it: every fold collides identically, so the key sets
  still match. The rate is not otherwise recorded - F-06-8 shows `EVAL_MASK_RATES_EXTRA` never
  reaches the params or `hyperparameters.json` - so the key name is the only record of which
  rate produced the number, and it is lossy.
- **Consequence:** A sweep over close rates (0.02/0.025/0.03, or anything sub-percent)
  silently reports fewer series than were computed, attributed to the wrong rate. Unlikely
  with the round rates the README suggests; unrecoverable when it happens.
- **Direction:** Key on the requested rate itself rather than a rounded percent (`rate_0p025`,
  or the value formatted without loss), and refuse a duplicate key rather than overwriting it
  - the collision is detectable at the point the prefix is built.

### F-06-8 - `hyperparameters.json` claims to record "the values this run used, and only those" but omits `EVAL_MASK_RATES_EXTRA`, which `--promote_best` then deletes from the config file

- **Kind:** bug
- **Severity:** low
- **Where:** `src/training/artifacts.py:35` (`write_hyperparameters`), mirrored by
  `src/training/config.py:97-105` (`complete_configuration`)
- **Evidence:** The docstring is explicit:

  > Record the values this run used, and only those. A file claiming a fine-tuning rate on a
  > run that never fine-tuned misleads whoever reads it later ...

  The imputation branch writes `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`,
  `LAMBDA_NUM`, `EVAL_MASK_RATE` - not `EVAL_MASK_RATES_EXTRA`, which
  `Hyperparameters.from_mapping` does read (`types.py:171-176`) and which the decode stage
  does consume (`decoding.py:162`). `complete_configuration` has the same gap and a stronger
  claim ("The same key set is what a promoted configuration file holds, so a promoted file
  names every value the task will train with, held ones included"), and
  `opt.py:promote_best_configuration` opens the task's config file with `'w'` and writes
  exactly that mapping. `--score_null_path`, which also changes what the run scores and what
  goes into the ledger, is likewise neither a param nor a tag. The one test that mentions the
  key, `tests/unit/test_opt_search_space.py:109`, asserts only that `define_search_space`
  never *samples* it ("A search never tunes how hard its own exam is") - a different and
  correct concern. Nothing tests that the run record names it.
- **Consequence:** A hand-written `EVAL_MASK_RATES_EXTRA` in
  `datasets/hiperparams/<base>/<dataset>.imputation.json` is erased the next time a study is
  promoted for that variant - the two promoted files present today
  (`credit-g_20nan.imputation.json`, `kr-vs-kp_20nan.imputation.json`) contain
  `EVAL_MASK_RATE` and no extras, which is what a rewrite leaves behind. After that the
  diagnostic rates stop being scored and nothing in the run record says they ever were,
  because the only place the rates were named was the metric key F-06-7 rounds.
- **Direction:** Include `EVAL_MASK_RATES_EXTRA` in `complete_configuration`, and therefore in
  both the params and the promoted file; a `[]` default is as honest as a scalar default, and
  the key set stays complete under its own stated rule. `score_null_path` is a runtime switch
  rather than a hyperparameter, so it belongs as a tag on the run.

## Checked and cleared

- **No `cv/test/*` key is shared between the two tasks.** The census in F-06-4 (probe 5, over
  the real store, `run_role = parent`) returns *zero* shared `cv/test/*` keys. The imputation
  family (`cv/test/impute/masked/...`, `cv/test/impute/induced/...`) cannot collide with
  `cv/test/f1_macro` or `cv/test/accuracy`. The overlap is confined to timing and the
  genuinely shared pre-training loss bands.
- **Stage loss bands are keyed by the task's own stage.** `TaskSpec.loss_keys` is
  `pretrain/{train,val}_loss` plus `f"{self.stage}/{train,val}_loss"`, so the parent logs
  `cv/decode/train_loss/*` for imputation and `cv/finetune/train_loss/*` for classification -
  different series, no silent mixing. `cv/pretrain/*` is shared and should be: the
  pre-training stage is identical between tasks.
- **`_diagnostic_roles` orients correctly for a minimised metric.** With
  `direction = "minimize"`, `sign = -1.0`; `worst = min(records, key=(-metric, fold))` selects
  the largest `impute_score` (most error relative to naive) and
  `best = min(records, key=(+metric, fold))` the smallest. Ties resolve to the lowest fold
  number for both roles, as the comment claims. Reversing this would have silently swapped the
  two diagnostic children on every imputation run.
- **The manifest and the diagnostic children carry the imputation record.**
  `write_cv_tracking_artifacts` keys the ranking section off `task.ranking_metric_short_name`,
  so the file holds `impute_score_ranking` for imputation while the classification file keeps
  `f1_macro_ranking` byte for byte; `retained_artifact_paths` names the selected folds' preview
  and ledger, and `_log_diagnostic_children` replays them to MLflow under `imputation/`.
  Children now also carry `task`, `lr_scheduler` and `is_optuna` (new on this branch) -
  verified in the store: 2 `best_fold` + 2 `worst_fold` imputation children, all tagged.
- **Every per-population and per-rate pooled number reaches MLflow.** `impute/masked/*`,
  `impute/induced/*`, `impute/induced/null_token/*` and `impute/masked/rate_<pct>/*` all land
  in `FoldResult.metrics`, so all are summarised into
  `cv/test/.../{mean,ci95_lower,ci95_upper,std,min,max,fold_count}` and all appear in
  `raw_fold_metrics.csv` and `cv_summary.json`. The headline does not hide the populations;
  what it hides is the denominator (F-06-2).
- **The induced population's cell accounting reconciles exactly on real data.** From the logged
  row for `credit-g_20nan`: masked cells 98 + 185 = 283, `realised_rate = 0.17900063251106893`,
  so eligible = 283 / 0.179... = 1581.0 exactly. Induced cells 151 + 268 = 419.
  1581 + 419 = 2000 = 100 test rows x 20 features. Probe 4 counted the real NaN cells on that
  test split: **419**. The `realised_rate` definition and the induced population size are both
  verifiable from the logged metrics alone.
- **The preview's `[MASK]` claim is truthful for the masked population.** `preprocess_table`
  sets `dynamic_mask[null_values] = False` and the "at least one mask per row" fallback draws
  only from `non_null_indices` (`src/utils.py:59-74`), so a scored `masked` cell was always
  observed in the variant. The header's "distinguishes a cell hidden for scoring from one the
  dataset was already missing" is accurate for that population. (Known finding 4 covers the
  `induced_null_token` label; F-06-3 covers the duplicate.)
- **`_to_original_units` indexes the scaler correctly.** `order = list(numerical_columns)` and
  `scaler.fit_transform(frame[numerical_columns])` (`data.py:71`) use the same list in the same
  order, so `scaler.scale_[index]` / `mean_[index]` match the column. A mismatch here would
  have un-scaled every preview number against the wrong column.
- **The ledger has enough precision to reproduce a score.** `ledger.to_csv` uses pandas'
  default float formatting (`repr`), and `actual`/`imputed` arrive as Python floats from
  `.tolist()` on float32 tensors, so they round-trip exactly. `rmse_num_z`, `mae_num_z`,
  `acc_cat` and `macro_f1_cat` are all recomputable from `imputation_fold_N_cells.csv`.
  `impute_score` is not - see F-06-2.
- **An empty scored-cell frame does not crash the preview.** Probe 2:
  `write_imputation_preview` on a zero-row frame emits `_No cells were scored._` rather than
  raising out of `generator.choice`.
- **`optuna/objective_value` shares one key across tasks with opposite directions** -
  `f1_macro` (maximize, test split) for classification, `validation/impute/masked/impute_score`
  (minimize, validation split) for imputation. Not reported: `README.md:381` states it
  explicitly, `tags.task` and `tags.is_optuna` separate the runs, and the store currently holds
  Optuna trials for one task only.

## Open questions

- **Do the five clobbered `metrics/*.csv` files matter, or were those classification baselines
  already published elsewhere?** The MLflow store has 87 classification parents, so the numbers
  probably survive there - but `metrics/` is where `compute_cv_summary` reads from and where a
  reader looks first. Settle by checking whether each of the five variants has a
  `run_role = parent`, `task = classification` run whose `cv/test/f1_macro/mean` matches what
  the file used to say; if not, they are gone.
- **How often does a real fold actually lose a metric key (F-06-6)?** I reproduced the abort
  synthetically but could not trigger it from data. Settle by logging the per-population
  `n_num_cells`/`n_cat_cells` of every fold of an `_80nan` run with a small
  `EVAL_MASK_RATES_EXTRA` entry and checking whether any lands at zero.
- **Whether the `results/` artifacts for non-diagnostic folds are discoverable from MLflow at
  all.** `retained_artifact_paths` gives the local `source_path` of the two selected folds,
  which reveals the timestamped directory, but nothing logs that directory as a param or tag.
  Settle by checking whether a reader starting from a parent run in the MLflow UI can reach
  `results/<dataset>/<timestamp>/imputation/imputation_fold_3_cells.csv` without guessing the
  timestamp.
