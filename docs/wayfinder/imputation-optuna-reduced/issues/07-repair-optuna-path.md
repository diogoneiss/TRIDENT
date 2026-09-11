# 07. Task: repair the imputation Optuna path and prove it with a two-trial study

Type: task
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (AFK), commit `161fc92`
Blocked by: none

## Question

The shipped imputation Optuna path (previous map ticket 13, commit `356bcca`) cannot
score a single trial. Repair it, so that ticket 08's plan can build on a path that works
and so that the kr-vs-kp cost estimate in ticket 03 becomes a measurement. Nothing here is
a decision; every item was verified against the code while charting.

**Defects to repair, all in `opt.py` unless noted:**

1. **The task never reaches a trial.** `ObjectiveFunctionWrapper.__call__` builds the
   trial namespace without `task`; `resolve_training_request` (`src/training/config.py`)
   falls back to `DEFAULT_TASK`, so every trial trains a classification model with the
   decode hyperparameters ignored, then `metrics[self.ranking_metric]` raises `KeyError`
   on `impute/masked/impute_score` and the trial is pruned. A 50-trial imputation study
   ends with zero scored trials. `final_args` in `run_hyperparameter_optimization`
   (`--retrain_best`) has the same gap.
2. **The end-of-study write ignores the task.** The block commented "2. Also save to the
   standard hiperparams directory for model loading" writes
   `datasets/hiperparams/<base>/<dataset>.json` unconditionally, so a finished imputation
   study overwrites that dataset's classification configuration. Only the per-trial
   `save_best_params` honours ticket 13 decision 5. Repair to the same rule as
   `save_best_params` (study directory only for imputation); ticket 04's `--promote_best`
   later replaces both writes and is *not* part of this ticket.
3. **The study parent carries neither `task` nor `is_optuna`.** It builds tags with
   `build_run_tags` directly; `_execution_tags` in `src/training/tracking.py` already sets
   both for every other run kind, and ADR 0004 decision 7 names the study parent as the
   third call site. Build the study parent's tags through the helper.
4. **Unseeded sampler.** `optuna.create_study` takes no sampler; pass
   `TPESampler(seed=args.seed)` (ticket 03).
5. Cosmetic: the final log line says "Best F1 macro" whatever the task; name the ranking
   metric.

**Why the tests did not catch it**: `tests/unit/test_opt_search_space.py` exercises
`define_search_space` and `save_best_params` in isolation, and
`tests/unit/test_opt_tracking.py` stubs `train_main`, so no test ever built the trial
namespace and read `task` from it. The new tests must go through the namespace the
objective actually builds (a stubbed `train_main` that records the namespace it received
is enough) and must assert the study parent's tags in the temporary MLflow backend.

**Resolution criterion, in this order**:

1. Failing tests for defects 1 to 4, then green, under `mattpocock-skills:tdd`; the unit
   suite and both integration fixtures unchanged and passing.
2. **A real two-trial study** against a scratch tracking store so the shared `mlflow.db`
   stays clean:

   ```
   MLFLOW_TRACKING_URI=sqlite:///<scratchpad>/proof.db \
   uv run --python 3.10 python main.py --dataset_name credit-g_20nan --task imputation \
     --use_optuna --n_trials 2 --lr_scheduler cosine \
     --output_dir <scratchpad>/results --metrics_dir <scratchpad>/metrics
   ```

   Expected: both trials `trial_status = success` with `optuna/objective_value` logged;
   the study parent tagged `task = imputation`, `is_optuna = true`; every trial tagged the
   same; `best_hyperparameters.json` inside the study directory; `datasets/hiperparams/`
   untouched (it holds no file today, so any file appearing is the clobber). This runs the
   *current* full imputation space (`EPOCHS_PRE` 20..60), so it costs seconds per trial.
3. **Time one default-configuration single-split imputation run on each of
   `credit-g_20nan` and `kr-vs-kp_20nan`**
   (`--task imputation --lr_scheduler cosine --disable_mlflow --metrics_dir <scratchpad>`)
   and record `time/pretrain_seconds` plus `time/decode_seconds` for both here. The
   credit-g figure against the measured 38.6 s per 2-fold fold gives the
   single-split-to-fold cost ratio the budget in ticket 03 is missing (a fold trains on
   about 40% of the rows, the single split on 80%); the kr-vs-kp figure replaces the 150 s
   estimate. Update ticket 03's table, the map's cost fact and the Decisions line, and say
   whether the six-study total lands nearer eleven or twenty-two hours.

The answer records the commit, the two trials' objective values, the study parent's tags
as read back from the scratch store, and the kr-vs-kp timing.

## Answer

Resolved 2026-09-10, commit `161fc92` on `feat/imputation-task`: five red-green slices in
`tests/unit/test_opt_tracking.py`, unit suite 141 green, both integration fixtures
unedited and green.

**Repaired, defects 1 to 5 from the question:**

| Defect | Repair | Test that pins it |
|---|---|---|
| 1. task never reaches a trial | `args.task = self.task` on the trial namespace, `final_args.task = task.name` on the retrain | an imputation study passes `task=imputation` to every trial and the retrain (the stub returns the metrics of whichever task it was asked for, so the red run reproduced the production symptom: both trials pruned on the missing key) |
| 2. end-of-study write ignores the task | same rule as `save_best_params`: imputation stays inside the study directory | no `datasets/hiperparams/` after a finished imputation study; `best_hyperparameters.json` inside the study directory |
| 3. study parent lacks `task` / `is_optuna` | `_execution_tags` promoted to `execution_tags` (`src/training/tracking.py`) and used for the study parent; the pre-runner trial tags carry both too, so a trial that crashes before the runner is still a run of a known task | both tags on the study parent and every trial, read back from the temporary store |
| 4. unseeded sampler | `TPESampler(seed=args.seed)` | two studies with one seed sample identical trials (separate output directories, because the study storage is named to the second and a second study in the same second resumes the first) |
| 5. "Best F1 macro" | names the ranking metric | none, cosmetic |

**A sixth defect, found by the proof run and repaired in the same commit.** Since the
head-count constraint (commit `356bcca`), `define_search_space` sampled the per-head width
under the name `DIM` and multiplied it by `HEADS`, so Optuna's own record of `DIM` was the
multiplier: the proof's trial 0 logged `DIM: 22` with 8 heads for a trained width of 176.
`study.best_params` fed `--retrain_best`, `best_hyperparameters.json`, the shared-config
write and the `best_` params on the study parent, so a retrain built a model the winning
trial never ran, with a head count that no longer divided the width. `--retrain_best` was
therefore broken for **both** tasks between `356bcca` and `161fc92`; no run in the store
used it. Now the sampled value is named `HEAD_DIM`, the trained configuration is stored on
the trial as the `hyperparameters` user attribute, and every consumer reads
`study.best_trial.user_attrs["hyperparameters"]`. Test: the retrain namespace, the saved
file and the study's `best_DIM` all equal the winning trial's trained configuration.
Ticket 08 records the rename in the ADR and the broken interval in `docs/BACKLOG.md`.

**Proof run** (`main.py --dataset_name credit-g_20nan --task imputation --use_optuna
--n_trials 2 --lr_scheduler cosine`, scratch store `scratchpad/proof/proof.db`, the
current full space): trial 0 objective 1.0099, trial 1 1.0175 (tiny random configurations
at 30 to 50 epochs, near baseline parity as expected), both `trial_status = success`. Read
back with `MlflowClient`: study parent `optuna_credit-g_20nan_20260910_231828` tagged
`run_role = optuna_study`, `task = imputation`, `is_optuna = true`, `lr_scheduler = cosine`,
`n_trials = 2`, with `optuna/best_objective_value = 1.0099` and
`optuna/best_trial_number = 0`; both trials `task = imputation`, `is_optuna = true`,
`best_trial = true` on trial 0. `best_hyperparameters.json` and `credit-g_20nan.json` sit
inside `optuna_20260910_231828/`; `datasets/hiperparams/` holds no file; the log says
"Imputation study: not writing datasets/hiperparams, which classification reads".

**Timing** (single split, defaults, `cosine`, RTX 3050 Laptop GPU). Recorded against a
scratch store `scratchpad/timing/timing.db` rather than with `--disable_mlflow` as the
question said, because stage timings are tracked metrics that only the tracker writes.

| Run | Pre-training | Decode | Total | `impute_score` masked / induced |
|---|---|---|---|---|
| credit-g_20nan | 34.4 s | 31.5 s | 65.9 s | 0.896 / 0.932 |
| kr-vs-kp_20nan | 141.2 s | 151.9 s | 293.1 s | 0.724 / 0.667 |

credit-g against the 38.6 s per 2-fold fold in the store gives a ratio of 1.7; spambase
projects to about 500 s per trial; the six studies at 40 trials total about nineteen
hours. Ticket 03's table, the map's cost fact and the Decisions line are updated. Two
observations for ticket 08: the decode stage logs its timing under `time/finetune_seconds`
(no decode key exists), which the ADR should state; and kr-vs-kp at the defaults already
scores 0.72 masked, well below parity, so the mode baseline is weak on that table.

## Comments
