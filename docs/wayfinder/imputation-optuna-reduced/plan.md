# Reduced Optuna Search for Imputation: Implementation Plan

> **How to execute:** Each task below has an execution ticket under
> [`docs/tickets/imputation-optuna-reduced/`](../../tickets/imputation-optuna-reduced/spec.md);
> pick a ticket by path and work it with `mattpocock-skills:tdd`. Every task is shaped as
> one red-green cycle: write the failing tests, run them to see them fail, implement the
> minimum, run them to see them pass, then run a short real study or training as the task
> names, commit. Respect the task order: the four code tasks all edit `opt.py`, so they
> are serialised, and the studies (task 5) need all four. Steps use checkbox (`- [ ]`)
> syntax for tracking. The wayfinder map that produced this plan is [`map.md`](map.md);
> [ADR 0005](../../adr/0005-reduced-optuna-search-for-imputation.md) is the specification.

**Goal:** A `--search_space reduced` profile that samples only `PROB_MASCARA`,
`LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and, on mixed tables, `LAMBDA_NUM`; an
imputation search objective scored on the validation split; a task-keyed promoted
configuration written only under `--promote_best`; an importance artifact per study; then
six studies and twelve comparison runs that say whether tuning helped.

**Architecture:** ADR 0005 is the specification; this plan orders its decisions so every
task leaves the study runnable. The profile lands first because every later task reads
the profile-aware `define_search_space`. The validation objective is a programmatic
request field, so ordinary runs never carry it. The promotion path replaces both writes
to `datasets/hiperparams/` at once, so the shared file is never half-governed by the
flag. The launcher script is the only place the six study commands and the twelve
comparison commands are spelled out.

**Tech Stack:** Python 3.10, PyTorch, Optuna, MLflow, pandas, pytest, uv, PowerShell for
the launcher.

## Global Constraints

- Use Python 3.10 through `uv run --python 3.10 <command>`.
- Classification training behaviour is untouched. `tests/integration/test_vehicle_regression.py`
  and `tests/integration/test_credit_g_imputation_regression.py` pass **with no edit**
  after every task that touches the training package.
- Every new function argument, dataclass field and flag defaults to today's behaviour:
  `define_search_space(trial, task="classification", profile="full", mixed_columns=True)`
  must reproduce the current full space; the request field for the validation score
  defaults to off; `--search_space` defaults to `None` and resolves per task.
- `src/training/data.py` is a protected file: the column-mix extraction is pure and
  changes no draw, no order and no value.
- Tests at pre-agreed public seams only; expected values are independent literals.
- `_DisabledMlflow` in `opt.py` must grow whatever MLflow calls a task adds
  (`log_dict`, `log_artifact`), or the disabled-tracking test breaks.
- Real runs in a task go to a scratch tracking store (`MLFLOW_TRACKING_URI=sqlite:///C:/...`,
  drive letter and forward slashes) and a scratch `--output_dir` / `--metrics_dir`, and
  their artifact folders under `mlruns/` are removed afterwards; the shared `mlflow.db` is
  touched only by task 5 and task 6, which are the real experiments.
- Preserve unrelated worktree changes (`metrics/`, `results/`, MLflow files, user edits).
- Commit after each task, only that task's files, with the session's attribution lines.

## File structure

| File | Responsibility |
|---|---|
| `src/training/types.py` | `SEARCH_SPACE_PROFILES`; `TaskSpec.search_objective`; `TrainingRequest.score_search_objective` (programmatic, defaulted) and `TrainingRequest.config_source`. |
| `src/training/config.py` | `--search_space`, `--promote_best`; parse-time rejection of `reduced` with classification; task-keyed lookup in `_load_base_hyperparameters` recording the source; `resolve_training_request` reads the programmatic field. |
| `src/training/data.py` | `declared_column_types(columns, label_column, base_dataset_name)` extracted from `prepare_dataset`, pure. |
| `src/training/decoding.py` | Validation-split scoring behind the request field; `validation/` metric family. |
| `src/training/tracking.py` | `config_source` logged as a param at both call sites. |
| `src/mlflow_utils.py` | `SEARCH_SPACE_TAG`. |
| `opt.py` | Profile-aware `define_search_space`; column mix once per study; `search_space` tag and param; objective from `search_objective` and the request field; study-directory-only running best; `--promote_best` writing the complete configuration; importance artifact; `__main__` on `build_training_parser`. |
| `imputation_studies.ps1` | Launcher: six studies with promotion, twelve comparison runs, `-DryRun`. |
| `tests/unit/test_opt_search_space.py`, `tests/unit/test_opt_tracking.py`, `tests/unit/test_training_config.py`, `tests/unit/test_training_tasks.py`, `tests/unit/test_training_decoding.py`, `tests/unit/test_training_data.py` | The seams named per task. |
| `README.md`, `docs/BACKLOG.md`, `docs/adr/0005-...md` | Flags, lookup order, profile, MLflow filters; I2 closed; results recorded in the ADR's status. |

## Task 1: Search-space profiles

**Files:**

- Modify: `opt.py`, `src/training/types.py`, `src/training/config.py`, `src/mlflow_utils.py`, `src/training/data.py`
- Modify tests: `tests/unit/test_opt_search_space.py`, `tests/unit/test_opt_tracking.py`, `tests/unit/test_training_config.py`, `tests/unit/test_training_data.py`

**Interfaces:**

- `define_search_space(trial, task="classification", profile="full", mixed_columns=True) -> dict`.
- `declared_column_types(columns, label_column, base_dataset_name) -> tuple[list[str], list[str]]` in `data.py`; `prepare_dataset` calls it and behaves identically.
- `--search_space {full,reduced}` on `build_training_parser`, default `None`; `opt.run_hyperparameter_optimization` resolves `None` to `reduced` for imputation and `full` otherwise, and computes the column mix once from the dataset's header and the declared list.
- `SEARCH_SPACE_TAG = "search_space"`; tag plus param on the study parent and every trial.
- `opt.py`'s `__main__` uses `build_training_parser()` (with `--use_optuna` implied) so it accepts every flag `main.py` accepts.

- [ ] **Step 1: Write failing tests**

~~~
def test_the_reduced_profile_samples_only_what_moves_the_decoder():
    # imputation + reduced + mixed: exactly {PROB_MASCARA, LR_DECODE, WEIGHT_DECAY_DECODE, DROPOUT, LAMBDA_NUM}
    # imputation + reduced + not mixed: the same without LAMBDA_NUM
    # everything else takes the dataclass default, so no DIM/HEADS/EPOCHS_* key is present

def test_the_full_profile_is_the_space_decided_before_this_effort():
    # imputation + full + mixed reproduces the fifteen keys (with LAMBDA_NUM); classification + full the fourteen

def test_the_decode_learning_rate_can_exceed_its_default_in_both_profiles():
    # study.trials[0].distributions["LR_DECODE"].high == 1e-2 for reduced and for full

def test_lambda_num_is_sampled_only_where_both_column_types_exist():
    # full + not mixed omits LAMBDA_NUM

def test_reduced_is_refused_for_classification_at_parse_time():
    # validate_parsed_args(--search_space reduced --task classification) -> SystemExit naming both flags

def test_declared_column_types_follow_the_pipelines_declaration_not_dtypes():
    # electricity: (["day"], seven numericals); vehicle: ([], eighteen); credit-g: (13, 7)

def test_every_run_a_study_opens_says_which_profile_it_sampled():
    # temporary backend: study parent and trials carry search_space tag and param;
    # imputation without the flag -> "reduced", classification without the flag -> "full"

def test_opt_accepts_every_training_flag():
    # opt.build_parser().parse_args(["--dataset_name", "x", "--task", "imputation", "--search_space", "reduced"])
~~~

- [ ] **Step 2: Run them to see them fail**

`uv run --python 3.10 pytest tests/unit/test_opt_search_space.py tests/unit/test_opt_tracking.py tests/unit/test_training_config.py tests/unit/test_training_data.py -q`

- [ ] **Step 3: Implement**

`types.py`: `SEARCH_SPACE_PROFILES`. `config.py`: the flag and the rejection. `data.py`: extract the helper; `prepare_dataset` calls it. `mlflow_utils.py`: the tag. `opt.py`: the profile branch (reduced returns only sampled keys; full keeps every key with the two corrections), the once-per-study column mix (header via `pd.read_csv(nrows=0)` on the dataset spec's source path), the tag and param, `__main__` on the training parser.

- [ ] **Step 4: Verify, including the classification fixture**

`uv run --python 3.10 pytest -m "not integration" -q`, then `uv run --python 3.10 pytest -m integration -q` (the `data.py` extraction must leave both fixtures unedited). Then one real two-trial reduced study on `credit-g_20nan` against a scratch store; confirm the trial params show only the sampled keys as `best_` params and the study carries `search_space = reduced`.

- [ ] **Step 5: Commit**

## Task 2: The search objective on the validation split

**Files:**

- Modify: `src/training/types.py`, `src/training/decoding.py`, `src/training/config.py`, `opt.py`
- Modify tests: `tests/unit/test_training_tasks.py`, `tests/unit/test_training_decoding.py`, `tests/unit/test_opt_tracking.py`

**Interfaces:**

- `TaskSpec.search_objective`: `"validation/impute/masked/impute_score"` for imputation, `"f1_macro"` for classification.
- `TrainingRequest.score_search_objective: bool = False`, programmatic only (`resolve_training_request` reads `getattr(args, "score_search_objective", False)`); `opt.py` sets it on imputation trial namespaces only.
- `train_and_evaluate_decoder(..., score_search_objective=False)`: when on, scores `hidden_validation` against `clean_validation` with the training-fold baselines, logs `validation/impute/masked/<name>` through the tracker, and adds `validation/impute/masked/impute_score` to the fold metrics; when off, no `validation/` key exists anywhere.
- `ObjectiveFunctionWrapper` reads `task.search_objective` instead of `ranking_metric`.

- [ ] **Step 1: Write failing tests**

~~~
def test_each_task_declares_its_search_objective():
    # imputation -> validation/impute/masked/impute_score; classification -> f1_macro

def test_the_decode_stage_scores_the_validation_split_only_when_asked():
    # tiny synthetic frame: with the flag, metrics hold validation/impute/masked/impute_score
    # and the tracker saw validation/impute/masked/impute_score; without it, no key starts with "validation/"

def test_an_imputation_trial_is_ranked_by_its_validation_score_not_its_test_score():
    # stub returns validation 0.3 and test 0.9 for one trial, 0.6 / 0.2 for the other;
    # optuna/best_objective_value == 0.3 and best_trial_number == that trial;
    # every imputation trial namespace has score_search_objective True, the retrain does not
~~~

- [ ] **Step 2: Run them to see them fail**

- [ ] **Step 3: Implement**

The scorer already has `_score_population` and `score_cells`; the validation pass reuses them on the encodings the loss already holds. Keep the `test/` prefixing call untouched; log the `validation/` family in its own call.

- [ ] **Step 4: Verify, including both fixtures**

Unit suite, both fixtures, then a real two-trial reduced study on `credit-g_20nan` (scratch store): `optuna/objective_value` on each trial equals the trial's `validation/impute/masked/impute_score`, and the trial's `test/impute/masked/impute_score` differs from it.

- [ ] **Step 5: Commit**

## Task 3: Task-keyed lookup and explicit promotion

**Files:**

- Modify: `src/training/config.py`, `src/training/types.py`, `src/training/tracking.py`, `opt.py`, `README.md`, `docs/BACKLOG.md`
- Modify tests: `tests/unit/test_training_config.py`, `tests/unit/test_training_tracking.py`, `tests/unit/test_opt_tracking.py`

**Interfaces:**

- `_load_base_hyperparameters(args)`: for `--task imputation`, `datasets/hiperparams/<base>/<dataset>.imputation.json`, then `<dataset>.json`, then defaults; classification unchanged. It records what it used on `TrainingRequest.config_source` (`"defaults"`, `"override"`, or the relative path), logged as the `config_source` param at both tracker call sites.
- `--promote_best` (store_true): at the end of the study, write the **complete** configuration (the task's key set from `logged_hyperparameters` with resolved values, `LR_SCHEDULER` included) to the task-keyed file for imputation or the shared file for classification. Without it, nothing outside the study directory is written for either task; `save_best_params` always writes inside the study directory.

- [ ] **Step 1: Write failing tests**

~~~
def test_an_imputation_run_prefers_its_own_configuration_and_falls_back_to_the_shared_one():
    # tmp cwd with both files: imputation loads .imputation.json; classification loads the shared one;
    # only the shared file present: imputation loads it; none: defaults

def test_every_run_says_where_its_configuration_came_from():
    # temporary backend: params.config_source is "defaults" / the relative path / "override"

def test_a_classification_study_no_longer_writes_the_shared_file_on_its_own():
    # stubbed classification study, no flag: datasets/hiperparams absent; running best inside the study dir

def test_promotion_writes_a_complete_configuration_where_the_task_will_find_it():
    # imputation + --promote_best: <dataset>.imputation.json exists, holds every imputation key
    # (held values included) and LR_SCHEDULER == the study's schedule; the shared file is absent
    # classification + --promote_best: the shared file holds every classification key
~~~

- [ ] **Step 2: Run them to see them fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Verify, then a real promotion**

Unit suite and both fixtures. Then a two-trial reduced study on `credit-g_20nan --promote_best` against a scratch tracking store. The promoted path is relative to the repository, so the file lands in the real `datasets/hiperparams/`: confirm `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json` appears with the complete key set, then delete it, because the real promoted files come from task 5. Document the lookup order in `README.md` ("Configuration System") and mark I2 fixed in `docs/BACKLOG.md`.

- [ ] **Step 5: Commit**

## Task 4: The importance artifact

**Files:**

- Modify: `opt.py`
- Modify tests: `tests/unit/test_opt_tracking.py`

**Interfaces:**

- After `study.optimize`, `optuna.importance.get_param_importances(study)` logged as `optuna/importance/<knob>` metrics and an `importance.json` artifact on the study parent; guarded (`try`/`except`, a warning) so a study with fewer than two completed trials or one distinct parameter logs nothing and does not crash. `_DisabledMlflow` gains `log_dict` and `log_artifact` no-ops.

- [ ] **Step 1: Write failing tests**

~~~
def test_a_finished_study_ranks_the_knobs_it_sampled():
    # stubbed three-trial study: optuna/importance/<knob> present for every sampled knob, values sum to ~1, importance.json listed among the study's artifacts

def test_a_study_too_small_to_rank_still_finishes():
    # one-trial study: no importance metric, no exception, disabled tracking still runs
~~~

- [ ] **Step 2: Run them to see them fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Verify**

Unit suite; a real three-trial reduced study on `credit-g_20nan` (scratch store) shows the importances on the study parent.

- [ ] **Step 5: Commit**

## Task 5: The launcher and the six studies

**Files:**

- Create: `imputation_studies.ps1` (root, beside `experiment.ps1`, same `[CmdletBinding()]` / `-DryRun` shape)

- [ ] **Step 1: Write the launcher**

Parameters: `-Datasets @("credit-g", "kr-vs-kp", "spambase")`, `-Variants @("20nan", "40nan")`, `-Trials 40`, `-Seed 42`, `-Scheduler cosine`, `-DryRun`, `-Compare` (switch: run the twelve comparison runs of task 6 instead of the studies). Each study:

~~~
uv run --python 3.10 python main.py --dataset_name <base>_<variant> --task imputation --use_optuna --search_space reduced --n_trials <Trials> --lr_scheduler <Scheduler> --seed <Seed> --promote_best
~~~

Each comparison pair (task 6): the same `--task imputation --cv_folds 5 --lr_scheduler cosine --seed 42` twice, once with the promoted file in place and once with it moved aside (`config_source` distinguishes them in the store; the script restores the file afterwards).

- [ ] **Step 2: Dry-run, then run the six studies**

`.\imputation_studies.ps1 -DryRun`, then `.\imputation_studies.ps1`. Against the shared `mlflow.db`: these are the experiment. Budget about nineteen hours; the studies are independent, so they may be split across sessions or machines with the same seed.

- [ ] **Step 3: Record**

Six promoted files under `datasets/hiperparams/`; in the execution ticket's comments, per study: the best objective, the winning configuration, the importances. Commit the six files.

## Task 6: The comparison and the documentation

**Files:**

- Modify: `README.md`, `docs/BACKLOG.md`, `docs/adr/0005-reduced-optuna-search-for-imputation.md`

- [ ] **Step 1: Run the twelve comparison runs**

`.\imputation_studies.ps1 -Compare`. About two hours.

- [ ] **Step 2: Build the table**

One row per pair: `cv/test/impute/induced/impute_score/mean` with its interval, `cv/test/impute/masked/impute_score/mean`, for the promoted and the default run, filtered as ADR 0005's "How to compare" says. Record it in the ADR's status section with the date, and say per pair whether the promoted interval excludes the default mean.

- [ ] **Step 3: Documentation**

README: `--search_space`, `--promote_best`, the lookup order, `search_space` and `config_source` in "MLflow Cross-Validation Comparisons", the validation-scored objective note on the study run. BACKLOG: I2 in "Fixed already" (if task 3 did not already do it). `graphify update .`.

- [ ] **Step 4: Review specification coverage**

Walk ADR 0005's seven decisions and confirm each has a test or a recorded run.

- [ ] **Step 5: Commit**
