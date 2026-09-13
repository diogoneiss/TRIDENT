# 07 - The Optuna search for imputation

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `opt.py` (all 603 lines, close reading of 81-150, 187-325, 328-346, 349-368,
371-533), `docs/adr/0005-reduced-optuna-search-for-imputation.md` (all),
`docs/tickets/imputation-optuna-reduced/issues/02-validation-search-objective.md` (all),
`tests/unit/test_opt_search_space.py` (1-60, headings), `tests/unit/test_opt_tracking.py`
(headings + 382-460). Supporting reads outside the dimension, for the objective and the
promotion path only: `src/training/config.py:21-135`, `src/training/types.py:18-180`,
`src/training/decoding.py:32-230,279-340`, `src/training/imputation_metrics.py` (all),
`src/training/data.py:132-230`, `tests/integration/test_credit_g_imputation_regression.py`,
`experiment_imputation.ps1` (uncommitted).

**Method:** graphify orientation, then source. Then five empirical checks, all read-only:

1. Resolved the real config lookup with the working tree as it stands
   (`load_hyperparameters` on `credit-g_20nan`, both tasks).
2. Reproduced the cross-task fallback in an isolated scratch cwd with a synthetic
   `foo_20nan.json`.
3. Counted the cells the search objective is actually computed on, by calling the real
   `prepare_dataset` / `build_folds` / `evaluation_mask` on the real tables, at the
   parameters the studies actually ran with: `seed = 42` and `fold_ordinal = 1`
   (`src/training/runner.py:67` enumerates folds from 1; the seed is from the launch log,
   `results/imputation_studies_20260911_011008.out`, which records
   `--search_space reduced --n_trials 40 --lr_scheduler cosine --seed 42 --promote_best`
   for all five studies).
4. Reproduced Optuna's fANOVA behaviour on a `HEAD_DIM`-style dependent range with the
   installed optuna 4.9.0.
5. **Read the user's own five Optuna study databases under `results/*/optuna_*/optuna_study.db`
   read-only** (`sqlite3` with `?mode=ro`, never `optuna.load_study`, which would migrate
   them). Four finished 40-trial studies plus one in flight. This is what turns the
   headline finding from an argument into a measurement.

No training run; nothing here needed one. No pytest. What I could not check: whether the
validation objective correlates at all with the headline `cv/test/impute/induced/impute_score`
— nothing in the repo measures that, and it is the open question below.

## Findings

### F-07-1 - The search objective cannot resolve the configurations it is ranking; two of the four promoted files were decided by a tie-break, not by the objective

- **Kind:** methodology
- **Severity:** critical
- **Where:** `opt.py:266` (`score = metrics[self.search_objective]`), objective defined at
  `src/training/types.py:71`, computed at `src/training/decoding.py:204-210`
- **Evidence:** the objective is `validation/impute/masked/impute_score`, scored on the
  cells `evaluation_mask` hides in the predefined split's validation rows. That validation
  split is 10% of each table, so the population is tiny. Measured with the real code:

  | variant | val rows | scored validation cells |
  |---|---|---|
  | `credit-g_20nan` | 100 | **283** (187 categorical, 96 numerical) |
  | `credit-g_40nan` | 100 | **179** (118 categorical, 61 numerical) |
  | `kr-vs-kp_20nan` | 320 | 1511 (all categorical) |
  | `kr-vs-kp_40nan` | 320 | 910 (all categorical) |
  | `spambase_20nan` | 461 | 3390 (all numerical) |

  On an all-categorical table `impute_score` collapses to `(1 - acc_cat) / naive_error`,
  which is a count of wrong cells divided by a constant — a lattice, not a continuum.
  The user's own finished studies show exactly that:

  ```
  study                  trials  distinct objective values   best == 2nd?
  credit-g_20nan            40            40                    no
  credit-g_40nan            40            40                    no
  kr-vs-kp_20nan            40            20                    YES
  kr-vs-kp_40nan            40            20                    YES
  spambase_20nan (running)  11            11                    no
  ```

  `kr-vs-kp_20nan` top of the table, exact values from `trial_values`:

  ```
  trial 17  obj 0.67032967033  DROPOUT=0.2  LR_DECODE=0.001742  PROB_MASCARA=0.2  WD=0.003901
  trial 29  obj 0.67032967033  DROPOUT=0.2  LR_DECODE=0.005927  PROB_MASCARA=0.5  WD=0.005487
  trial  0  obj 0.673992673993 DROPOUT=0.3  LR_DECODE=0.007969  PROB_MASCARA=0.3  WD=0.00157
  trial 27  obj 0.673992673993 DROPOUT=0.2  LR_DECODE=0.006552  PROB_MASCARA=0.5  WD=0.00993
  ```

  Consecutive distinct values differ by exactly `1/273 = 0.0036630`. The arithmetic closes
  exactly: 1511 scored cells, the mode baseline wrong on 273 of them, so one extra
  correctly imputed cell moves `impute_score` by `(1/1511)/(273/1511) = 1/273`, and the
  winner's `0.67032967033 = 183/273` means it got 183 of 1511 cells wrong. So **the top
  four configurations of a 2-hour-39-minute, 40-trial study are separated by one cell out
  of 1511; the top two by zero.** `study.best_trial` breaks the tie on trial order, so
  trial 17 won over trial 29 — a configuration with a 3.4x lower decode learning rate and
  a mask rate of 0.2 instead of 0.5. The promoted file confirms it:
  `datasets/hiperparams/kr-vs-kp/kr-vs-kp_20nan.imputation.json` holds
  `LR_DECODE = 0.0017418753352176958`, trial 17's value. `kr-vs-kp_40nan` is the same story
  at a coarser quantum: 910 scored cells, baseline wrong on 179, quantum `1/179 = 0.005587`,
  best and 2nd both exactly `0.648044692737` (116 of 910 wrong), promoted
  `LR_DECODE = 0.0016986...` = trial 32, over trial 38's `0.002784`. Wall clock from the
  launch log: `kr-vs-kp_20nan` 02:37:23 to 05:16:24, `kr-vs-kp_40nan` 05:16:38 to 07:57:20.

  `credit-g` does not tie only because the numerical RMSE term is continuous, but its
  top-five band is 0.0384 wide against a whole-study span of 0.219 — and the two variants
  of the *same base table* promoted `LAMBDA_NUM = 2.74` and `LAMBDA_NUM = 0.23`, a 12x
  disagreement about the loss balance of the same twenty columns, with `PROB_MASCARA` 0.4
  vs 0.2 and `DROPOUT` 0.3 vs 0.2. That is what an unidentified argmax looks like.
- **Consequence:** ADR 0005 decision 4 commits nineteen GPU-hours to six studies, and
  decision 6 then reports whether "the tuning helped". For `kr-vs-kp_20nan` and
  `kr-vs-kp_40nan` — two of the three protocol datasets, and the only all-categorical one
  — the promoted configuration is an arbitrary pick among tied trials. Any downstream
  claim that the promoted configuration beats the defaults on those variants is a claim
  about a coin flip. `credit-g`'s promotion is not tied but is chosen on 283 and 179
  cells respectively, and nothing anywhere measures whether the ordering survives a
  second mask draw.
- **Direction:** the objective needs more cells before it needs more trials. Three levers,
  cheapest first: score the objective over several `evaluation_mask` draws (the helper
  already takes a `fold` argument and restores the global RNG, so `mean` over
  `fold = 0..k` costs k forward passes, not k trainings); or score the objective on the
  *union* of validation and training-fold held-out cells; or run trials at `--cv_folds`
  and average the per-fold validation score. Separately, a tie in `study.best_trial`
  should be reported, not silently broken — log the count of trials within one objective
  quantum of the winner on the study parent, and refuse `--promote_best` when that count
  is greater than one.

### F-07-2 - The objective is measured on exactly the cells that chose the checkpoint, so `optuna/best_objective_value` is a minimum-of-minima, not a validation score

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/decoding.py:76-79` (the fixed `hidden_validation`), `:114-127`
  (per-epoch loss on it, `best_state` kept at its argmin), `:204-206` (the objective
  scored on the same two tensors)
- **Evidence:** `hidden_validation` / `clean_validation` are encoded once and never
  re-rolled. Every decode epoch computes `model(hidden_validation, clean_validation)` and
  keeps `best_state` at the epoch with the lowest value; the loop then restores that
  checkpoint and `score_search_objective` scores **the same tensors**:

  ```python
  validation_cells = _score_population(model, hidden_validation, clean_validation, "masked")
  ```

  So the trial's reported objective is the score at the epoch whose *loss* on those same
  cells was lowest — a selection over epochs made on the very population the number then
  describes — and the study's `optuna/best_objective_value` is a minimum over 40 such
  selected values. At the held `EPOCHS_DECODE = 150` that is 150 epochs of selection per
  trial, 6000 per study, on 283 cells for `credit-g_20nan`. (Loss and `impute_score` are
  not the same functional, so the epoch chosen is not exactly the score's argmin; the
  direction of the bias is the same, because they are monotonically related through the
  same errors on the same cells.)
- **Consequence:** ADR 0005's "How to compare" section says `optuna/best_objective_value`
  "is a **validation-split** score and is not comparable with any `test/` number" — true,
  but it understates the problem: it is not comparable with an honest validation score
  either, and it will read systematically better than the promoted configuration's real
  performance. Anyone reading the study parent and the CV comparison side by side will see
  the CV number come out worse and reach for a story about folds. In the `full` profile
  this becomes a search bias rather than only a reporting one: `EPOCHS_DECODE` is sampled
  20..60 (`opt.py:134`), so a trial with 60 epochs gets three times as many chances at the
  validation minimum as one with 20 and wins partly for that reason. The reduced profile
  holds it at 150 and so is immune to the bias *between* trials, only to the reporting
  one.
- **Direction:** split the decode validation cells once per fold into a selection half and
  a scoring half — two `evaluation_mask` draws with disjoint `fold` arguments, or a row
  split of `fold.validation_indices`. The checkpoint watches one, the objective reads the
  other. Cheap: the tensors are already encoded once per fold and the eval path is a
  single forward pass. Failing that, state on the study parent that the objective is
  best-of-`EPOCHS_DECODE`.

### F-07-3 - There is no way to run the defaults arm of ADR 0005's own comparison protocol once a configuration has been promoted

- **Kind:** design
- **Severity:** high
- **Where:** `src/training/config.py:47-57` (candidate lookup, no opt-out),
  `opt.py:525-530` (`--promote_best`), `src/training/config.py:186-280` (the parser has no
  flag to ignore a promoted file)
- **Evidence:** `_load_base_hyperparameters` prefers the on-disk file whenever it exists
  and there is no CLI escape — backlog I1's `--hyperparams <path>` is explicitly out of
  scope (ADR 0005 "Considered options"). The four promoted files already exist in the
  working tree, so this is not hypothetical:

  ```
  $ python -c "... load_hyperparameters(Namespace(dataset_name='credit-g_20nan', task='imputation'))"
  imputation source: datasets/hiperparams/credit-g/credit-g_20nan.imputation.json
    dropout 0.30000000000000004 prob_mask 0.4 lr_decode 0.002538672990101232 sched cosine
  ```

  ADR 0005 decision 6 prescribes "one 5-fold cross-validation run with the promoted
  configuration against one with the defaults ... so the configuration is the only
  difference". On `credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan` and
  `kr-vs-kp_40nan` the defaults arm cannot be launched today: every
  `main.py --task imputation` invocation on those variants silently picks up the promoted
  file. The only way to get the defaults arm is to move the file out of the way and
  remember to move it back — an untracked, manual, unlogged step in the middle of a
  twelve-run protocol.

  The same precedence silently defeats a tool already in the tree.
  `experiment_imputation.ps1` backs up, overwrites and restores
  `datasets/hiperparams/<base>/<dataset>.json` (`Get-HyperparamsPath`) and prints
  `[setup] <path> -> EPOCHS_PRE=200`, but it runs `--task imputation`, whose first
  candidate is `<dataset>.imputation.json`. Point that script at any promoted variant and
  the setup line is a lie: the run uses the promoted configuration and the script's
  epoch override never reaches it.
- **Consequence:** the headline comparison of the whole effort — promoted vs defaults on
  `cv/test/impute/induced/impute_score/mean` — is either not runnable in the order the
  ADR implies, or runnable only via an off-the-record file move whose success leaves no
  trace in MLflow beyond `params.config_source`. If someone forgets the move, both arms
  are the promoted configuration and the comparison reports "no difference".
- **Direction:** a `--hyperparams {path|defaults}` (or a plain `--ignore_promoted`) flag
  that short-circuits the candidate list, so the defaults arm is a flag rather than a
  filesystem manoeuvre. It is backlog I1 and it is load-bearing for decision 6, not a
  nice-to-have. Whatever the flag, `config_source` already exists to record which arm ran.

### F-07-4 - An imputation run with no promoted file of its own silently loads classification's tuned configuration

- **Kind:** design
- **Severity:** high
- **Where:** `src/training/config.py:47-57`
- **Evidence:**

  ```python
  candidates = [hyperparameter_file(dataset_name, task)]
  if task != DEFAULT_TASK:
      candidates.append(hyperparameter_file(dataset_name, DEFAULT_TASK))
  ```

  Reproduced in an isolated scratch cwd with only a classification file present:

  ```
  imputation run, only a CLASSIFICATION file present -> datasets/hiperparams/foo/foo_20nan.json
    dim 64 dropout 0.45 prob_mask 0.11 epochs_pre 7 sched plateau
  ```

  Every shared key is taken wholesale: architecture, `EPOCHS_PRE`, `PROB_MASCARA`,
  `DROPOUT`, and `LR_SCHEDULER` (which then also sets the imputation run's schedule
  whenever `--lr_scheduler` is omitted). `EPOCHS_DECODE`, absent from a classification
  file, falls back to 150 — so the run trains 7 pre-training epochs and 150 decode epochs,
  a configuration no one chose.

  This is specified behaviour, not an oversight — ADR 0005 decision 5 says
  "`--task imputation` reads `<dataset>.imputation.json` first, then the shared
  `<dataset>.json`, then the defaults" — which is why I file it as design. The history
  explains why it looks harmless: on `main` the loader was a single unconditional path,
  `datasets/hiperparams/<base>/<dataset>.json`, with no task at all
  (`git show main:src/training/config.py`, lines 16-22), so the fallback is literally
  pre-branch behaviour preserved. It was harmless on `main` because there was only one
  task; the branch makes it a cross-task channel.
- **Consequence:** ADR 0005 decision 5 is careful in one direction — "Classification never
  reads the task-keyed file, so promoting an imputation result cannot retune it" — and
  silent in the other. A classification study run with `--promote_best` retunes every
  later imputation run on that variant, including `PROB_MASCARA`, which is the single knob
  the reduced imputation profile exists to tune, and including the learning-rate schedule,
  which ADR 0003 spent an effort making explicit and taggable. `tests/unit/test_opt_tracking.py`
  pins only that promotion writes the right file and leaves the other alone
  (`test_promotion_writes_a_complete_*`); nothing pins that *loading* is isolated per task.
  Concretely: once anyone promotes a classification study for `credit-g_20nan`, decision 6's
  "defaults" imputation arm on that variant stops being the defaults, and `config_source`
  is the only clue — a path that looks perfectly legitimate.
- **Direction:** the fallback bought backward compatibility for a case that cannot exist —
  no shared file was in the tree when ADR 0005 was written, and the ADR says so
  ("no configuration JSON exists under `datasets/hiperparams/`"). Drop it, so an imputation
  run with no imputation file uses the defaults; or, if it is kept, restrict it to the keys
  both tasks share *and* log a warning naming the task the file was written for. Either way
  add the missing test: a promoted classification file must not change what an imputation
  run trains with. The existing `test_promotion_writes_a_complete_*` pair pins the write
  side of decision 5 and nothing pins the read side.

### F-07-5 - fANOVA silently drops `HEAD_DIM` from the importance ranking, so the check on the reduction cannot see the model width

- **Kind:** bug
- **Severity:** medium
- **Where:** `opt.py:360` (`optuna.importance.get_param_importances(study)`), space defined
  at `opt.py:115-116`
- **Evidence:** the full profile samples a dependent range:

  ```python
  heads = trial.suggest_categorical('HEADS', [4, 8, 16])
  dim = trial.suggest_int('HEAD_DIM', 64 // heads, 256 // heads) * heads
  ```

  so `HEAD_DIM`'s `IntDistribution` differs per trial (`[4,16]` at 16 heads, `[16,64]` at
  4). `get_param_importances` with `params=None` takes its distributions from the
  *intersection* search space, which requires an identical distribution across trials.
  Reproduced with the installed optuna 4.9.0 on a 40-trial study using exactly this
  construction:

  ```
  ranked knobs: ['DROPOUT', 'HEADS']
  intersection space: ['DROPOUT', 'HEADS']
  ```

  `HEAD_DIM` is absent from both. `log_param_importances` catches only `ValueError` and
  `RuntimeError`, and nothing was raised — the knob is dropped silently, and
  `importance.json` simply has one fewer key than the study sampled.
- **Consequence:** ADR 0005 decision 6 makes the importance ranking the instrument that
  checks the reduction, and defers "a full-profile pilot with fANOVA to rank the held
  ones" to the fog. When that pilot runs, the ranking it produces will be missing the
  model width — the largest architectural knob and one of the five the reduced profile
  holds — with no warning anywhere. The pilot would then "find" that width does not
  matter, by omission. This affects every classification study on the branch too, since
  they all use the full profile.
- **Direction:** pass an explicit `params=` list to `get_param_importances`, or sample the
  width on a fixed grid so its distribution is trial-independent (e.g. suggest `DIM` from a
  categorical of widths and derive the legal head counts from it, inverting the current
  dependency). Either way, assert in `log_param_importances` that the ranked key set equals
  the sampled key set and warn when it does not.

### F-07-6 - The rows the search selects on are scored again as test rows in the 5-fold comparison run

- **Kind:** methodology
- **Severity:** medium
- **Where:** `opt.py:187-229` (trial namespace sets no `cv_folds`, so
  `resolve_training_request` reads `None` and `build_folds` returns the predefined split),
  ADR 0005 decision 6 (`--cv_folds 5` for both comparison arms)
- **Evidence:** the predefined split is 80/10/10 for every dataset in the protocol
  (`credit-g` 800/100/100, `kr-vs-kp` 2556/320/320, `spambase` 3679/461/461; val ∩ test = 0,
  checked). A trial is a single-split run, so the search's signal comes from exactly those
  100 / 320 / 461 validation rows. The comparison run is `--cv_folds 5`, which repartitions
  the whole table with `StratifiedKFold(shuffle=True, random_state=seed)`
  (`src/training/data.py:183-191`), so every row is a test row in exactly one fold — the
  selection rows included.
- **Consequence:** about 10% of the rows behind
  `cv/test/impute/induced/impute_score/mean` were used to choose the hyperparameters being
  evaluated, so the promoted arm's headline number is optimistically biased relative to a
  clean held-out estimate. I rate this **medium, not the headline**, and deliberately:
  the channel is four or five hyperparameters, the CV run re-masks at a different
  `fold_ordinal` so the actual scored cells differ, and the contaminated share is 10% of
  rows rather than all of them. It is a caveat to state, not a result-invalidating leak —
  unlike F-07-1, which is.
- **Direction:** either run the trials themselves at `--cv_folds 5` with the same seed, so
  search and comparison use the same partition and the leak is total and visible (bad), or
  hold the predefined split's validation rows out of the comparison run — simplest as a
  sentence in ADR 0005's "How to compare" recording the 10% overlap so the next reader
  does not have to rediscover it. There is no cheap way to make it zero without a third
  split.

### F-07-7 - A configuration is promoted on the search objective alone; nothing produces a held-out number for the winner before it is published

- **Kind:** design
- **Severity:** medium
- **Where:** `opt.py:525-530` (promotion), `opt.py:535-573` (retrain, a separate opt-in
  that runs *after* promotion and gates nothing)
- **Evidence:** `--promote_best` writes the file immediately after the study loop, using
  only `study.best_trial.user_attrs["hyperparameters"]`. `--retrain_best` is documented as
  independent ("`--promote_best` and `--retrain_best` are independent", ADR 0005 decision
  5) and runs afterwards; its metrics go to `best_metrics.json` and an MLflow run, and
  nothing reads them. The retrain is also a single-split run at the same seed
  (`final_args` sets no `cv_folds`), so even when it is enabled its held-out estimate comes
  from 100 rows on `credit-g`.
- **Consequence:** combined with F-07-1 and F-07-2, a configuration reaches
  `datasets/hiperparams/` on the strength of one selection-biased statistic computed on a
  few hundred cells, with no check at any point that it beats the defaults on anything.
  The four files in the working tree got there that way. The first evidence that the
  tuning helped arrives only in decision 6's separate comparison protocol — which F-07-3
  says cannot currently be run on those four variants.
- **Direction:** make promotion conditional: run the winner and the defaults on the
  predefined split's *test* rows (the retrain already trains the winner; a defaults run is
  one more) and refuse to promote when the winner does not beat the defaults by more than
  the objective's own quantum. If that is too strong, at minimum log the winner's
  held-out `test/impute/masked/impute_score` on the study parent beside
  `optuna/best_objective_value`, so the two can be read together.

### F-07-8 - `opt.py` hardcodes `label_column = 'class'` for trials and the retrain while honouring `--label_column` for the column-mix decision

- **Kind:** bug
- **Severity:** low
- **Where:** `opt.py:203` (`args.label_column = 'class'`), `opt.py:544`
  (`final_args.label_column = 'class'`) vs `opt.py:422` (`getattr(args, "label_column", None) or "class"`)
- **Evidence:** the study computes `mixed_columns` from `declared_column_types(header,
  getattr(args, "label_column", None) or "class", base)` — honouring the flag — and then
  builds every trial namespace with the literal `'class'`. The parser accepts
  `--label_column` (`src/training/config.py:214`).
- **Consequence:** a study launched with `--label_column target` decides whether to sample
  `LAMBDA_NUM` using `target` as the label, then trains every trial with `class` as the
  label — treating the real label as a feature and `class` as absent. On today's datasets
  the label is always `class`, so nothing is broken now; the defect is that the flag is
  accepted and half-ignored, and the two halves disagree.
- **Direction:** copy the resolved label column onto the trial and retrain namespaces the
  same way `task`, `seed` and `lr_scheduler` are copied, or reject `--label_column` in
  `opt.py`'s `__main__` next to the existing `--all` rejection.

### F-07-9 - A study cannot be resumed, and the studies this ADR budgets are exactly the ones that need to be

- **Kind:** design
- **Severity:** low
- **Where:** `opt.py:388-390` (timestamped `optuna_dir`), `opt.py:431-441`
  (`storage` inside it, `load_if_exists=True`)
- **Evidence:** a prior review cleared the timestamped directory as the reason
  `load_if_exists` never resumes — I am not disputing that, only stating its cost. The
  storage URI is `sqlite:///{optuna_dir}/optuna_study.db` with `optuna_dir` stamped at
  invocation, so `load_if_exists=True` is inert and there is no flag to point a new
  invocation at an existing store. ADR 0005 decision 4 budgets `spambase` at about 11
  hours for 40 trials, and the study database in the tree right now
  (`results/spambase_20nan/optuna_20260911_075737/optuna_study.db`) shows 11 COMPLETE and
  1 RUNNING — the case in point. Only `optuna.TrialPruned` is caught (`opt.py:302`);
  anything raised outside the objective's `try` (a tracking-store hiccup in
  `mlflow.start_run` at `opt.py:251`, a Ctrl-C) ends the study with its completed trials
  stranded in a store nothing will ever open again.
- **Direction:** a `--study_dir <path>` (or `--resume`) that reuses an existing
  `optuna_dir` instead of stamping a new one, which makes `load_if_exists=True` mean
  something and costs nothing else. The timestamp stays the default.

### F-07-10 - The study's running-best file uses classification's promoted filename and holds only the sampled subset

- **Kind:** design
- **Severity:** low
- **Where:** `opt.py:317` (`save_path = save_dir / f"{self.dataset_name}.json"`),
  `opt.py:517-519` (`best_hyperparameters.json` written with the same content)
- **Evidence:** an imputation study leaves
  `results/<dataset>/optuna_<ts>/<dataset>.json` containing the five sampled keys, and
  `best_hyperparameters.json` containing the same mapping. `<dataset>.json` is character
  for character the filename `_load_base_hyperparameters` reads for **classification**
  (`src/training/config.py:68`). The file is also partial, which contradicts decision 5's
  "a promoted file is **complete** ... so the file cannot silently move if a default
  changes" — `Hyperparameters.from_mapping` will fill the other fifteen keys from whatever
  the dataclass defaults are at read time. The log line that announces it
  (`opt.py:321-324`) labels the value with `self.ranking_metric`
  (`impute/masked/impute_score`) although the number is `self.search_objective`, the
  validation score.
- **Consequence:** two files in the study directory with identical content and different
  names, one of them named so that copying it one directory over — the obvious way to
  "use this configuration" — hands an imputation study's result to every future
  classification run, partial and default-sensitive. Low because it needs a human to copy
  it, but the name is the trap.
- **Direction:** name the running-best file after what it is and what wrote it
  (`best_so_far.json`, or `<dataset>.<task>.json` to match the promotion path), write it
  through `complete_configuration` like `promote_best_configuration` does, drop the
  duplicate, and label the log line with the search objective.

## Checked and cleared

- **The promoted filename matches what the loader looks for.** Both
  `promote_best_configuration` (`opt.py:342`) and `_load_base_hyperparameters`
  (`src/training/config.py:49`) go through the same `hyperparameter_file(dataset_name, task)`
  helper, so `<dataset>.imputation.json` is written and read by one definition and cannot
  drift. Verified end to end against the real tree: loading `credit-g_20nan` for
  `--task imputation` returns
  `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json` and the values in it match
  the winning trial in that dataset's study database (`LR_DECODE = 0.002538672990101232`
  = trial 30). The base-directory derivations agree too (`split("_", 1)[0]` in the helper,
  `split('_')[0]` at `opt.py:417`) for every dataset in the tree.
- **Promotion cannot move either regression fixture.** Both
  `tests/integration/test_credit_g_imputation_regression.py:36-38` and
  `tests/integration/test_vehicle_regression.py` construct `TrainingRequest` directly with
  `Hyperparameters.from_mapping(fixture["hyperparameters"])`, never touching
  `load_hyperparameters`. I specifically expected the freshly promoted
  `credit-g_20nan.imputation.json` to invalidate the imputation baseline; it cannot.
- **The objective and the test metric are in the same units.** `_score_population`
  (`src/training/decoding.py:279-330`) keeps `actual` in the scaled space for both
  populations; `actual_original` is written only for the preview and `_exact`'s docstring
  says it "never reaches a metric". The validation call passing no `raw_truth` therefore
  changes nothing about the number. No scaled-vs-original mismatch between the search
  objective and the reported score.
- **The pruner is inert, as decision 4 says.** `create_study` takes the default
  `MedianPruner`, but `trial.should_prune()` is never called — only `trial.report(score,
  step=0)` after training finishes (`opt.py:273`). No trial is ever stopped early.
- **No trial has actually failed.** The `TrialPruned`-on-exception path (known finding 6)
  has not fired: all four finished studies show 40/40 `COMPLETE`, and the in-flight one 11
  COMPLETE + 1 RUNNING. So the effective trial count has not silently shrunk in practice,
  and I have no new evidence to add to that known finding beyond noting that no
  completed-trial count is logged on the study parent (`n_trials` records what was asked
  for, not what finished).
- **The predefined validation and test splits are disjoint**, for all three protocol
  datasets (`len(set(val) & set(test)) == 0`). A trial's objective genuinely never reads
  the predefined split's test rows, which is what ADR 0005 decision 3 claims. F-07-6 is
  about a *different* split (the 5-fold repartition), not about this one.
- **The schedule is consistent across the trial, its tag and the promoted file.** All three
  resolve `getattr(args, "lr_scheduler", None) or DEFAULT_LR_SCHEDULER` from the same
  command-line value (`opt.py:225`, `:243`, `:528`), and `load_hyperparameters` lets
  `--lr_scheduler` win over a file, so a promoted configuration can be re-run under another
  schedule without editing it.
- **The retrain opens a normal top-level run.** `study_run_id = parent_run.info.run_id` and
  everything after it (`opt.py:499`) is outside the parent run's `with`, so
  `--retrain_best` is not nested under the study and carries
  `mlflow_tags={"optuna_study_run_id": ...}` as its only link. It also uses
  `hyperparams_override`, so it is unaffected by F-07-3's precedence problem and its
  `config_source` reads `override`, distinguishable from both comparison arms.
- **`mixed_columns` gating works on real data.** `kr-vs-kp` is all-categorical, and both
  its promoted files carry `LAMBDA_NUM = 1.0` (the held default) while both `credit-g`
  files carry sampled values — so the conditional dimension behaved as ADR 0005 decision 2
  describes, on the user's actual studies, not only in the unit test.
- **The reduced profile's held values are the dataclass defaults, and the promoted files
  prove it.** `DIM 128`, `HIDDEN_DIM 16`, `HEADS 16`, `LAYERS 2`, `DIM_FEED 32`,
  `EPOCHS_PRE 300`, `BATCH 256`, `LR_PRE 0.00034`, `WEIGHT_DECAY_PRE 0.005`,
  `EPOCHS_DECODE 150`, `EVAL_MASK_RATE 0.2` appear identically in all four files. The
  reduction is arbitrary in the sense that no evidence ranked these knobs — ADR 0005 says
  so plainly ("the user was offered deliberate values and kept the defaults") and defers
  the pilot that would justify it — so I record it as a stated, documented choice rather
  than a finding, with the caveat that F-07-5 means the deferred pilot will not be able to
  rank the width when it runs.
- **Trial-to-trial differences are not seed noise, by construction.** Every trial runs at
  the same `args.seed`, so the split, the pre-training initialisation and the fixed
  validation mask are identical across trials; the only thing that varies is the sampled
  configuration. That is the right design, and it is why F-07-1 is about the *resolution*
  of the objective rather than about seed variance.

## Open questions

- **Does the search objective predict the headline metric at all?** The objective is
  `validation/impute/masked/impute_score` (artificially masked cells in observed data); the
  ADR's headline is `cv/test/impute/induced/impute_score/mean` (cells that are genuinely
  missing in the variant, truth from the complete sibling). Nothing in the repo measures
  the correlation between them. Ticket 02's own proof run shows the *masked* validation and
  test scores differing by 0.005 and 0.001 on two trials, which says nothing about the
  induced population. What would settle it: score the induced population on the validation
  rows too (the complete sibling is already loaded and row-aligned) for a handful of trials
  and rank-correlate the two objectives across a finished study's 40 trials. If they do not
  agree, the reduced search is optimising the wrong thing and F-07-1 is the smaller problem.
- **Does the trial ordering survive a second mask draw?** `evaluation_mask` takes a `fold`
  argument and restores the global RNG, so re-scoring the four finished studies' saved
  configurations at `fold = 1, 2` would cost a handful of trainings and would directly
  measure whether the top-five band in F-07-1 is signal. I did not spend the training
  budget on it; it is the single most informative experiment available here.
- **Is `mean_mode_baselines` comparing like with like for categorical columns?**
  `baselines[column] = str(training_fold[column].mode().iloc[0])` against
  `cells["actual"]` from `encoder.inverse_transform(...)`. If the frame holds encoded
  codes and `actual` holds decoded labels, the naive categorical error is pinned at 1.0 and
  every categorical `impute_score` is just the model's error rate. That would not change
  trial *ranking* (it is a monotone rescaling), so it does not move any finding here, but
  it changes what "1.0 is baseline parity" means. It belongs to the metrics dimension;
  flagging it only so it is not lost.
