# 07 - Optuna search for imputation: verdicts

_Adversarial verification of `07-optuna-imputation.md`. 2026-09-11._

**Method:** graphify orientation, then source: `opt.py` (all 603 lines), `src/training/config.py:21-280`,
`src/training/types.py:18-180`, `src/training/decoding.py:30-345`, `src/models.py:154-210`,
`src/embedder.py:161-200`, `src/training/data.py:132-230`, `src/training/imputation_metrics.py`
(`score_cells`), `tests/unit/test_training_config.py:276-335`, `tests/unit/test_opt_tracking.py`
(headings), `docs/adr/0005-reduced-optuna-search-for-imputation.md` (all),
`experiment_imputation.ps1` (all, uncommitted; `git diff` is a one-line `--task imputation` insert),
and — **not in the critique's scope list, and decisive for F-07-3** — the committed launcher
`imputation_studies.ps1` (all 145 lines; `d039c47`, `d69e344`; tracked and clean).

Everything below was re-measured independently. Read-only throughout: the five study databases and
`mlflow.db` were opened with `sqlite3` `?mode=ro`, never through `optuna.load_study` or the MLflow
client. No training run, no pytest.

1. **The study databases.** `sqlite3 file:results/*/optuna_*/optuna_study.db?mode=ro`, joining
   `trials` to `trial_values` and `trial_params`:

   ```
   results\credit-g_20nan\optuna_20260911_011020   trials 40 complete 40 distinct 40
     top6: (30,0.7763167473904415) (17,0.7915938240278412) (23,0.7979530022684177) ...
   results\credit-g_40nan\optuna_20260911_015509   trials 40 complete 40 distinct 40
     top6: (2,0.8050950011944125) (33,0.8153436328087034) (30,0.821648480584557) ...
   results\kr-vs-kp_20nan\optuna_20260911_023724   trials 40 complete 40 distinct 20
     top6: (17,0.6703296703296705) (29,0.6703296703296705) (0,0.673992673992674) (27,0.673992673992674) ...
   results\kr-vs-kp_40nan\optuna_20260911_051639   trials 40 complete 40 distinct 20
     top6: (32,0.6480446927374302) (38,0.6480446927374302) (11,0.653631284916201) (13,0.653631284916201) ...
   results\spambase_20nan\optuna_20260911_075737   trials 21 complete 20 distinct 20   (still in flight)
   ```

2. **Scored-cell counts**, with the real `prepare_dataset` / `build_folds` / `evaluation_mask` at the
   parameters the launch log records (`results/imputation_studies_20260911_011008.out`:
   `--search_space reduced --n_trials 40 --lr_scheduler cosine --seed 42 --promote_best`, and
   `src/training/runner.py:67` enumerates folds from 1):

   ```
   credit-g_20nan: train=800 val=100 test=100 val_masked_cells=283 (cat 187, num 96) val_int_test=0
   credit-g_40nan: train=800 val=100 test=100 val_masked_cells=179 (cat 118, num 61) val_int_test=0
   kr-vs-kp_20nan: train=2556 val=320 test=320 val_masked_cells=1511 (cat 1511, num 0) val_int_test=0
   kr-vs-kp_40nan: train=2556 val=320 test=320 val_masked_cells=910 (cat 910, num 0) val_int_test=0
   spambase_20nan: train=3679 val=461 test=461 val_masked_cells=3390 (cat 0, num 3390) val_int_test=0
   ```

3. **fANOVA on a `HEAD_DIM`-style dependent range**, installed optuna 4.9.0, 40 trials built exactly
   as `opt.py:115-116` builds them:

   ```
   optuna 4.9.0
   ranked knobs: ['DROPOUT', 'HEADS']
   intersection space: ['DROPOUT', 'HEADS']
   sampled knobs: ['DROPOUT', 'HEADS', 'HEAD_DIM']
   ```

4. **Config resolution against the tree as it stands**, and the cross-task fallback in an isolated
   scratch cwd holding only a classification file.

5. **The new measurement the critique did not make.** Every trial logged
   `test/impute/masked/impute_score` and `test/impute/induced/impute_score` to `mlflow.db` beside its
   objective. 180 reduced-profile imputation trials carry all three. Rank-correlating them is free and
   answers the critique's own first open question; it is the basis of "What this critique missed" and
   of the F-07-7 verdict.

## Verdicts

### F-07-1 - The validation objective is too coarse to separate the configurations it ranks, and two of the four promoted files were decided by trial order rather than by the objective
- **Verdict:** SOUND
- **Severity after review:** critical
- **Basis:** every number in the finding reproduces exactly.
  - Cell counts: 283 / 179 / 1511 / 910 / 3390, split as the critique states (see Method 2).
  - The lattice: `score_cells` (`src/training/imputation_metrics.py`) computes
    `score += (len(categorical)/total) * _ratio(1.0 - metrics["acc_cat"], naive_error)`, so on an
    all-categorical table `impute_score` is `wrong_cells / (1511 * naive_error)` — integers over a
    constant. Consecutive distinct values in `kr-vs-kp_20nan` differ by
    `0.673992673992674 - 0.6703296703296705 = 0.0036630036630035 = 1/273` exactly, and
    `0.6703296703296705 * 273 = 183.0`. `kr-vs-kp_40nan`:
    `0.653631284916201 - 0.6480446927374302 = 0.0055865921787708 = 1/179`, and
    `0.6480446927374302 * 179 = 116.0`.
  - The ties: 40 trials, 20 distinct values, best == 2nd bit-for-bit in both kr-vs-kp studies.
  - The tie-break and the promoted files, from `trial_params` against `datasets/hiperparams/`:
    `kr-vs-kp_20nan` trial 17 `LR_DECODE=0.0017418753352176958 PROB_MASCARA=0.2` vs trial 29
    `0.005926803304179926 / 0.5`; the promoted file holds trial 17's values.
    `kr-vs-kp_40nan` trial 32 `0.0016986087972445047 / 0.3 / DROPOUT 0.1` vs trial 38
    `0.0027844921145462236 / 0.4 / 0.2`; the promoted file holds trial 32's.
  - credit-g's 12x `LAMBDA_NUM` disagreement across variants of one table: 2.738287642961479
    (`_20nan`) vs 0.23102018878452935 (`_40nan`), confirmed in both promoted files.
- **Correction:** none — if anything the finding understates itself. I measured what the tie-break
  actually cost on the metric ADR 0005 decision 6 names as the headline, using the trials' own
  MLflow records:

  ```
  kr-vs-kp_20nan trial 17: obj 0.6703296703  test_masked 0.6918  test_induced 0.6043   <- promoted
  kr-vs-kp_20nan trial 29: obj 0.6703296703  test_masked 0.6882  test_induced 0.6595
  kr-vs-kp_40nan trial 32: obj 0.6480446927  test_masked 0.7692  test_induced 0.8092   <- promoted
  kr-vs-kp_40nan trial 38: obj 0.6480446927  test_masked 0.8013  test_induced 0.7814
  ```

  On `kr-vs-kp_20nan` the coin landed well (0.6043 vs 0.6595); on `kr-vs-kp_40nan` it landed badly —
  trial order promoted the configuration that is 0.028 *worse* on `impute/induced/impute_score`, a
  difference of roughly 30-40 cells out of 4594. I did not test that for significance and it is a
  single split at a single seed, so read it as direction, not as a proven loss. What it does show
  concretely is that the two tied configurations are not interchangeable on the metric decision 6
  reports, which is the part the objective had no way to see.

### F-07-2 - The checkpoint is selected on exactly the cells the objective is then scored on, so `optuna/best_objective_value` is a best-of-`EPOCHS_DECODE` minimum, not a plain validation score
- **Verdict:** SOUND
- **Severity after review:** high
- **Basis:** the code path is unambiguous end to end in `src/training/decoding.py`.
  `hidden_validation` / `clean_validation` are encoded once per fold (`:71-79`) and never re-rolled.
  Each epoch, `validation_loss, _ = model(hidden_validation, clean_validation)` (`:112-114`), and
  `if validation_losses[-1] < best_validation_loss: ... best_state = {...}` (`:125-127`). After the
  loop `model.load_state_dict(best_state)` (`:129-130`), and then:

  ```python
  validation_cells = _score_population(model, hidden_validation, clean_validation, "masked")
  ```

  (`:204-206`). The selection statistic is not merely on the same rows but on the same *cells*:
  `TridentDecoder.forward` restricts its loss to `mask = hidden.masked_positions`
  (`src/models.py:165`), and `masked_positions` is `(df[col] == "[MASK]")` (`src/embedder.py:190-192`)
  — the identical 283 / 179 / 1511 / 910 / 3390 cells. `EPOCHS_DECODE` is held at 150 in the reduced
  profile, so that is 150 selections per trial and 6000 per study; `opt.py:134` does sample it 20..60
  in the `full` profile, so there the unequal budget becomes a between-trial bias too, exactly as
  stated.
- **Correction:** none to the premise. One supporting measurement, offered as suggestive rather than
  decisive (validation and test are different rows, so part of any gap is row variation): across the
  180 reduced-profile trials the objective reads better than the same trial's test masked score with
  near-perfect consistency —

  ```
  credit-g_20nan: mean val 0.8676  mean test_masked 0.9066  delta -0.0390  (val better on 36/40)
  credit-g_40nan: mean val 0.8894  mean test_masked 0.9310  delta -0.0416  (val better on 34/40)
  kr-vs-kp_20nan: mean val 0.7094  mean test_masked 0.7215  delta -0.0121  (val better on 25/40)
  kr-vs-kp_40nan: mean val 0.7049  mean test_masked 0.7691  delta -0.0642  (val better on 38/40)
  spambase_20nan: mean val 0.9198  mean test_masked 0.9642  delta -0.0444  (val better on 20/20)
  ```

  Worth stating explicitly, because the finding does not: the *test* split stays clean. The checkpoint
  is chosen on validation cells and the test populations are disjoint rows, so `test/` and `cv/test/`
  numbers are unaffected. What is contaminated is the search objective itself and the
  `optuna/best_objective_value` the ADR publishes.

### F-07-3 - Once a configuration is promoted there is no way to run the defaults arm of ADR 0005 decision 6's own comparison
- **Verdict:** UNSOUND
- **Severity after review:** low
- **Basis:** the premise about the CLI is true. `_load_base_hyperparameters`
  (`src/training/config.py:38-59`) prefers the on-disk file whenever it exists, and
  `build_training_parser` (`:186-280`) has no flag that skips it — I read every argument;
  `hyperparams_override` is programmatic only. Reproduced against the real tree:

  ```
  imputation     -> datasets/hiperparams/credit-g/credit-g_20nan.imputation.json
                    dropout 0.30000000000000004 prob_mask 0.4 lr_decode 0.002538672990101232 sched cosine
  classification -> defaults
  ```

  The conclusion drawn from it does not hold. The critique's scope list names
  `experiment_imputation.ps1` but never `imputation_studies.ps1` — the committed launcher ADR 0005
  decision 6 actually refers to ("A launcher script in the shape of `experiment.ps1` runs the six
  studies with promotion and the twelve comparison runs"), tracked and clean at `d039c47` / `d69e344`.
  Its `-Compare` mode is exactly decision 6's protocol, and it already does the file move — as code,
  not as a human step:

  ```powershell
  # The comparison: the same run twice, once reading the promoted file and once
  # with it moved aside so the lookup falls through to the defaults (or to the
  # shared file, if one exists: params.config_source on the run says which).
  ...
  $aside = "$promoted.aside"
  Write-Host "[setup] $promoted -> $aside"
  if (-not $DryRun) { Move-Item $promoted $aside -Force }
  try   { $status = Invoke-Trainer $common }
  finally {
      # Put the promoted file back even on Ctrl+C.
      if (-not $DryRun -and (Test-Path $aside)) { Move-Item $aside $promoted -Force }
  }
  ```

  It also refuses to run the comparison at all when no promoted file exists
  (`[skip] ${dataset}: no promoted configuration ... run the study first`), and its summary prints the
  MLflow filter "grouped by `params.config_source`".
- **Correction:** three of the finding's four claims are false against that script. "The defaults arm
  cannot be launched today" — it can, with `.\imputation_studies.ps1 -Compare`. "An untracked, manual,
  unlogged step" — it is tracked, automated, and printed as `[setup] ... -> ...aside`. "If someone
  forgets the move, both arms are the promoted configuration" — the script cannot forget, and a
  `try/finally` restores the file on Ctrl+C. The "silent" framing was already overstated on its own
  terms: `config_source` is a dense param and ADR 0005's "How to compare" says it separates the arms,
  so two identical values would be visible to anyone following the ADR.
  What survives is a genuine but small design note: a file move is a weaker mechanism than a flag —
  a hard kill or power loss between the two `Move-Item` calls leaves `<dataset>.imputation.json.aside`
  and no promoted file — and any entry point other than this one launcher (a human typing `main.py`)
  still has no escape from the precedence. That is worth backlog I1, at low.
  The `experiment_imputation.ps1` half of the finding is factually right — `Get-HyperparamsPath`
  returns `"$dataset.json"` (line 27) while the run is `main.py --task imputation` (line 60), whose
  first candidate is `<dataset>.imputation.json` — but it is about the wrong script. That file is an
  uncommitted one-line edit of the ADR 0003 schedule sweep (its own summary still prints
  `cv/test/f1_macro/mean`), its default `-Datasets` are `vehicle_00nan, credit-g_00nan` which have no
  promoted files, and the classification key set it writes (`EPOCH_FINE`, `LR_FINE`, `LABELS`, no
  `EPOCHS_DECODE`) only reaches an imputation run at all *because* of the F-07-4 fallback. It is
  scratch, not the protocol.

### F-07-4 - An imputation run with no promoted file of its own loads classification's tuned configuration wholesale
- **Verdict:** SOUND
- **Severity after review:** medium
- **Basis:** the candidate list is as quoted (`src/training/config.py:47-51`). Reproduced in an
  isolated scratch cwd containing only `datasets/hiperparams/foo/foo_20nan.json`:

  ```
  imputation run, only a CLASSIFICATION file present -> datasets/hiperparams/foo/foo_20nan.json
    dim 64 dropout 0.45 prob_mask 0.11 epochs_pre 7 sched plateau epochs_decode 150 lr_decode 0.001
  ```

  `PROB_MASCARA`, `EPOCHS_PRE` and `LR_SCHEDULER` are in `complete_configuration`'s `shared` block, so
  a promoted classification file does carry all three into an imputation run, and `--lr_scheduler`
  only wins when explicitly passed (`load_hyperparameters`, `:29-33`). The full classification profile
  samples `EPOCHS_PRE` in 20..60 while the reduced imputation profile holds it at 300, so the
  cross-task channel is not cosmetic: it would cut pre-training by a factor of five to fifteen.
- **Correction:** two things cut this from high.
  (a) The critique says "nothing pins that *loading* is isolated per task" and prescribes "add the
  missing test". A test exists and pins the *opposite*, deliberately:
  `tests/unit/test_training_config.py:276`,
  `test_an_imputation_run_prefers_its_own_configuration_and_falls_back_to_the_shared_one`, whose body
  is `task_keyed.unlink(); assert request("imputation").hyperparameters.dimension == 32` and whose
  docstring states the rationale. So this is not an untested corner; it is a pinned contract, and the
  correct ask is to change the decision, not to add coverage.
  (b) Blast radius today is zero: `find datasets/hiperparams -type f` returns four files, all
  `.imputation.json`; no shared `<dataset>.json` exists anywhere in the tree, and `config_source` is
  dense, so the day one appears the cross-task load is legible on every affected run. The committed
  launcher also shows the author reasoning about exactly this case — its comparison comment reads
  "so the lookup falls through to the defaults (**or to the shared file, if one exists**:
  `params.config_source` on the run says which)" — so it is a known, managed behaviour rather than an
  unnoticed one.
  Medium, not high, and medium rather than low only because the inheritance is quietly large: the
  full classification profile's `EPOCHS_PRE` range would replace the held 300 with 20..60, a five- to
  fifteen-fold cut to pre-training on a run nobody configured that way, one `--promote_best` away from
  being live.

### F-07-5 - fANOVA silently drops `HEAD_DIM`, so the importance ranking cannot see the model width
- **Verdict:** CONFIRMED
- **Severity after review:** low
- **Basis:** reproduced on the installed optuna 4.9.0 with `opt.py:115-116`'s exact construction
  (`heads = suggest_categorical('HEADS',[4,8,16]); suggest_int('HEAD_DIM', 64//heads, 256//heads)`):
  ranked knobs `['DROPOUT', 'HEADS']`, intersection space `['DROPOUT', 'HEADS']`, sampled knobs
  `['DROPOUT', 'HEADS', 'HEAD_DIM']`. No exception, so `log_param_importances`' `except (ValueError,
  RuntimeError)` guard (`opt.py:360-362`) never fires and `importance.json` is one key short with no
  warning. The mechanism is exactly as described: `get_param_importances(study)` with `params=None`
  takes distributions from the intersection search space, which requires an identical distribution
  across trials, and `HEAD_DIM`'s `IntDistribution` is `[4,16]` at 16 heads and `[16,64]` at 4.
- **Correction:** severity cut from medium. The bug cannot touch this dimension: the `reduced`
  profile never samples `HEAD_DIM`, and all five knobs it does sample have trial-independent
  distributions (`LAMBDA_NUM`'s conditional is resolved once per study from `mixed_columns`, so it is
  present on all trials or none). It bites only the `full` profile, and no full-profile study exists —
  `mlflow.db` holds five `optuna_study` runs, all imputation/reduced. The product is a diagnostic
  artifact nothing automated reads. A real, reproducible, silent bug with no current consequence: low.

### F-07-6 - The predefined validation rows the search selects on are scored again as test rows in the 5-fold comparison
- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** premise confirmed on both sides. The trial namespace (`opt.py:198-229`) is a bare
  `class Args: pass` that never sets `cv_folds`, so `resolve_training_request` reads
  `getattr(args, "cv_folds", None) -> None` and `build_folds` takes the predefined branch
  (`src/training/data.py:150-166`). Split sizes and `val ∩ test = 0` measured for all three protocol
  datasets (Method 2). ADR 0005 decision 6 runs `--cv_folds 5`, which reaches
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` (`data.py:183-191`) and partitions
  every row into exactly one test fold.
- **Correction:** severity cut from medium. The finding self-moderates well and its own prescribed
  remedy is "a sentence in ADR 0005's How to compare" — a caveat, and a caveat is a low-severity item.
  Three further dampeners the finding names but does not weigh: the channel is four or five
  hyperparameters; the CV run re-masks at a different `fold_ordinal` so no scored cell recurs; and the
  headline population is `impute/induced`, which the search never touched at all in any split. It
  belongs in the ADR, not near the top of a findings list.

### F-07-7 - A configuration is promoted on the search objective alone, with no held-out number for the winner
- **Verdict:** UNSOUND
- **Severity after review:** medium (for the corrected version below)
- **Basis:** the mechanics are right — `--promote_best` writes the file immediately after the study
  loop from `study.best_trial.user_attrs["hyperparameters"]` (`opt.py:525-531`); `--retrain_best`
  runs afterwards (`:535-573`), gates nothing, sets no `cv_folds`, and its `best_metrics.json`
  (`opt.py:568`) is read by nothing (`grep` over `*.py`, `*.ps1`, `*.md` finds only the writer and a
  ticket that mentions it). But the headline claim — "nothing **produces** a held-out number for the
  winner before it is published" — is false, and the word carries the finding's weight.
  `train_and_evaluate_decoder` always scores the test split; `score_search_objective` only *adds* the
  `validation/` family. So every trial, the winner included, computes and logs a genuine held-out
  score on the predefined test rows before promotion ever runs. All 180 reduced-profile trial runs in
  `mlflow.db` carry
  `test/impute/masked/impute_score` and `test/impute/induced/impute_score`, on 283..5202 cells:

  ```
  credit-g_20nan  test/impute/induced n_cat 268  n_num 151
  credit-g_40nan  test/impute/induced n_cat 509  n_num 262
  kr-vs-kp_20nan  test/impute/induced n_cat 2326
  kr-vs-kp_40nan  test/impute/induced n_cat 4594
  spambase_20nan  test/impute/induced n_num 5202
  ```
- **Correction:** the true finding is narrower in premise and sharper in consequence: **the held-out
  number exists on the winning trial's own run and nothing reads it** — not the promotion gate, not
  the study parent, not `importance.json`. Had it been read, it would have chosen differently. Ranking
  each study's trials by `test/impute/induced/impute_score` and locating the promoted winner:

  ```
  credit-g_20nan  winner trial 30  induced 0.9485  ranks 31/40   (best in study 0.8950, trial 13)
  credit-g_40nan  winner trial  2  induced 0.9562  ranks 21/40   (best in study 0.9206, trial 24)
  kr-vs-kp_20nan  winner trial 17  induced 0.6043  ranks  1/40
  kr-vs-kp_40nan  winner trial 32  induced 0.8092  ranks 33/40   (best in study 0.7343, trial 25)
  spambase_20nan  winner trial 16  induced 0.9063  ranks  2/20   (in flight)
  ```

  Three of the four promoted configurations sit in the bottom half of their own study on the metric
  decision 6 declares the headline. So the finding's *direction* ("log the winner's held-out score on
  the study parent beside `optuna/best_objective_value`") is right and cheap — the number is already
  computed — but its stated premise is not, and its stronger form belongs with F-07-1 rather than as a
  separate design note. Medium, as a corrected finding.

### F-07-8 - `opt.py` honours `--label_column` for the column-mix decision and hardcodes `'class'` for every trial and the retrain
- **Verdict:** CONFIRMED
- **Severity after review:** low
- **Basis:** verified at the exact lines. `opt.py:422`
  `header, getattr(args, "label_column", None) or "class", base_dataset_name` decides `mixed_columns`
  and therefore whether `LAMBDA_NUM` gets a dimension; `opt.py:203` `args.label_column = 'class'` and
  `opt.py:544` `final_args.label_column = 'class'`. The parser accepts the flag
  (`src/training/config.py:214`, `default=None`) and `DatasetSpec.from_name` resolves
  `label_column or "class"` (`src/training/types.py:187-192`), so the literal is not merely a
  redundant default — it overrides a user-supplied value. `opt.py`'s `__main__` rejects only `--all`.
- **Correction:** none. Two paths in one file disagree about the same input, which is a defect rather
  than ergonomics. Low is right and is what the critique already assigned: every processed dataset uses
  `class`, so nothing today can trip it.

### F-07-9 - A study cannot be resumed; `load_if_exists=True` is inert
- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** `timestamp = datetime.datetime.now().strftime(...)` (`opt.py:388`), `optuna_dir` built
  from it (`:389`), `storage_name = f"sqlite:///{optuna_dir}/optuna_study.db"` (`:431`),
  `load_if_exists=True` (`:437`) — the URI is unique per invocation, so the flag can never match, and
  no parser argument points at an existing store. The case in point is live:
  `results/spambase_20nan/optuna_20260911_075737/optuna_study.db` currently holds 20 COMPLETE and 1
  RUNNING against decision 4's 40-trial, ~11-hour budget.
- **Correction:** one wording slip. "Only `optuna.TrialPruned` is caught (`opt.py:302`)" inverts the
  line: `opt.py:302` *raises* `raise optuna.TrialPruned() from e`, inside the objective's own
  `except Exception`. The accurate statement is that `study.optimize` is called without `catch=`
  (`opt.py:478`), so anything raised outside that `try` — the `with mlflow.start_run(...)` at
  `opt.py:251`, or a `KeyboardInterrupt` — terminates the study and strands its completed trials in a
  store nothing will reopen. The conclusion is unchanged. The critique correctly flags this as stating
  the cost of an already-cleared decision, not disputing it; low.

### F-07-10 - The study's running-best file carries classification's promoted filename and holds only the sampled subset
- **Verdict:** SOUND
- **Severity after review:** low
- **Basis:** `save_path = save_dir / f"{self.dataset_name}.json"` (`opt.py:317`), which is exactly
  what `hyperparameter_file(dataset_name, "classification")` produces
  (`src/training/config.py:61-69`, `suffix = ".json" if task == DEFAULT_TASK`). Confirmed on the real
  study directories — each holds two files with byte-identical, partial content:

  ```
  results/credit-g_20nan/optuna_20260911_011020/best_hyperparameters.json
  results/credit-g_20nan/optuna_20260911_011020/credit-g_20nan.json
    both: {PROB_MASCARA, LR_DECODE, WEIGHT_DECAY_DECODE, DROPOUT, LAMBDA_NUM}   (5 of 17 keys)
  ```

  Against decision 5's "a promoted file is **complete** ... so the file cannot silently move if a
  default changes", and `Hyperparameters.from_mapping` would fill the other twelve from the dataclass
  defaults at read time. The mislabelled log line is real too (`opt.py:321-324` interpolates
  `self.ranking_metric`, `impute/masked/impute_score`, next to `self.best_score`, which is
  `self.search_objective`).
- **Correction:** none; low is right. The trap needs a human to copy the file one directory over, and
  the log-line label on its own would be a wording nit — it earns its place only as part of the
  filename finding.

## What this critique missed

**The critique's first open question is answerable today, from data already in the tree, and the
answer is larger than F-07-1.** The critique says "nothing in the repo measures" whether the search
objective predicts the headline metric, and defers it. But every trial already logged both numbers:
`mlflow.db` holds 181 `optuna_trial` runs, 180 of them reduced-profile imputation trials carrying
`validation/impute/masked/impute_score`, `test/impute/masked/impute_score` and
`test/impute/induced/impute_score` together. Rank-correlating them costs one read-only SQL query:

```
                      objective vs test masked   objective vs test INDUCED (the headline population)
credit-g_20nan  n=40           rho +0.611                   rho +0.124
credit-g_40nan  n=40           rho +0.563                   rho +0.568
kr-vs-kp_20nan  n=40           rho +0.418                   rho +0.340
kr-vs-kp_40nan  n=40           rho +0.071                   rho +0.372
spambase_20nan  n=20           rho +0.783                   rho +0.812
```

Read those against the noise floor: at n=40 a Spearman rho is only distinguishable from zero beyond
about +-0.31 (1.96/sqrt(39)), at n=20 beyond about +-0.45. So +0.568 and +0.812 are real; +0.124 and
+0.071 are indistinguishable from no relationship at all; +0.34 and +0.37 are borderline. On
`credit-g_20nan` the objective and the population decision 6 reports are, on this evidence, unrelated;
on `kr-vs-kp_40nan` the objective does not even track the *same* population on the test split.

The consequence is the one the critique itself named as decisive and left unmeasured: the promoted
winners rank 31/40, 21/40, 33/40 and 1/40 on `test/impute/induced/impute_score`, giving up 52%, 42%,
66% and 0% of their own study's achievable induced range. `kr-vs-kp_40nan` is the worst on both
counts, so this does not overturn F-07-1's emphasis on the lattice. What it adds is that the lattice
is not the *only* failure mode: `credit-g_20nan`'s objective is continuous, never ties, and is still
nearly uninformative about the headline (winner 0.9485 in a spread of 0.8950..0.9969). A reader of
F-07-1 alone could conclude that credit-g is the safe case because its objective resolves. It
resolves, and it still selects near the middle of the pack.

**A better-resolved objective is already computed on every trial and was not considered.** The
critique's "Direction" for F-07-1 lists three levers, all of which cost extra forward passes or extra
trainings. A fourth is free in cells: the induced population is 419 / 771 / 2326 / 4594 / 5202 cells —
1.5x to 5x the masked validation population it is being ranked on — it is the exact population the
headline uses, and `_score_induced_missing` already runs on every trial. It cannot be used as it
stands, because it scores the test split and decision 3 forbids that, but moving it to the validation
rows is a small change rather than a new mechanism: `_score_induced_missing`
(`src/training/decoding.py:227-265`) takes the frame as a parameter and only hardcodes
`fold.test_indices` for the truth rows. That single change would fix the resolution problem (F-07-1),
give the objective the same population as the headline (the correlation problem above), and cost one
forward pass. It deserves to be on the list ahead of "score over several `evaluation_mask` draws".

**The lattice damages the search, not just the promotion.** F-07-1 stops at `study.best_trial`
breaking the tie. But `TPESampler` splits observed trials into a good and a bad group at a quantile of
the objective; with 20 distinct values over 40 trials, that boundary falls inside a block of tied
trials and the split is partly arbitrary. The 30 trials that ran after TPE left its 10 random startup
trials were being steered by a signal with ~20 levels, several of which are indistinguishable. So a
coarse objective does not merely pick badly at the end — it degrades the 2h39m of search that
preceded the pick. That is a separate argument for fixing the objective before buying more trials, and
it strengthens the finding's own "more cells before more trials" direction.

**The committed launcher was never read, and it is the script the ADR's protocol lives in.** The
critique's scope list covers `opt.py`, the ADR, one ticket, two test files and the *uncommitted*
`experiment_imputation.ps1`, plus the launch *log* `results/imputation_studies_20260911_011008.out` —
but not `imputation_studies.ps1`, the tracked script that produced that log. That omission is what
made F-07-3 wrong, and it hid two other facts. Its `-Compare` mode is decision 6's protocol
implemented, including the defaults arm and a `try/finally` restore. And its default `-Variants` are
`@("20nan","40nan")` over `@("credit-g","kr-vs-kp","spambase")`, so `spambase_40nan` is not missing
from the protocol — it is queued behind `spambase_20nan`, which is still in flight at 20 of 40 trials.
The right statement is "not yet started", not "one study short". That does sharpen F-07-9 rather than
blunt it: with no resume, an interruption of the in-flight study loses its 20 completed trials and the
~5.5 hours in them, and the batch has ~16 hours of spambase still to run.

**Whoever acts on this review should re-read the launcher before touching the precedence.** Any change
to `_load_base_hyperparameters` that F-07-3 or F-07-4 might motivate has to keep
`imputation_studies.ps1 -Compare` working, because the comparison it runs is the effort's deliverable.
A `--hyperparams defaults` flag would let that script drop the `Move-Item` pair entirely, which is the
version of backlog I1 worth writing down.

**Not checked, and it bounds everything above.** All of my held-out rankings come from single-split,
single-seed trial runs. A rank of 31/40 on 419 induced cells is a real signal about the winner's
mediocrity but not a precise one, and I did not spend training budget re-running any configuration at
a second seed or a second mask draw — which remains, as the critique says, the single most informative
experiment available here. I also did not check the critique's "Checked and cleared" items beyond the
ones a finding depends on.
