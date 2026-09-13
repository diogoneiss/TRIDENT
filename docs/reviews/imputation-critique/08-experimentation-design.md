# 08 - Experimental design of the launchers

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `experiment_imputation.ps1` (committed at HEAD and the working-tree version), `imputation_studies.ps1:1-142`, `experiment.ps1:1-101` for contrast. Supporting reads: `src/training/config.py:21-69` and `:77-112`, `src/training/types.py:10-31`, `:62-71`, `:90-176`, `src/training/data.py:92-110`, `:132-211`, `src/training/decoding.py:32-215`, `src/training/summary.py:31-46`, `:148-158`, `src/training/artifacts.py:28-89`, `opt.py:82-150`, `:200-300`, `:328-348`, `:371-430`, `docs/adr/0005-reduced-optuna-search-for-imputation.md:100-195`.

**Method:** read the three launchers and every code path they reach that decides which configuration a run loads and which metrics it emits. Ran five read-only empirical checks: (1) the real `load_hyperparameters` on `credit-g_20nan --task imputation` against the exact JSON the sweep writes; (2) `StratifiedKFold(5, shuffle=True, random_state=42)` against the predefined split files for the three study datasets, counting row overlap; (3) Spearman correlation between the Optuna search objective and the ADR 0005 headline metric over the eleven preserved trial `metrics.csv` files of the in-flight `spambase_20nan` study; (4) fold-to-fold spread of the headline metric on the two existing multi-fold imputation runs; (5) `git show HEAD:experiment_imputation.ps1 | diff` against `experiment.ps1`.

**Not checked:** I deliberately did **not** run the one permitted training run. `results/imputation_studies_20260911_011008.err` was last written at 09:41 today and `results/spambase_20nan/optuna_20260911_075737/` is live — the six-study sweep is executing right now, and a second training process would contend with it for the GPU and write into `results/`. I also did not open `mlflow.db`, which that run holds. Every claim below rests on source, on files the running experiment already wrote, or on the real loader called in isolation.

## Findings

### F-08-1 - The committed `experiment_imputation.ps1` is a byte-for-byte copy of `experiment.ps1` and runs classification, not imputation

- **Kind:** bug
- **Severity:** high
- **Where:** `experiment_imputation.ps1:60` (added in commit `9a38b90`)
- **Evidence:**

  ```
  $ git show HEAD:experiment_imputation.ps1 | tr -d '\r' > a
  $ git show HEAD:experiment.ps1            | tr -d '\r' > b
  $ diff a b
  IDENTICAL at HEAD: experiment_imputation.ps1 == experiment.ps1
  ```

  `git log --diff-filter=A --no-renames -- experiment_imputation.ps1` → `9a38b90 feat(optuna): rank imputation trials on the validation split`, a commit otherwise about `opt.py`/`decoding.py`. The *only* thing that makes this file an imputation launcher is the uncommitted working-tree edit, `git diff -- experiment_imputation.ps1`, which inserts `"--task", "imputation"` into the argument array. The header (`Learning-rate schedule sweep (ADR 0003), tier 1`), the usage lines (`.\experiment.ps1`), the JSON builder, the datasets, the seed and the closing MLflow recipe were never adapted — the diff against `experiment.ps1` is exactly two tokens on one line.
- **Consequence:** anyone who checked the branch out and ran `.\experiment_imputation.ps1` — before today's edit, or on any clone, or on the CI/paper-reproduction path — got eight **classification** runs on `vehicle_00nan` and `credit-g_00nan` at seed 420, tagged `task=classification`, indistinguishable in MLflow from `experiment.ps1`'s own runs except by start time. Two launchers writing identical runs into the same store is worse than one: an operator filtering `tags.lr_scheduler` sees sixteen runs where the design called for eight, with no tag that separates them. Nothing would catch this: `-DryRun` prints the command line without asserting anything about it, and no test covers the `.ps1` files.
- **Direction:** the file needs to be adapted, not annotated. Its own header and usage lines; `Get-HyperparamsPath` returning `<dataset>.imputation.json` (see F-08-6); a JSON built from the imputation key set (see F-08-7); imputation datasets rather than `_00nan` siblings; and a closing recipe naming an imputation metric. Its inherited seed-420/no-`cosine_legacy`-arm problem is known finding #8 and is not re-reported here, but it applies verbatim.

### F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the experiment cannot test the decision it exists to test

- **Kind:** methodology
- **Severity:** high
- **Where:** `imputation_studies.ps1:62` (and the `param()` block, `:18-28`)
- **Evidence:** the study argument array is

  ```powershell
  $studyArgs = @(
      "--dataset_name", $dataset, "--task", "imputation", "--use_optuna",
      "--search_space", "reduced", "--n_trials", $Trials,
      "--lr_scheduler", $Scheduler, "--seed", $Seed
  )
  ```

  `reduced` is a string literal. The `param()` block exposes `-Datasets`, `-Variants`, `-Trials`, `-Seed`, `-Scheduler`, `-CvFolds`, `-Compare`, `-NoPromote`, `-DryRun` — there is no `-SearchSpace`. `--search_space` is a real CLI choice (`config.py:265-277`, `SEARCH_SPACE_PROFILES = ("full", "reduced")`) and `full` is defined for imputation (`opt.py:112-148` adds `EPOCHS_DECODE`/`LR_DECODE`/`WEIGHT_DECAY_DECODE` under `task == 'imputation'`). The script parameterises `-Scheduler`, the ADR 0003 factor, but not the ADR 0005 factor.
- **Consequence:** ADR 0005's decision is "sample only the decode-stage knobs instead of every knob". The twelve `-Compare` runs pit *reduced-promoted* against *task defaults*. Whatever they show — even a clean, large win — is evidence that **tuning at all** helps, not that **reducing the space** helps. The counterfactual that ADR 0005 rejected (a 40-trial `full` study on the same six pairs) is never run, so the reduction is adopted on an argument the experiment does not address. A `full` study at the same budget could plausibly win: it also samples `EPOCHS_DECODE`, which `reduced` pins at 150 in all four promoted files, and 40 trials over the nine-knob `full` space is not obviously worse than 40 over the five-knob one.
- **Direction:** add `-SearchSpace` (default `reduced`) and a matched `full` arm on the same six pairs at the same `-Trials`, `-Seed` and schedule; the promoted paths already collide (`hyperparameter_file` is profile-blind), so the full arm needs `-NoPromote` plus an explicit config path, or a profile segment in the promoted filename.

### F-08-3 - The studies never cross-validate: hyperparameters are chosen on one fixed validation split whose rows then make up 8.5-13% of every CV test fold they are evaluated on

- **Kind:** methodology
- **Severity:** high
- **Where:** `imputation_studies.ps1:60-64` (no `--cv_folds`) and `opt.py:200-227` (`args` has no `cv_folds`)
- **Evidence:** `-CvFolds` is declared at `:24` and referenced exactly once, at `:83`, inside the `-Compare` branch. The study branch never passes it. `opt.py`'s per-trial `Args` object sets `dataset_name, label_column, output_dir, metrics_dir, disable_mlflow, plot_losses, save_model, seed, task, score_search_objective, hyperparams_override, lr_scheduler, mlflow_run_role` — no `cv_folds` — so `resolve_training_request`'s `getattr(args, "cv_folds", None)` yields `None` and `build_folds` (`data.py:155-170`) takes the single predefined split. `opt.py:452` confirms it: the study parent is tagged `cv_folds=None`. The trial's checkpoint and score come from `hidden_validation = embedder.encode(evaluation_mask(validation_frame, rate, seed, fold_ordinal), ...)` (`decoding.py:77-82`), drawn once per study, so all 40 trials are ranked on the *same* cells of the *same* rows. Measured:

  ```
  credit-g_20nan : rows=1000 predefined train=800 val=100 test=100
      observed cells in val split = 1607 -> ~321 scored cells rank all 40 trials
      predefined-VAL rows landing in each CV test fold (seed 42): [17,18,26,22,17] = 100/100
      share of each CV test fold that was selection data: [.085,.090,.130,.110,.085]
  kr-vs-kp_20nan : val=320 -> ~1845 scored cells; shares [.089,.106,.110,.088,.108]
  spambase_20nan : val=461 -> ~4185 scored cells; shares [.098,.107,.099,.111,.087]
  ```

  `build_folds` with `cv_folds=5` runs `StratifiedKFold(5, shuffle=True, random_state=seed)` over the **whole** frame (`data.py:183-191`), so every predefined-split row, validation and test alike, reappears in some CV test fold.
- **Consequence:** two distinct effects, both pointing the same way. (a) *Winner's curse*: the promoted configuration is the argmax over 40 draws of a score computed on ~321 cells (credit-g), one mask draw, one split. The reported `optuna/best_objective_value` overstates what the configuration does, and the comparison run is the first honest re-measurement. (b) *Selection leakage*: 8.5-13% of the rows in each CV test fold of the comparison are the rows that chose the configuration. Small, but one-directional — it can only flatter the promoted arm, never the defaults arm, which saw none of them. Neither effect is estimated anywhere, and the `-Compare` summary presents the result as if the promoted arm had been evaluated on untouched data.
- **Direction:** either pass `--cv_folds` to the study so selection and evaluation use the same protocol and the winner's curse is at least averaged over folds, or — better for the leakage — hold a partition out of the study entirely and evaluate the comparison only there. Nested CV (promote per outer fold) is the clean form and costs 5x the study budget.

### F-08-4 - The twelve comparison runs cannot support a generalisation claim: one seed, one missingness draw, unpaired t-intervals over five correlated folds, six of them read at once

- **Kind:** methodology
- **Severity:** high
- **Where:** `imputation_studies.ps1:82-133`
- **Evidence:** both arms run `--cv_folds 5 --lr_scheduler cosine --seed 42`, and the closing line tells the reader to compare `cv/test/impute/induced/impute_score/mean with ci95_lower / ci95_upper`. Those bounds come from `summary._summarize_metric` → `_student_t_interval` over the five per-fold values of a **single arm** (`summary.py:148-158`). Nothing computes the per-fold **difference** between arms, even though the folds are identical (same `random_state=seed`) and a paired difference is available for free. Measured fold spread on the two multi-fold imputation runs that exist:

  ```
  credit-g_20nan  folds=2 mean=0.9164 sd=0.0023  vals=[0.9180, 0.9147]
  spambase_20nan  folds=2 mean=0.8990 sd=0.0058  vals=[0.9031, 0.8949]
  ```

  The seed also fixes the CV partition, the torch init and the evaluation masks; the `_20nan`/`_40nan` tables are one fixed draw of missingness shipped in `datasets/processed_datasets/`, identical for both arms.
- **Consequence:** precisely what the design *can* support: "on these six tables, with this one missingness draw, this one 5-fold partition at seed 42, under the cosine schedule, the promoted configuration's mean induced `impute_score` is X and the defaults arm's is Y." That is a descriptive, fully conditional statement. What it *cannot* support: (a) that the advantage survives another seed — seed 42 is the only draw, and CV fold variance is not resampling variance (the five training sets overlap 75% pairwise, so a t-interval over them systematically understates how much the number would move under a different partition; there is no unbiased estimator of k-fold CV variance); (b) that it survives another missingness draw — there is only one; (c) a 95% coverage statement about the *difference*, which is what the claim is about, since two overlapping single-arm intervals are not a test of a difference and discard the pairing that the shared folds give for free; (d) any of it at 95% across the family — six pairs are judged simultaneously with no correction, so under the null the chance that at least one pair shows a "significant" win is about 26%, not 5%. The bare fold spreads above (sd 0.002-0.006) mean the intervals will look reassuringly tight; that tightness is the misleading part, because it measures fold wobble on one table and is read as generalisation.
- **Direction:** in order of cost. (1) Report the per-fold paired difference and its interval instead of two overlapping single-arm intervals — free, uses runs you already have. (2) Repeat the twelve comparison runs at three or more seeds and report the seed-to-seed spread alongside the fold spread; this is the only thing that turns "on this partition" into "on this table". (3) Say in the write-up that six pairs are being read at once, and either correct or present them as six separate conditional observations rather than one verdict.

### F-08-5 - The search optimises the masked proxy; the comparison is judged on the induced population, and the study's winner is not the best trial on it

- **Kind:** design
- **Severity:** medium
- **Where:** `imputation_studies.ps1:133` vs `src/training/types.py:66-71`
- **Evidence:** `TaskSpec("imputation", ranking_metric="impute/masked/impute_score", ..., search_objective="validation/impute/masked/impute_score")`. The `masked` population is cells the scorer *hides* among cells the variant observes (`decoding.py:137-146`); `induced` is the variant's genuinely missing cells, ground-truthed from the complete sibling (`decoding.py:174-182`). ADR 0005 states both sides explicitly — `:111-112` "the objective mirrors the ranking metric's population: masked cells only; the induced population stays a test-fold report", and `:161` / `:191` "Headline `cv/test/impute/induced/impute_score/mean`". So the launcher is faithful to the ADR; the ADR decides to select on one population and report on another, and offers no evidence that the two agree. I measured the agreement on the eleven preserved trials of the live `spambase_20nan` study (`results/spambase_20nan/temp_trials/spambase_20nan/*/metrics.csv`):

  ```
  n = 11
  spearman(objective, headline induced)      = 0.945  (p = 1.1e-05)
  spearman(objective, companion masked-test) = 0.827  (p = 1.7e-03)
  trial the search would pick (min objective): obj=0.9037  headline=0.9388
  trial best on the headline                 : obj=0.9093  headline=0.9053
  headline rank of the search winner: 3 of 11
  ```
- **Consequence:** the good news first, and it partly clears the concern: the proxy tracks the headline well (rank correlation 0.945), so the search is not optimising something unrelated. The problem is the residual. Over eleven trials the search's pick was third on the headline, and 0.033 worse than the headline-best trial. The headline spread over all eleven trials was 0.905 to 0.998. A selection error of 0.033 is the same order as the entire tuning gain the twelve comparison runs are trying to resolve, and it is *not* noise that averages out — it is a systematic consequence of ranking on a different population, so it recurs in every one of the six studies. Concretely: a pair can come back showing "promoted barely beats defaults on induced" when a differently-ranked trial from the same 40 would have won clearly, and the write-up would read that as a weak search rather than a mis-specified objective.
- **Direction:** either rank trials on a validation-split *induced* score (the complete sibling is already loaded; the validation split has genuinely missing cells in every `_XXnan` variant) so the objective and the headline name the same population, or keep the masked objective and make the headline the masked companion, so the reported number is the one the search maximised. Reporting both is fine; selecting on one and headlining the other needs the correlation above stated in the ADR, with its residual.

### F-08-6 - `experiment_imputation.ps1` writes the classification-keyed config file, which `--task imputation` reads only as a fallback — on the nan variants it is shadowed and silently ignored

- **Kind:** bug
- **Severity:** medium
- **Where:** `experiment_imputation.ps1:24-28` and `:52`
- **Evidence:** `Get-HyperparamsPath` returns `datasets/hiperparams/<base>/<dataset>.json`. `config._load_base_hyperparameters:49-56` builds `candidates = [hyperparameter_file(name, task)]` and appends the classification file **only as a fallback**, so an imputation run reads `<dataset>.imputation.json` first. Run against the real loader on the current tree:

  ```
  path the script writes to       : datasets/hiperparams/credit-g/credit-g_20nan.json
  path an imputation run reads 1st: datasets/hiperparams/credit-g/credit-g_20nan.imputation.json
  actual config_source for --dataset_name credit-g_20nan --task imputation TODAY:
      datasets/hiperparams/credit-g/credit-g_20nan.imputation.json
      -> EPOCHS_PRE it would train with: 300   (the sweep asked for 200)
  ```

  Meanwhile line 52 prints `[setup] $path -> EPOCHS_PRE=$PretrainEpochs EPOCH_FINE=$FinetuneEpochs` unconditionally.
- **Consequence:** two failures that face opposite directions. (a) `experiment_imputation.ps1 -Datasets @("credit-g_20nan")` — the natural invocation once the script is an imputation launcher, since `_00nan` has nothing to impute — reports `EPOCHS_PRE=200` and trains at 300, because `imputation_studies.ps1` promoted a file that shadows it. Worse, the sweep then silently runs at the *promoted* dropout, mask probability and decode rates, so the four schedule arms are not the controlled comparison the script claims; they are four runs of somebody else's tuned configuration. (b) The reverse: `<dataset>.json` is exactly the fallback that `imputation_studies.ps1:79-81` warns about. If a sweep is interrupted and leaves one behind, `-Compare`'s "defaults" arm loads it instead of the task defaults, while the on-screen summary table still records `Run = "defaults"`. Only `params.config_source` in MLflow tells the truth, and the summary line never mentions checking it for the defaults arm specifically.
- **Direction:** have the sweep write `hyperparameter_file(dataset, "imputation")`, i.e. `<dataset>.imputation.json`, and refuse to start if a promoted file is already there rather than silently sitting behind it. For `-Compare`, assert `config_source == "defaults"` on the control arm before recording it as defaults.

### F-08-7 - The sweep runs imputation on complete variants, where the headline population does not exist, and then prints a classification metric recipe

- **Kind:** bug
- **Severity:** medium
- **Where:** `experiment_imputation.ps1:12` (`$Datasets = @("vehicle_00nan", "credit-g_00nan")`) and `:99-100`
- **Evidence:** `data.load_complete_sibling:92-110` — `complete = <base>_00nan.csv`; `if complete == variant or not complete.exists(): return None`. A `_00nan` dataset is its own sibling, so it returns `None`, and `decoding.py:174` (`if complete_sibling is not None:`) skips the entire induced block. The metric header of a real `_20nan` imputation run confirms which keys exist at all:

  ```
  impute/masked/*  impute/induced/*  validation/impute/masked/*
  ```

  No `f1_macro` anywhere — an imputation fold's metric dict is built only from `impute/...` keys (`decoding.py:149-197`). The script's closing lines nonetheless say `Metric: cv/test/f1_macro/mean with cv/test/f1_macro/ci95_lower and ci95_upper`. The JSON it writes is also the classification key set: `EPOCH_FINE` maps to `finetuning_epochs`, never read by the decode stage, and `LABELS` to a classifier head that is never built; the decode knobs are absent, so `EPOCHS_DECODE=150`, `LR_DECODE=0.001`, `WEIGHT_DECAY_DECODE=0.0019`, `LAMBDA_NUM=1.0`, `EVAL_MASK_RATE=0.2` all fall to defaults (verified against `Hyperparameters.from_mapping`).
- **Consequence:** once F-08-1 is fixed and the script really runs imputation, its eight runs produce no `impute/induced/*` metric at all — the ADR 0005 headline is simply absent — and the operator, following the script's own printed recipe, queries `cv/test/f1_macro/mean` and gets an empty column for every run. The schedule question for imputation ends up answered, if at all, only on the synthetic masked proxy on tables with no real gaps, which is the configuration furthest from the task. `-FinetuneEpochs` is a parameter the operator can set that changes nothing, and the `[setup]` line advertises it.
- **Direction:** default the sweep to nan variants so the induced population exists; build the JSON from `config.complete_configuration(values, "imputation")` so the key set is the task's and the decode knobs are named rather than defaulted; drop `-FinetuneEpochs` and add `-DecodeEpochs`; print `cv/test/impute/induced/impute_score/mean` with its bounds.

### F-08-8 - Neither launcher isolates `--metrics_dir`, so 240 Optuna trials and then the defaults arm overwrite the per-dataset metrics files

- **Kind:** design
- **Severity:** medium
- **Where:** `imputation_studies.ps1:42-48` (`Invoke-Trainer` passes no `--metrics_dir`), `opt.py:206-208`, `src/training/artifacts.py:85-88`
- **Evidence:** `opt.py` deliberately isolates the trial's results — `temp_dir = Path(self.output_dir)/dataset/"temp_trials"`, commented "Use a temporary directory for trials - we will only save the best one" — but the very next line passes the metrics directory straight through: `args.metrics_dir = getattr(self, "metrics_dir", "metrics")`, which the launcher never sets, so it is `metrics`. `ArtifactWriter.write_metrics` then does `frame.to_csv(root_path, index=False)` on `metrics/<dataset>_metrics.csv` — a plain overwrite, not an append. The live run's log agrees, once per trial: `Metrics also saved to: metrics\credit-g_20nan_metrics.csv`. `metrics/credit-g_20nan_metrics.csv` is currently two lines: a header and one row.
- **Consequence:** each of the 40 trials in a study overwrites the dataset's canonical metrics file; after the study it holds trial 39's single-split numbers, which are not a comparable run at all. In `-Compare` mode the same applies between arms: the defaults run executes second and overwrites the promoted run's row, so after the twelve runs the local metrics tree holds only defaults-arm numbers for all six pairs and the promoted half of the comparison survives on disk only inside timestamped `results/` directories and in MLflow. `metrics/` is gitignored so nothing is committed wrongly, but AGENTS.md asks explicitly to preserve generated `metrics/`, and the project's own launcher is what destroys it.
- **Direction:** give the trials their own metrics directory beside `temp_trials` in `opt.py`, and have `imputation_studies.ps1` pass a per-arm `--metrics_dir` in `-Compare` mode (e.g. `metrics/compare/<dataset>/promoted` and `.../defaults`) so both arms survive on disk and can be diffed without MLflow.

### F-08-9 - `-Scheduler X -Compare` compares a cosine-tuned configuration against defaults, both forced to run under X

- **Kind:** design
- **Severity:** low
- **Where:** `imputation_studies.ps1:9-11` and `:82-85`
- **Evidence:** the usage block suggests `.\imputation_studies.ps1 -Scheduler plateau -NoPromote` for "exploration". `-Compare` uses the same `-Scheduler` parameter and passes it as `--lr_scheduler`, and `config.load_hyperparameters:32-34` makes the flag win over the file: `hyperparameters = dataclasses.replace(hyperparameters, lr_scheduler=lr_scheduler)`. The promoted files carry `"LR_SCHEDULER": "cosine"` because `promote_best_configuration` stamps the study's schedule.
- **Consequence:** the natural follow-up to the suggested exploration — `-Scheduler plateau -Compare` — runs the *cosine*-tuned promoted configuration under `plateau` against defaults under `plateau`. That answers nothing about plateau (the configuration was never tuned for it) and nothing about the promoted configuration (it is not being run as promoted). The result is readable as a plateau result and is not one, and the only trace is `params.config_source` naming a file whose `LR_SCHEDULER` was overridden.
- **Direction:** in `-Compare`, refuse or warn when `-Scheduler` differs from the promoted file's `LR_SCHEDULER`, or key the promoted filename by schedule so a plateau comparison can only read a plateau-tuned file.

### F-08-10 - `experiment_imputation.ps1` takes its backups and writes the config outside the `try`, so a mid-setup failure leaves a clobbered file with no restore

- **Kind:** bug
- **Severity:** low
- **Where:** `experiment_imputation.ps1:42-53` (setup loop) vs `:57` (`try`) / `:80-93` (`finally`)
- **Evidence:** the loop that records `$backups[$path]` and calls `[System.IO.File]::WriteAllText(...)` runs at lines 43-53. The `try` opens at line 57 and the restoring `finally` at line 80. `$ErrorActionPreference = "Stop"` is in force, so any failure inside the setup loop — a locked file, a read-only directory, a `New-Item` failure on the second dataset after the first has already been written — terminates the script before the `finally` exists.
- **Consequence:** the first dataset's `<dataset>.json` is left holding the sweep's configuration, permanently. Since nothing under `datasets/hiperparams/` is tracked (`git ls-tree -r main -- datasets/hiperparams` is empty), there is no `git checkout` to recover with, and the leftover file then silently becomes the config for every later imputation run of that variant via the fallback described in F-08-6.
- **Direction:** move the setup loop inside the `try`, so the `finally` covers a partially-applied setup; record the backup before the write for each dataset so a half-finished loop still restores what it touched.

## Checked and cleared

- **Untracked `datasets/hiperparams/` is not a crashed restore.** It holds exactly four files, all `<dataset>.imputation.json`, for `credit-g_20nan`, `credit-g_40nan`, `kr-vs-kp_20nan`, `kr-vs-kp_40nan` — the first four of the six pairs, in the script's own iteration order. `results/imputation_studies_20260911_011008.err` was last written at 09:41 and `results/spambase_20nan/optuna_20260911_075737/` exists with a live `temp_trials`: the fifth study is running now. These are normal `--promote_best` output from an in-flight sweep, which `imputation_studies.ps1:139` tells the operator to commit. No `<dataset>.json` is present, so no sweep's restore step was skipped.
- **`imputation_studies.ps1`'s move-aside restore is exception-safe.** `Move-Item $promoted $aside -Force` sits *outside* the `try` (`:107`) and the restore inside the `finally` (`:113-117`) is guarded by `Test-Path $aside`, so an interrupted run restores, a failure before the move has nothing to restore, and a stale `.aside` from a previous crash is overwritten by `-Force` and then moved back. Idempotent in every ordering I could construct. (F-08-10 is the *other* script.)
- **Both comparison arms genuinely share the schedule.** The promoted files carry `"LR_SCHEDULER": "cosine"` and `-Compare` passes `--lr_scheduler cosine`, which `config.py:32-34` applies to both arms, so the defaults arm's `cosine_legacy` default is overridden identically. The arms differ only in `DROPOUT`, `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE` and (on mixed tables) `LAMBDA_NUM` — exactly the reduced space.
- **Today's `-Compare` defaults arm really is defaults.** `git ls-tree -r main --name-only -- datasets/hiperparams` and the same on `HEAD` are both empty: no classification config file has ever been tracked, and none exists in the worktree. So the fallback in `config.py:50-51` currently finds nothing and `config_source` will read `"defaults"`. F-08-6(b) is a live hazard, not a present error.
- **`LAMBDA_NUM` is correctly held on single-type tables.** `opt.py:106-107` gates it on `mixed_columns`, computed once per study from the declared column types (`opt.py:421-424`), not from dtypes. Both `kr-vs-kp` promoted files show `"LAMBDA_NUM": 1.0` (the default) while both `credit-g` files show tuned values — the gate works, and 40 trials are not spent on an inert dimension.
- **Run artifacts are not clobbered between arms or schedule steps.** `ArtifactWriter.__init__` builds `results_dir = output_dir / dataset_name / <timestamp>`, so the two comparison arms and the four schedule arms each get their own directory. Only the shared `metrics/` file is overwritten (F-08-8).
- **The suggested MLflow filters do not pool across datasets.** `get_or_create_experiment(args.dataset_name)` gives each dataset variant its own experiment, so `grouped by params.config_source` runs within one table and the `impute_score` ratio — which is relative to a per-fold mean/mode baseline — is only ever compared where the baseline is shared.
- **No trials failed in the four completed studies.** `grep -c "failed with error"` over the 6 MB study log returns 0, so the reduced space is not quietly losing trials to crashes and the effective budget really is 40 per pair.
- **`$PSNativeCommandUseErrorActionPreference` does not abort `experiment_imputation.ps1` on this host.** `imputation_studies.ps1:33` sets it to `$false` with a comment about PowerShell 7.4 turning a non-zero native exit code into a terminating error under `"Stop"`; `experiment.ps1`/`experiment_imputation.ps1` have no such guard, which looked like a first-failure abort. On this machine (PowerShell 7.6.6) the variable already defaults to `False`, and `$ErrorActionPreference='Stop'; & cmd /c "exit 3"` did not throw. The guard's absence is a portability gap, not a defect here, so I am not reporting it.

## Open questions

- **Does the missingness in `_XXnan` come from one MCAR draw per variant, and is it the same draw across levels?** `generate_splits.py` is where this would be settled. It decides whether the six pairs are six independent observations or two families of three (20/40 of the same base table), which changes both the multiplicity correction in F-08-4 and whether a second missingness draw is even available as a repetition axis.
- **What is the seed-to-seed spread of `cv/test/impute/induced/impute_score/mean`?** The fold spread I measured (sd 0.002-0.006 over two folds) is the wrong variance and is a 1-dof estimate besides. Two extra `-Compare` runs of a single pair at seeds 43 and 44 would settle whether the tuning gain survives a partition change — it is the cheapest evidence that would upgrade F-08-4 from "cannot support" to "supports, with this spread".
- **Will the spambase studies complete?** Two of six promoted files are still missing. If either spambase study dies, `-Compare` silently skips that pair (`imputation_studies.ps1:86-90`) and the comparison reports four or five pairs with no record of why — a survivorship filter on which pairs reach the write-up. Worth checking the two summary tables against each other before drawing the conclusion.
- **Does the masked/induced rank correlation hold on the mixed-type tables?** I could only measure it on `spambase_20nan` (all-numerical, 11 trials). `credit-g` is mixed and `kr-vs-kp` all-categorical; their `temp_trials` directories were removed when the studies finished. Re-running one study with the trial artifacts retained, or reading the per-trial runs out of `mlflow.db` once the sweep is idle, would tell whether F-08-5's residual is larger where `LAMBDA_NUM` is in play.
