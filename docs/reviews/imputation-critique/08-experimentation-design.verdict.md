# 08 - Experimental design of the launchers: verdicts

_Adversarial verification of `08-experimentation-design.md`. 2026-09-11._

**Method:** read `experiment_imputation.ps1` (HEAD and worktree), `imputation_studies.ps1`,
`experiment.ps1`, `src/training/{config,types,data,decoding,summary,artifacts}.py`,
`opt.py:75-240` and `:320-470` and `:570-585`, `docs/adr/0005-reduced-optuna-search-for-imputation.md`
in full, and two sources the critique never opened:
`docs/tickets/imputation-optuna-reduced/issues/05-launcher-and-studies.md` and
`.../06-comparison-and-docs.md`. No training run was started: the six-study sweep is live on
the GPU (`results/spambase_20nan/temp_trials/` gained trials while I worked) and every claim
here is settled by calling the real loader, the real splitter, or reading files the running
sweep already wrote.

Commands and output that decided verdicts.

```
$ git show HEAD:experiment_imputation.ps1 | tr -d '\r' > a; git show HEAD:experiment.ps1 | tr -d '\r' > b; diff a b
IDENTICAL_AT_HEAD                                   (no output from diff)
$ git log --diff-filter=A --no-renames --oneline -- experiment_imputation.ps1
9a38b90 feat(optuna): rank imputation trials on the validation split
$ git log -1 --format='%ci' 9a38b90            -> 2026-09-11 00:21:58 -0300
$ ls -1 results/vehicle_00nan/  | tail -1      -> 20260909_214200
$ ls -1 results/credit-g_00nan/ | tail -1      -> 20260909_215425      (both predate the commit by 2 days)
$ ls .github/workflows                         -> No such file or directory   (there is no CI)
$ git ls-tree -r main --name-only -- datasets/hiperparams | wc -l   -> 0   (same on HEAD)
$ grep -rn "paired" --include=*.py .           -> hits only under ./.venv/; nothing in this project
```

The real config loader and the real sibling loader, on the current tree:

```
credit-g_20nan --task imputation
    config_source = datasets/hiperparams/credit-g/credit-g_20nan.imputation.json
    EPOCHS_PRE = 300  DROPOUT = 0.30  PROB_MASCARA = 0.4     (the sweep JSON asks for EPOCHS_PRE 200)
Hyperparameters.from_mapping(<the JSON New-HyperparamsJson writes>)
    -> decode_epochs 150  lr_decode 0.001  wd_decode 0.0019  lambda_num 1.0  eval_mask_rate 0.2   (all defaults)
load_complete_sibling: vehicle_00nan -> None   credit-g_00nan -> None   credit-g_20nan -> (1000, 21)
```

Fold overlap, real `prepare_dataset` + `StratifiedKFold(5, shuffle=True, random_state=42)`:

```
credit-g_20nan  800/100/100    val rows per CV test fold [17,18,26,22,17]=100/100  shares [.085,.090,.130,.110,.085]
kr-vs-kp_20nan  2556/320/320                             [57,68,70,56,69]=320/320  shares [.089,.106,.110,.088,.108]
spambase_20nan  3679/461/461                             [90,98,91,102,80]=461/461 shares [.098,.107,.099,.111,.087]
```

Search objective vs headline on the live `spambase_20nan` study, the critic's snapshot and mine:

```
first 11 trials: rho=0.945  pick obj=0.9037 ind=0.9388  best-ind=0.9053  rank=3/11  gap=0.0334   <- critic
first 20 trials: rho=0.812  pick obj=0.8993 ind=0.9063  best-ind=0.9053  rank=2/20  gap=0.0009   <- nine trials later
```

Missingness draws (the critique's own open question, settled):

```
credit-g: missing20=4000  missing40=8000  nested=False  cells missing at 20 but observed at 40 = 2459
kr-vs-kp: missing20=23004 missing40=46008 nested=False  13764
spambase: missing20=52440 missing40=104880 nested=False 31562
```

## Verdicts

### F-08-1 - The committed `experiment_imputation.ps1` is byte-identical to `experiment.ps1`; only an uncommitted edit makes it an imputation launcher

- **Verdict:** CONFIRMED
- **Severity after review:** medium (critique said high)
- **Basis:** the diff at HEAD is empty; the only difference in the worktree is two tokens,
  `"--task", "imputation"`, inserted into the argument array at line 60. Everything else is
  unadapted: header "Learning-rate schedule sweep (ADR 0003), tier 1", usage lines naming
  `.\experiment.ps1`, `$Datasets = @("vehicle_00nan", "credit-g_00nan")`, the
  classification JSON builder, `-FinetuneEpochs`, the `cv/test/f1_macro` closing recipe.
  `-DryRun` prints the command and asserts nothing; no test touches the `.ps1` files.
- **Correction:** the digest states as fact that *"Running the committed file produced eight
  classification runs on vehicle_00nan/credit-g_00nan at seed 420, tagged
  task=classification"*. No such runs exist. The newest result directory for either default
  dataset is `20260909_21…`, two days before commit `9a38b90` (2026-09-11 00:21:58) created
  the file — the committed launcher has never been executed. The critique's own prose is
  conditional ("anyone who … *would have* got"), so this is a digest inflation, but the
  consequence paragraph also invokes "the CI/paper-reproduction path" and there is no CI:
  `.github/workflows` does not exist, and none of the three launchers is on `main`.
  Ticket 05's Comments record the file as *"work in progress in the working copy … do not
  overwrite it"*, so it was committed mid-adaptation rather than shipped as finished. What
  is left is real and worth fixing: a file at HEAD that silently runs the wrong task, on a
  branch that is meant to merge, with no test and no guard. Medium, not high.

### F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the twelve comparison runs cannot test reduced-vs-full

- **Verdict:** UNSOUND
- **Severity after review:** low (critique said high)
- **Basis (premise, true):** line 62 is the literal `"--search_space", "reduced"`; the
  `param()` block has no `-SearchSpace`; `full` is a real imputation profile
  (`opt.py:135-148` adds `EPOCHS_DECODE`/`LR_DECODE`/`WEIGHT_DECAY_DECODE` under
  `task == 'imputation'`) and `config.py:265-277` exposes the flag. All verified.
- **Correction (reasoning, does not follow):** the critique's consequence is that "the
  experiment cannot test the decision it exists to test" and "the reduction is adopted on
  an argument the experiment does not address". ADR 0005 never claims otherwise. Its
  decision 6 is titled **"Checking the reduction, and proving the tuning helped"** and names
  two different mechanisms for the two things: fANOVA `optuna/importance/<knob>` on the
  study parent checks the reduction; the twelve promoted-vs-defaults runs prove the tuning
  helped. Under **Considered options** the ADR lists *"A full-profile pilot with fANOVA to
  rank the held knobs. **Deferred:** fifteen dimensions at 50 trials rank noisily."* The
  counterfactual is explicitly deferred with a stated reason, not silently skipped, and the
  launcher implements the protocol the ADR prescribes. Assigning the twelve runs a purpose
  the ADR assigns elsewhere is the gap.
  The proposed remedy is also not the matched arm it claims to be: the `full` profile
  samples `EPOCHS_PRE` in 20..60 and `EPOCHS_DECODE` in 20..60 (`opt.py:124,134`) while
  `reduced` holds them at the task defaults 300 and 150. A "matched `full` arm at the same
  `-Trials`, `-Seed` and schedule" would therefore promote a configuration pre-trained for
  at most 60 epochs against a reduced winner pre-trained for 300 — it would test the epoch
  budget at least as much as the width of the space, which is exactly the confound ADR 0005
  decision 2 held the epoch counts to avoid ("a trial costs what the promoted configuration
  will cost to run").
  The residual truth, and why this survives at low rather than none: fANOVA ranks only the
  knobs a study **sampled**, so it can expose an over-inclusion (a sampled knob that turns
  out inert — the mask rate, importance 0.02-0.04 in the three finished studies) but can
  never expose an under-inclusion (a held knob that would have mattered). The ADR says this
  itself. The reduction therefore rests on a principle plus a partial check, and the branch
  ships no way to run the missing half without editing the script.

### F-08-3 - Optuna studies never cross-validate; the config is chosen on one fixed validation split whose rows then form 8.5-13% of every CV test fold

- **Verdict:** SOUND
- **Severity after review:** medium (critique said high)
- **Basis:** every premise reproduced. `-CvFolds` is declared at `:24` and used only at
  `:83` in the `-Compare` branch; the study argument array `:60-64` omits `--cv_folds`;
  `opt.py`'s per-trial `Args` never sets `cv_folds`, so `resolve_training_request`'s
  `getattr(args, "cv_folds", None)` is `None` and `build_folds` (`data.py:155-170`) returns
  the single predefined split; `opt.py:452` tags the study parent `cv_folds=None`. Every
  trial is ranked on `hidden_validation`, drawn once per fold from a seeded stream
  (`data.evaluation_mask`), against `clean_validation`. The row-overlap table above is my
  own run of the real splitter and matches the critique exactly: 100/100, 320/320 and
  461/461 predefined-validation rows land in some CV test fold, at 8.5-13% of each fold.
- **Correction (two, both shrinking it):** the scored-cell count is not an estimate. The
  trial row the sweep left in `metrics/credit-g_20nan_metrics.csv` gives it directly:
  `validation/impute/masked/n_num_cells = 96`, `n_cat_cells = 187` — **283** cells rank all
  40 credit-g trials, not "~321". Second and more substantive: the critique never credits
  **common random numbers**. `args.seed = self.seed` for every trial and the sampler is
  `TPESampler(seed=args.seed)`, so all 40 trials share one evaluation mask, one split and
  one torch init. The noise in each trial's score is therefore almost perfectly correlated
  across trials, which is precisely the variance-reduction trick that makes a *ranking* far
  more reliable than 283 independent cells would suggest. It does nothing for the absolute
  level (`optuna/best_objective_value` still overstates, as the critique says) and nothing
  for the leakage, but it is the reason the winner's-curse half of this finding is weaker
  than stated.
  The leakage half is real, one-directional, and small: the channel is four or five
  hyperparameter numbers, carried into 8.5-13% of each test fold. Medium.

### F-08-4 - Two overlapping unpaired single-arm t-intervals over five correlated folds, one seed, one missingness draw, six pairs read at once

- **Verdict:** SOUND
- **Severity after review:** medium (critique said high)
- **Basis, and the part that gets sharper:** `summary._summarize_metric` →
  `_student_t_interval` over the five per-fold values of **one** arm is the only interval in
  the repository. `grep -rn "paired" --include=*.py .` returns nothing outside `.venv`:
  nothing anywhere computes a per-fold difference between two runs, although the folds are
  identical between arms because both pass `--seed 42`. Meanwhile ADR 0005:163 and ticket
  06 both call the design *"the glossary's **paired** single-seed CV protocol: fold-level
  comparisons are paired"*. So the declared protocol is paired and neither the code nor the
  launcher's printed recipe (`imputation_studies.ps1:133`) produces a paired quantity. That
  mismatch between the documented protocol and what an operator can actually read is the
  strongest thing in this finding, and the critique reaches it only as remedy (1).
- **Correction:** the claim the critique leads with — that the design "cannot support a
  generalisation claim" over seeds — is a limit the project has already declared, not a
  hidden flaw. Ticket 06: *"conclusions do not quantify seed variability. Multi-seed
  robustness is out of scope."* Criticising a stated scope boundary as if it were an
  oversight is what pulls this to high; the implementation/documentation mismatch and the
  multiplicity point are what keep it at medium.
  I also settled the critique's own open question, which bears on its multiplicity
  arithmetic: the `_20nan` and `_40nan` masks are **independent draws, not nested** — on
  credit-g, 2459 of the 4000 cells missing at 20% are observed at 40%. So the six pairs are
  three base tables × two independent missingness draws laid over the same rows and columns.
  That is weak replication over the missingness draw (better than the critique's "there is
  only one"), and it is *not* six independent observations either, so the uncorrected
  six-way read the critique flags stands — with an effective family somewhere between three
  and six, not six.

### F-08-5 - The search ranks on the masked proxy while the comparison headlines the induced population; the search winner is not the best trial on the headline

- **Verdict:** UNSOUND
- **Severity after review:** low (critique said medium)
- **Basis (premise, true):** `TaskSpec("imputation", ranking_metric="impute/masked/impute_score",
  …, search_objective="validation/impute/masked/impute_score")` at `types.py:66-71`; masked
  cells are hidden among observed cells (`decoding.py:137-146`), induced cells are the
  variant's real gaps ground-truthed from the complete sibling (`decoding.py:174-182`); ADR
  0005 decision 3 says "the objective mirrors the ranking metric's population: masked cells
  only; the induced population stays a test-fold report" and decision 6 headlines
  `cv/test/impute/induced/impute_score/mean`. The launcher is faithful to the ADR, and the
  ADR does select on one population and report on another without stating the agreement.
- **Correction (the consequence does not hold):** I reproduced the critique's numbers to the
  digit on the first 11 preserved trials — rho 0.945, search pick obj 0.9037 / induced
  0.9388, rank 3/11, gap 0.0334. The study had produced nine more trials by the time I read
  it, and the picture inverts: rho 0.812, search pick obj 0.8993 / induced 0.9063, **rank
  2/20, gap 0.0009**. The regret fell by a factor of 37. The critique's consequence —
  "a selection error of 0.033 … is the same order as the entire tuning gain … and it is
  *not* noise that averages out — it is a systematic consequence" — is contradicted by nine
  further trials of the very study it measured. What was measured was a mid-flight snapshot
  of a running search, presented as a stable property.
  Do not over-correct: rho *fell* from 0.945 to 0.812, so the proxy genuinely is imperfect,
  the study is still running and the final figure is unknown, and one study on one
  all-numerical table cannot establish agreement on the mixed and all-categorical tables
  either. The design asymmetry is real and the ADR should state the correlation with its
  residual. But it is a documentation gap of low consequence, not a mis-specified objective
  costing the experiment its effect size.

### F-08-6 - The sweep writes `<dataset>.json`, which `--task imputation` reads only as a fallback, so on nan variants its config is silently shadowed while `[setup]` claims it was applied

- **Verdict:** CONFIRMED
- **Severity after review:** low (critique said medium)
- **Basis:** `Get-HyperparamsPath` (lines 24-28) returns `datasets/hiperparams/<base>/<dataset>.json`;
  `config._load_base_hyperparameters:49-51` builds `candidates = [hyperparameter_file(name, task)]`
  and appends the classification file only behind it. Run against the real loader:
  `--dataset_name credit-g_20nan --task imputation` resolves `config_source` to
  `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json` with `EPOCHS_PRE = 300`
  while the sweep JSON asks for 200, and line 52 prints
  `[setup] … -> EPOCHS_PRE=200 EPOCH_FINE=150` unconditionally. The four schedule arms
  would run at the promoted `DROPOUT 0.30` / `PROB_MASCARA 0.4` / `LR_DECODE 0.00254`.
- **Correction:** severity down one step, for two reasons. The hazard is latent behind
  F-08-1: on the committed datasets (`vehicle_00nan`, `credit-g_00nan`) no imputation-keyed
  file exists, so `config_source` is `defaults` and the sweep's file is read normally; the
  shadowing needs someone to first fix the task flag *and* point `-Datasets` at a nan
  variant. And the reverse hazard (b) is not live: `ls -R datasets/hiperparams` shows only
  four `.imputation.json` files, and `git ls-tree` on both `main` and `HEAD` returns nothing
  under that directory, so no `<dataset>.json` exists anywhere to be picked up as a
  "defaults" arm. One defect, one fix, shared with F-08-1 and F-08-7.

### F-08-7 - The sweep targets `_00nan`, where no induced metric exists at all, then prints an `f1_macro` recipe an imputation run never emits

- **Verdict:** CONFIRMED
- **Severity after review:** low (critique said medium)
- **Basis:** `$Datasets = @("vehicle_00nan", "credit-g_00nan")` at line 12.
  `load_complete_sibling` returns `None` for both — verified by calling it — so
  `decoding.py:174` (`if complete_sibling is not None:`) skips the entire induced block and
  the ADR 0005 headline population does not exist. The metric header of a real `_20nan`
  imputation run holds only `impute/masked/*`, `impute/induced/*` and
  `validation/impute/masked/*` keys; there is no `f1_macro` anywhere, and lines 99-100 tell
  the operator to read `cv/test/f1_macro/mean`. `Hyperparameters.from_mapping` on the JSON
  the script writes leaves `decode_epochs 150`, `lr_decode 0.001`, `wd_decode 0.0019`,
  `lambda_num 1.0`, `eval_mask_rate 0.2` at defaults, and `EPOCH_FINE`/`LABELS` name a stage
  and a head an imputation run never builds.
- **Correction:** severity only. This is the same unadapted-copy defect as F-08-1 and F-08-6,
  in a file that has never been run in this form, on a branch with no CI. Reporting one
  half-finished script as three medium findings and one low overstates the area's defect
  density; the four together are one fix of maybe thirty lines.

### F-08-8 - Neither launcher isolates `--metrics_dir`, so every Optuna trial and then the defaults arm overwrite `metrics/<dataset>_metrics.csv`

- **Verdict:** SOUND
- **Severity after review:** medium (critique said medium)
- **Basis:** `opt.py:206` isolates results into `temp_trials` with the comment "we will only
  save the best one" and `:208` passes the metrics directory straight through
  (`args.metrics_dir = getattr(self, "metrics_dir", "metrics")`);
  `run_hyperparameter_optimization` sets `objective.metrics_dir = getattr(args, "metrics_dir", "metrics")`
  and `imputation_studies.ps1`'s `Invoke-Trainer` never passes the flag, so it is `metrics`.
  `ArtifactWriter.write_metrics` does `frame.to_csv(root_path, index=False)` — a plain
  overwrite. The decisive evidence is on disk and I did not have to reason about it:
  `metrics/credit-g_20nan_metrics.csv` is a header plus **one row whose `fold` is
  `single_split` and which carries `validation/impute/masked/*` columns**. Only an Optuna
  trial emits a `validation/` key (`score_search_objective` is set programmatically by
  `opt.py` alone, `decoding.py:194-205`), and only a predefined-split run emits
  `single_split`. So the canonical metrics file for that dataset currently holds one
  trial's numbers, not any comparable run's. `kr-vs-kp_20nan`, `kr-vs-kp_40nan` and
  `credit-g_40nan` show the same shape and timestamps inside the sweep window.
  In `-Compare` the promoted arm runs first (`:95`) and the defaults arm second (`:110`), so
  after the twelve runs only defaults-arm rows survive on disk. `metrics/` is gitignored,
  but AGENTS.md asks explicitly to preserve generated `metrics/`.
- **Correction:** none to the claim. One addition to its consequence: the queued
  `-Scheduler plateau -NoPromote` exploration recorded in ticket 05's Comments runs the six
  studies *again* after the `-Compare` runs, so it will overwrite these files a third time.
  Whatever the comparison leaves behind on disk is transient unless `--metrics_dir` is fixed
  before that queue drains.

### F-08-9 - `-Scheduler X -Compare` runs the cosine-tuned promoted configuration under X, so the result reads as an X result and is not one

- **Verdict:** SOUND
- **Severity after review:** low (critique said low)
- **Basis:** `-Compare` reuses `-Scheduler` as `--lr_scheduler` at `:84`, and
  `config.load_hyperparameters:30-33` makes the flag win over the file
  (`dataclasses.replace(hyperparameters, lr_scheduler=lr_scheduler)`).
  `promote_best_configuration` stamps the study's schedule into the promoted file, and all
  four files on disk read `"LR_SCHEDULER": "cosine"`. So `-Scheduler plateau -Compare`
  compares a cosine-tuned configuration under plateau against defaults under plateau; only
  `params.config_source` records that the file's own schedule was overridden.
- **Correction:** none; the severity the critique assigned is right. Worth noting that the
  exploration actually queued (ticket 05 Comments) is `-Scheduler plateau -NoPromote`, the
  *study* branch, which never touches promoted files — so the hazard is a plausible next
  step rather than something already scheduled.

### F-08-10 - Backups and the config write sit outside the `try`, so a mid-setup failure leaves a clobbered file with no restore

- **Verdict:** CONFIRMED
- **Severity after review:** low (critique said low)
- **Basis:** the loop that records `$backups[$path]` and calls
  `[System.IO.File]::WriteAllText` is lines 43-53; `try` opens at 57 and the restoring
  `finally` at 80-93; `$ErrorActionPreference = "Stop"` is set at 21. A failure on the
  second dataset after the first has been written — locked file, read-only directory, a
  `New-Item` failure — terminates before any `finally` exists. `git ls-tree -r` on `main`
  and on `HEAD` under `datasets/hiperparams` both return nothing, so there is no tracked
  copy to restore from.
- **Correction:** none to the claim; one sharpening that makes the consequence *worse*, not
  better. Because no `<dataset>.json` exists for any dataset today, `$backups[$path]` is
  `$null` for every entry, so even the successful path's "restore" is a delete. A crash
  inside the loop therefore leaves a file that nothing in the repository or in git can
  distinguish from an intentional configuration, and it becomes the fallback config for
  every later imputation run of that variant (F-08-6b). Still low: the window is two file
  writes, in a script nobody has run.

## What this critique missed

- **Four findings are one defect.** F-08-1, F-08-6, F-08-7 and F-08-10 are all
  "`experiment_imputation.ps1` is an unadapted copy of `experiment.ps1`", reported once per
  symptom. Ticket 05's Comments say so in as many words — *"the user is adapting a copy of
  `experiment.ps1` … work in progress in the working copy"* — and the critique cites that
  ticket's sibling but never opens it. A reader counting findings sees four defects in this
  area where there is one incomplete file and one fix. The dimension's real weight sits in
  F-08-3, F-08-4 and F-08-8.
- **The proposed `full` counterfactual is not a matched arm.** F-08-2's remedy asks for a
  `full` study at the same trials, seed and schedule. `full` samples `EPOCHS_PRE` 20..60 and
  `EPOCHS_DECODE` 20..60; `reduced` holds them at 300 and 150. That arm would promote a
  configuration pre-trained for a fifth as long and would confound space width with epoch
  budget. Running it honestly means first pinning the epoch ranges in `full`, which the
  critique does not notice it needs.
- **Common random numbers are never credited.** All 40 trials of a study share one seed, so
  they share one evaluation mask, one split and one initialisation. That is a deliberate
  variance-reduction choice which makes the trial *ranking* substantially more reliable than
  the 283-cell objective suggests, and it is the main reason F-08-3's winner's-curse half is
  weaker than argued. It also means the promoted configuration is fitted to one mask
  realisation, which is the half that survives — the critique conflates the two.
- **Measuring an in-flight study yields a moving target.** F-08-5's headline number is drawn
  from `results/spambase_20nan/temp_trials/`, which was growing while the critique was
  written and while I verified it. Nine more trials moved the regret from 0.0334 to 0.0009.
  Worse, `opt.py:575-581` `rmtree`s `temp_trials` when a study finishes — which is *why*
  only spambase could be measured, and why any number read from there must be dated and
  re-read, or read from `mlflow.db` where the trial runs persist. The critique lists the
  missing tables as an open question without tracing the cause.
- **`-DryRun` is blind to exactly the survivorship risk the critique raises.**
  `imputation_studies.ps1:86` guards the promoted-file check with `if (-not $DryRun -and …)`,
  so a dry run prints both arms for all six pairs regardless of which promoted files exist.
  The critique's own open question — "if either spambase study dies, `-Compare` silently
  skips that pair … with no record of why" — has an obvious operator answer (dry-run first),
  and that answer does not work. A one-line change (`Test-Path` unconditionally, skip the
  execution rather than the check) would make the dry run report the survivorship filter
  before two hours of GPU time are spent.
- **The `_20nan`/`_40nan` question is answerable in ten seconds and changes the arithmetic.**
  Left open in the critique; the masks are independent draws, not nested (2459 of credit-g's
  4000 20%-missing cells are observed at 40%). So the six pairs give two independent
  missingness draws per table and the family is effectively three-to-six, not six — which
  softens the ~26% family-wise figure and, more usefully, means a second missingness draw
  *is* already available as a repetition axis rather than needing to be generated.
