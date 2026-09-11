# 05. The launcher and the six studies

Status: in-progress (Claude, session of 2026-09-11; six studies running)
Blocked by: 04
Plan task: 5. ADR 0005 decision 4. Wayfinder ticket 03.

## Goal

`imputation_studies.ps1` at the repository root, in the shape of `experiment.ps1`
(`[CmdletBinding()]`, `-DryRun`), running the six studies with promotion, then the six
studies themselves, against the shared `mlflow.db`.

## What the script runs

Parameters: `-Datasets @("credit-g", "kr-vs-kp", "spambase")`, `-Variants @("20nan",
"40nan")`, `-Trials 40`, `-Seed 42`, `-Scheduler cosine`, `-DryRun`, `-Compare` (ticket
06's twelve runs instead of the studies). Per study:

```
uv run --python 3.10 python main.py --dataset_name <base>_<variant> --task imputation --use_optuna --search_space reduced --n_trials <Trials> --lr_scheduler <Scheduler> --seed <Seed> --promote_best
```

## Acceptance criteria

- [x] `-DryRun` prints the six commands and runs nothing.
- [ ] Six promoted files exist: `datasets/hiperparams/<base>/<base>_<variant>.imputation.json`
      for the three datasets and two variants, each complete, each committed.
- [ ] Six study parents in the store with `is_optuna = true`, `task = imputation`,
      `search_space = reduced`, `lr_scheduler = cosine`, 40 completed trials each, an
      `optuna/importance/*` set and a `best_trial = true` trial.
- [ ] Under `## Comments`, per study: the best objective (a validation-split score), the
      winning configuration, the importances, and the wall-clock time against the ADR's
      estimate (66 s, 293 s, about 500 s per trial).

## Constraints

- Budget about nineteen hours in total; the studies are independent, so they may be
  split across sessions or machines with the same seed, and 30 trials or `vehicle` in
  place of spambase are the levers the ADR names if the user pulls one.
- This ticket and ticket 06 are the only ones that write to the shared `mlflow.db`.

## Comments

- 2026-09-11 (from ticket 02): the user is adapting a copy of `experiment.ps1` as
  `experiment_imputation.ps1` at the repository root (a schedule sweep with
  `--task imputation`, work in progress in the working copy). Different purpose from this
  ticket's launcher, but read it before writing `imputation_studies.ps1` so the two share
  the same shape, and do not overwrite it.
