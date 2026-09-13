# 05. The launcher and the six studies

Status: done (`fcaadfb`) on `feat/imputation-task`, 2026-09-11
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
- [x] Six promoted files exist: `datasets/hiperparams/<base>/<base>_<variant>.imputation.json`
      for the three datasets and two variants, each complete, each committed.
- [x] Six study parents in the store with `is_optuna = true`, `task = imputation`,
      `search_space = reduced`, `lr_scheduler = cosine`, 40 completed trials each, an
      `optuna/importance/*` set and a `best_trial = true` trial.
- [x] Under `## Comments`, per study: the best objective (a validation-split score), the
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

- 2026-09-11, morning: three studies finished (credit-g_20nan 0.7763 in 44.6 min,
  credit-g_40nan 0.8051 in 42.0 min, kr-vs-kp_20nan 0.6703 in 159 min; 40/40 trials each,
  `best_trial` tagged, importances logged). In every study the decode learning rate is
  the dominant knob (0.39 to 0.52) and the worst trials cluster at its low end (median
  2.4e-4); the mask rate is near irrelevant (0.02 to 0.04). Full numbers land here when
  the six are done.
- 2026-09-11, morning: the user asked for the same six studies under `plateau`, to see
  whether the learning-rate finding depends on the schedule. Queued behind the cosine
  batch, after ticket 06's `-Compare` runs, as `imputation_studies.ps1 -Scheduler plateau
  -NoPromote` (new switch, `d69e344`): nothing promoted, so the cosine winners stay the
  promoted configurations; the plateau studies are told apart by `tags.lr_scheduler =
  plateau`. Exploration outside ADR 0005's plan; queue log `results/imputation_queue_20260911.*`.


- 2026-09-11 21:37, the six studies done; promoted files committed in `fcaadfb`. Every
  study parent: `is_optuna = true`, `task = imputation`, `search_space = reduced`,
  `lr_scheduler = cosine`, 40 of 40 trials successful, `best_trial = true` on one trial,
  `optuna/importance/*` logged. Launcher total 1227 min (20.5 h) against the ADR's
  about nineteen; the user's schedule sweep shared the GPU throughout.

  | Pair | Best objective (validation masked) | Winner: PROB_MASCARA / LR_DECODE / WEIGHT_DECAY_DECODE / DROPOUT / LAMBDA_NUM | Importances | Wall-clock | s per trial (ADR) |
  |---|---|---|---|---|---|
  | credit-g_20nan | 0.7763 (trial 30) | 0.4 / 2.54e-3 / 2.32e-3 / 0.3 / 2.74 | LR_DECODE 0.49, LAMBDA_NUM 0.33, WD 0.12, DROPOUT 0.05, PROB 0.02 | 44.6 min | 67 (66) |
  | credit-g_40nan | 0.8051 (trial 2) | 0.2 / 8.71e-3 / 3.14e-3 / 0.2 / 0.23 | LR_DECODE 0.52, WD 0.22, LAMBDA_NUM 0.18, PROB 0.04, DROPOUT 0.03 | 42.0 min | 63 (66) |
  | kr-vs-kp_20nan | 0.6703 (trial 17) | 0.2 / 1.74e-3 / 3.90e-3 / 0.2 / held | LR_DECODE 0.39, WD 0.30, DROPOUT 0.28, PROB 0.03 | 159.0 min | 239 (293) |
  | kr-vs-kp_40nan | 0.6480 (trial 32) | 0.3 / 1.70e-3 / 2.1e-4 / 0.1 / held | LR_DECODE 0.49, DROPOUT 0.22, PROB 0.18, WD 0.11 | 160.6 min | 241 (293) |
  | spambase_20nan | 0.8993 (trial 16) | 0.4 / 1.33e-3 / 4.0e-5 / 0.3 / held | DROPOUT 0.53, WD 0.25, LR_DECODE 0.21, PROB 0.01 | 423.4 min | 635 (about 500) |
  | spambase_40nan | 0.8996 (trial 28) | 0.5 / 1.08e-3 / 3.0e-4 / 0.2 / held | LR_DECODE 0.80, DROPOUT 0.10, WD 0.07, PROB 0.04 | 395.2 min | 593 (about 500) |

  What the six say about the reduction: the decode learning rate is the first knob in
  five studies of six (0.39 to 0.80) and third on spambase_20nan, where dropout leads
  (0.53); in every study the worst trials cluster at the low end of the rate range
  (bottom-eight median 2.4e-4, objectives near baseline parity on credit-g), while
  everything from about 1e-3 to 1e-2 is a plateau and the winners sit two to nine times
  above the default. The mask rate is near irrelevant everywhere (0.01 to 0.18, both
  top and bottom trials span its whole range). The loss balance on credit-g points both
  ways (2.7 at 20nan, 0.23 at 40nan). Whether the promoted files beat the defaults is
  ticket 06's question, on the induced score with intervals; the objective here is the
  validation-split masked score and credit-g's validation split is small.

  The queued runner started ticket 06's comparison at 21:37 and follows it with the
  plateau replication of the six studies (no promotion), per the earlier comment.

- 2026-09-12 14:56: the four cheap plateau studies are done and reach a **better**
  validation objective than their cosine twins on every one of them (credit-g_20nan
  0.750 against 0.776, credit-g_40nan 0.782 against 0.805, kr-vs-kp_20nan 0.637 against
  0.670, kr-vs-kp_40nan 0.587 against 0.648; lower is better). The clean cosine picture
  of the decode rate dissolves: the plateau winners scatter from 3.2e-4 to 8.3e-3, on
  kr-vs-kp_20nan the winner is at the bottom of the range, and on kr-vs-kp_40nan the
  worst eight trials have a *higher* median rate than the best eight. A schedule that
  adapts the rate during training makes its starting value matter less, which is the
  expected direction. spambase is still running; due about 23:00.

  Since the cosine batch showed validation gains not carrying to the induced benchmark,
  the user asked for the same paired comparison under plateau, queued behind the batch
  (scratch `queue_plateau_compare.ps1`, log `results/imputation_plateau_compare.*`). The
  plateau studies promoted nothing, so each plateau winner is staged as a complete
  configuration (the shipped `complete_configuration`, `LR_SCHEDULER` pinned to plateau)
  and swapped into the task-keyed path for the duration of its pair, with the committed
  cosine file restored in a `finally` afterwards. If it dies mid-pair,
  `git checkout -- datasets/hiperparams` restores the cosine winners. The twelve runs
  also give the plateau-versus-cosine comparison at the defaults for free, since the
  cosine defaults runs are already in the store.

- 2026-09-12 22:52, the plateau replication complete: six studies, 40 of 40 trials each,
  `tags.lr_scheduler = plateau`, nothing promoted. Total 1226 min, within a minute of the
  cosine batch's 1227.

  | Pair | Objective, cosine | Objective, plateau | Winner LR_DECODE, cosine | Winner LR_DECODE, plateau | Top knob, plateau |
  |---|---|---|---|---|---|
  | credit-g_20nan | 0.7763 | **0.7503** | 2.54e-3 | 3.82e-3 | LR_DECODE 0.54 |
  | credit-g_40nan | 0.8051 | **0.7820** | 8.71e-3 | 1.30e-3 | LAMBDA_NUM 0.39 |
  | kr-vs-kp_20nan | 0.6703 | **0.6374** | 1.74e-3 | 3.24e-4 | PROB_MASCARA 0.35 |
  | kr-vs-kp_40nan | 0.6480 | **0.5866** | 1.70e-3 | 8.26e-3 | LR_DECODE 0.39 |
  | spambase_20nan | 0.8993 | 0.8995 | 1.33e-3 | 2.08e-3 | DROPOUT 0.38 |
  | spambase_40nan | 0.8996 | 0.8999 | 1.08e-3 | 9.88e-4 | LR_DECODE 0.71 |

  Lower is better; bold marks the better of the two. Plateau reaches a better validation
  objective on the four small pairs, by 0.023 to 0.062, and ties on spambase (0.0002 and
  0.0003 the other way, far inside trial-to-trial noise). Every one of those four gaps is
  larger than any promoted-versus-default difference the cosine comparison measured,
  which is why the comparison under plateau is worth its two hours.

  On the rates, the cosine finding does **not** replicate. Under cosine every study put
  its worst trials at the bottom of the range and its winner above the default 1e-3.
  Under plateau the winners scatter from 3.24e-4 to 8.26e-3; kr-vs-kp_20nan wins at the
  very bottom of the range, and on kr-vs-kp_40nan the worst eight trials have a higher
  median rate (5.2e-3) than the best eight (4.5e-3), reversing the cosine ordering.
  The rate also stops being the first knob on three pairs (the loss balance, the mask
  rate and dropout take the top spot). A schedule that reduces the rate on plateau makes
  the starting value matter less, which is the expected direction and the reason the
  cosine studies' one clean finding should not be stated as a property of the task.

  Caveat carried from the cosine batch: these are validation-split objectives, and the
  cosine batch's validation gains did not carry to the induced test benchmark.

