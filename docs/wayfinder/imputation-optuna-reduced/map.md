# Map: Reduced Optuna search for the imputation task

Label: wayfinder:map
Charted: 2026-09-10
Tracker: local markdown. `AGENTS.md` keeps tickets, plans and decisions under `docs/`,
so this effort lives at `docs/wayfinder/imputation-optuna-reduced/`. Tickets are
`issues/NN-<slug>.md`; each carries `Type:`, `Status:` (`open` / `claimed` / `resolved`)
and `Blocked by:` lines. The frontier is every open, unblocked, unclaimed ticket, lowest
number first.

Work it with `/wayfinder docs/wayfinder/imputation-optuna-reduced/map.md` (optionally
naming a ticket). One decision per session.

Charting note: three grilling rounds in the charting session settled the six decision
tickets (01 to 06) on the spot. They are recorded as resolved so each decision and its
reasoning has one home. The repair (07) was resolved the same day in commit `161fc92`; the
destination (08) is the one open ticket.

## Destination

A decided **reduced hyperparameter search for the imputation task**, recorded as
ADR 0005 under `docs/adr/` plus an implementation plan beside this map at `plan.md`,
precise enough that the six studies can be launched from the plan without another
decision. It covers: which hyperparameters a study samples and which it holds at the
task defaults, with their ranges; the search objective and the split it is scored on;
the study protocol (datasets, variants, trial budget, schedule, seed); where a promoted
configuration lives and how a `--task imputation` run finds it; the search-space profile
flag and its MLflow record; the importance diagnostic; the protocol that shows whether
tuning helped; and the repair of the shipped Optuna path, which cannot produce an
imputation study today.

Planning only, with one exception: the repair ticket proves itself with a real two-trial
study. The six studies and the comparison runs are execution, launched from the plan.

## Notes

- **Domain**: TRIDENT. Read `README.md` (CLI, config schema, "Imputation" keys),
  `CONTEXT.md` (glossary; four terms from this effort are already in it: *search
  objective*, *search-space profile*, *held hyperparameter*, *promoted configuration*),
  `docs/adr/0004-imputation-decoder-task.md` decision 13 (what this effort amends) and
  the previous map's [ticket 13](../imputation-decoder/issues/13-optuna-for-imputation.md).
  Code: `opt.py`, `src/training/config.py` (`_load_base_hyperparameters`),
  `src/training/decoding.py`, `src/training/tracking.py` (`_execution_tags`,
  `OptunaTrialTracker`), tests `tests/unit/test_opt_search_space.py` and
  `tests/unit/test_opt_tracking.py`. For codebase questions run
  `graphify query "<question>"` first.
- **Skills each session should call**: `mattpocock-skills:grilling` and
  `mattpocock-skills:domain-modeling` for any grilling ticket;
  `mattpocock-skills:tdd` for the repair ticket's code, followed by the real run the
  ticket names (tests alone shipped the defects it repairs). Update `CONTEXT.md` inline
  as terms settle.
- **Standing preferences** (from `AGENTS.md` and prior decisions):
  - The classification task and its study stay untouched except where a ticket says
    otherwise and why. Classification training behaviour is protected; the
    `vehicle_00nan` and `credit-g_20nan` integration fixtures must pass unedited.
  - Behaviour choices ship as a flag with the old behaviour as default and the choice as
    an MLflow tag (ADR 0003 pattern). The user extended this to the *search space* even
    though the store holds no Optuna run to keep comparable (ticket 05).
  - Use `uv run --python 3.10 ...`; unit tests are `pytest -m "not integration"`.
  - Tickets, plans and decisions live under `docs/`; no external issues.
- **Facts established while charting** (verified against code, data and the MLflow
  store, so sessions need not re-derive them):
  - **The shipped imputation Optuna path could not score a trial** (the state at charting;
    repaired by ticket 07 in commit `161fc92`, which also found that Optuna's record of
    `DIM` was the per-head multiplier, breaking `--retrain_best` for both tasks since
    `356bcca`). `ObjectiveFunctionWrapper.__call__`
    (`opt.py`) builds the trial namespace without `task`, and `resolve_training_request`
    (`src/training/config.py`) falls back to `DEFAULT_TASK`, so every trial trains a
    classification model, then `metrics["impute/masked/impute_score"]` raises and the
    trial is pruned. `final_args` for `--retrain_best` has the same gap. The end-of-study
    write ("2. Also save to the standard hiperparams directory") is unconditional, so a
    finished imputation study still overwrites `datasets/hiperparams/<base>/<dataset>.json`
    (only the per-trial `save_best_params` honours ticket 13 decision 5). The study parent
    builds its tags with `build_run_tags` directly and carries neither `task` nor
    `is_optuna`, although `_execution_tags` in `src/training/tracking.py` already sets
    both for every other run kind. The final log line says "Best F1 macro" whatever the
    task. `tests/unit/test_opt_search_space.py` exercises `define_search_space` only,
    which is why all of this shipped green.
  - `optuna.create_study` was called without a sampler (seeded with the run seed by
    ticket 07) and is still called without a pruner; `trial.report(score, step=0)` is the
    only report.
  - A trial runs the **predefined single split** (no `cv_folds` on the trial namespace);
    the objective is `results[0].metrics[ranking_metric]`, which `decoding.py` computes on
    `fold.test_indices`. The decode stage already encodes a fixed validation mask
    (`hidden_validation`, `clean_validation`) once per fold for its loss.
  - `hyperparams_override` **replaces** the base configuration
    (`Hyperparameters.from_mapping`), so any key a profile does not sample takes the
    dataclass default. No configuration JSON exists anywhere under
    `datasets/hiperparams/` (two empty directories, `letter` and `train`); every run in
    the store used the defaults or an explicit `EPOCHS_PRE`.
  - The old full space samples `EPOCHS_PRE` in 20..60 and `EPOCHS_DECODE` in 20..60,
    against defaults of 300 and 150, so its trials cost five to ten times less than a
    default-configuration run. Holding both at the defaults makes a trial cost what the
    final run costs.
  - The decode stage keeps the **best-validation-loss checkpoint**; pre-training keeps
    its last epoch. The decode stage re-rolls masks at `PROB_MASCARA` every epoch, so
    that knob governs corruption in both stages. `LAMBDA_NUM` weights the numerical term
    against the categorical one, so it acts only on a table with both column types.
  - Datasets (features exclude the label):

    | Base | Rows | Categorical | Numerical |
    |---|---|---|---|
    | credit-g | 1000 | 13 | 7 |
    | kr-vs-kp | 3196 | 36 | 0 |
    | spambase | 4601 | 0 | 57 |
    | vehicle | 846 | 0 | 18 |
    | kc2 | 522 | 0 | 21 |
    | biodeg | 1055 | 0 | 41 |
    | pendigits | 10992 | 0 | 16 |
    | letter | 20000 | 0 | 16 |
    | electricity | 45312 | 1 | 7 |

    Column types follow the pipeline's declaration (`datasets/categorical_columns/<base>.txt`
    lists the categorical columns; `src/training/data.py` treats every other feature as
    numerical), **not pandas dtypes**: electricity's integer-coded `day` is declared
    categorical. credit-g and electricity are the two mixed tables, credit-g the only
    affordable one; kr-vs-kp is the only all-categorical one; the other six are
    numerical-only.
  - **Cost of one single-split imputation trial at the defaults**, RTX 3050 Laptop GPU,
    measured by ticket 07 on 2026-09-10 (`time/pretrain_seconds` + `time/finetune_seconds`,
    the decode stage reusing the latter key): credit-g_20nan 66 s (34 + 32), kr-vs-kp_20nan
    293 s (141 + 152). The 2-fold per-fold figures in the store (credit-g_20nan 38.6 s,
    spambase_20nan 290 s, `plateau`) understate a trial by 1.7, because a fold trains on
    about 40% of the rows and the single split on 80% (credit-g 800, kr-vs-kp 2556,
    spambase 3679 training rows); spambase therefore projects to about 500 s per trial.
    The decode stage costs as much as pre-training at the defaults (150 against 300 epochs,
    but per-column heads and a scoring pass per epoch).
  - MLflow store on 2026-09-10: 89 parent runs, **no Optuna run**, two imputation
    parents: `impute_credit-g_20nan` (masked `impute_score` 0.929, induced 0.916) and
    `impute_spambase_20nan` (0.896 / 0.899), both defaults, `plateau`, 2 folds. Neither
    serves as the comparison baseline in ticket 06, which fixes `cosine` and 5 folds.
  - The schedule is never sampled; every trial uses the command-line `--lr_scheduler`.
    ADR 0004 says to launch imputation experiments with `cosine`.

## Decisions so far

<!-- one line per resolved ticket: gist, then the link for detail -->

- [01. The held set and the ranges](issues/01-held-set-and-ranges.md): the reduced
  profile samples `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and, on
  mixed tables only, `LAMBDA_NUM`; everything else is held at the task default, including
  `EPOCHS_PRE` 300 and `EPOCHS_DECODE` 150. `LR_DECODE` widens to 1e-4..1e-2 because its
  default sat on the old upper bound; that widening and the `LAMBDA_NUM` conditional apply
  to the full profile too, so the profiles differ only in what is held.
- [02. The search objective is scored on the validation split](issues/02-search-objective-on-validation.md):
  an imputation trial's objective is the masked-population `impute_score` on the
  validation split of the predefined single split, one extra scoring pass over the mask
  the decode stage already holds; the test split stays unseen until retraining.
  Classification's study keeps scoring the test split.
- [03. Study protocol](issues/03-study-protocol.md): six studies, `credit-g`, `kr-vs-kp`
  and `spambase` on `_20nan` and `_40nan`, chosen for column-type coverage; 40 trials
  each, about nineteen hours in total once ticket 07 measured a single-split trial at 1.7
  times a 2-fold fold (30 trials: fourteen; vehicle for spambase: nine and a half);
  `cosine` schedule; seeded TPE with the run seed;
  predefined single split per trial; no pruner.
- [04. Where a promoted configuration lives](issues/04-promoted-configuration-lookup.md):
  `--task imputation` reads `datasets/hiperparams/<base>/<dataset>.imputation.json`
  first and falls back to the shared file, classification's lookup unchanged; a study
  writes outside its study directory only under an explicit `--promote_best`, for both
  tasks, which closes backlog I2.
- [05. The search-space profile flag](issues/05-search-space-profile-flag.md):
  `--search_space {full,reduced}`, defaulting to `reduced` for imputation and `full` for
  classification; `reduced` with classification is a parse-time rejection; recorded as a
  sparse `search_space` tag and param on the study parent and every trial, no backfill.
- [06. Checking the reduction and proving the tuning helped](issues/06-importance-and-comparison.md):
  fANOVA importances of the sampled knobs logged on the study parent; no full-profile
  pilot. For each of the six pairs, a 5-fold `cosine` run with the promoted configuration
  against a default-configuration run with the same flags, compared on the induced
  `impute_score` mean and its interval, filtered `is_optuna = false`.
- [07. Repair the imputation Optuna path](issues/07-repair-optuna-path.md): commit
  `161fc92`. Trials and the retrain now carry the task, the end-of-study write respects it,
  the study parent and pre-runner trial tags carry `task` and `is_optuna`, the sampler is
  seeded, and a sixth defect found by the proof run is fixed: the sampled per-head width is
  now `HEAD_DIM` and the trained configuration travels on the trial, so `--retrain_best`
  and every saved file carry what the winner ran. Proof: a two-trial credit-g study scored
  both trials on a scratch store. Measured: a single-split trial costs 66 s on credit-g and
  293 s on kr-vs-kp, 1.7 times a 2-fold fold.

## Not yet specified

- **Pruning on the decode validation score.** A `MedianPruner` needs per-epoch reports of
  the *objective*, which is the validation-split score, not the validation loss the
  checkpoint watches; whether reporting it every epoch is worth its cost, and how it
  interacts with best-checkpoint selection, is unsharp until the six studies show how
  many trials are clearly hopeless early.
- **A full-profile pilot with fANOVA** to check that the held knobs really rank low. Ruled
  out for now in ticket 06 because a 15-dimensional pilot at 50 trials ranks noisily; it
  returns if the reduced studies' importances look flat.

## Out of scope

- **The `_00nan`, `_60nan` and `_80nan` variants**, and whether a `_20nan` or `_40nan`
  promoted configuration should serve them. The user fixed the effort to 20 and 40.
- **The other six datasets.** Three cover every column-type composition the metric
  distinguishes; the rest are cost without a new axis.
- **A reduced classification profile** (ticket 05): `reduced` with `--task classification`
  is rejected until a separate effort defines it.
- **Backlog I1**, a general `--hyperparams <path>` flag: the task-keyed lookup in ticket 04
  is what a run needs; passing files by hand stays I1's business.
- **Multi-seed robustness** of the promoted configurations. The comparison follows the
  glossary's paired single-seed CV protocol.
- **Other samplers or multi-objective search.** Seeded TPE, one objective.
- **Tuning pre-training or the architecture for imputation.** Held by decision, not by
  oversight (ticket 01); revisit only if ticket 06's importances say otherwise, and then
  as a new effort.
