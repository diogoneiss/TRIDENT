# Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly

## Status

Accepted (2026-09-10). Charted and decided on the wayfinder map at
[`docs/wayfinder/imputation-optuna-reduced/map.md`](../wayfinder/imputation-optuna-reduced/map.md);
every numbered decision below links the ticket that holds its full reasoning and the
facts verified along the way. Implementation plan:
[`docs/wayfinder/imputation-optuna-reduced/plan.md`](../wayfinder/imputation-optuna-reduced/plan.md),
executed test-first through the tickets under
[`docs/tickets/imputation-optuna-reduced/`](../tickets/imputation-optuna-reduced/spec.md).

Amends [ADR 0004](0004-imputation-decoder-task.md) decision 13 (the imputation search
space and its ranges). Closes backlog item I2 when execution ticket 03 lands; leaves I1
open and out of scope. Records the repair of the shipped Optuna path (commit `161fc92`),
which also fixed `--retrain_best` for both tasks.

### Outcome (2026-09-12)

The six reduced studies ran on 2026-09-11 (40 trials each, `cosine`, seed 42; execution
ticket 05; promoted files in `fcaadfb`), followed by the paired 5-fold comparison of
decision 6 (ticket 06): per pair, one run reading the promoted file and one with it
moved aside, both `--task imputation --cv_folds 5 --lr_scheduler cosine --seed 42`, told
apart by `params.config_source`. Headline `cv/test/impute/induced/impute_score/mean` with
its 95% interval; lower is better, 1.0 is baseline parity; companion masked mean.

| Pair | Promoted, induced [CI] | Defaults, induced [CI] | Promoted, masked | Defaults, masked | Promoted interval excludes default mean |
|---|---|---|---|---|---|
| credit-g_20nan | 0.913 [0.868, 0.958] | 0.895 [0.868, 0.921] | 0.926 | 0.896 | no |
| credit-g_40nan | 0.968 [0.940, 0.995] | 0.962 [0.941, 0.983] | 0.965 | 0.974 | no |
| kr-vs-kp_20nan | 0.603 [0.559, 0.646] | 0.615 [0.584, 0.646] | 0.660 | 0.673 | no |
| kr-vs-kp_40nan | 0.778 [0.734, 0.822] | 0.756 [0.721, 0.791] | 0.802 | 0.775 | no |
| spambase_20nan | 0.886 [0.859, 0.913] | 0.887 [0.859, 0.914] | 0.879 | 0.882 | no |
| spambase_40nan | 0.930 [0.909, 0.952] | 0.927 [0.897, 0.957] | 0.931 | 0.921 | no |

**The reduced search did not beat the defaults on the induced benchmark.** No pair
separates; the promoted point estimate is worse on four pairs and better by at most
0.012 on the other two. The studies themselves say why: the decode learning rate
dominates the sampled space (first in five studies of six, 0.39 to 0.80), but the losses
all sit below 5e-4 and everything from about 1e-3 to 1e-2 is a plateau, so the default
1e-3 was already on it and the search had little to gain; the mask rate is near
irrelevant. On credit-g the validation-split objective also chose configurations that
generalise worse than the defaults (masked 0.926 against 0.896 at 20nan), which is what a
thousand-row validation split affords. The protocol is single-seed and paired, so seed
variability is not quantified. The promoted files remain as the record of the studies;
whether later `--task imputation` runs should keep reading them is a separate decision.
Per-study winners and importances are on
[execution ticket 05](../tickets/imputation-optuna-reduced/issues/05-launcher-and-studies.md).
Running the comparison also surfaced backlog B5 (an induced cell whose category the
variant never shows crashed a fold), fixed in `1cb0864`.

**Replicated under another schedule (2026-09-13).** At the user's request the six studies
and the twelve comparison runs were repeated end to end under `plateau` (no promotion;
`tags.lr_scheduler = plateau`). Tuning helps no more there: better on two pairs, worse on
three, tied on one, with exactly one interval separating out of twelve such comparisons.
The plateau studies beat the cosine studies on the *search objective* by 0.023 to 0.062 on
four pairs, and that advantage leaves no trace on the induced benchmark, where the two
schedules land within 0.013 of each other at the defaults. So decision 3's
validation-split objective is a poor proxy for the benchmark this work is judged on: it is
the part of this ADR the evidence argues against, and the thing to revisit before another
search is run. Table and reading on
[execution ticket 05](../tickets/imputation-optuna-reduced/issues/05-launcher-and-studies.md).

## Context

ADR 0004 gave the imputation task an Optuna study that samples fifteen hyperparameters:
the eleven shared with classification plus the four decode-stage ones. The user wants
degrees of freedom only where they matter for imputation, so that a study converges in a
budget a laptop GPU can afford and so that its result says something about the decoder
rather than about the whole architecture.

Facts that shaped the decisions, verified while charting:

- **The shipped path could not score a trial.** The trial and retrain namespaces in
  `opt.py` never carried the task, so an imputation study trained classification trials
  and pruned every one on the missing ranking metric; the end-of-study write still went
  to the shared configuration file classification reads; the study parent lacked the
  dense `task` and `is_optuna` tags; the sampler was unseeded. Repaired in `161fc92`
  ([ticket 07](../wayfinder/imputation-optuna-reduced/issues/07-repair-optuna-path.md)),
  whose proof run found a sixth defect: since the head-count constraint (`356bcca`),
  Optuna's own record of `DIM` was the per-head multiplier, and `study.best_params` fed
  `--retrain_best`, the saved file and the study's `best_` params.
- **No Optuna run exists in the MLflow store** and **no configuration JSON exists under
  `datasets/hiperparams/`**, so nothing published depends on the current search space or
  on the shared file, and a hyperparameter a study does not sample takes the dataclass
  default (`hyperparams_override` replaces the base configuration).
- **A trial scores the test half of the predefined split today**, for both tasks.
- **The decode stage keeps the best-validation-loss checkpoint** and pre-training keeps
  its last epoch; the decode stage re-rolls masks at `PROB_MASCARA` every epoch;
  `LAMBDA_NUM` weights the numerical loss term against the categorical one, so it acts
  only on a table with both column types.
- **Column types follow the pipeline's declaration**, `datasets/categorical_columns/<base>.txt`
  plus "every other feature is numerical", not pandas dtypes: credit-g (13 + 7) and
  electricity (1 + 7, its integer-coded `day`) are the mixed tables, kr-vs-kp the only
  all-categorical one, the other six numerical-only.
- **Cost of one single-split imputation trial at the defaults** on the user's RTX 3050
  Laptop GPU: credit-g_20nan 66 s, kr-vs-kp_20nan 293 s, both measured; spambase_20nan
  about 500 s, projected from its 290 s per 2-fold fold and the 1.7 ratio measured on
  credit-g. The old full space sampled `EPOCHS_PRE` and `EPOCHS_DECODE` in 20..60, so its
  trials were five to ten times cheaper than a default-configuration run.

## Decision

1. **Two search-space profiles behind a flag.** `--search_space {full,reduced}`, with
   `SEARCH_SPACE_PROFILES = ("full", "reduced")` in `src/training/types.py` in the
   `LR_SCHEDULER_NAMES` style. The flag defaults to `reduced` for `--task imputation` and
   `full` for classification; `--search_space reduced --task classification` is a
   parse-time rejection in `validate_parsed_args`, the `--score_null_path` pattern, until
   a separate effort defines that profile. The choice is recorded as a `search_space` tag
   and param on the study parent and on every trial, **sparse** (absent on non-Optuna
   runs) because the dense `is_optuna` tag already gates any filter on it; no backfill,
   since the store holds no Optuna run. The user chose the flag over replacing the
   imputation branch outright, extending the flag-and-tag preference of ADR 0003 to
   search spaces even with nothing to keep comparable.
   `opt.py`'s own `__main__` builds its parser from `build_training_parser`, so the two
   entry points cannot drift again: its private parser lacked `--task`, `--lr_scheduler`,
   `--disable_mlflow` and `--metrics_dir`, which is one reason ticket 07's defects shipped.
   ([ticket 05](../wayfinder/imputation-optuna-reduced/issues/05-search-space-profile-flag.md))

2. **What the reduced profile samples, and what it holds.** The principle: a knob earns a
   dimension if it governs the decode stage or the corruption the decoder learns from.

   | Freedom | Hyperparameter | Range | Why |
   |---|---|---|---|
   | sampled | `PROB_MASCARA` | 0.2..0.6, step 0.1 | corruption rate in both stages; the decoder learns from exactly these cells |
   | sampled | `LR_DECODE` | 1e-4..1e-2, log | the stage that produces the output; **widened**, the old 1e-5..1e-3 put the default 1e-3 on its upper bound |
   | sampled | `WEIGHT_DECAY_DECODE` | 1e-5..1e-2, log | same stage |
   | sampled | `DROPOUT` | 0.1..0.5, step 0.1 | the one regulariser acting on both stages |
   | sampled on mixed tables only | `LAMBDA_NUM` | 0.1..10, log | acts only where both column types exist; a pure loss scale elsewhere |
   | held | `DIM`, `HEADS`, `LAYERS`, `DIM_FEED`, `HIDDEN_DIM` | 128, 16, 2, 32, 16 | architecture, shared with classification, the paper's values |
   | held | `BATCH` | 256 | cost in both stages |
   | held | `EPOCHS_PRE`, `LR_PRE`, `WEIGHT_DECAY_PRE` | 300, 0.00034, 0.005 | pre-training is shared and protected; `EPOCHS_PRE` is a real knob (last epoch kept), held by decision |
   | held | `EPOCHS_DECODE` | 150 | a budget under best-checkpoint selection |

   Five dimensions on credit-g, four elsewhere. A held hyperparameter runs at the task
   default (the glossary's *held hyperparameter*); the user was offered deliberate values
   and kept the defaults. Holding the epoch counts at the defaults means a trial costs what
   the promoted configuration will cost to run, which is the consistent choice.
   The `LR_DECODE` widening and the `LAMBDA_NUM` conditional are corrections, so they apply
   to the `full` profile too; the profiles differ only in which shared knobs are held.
   The column mix comes from the pipeline's declaration through a pure helper extracted
   from `prepare_dataset` (both declared lists non-empty means mixed), computed once per
   study, never inferred from dtypes. In Optuna's storage the sampled per-head width is
   named `HEAD_DIM` (`161fc92`); the configuration a trial trains with, `DIM` included,
   travels on the trial as the `hyperparameters` user attribute.
   ([ticket 01](../wayfinder/imputation-optuna-reduced/issues/01-held-set-and-ranges.md))

3. **The imputation search is scored on the validation split.** `TaskSpec` gains a
   `search_objective` (the glossary's *search objective*): `validation/impute/masked/impute_score`
   for imputation, `f1_macro` for classification, which spells out today's test-split
   behaviour rather than changing it. The score is computed only when a new programmatic,
   defaulted `TrainingRequest` field asks for it (the `mlflow_run_role` pattern; `opt.py`
   sets it for imputation trials): `decoding.py` scores the fixed validation mask it
   already holds (`hidden_validation` against `clean_validation`, with the training-fold
   baselines), logs the result under `validation/` and merges it into the fold metrics
   only then, so ordinary and cross-validation runs never carry a `validation/` key and
   the `cv/test/` summariser never sees one. The objective mirrors the ranking metric's
   population: masked cells only; the induced population stays a test-fold report.
   `optuna/objective_value` keeps logging the objective under one key whatever the task.
   ([ticket 02](../wayfinder/imputation-optuna-reduced/issues/02-search-objective-on-validation.md))

4. **Study protocol.** Six studies: `credit-g`, `kr-vs-kp` and `spambase`, each on
   `_20nan` and `_40nan`, one study per (dataset, variant) pair writing its own per-variant
   configuration. The datasets cover the three column-type compositions the metric
   distinguishes (credit-g the only affordable mixed table, kr-vs-kp the only
   all-categorical one, spambase the widest numerical one). 40 trials each; `cosine`
   schedule, as ADR 0004 prescribes; `TPESampler(seed=<run seed>)`; the predefined single
   split per trial; no pruner.

   | Study | Trial cost | 40 trials, both variants |
   |---|---|---|
   | credit-g | 66 s, measured | about 1.5 h |
   | kr-vs-kp | 293 s, measured | about 6.5 h |
   | spambase | about 500 s, projected | about 11 h |

   About nineteen hours in total. The levers if that is too much: 30 trials (about
   fourteen hours) or `vehicle` in place of spambase (about nine and a half). The 40-trial
   decision stands until the user pulls one. `_00nan`, `_60nan` and `_80nan` are out of
   scope.
   ([ticket 03](../wayfinder/imputation-optuna-reduced/issues/03-study-protocol.md))

5. **A promoted configuration lives at a task-keyed path and is promoted explicitly.**
   `--task imputation` reads `datasets/hiperparams/<base>/<dataset>.imputation.json`
   first, then the shared `<dataset>.json`, then the defaults; classification's lookup is
   unchanged. A study always writes its running best and its final best inside its own
   timestamped study directory, for **both** tasks, and writes outside it only under an
   explicit `--promote_best`: the task-keyed file for imputation, the shared file for
   classification. That is backlog I2's suggested fix, and it closes I2. A promoted file is
   **complete**: the task's full key set with resolved values, held ones included, and
   `LR_SCHEDULER` set to the schedule the study ran under, so the file cannot silently move
   if a default changes. Every run logs a dense `config_source` param (`defaults`,
   `override`, or the relative path of the file it loaded) so tuned and default runs are
   told apart without a new tag. `--promote_best` and `--retrain_best` are independent.
   A general `--hyperparams <path>` flag (backlog I1) stays out of scope.
   ([ticket 04](../wayfinder/imputation-optuna-reduced/issues/04-promoted-configuration-lookup.md))

6. **Checking the reduction, and proving the tuning helped.** At the end of a study,
   `optuna.importance.get_param_importances(study)` (the default fANOVA evaluator) is
   logged on the study parent as `optuna/importance/<knob>` metrics and an
   `importance.json` artifact, guarded so a study with too few completed trials warns
   instead of crashing. It ranks only the sampled knobs; a full-profile pilot to rank the
   held ones is deferred to the fog. The comparison protocol: for each of the six pairs,
   one 5-fold cross-validation run with the promoted configuration against one with the
   defaults, both `--task imputation --cv_folds 5 --lr_scheduler cosine --seed 42`, so the
   configuration is the only difference; fresh default runs, because the two imputation
   runs already in the store used `plateau` and 2 folds. Headline
   `cv/test/impute/induced/impute_score/mean` with its `ci95_lower` / `ci95_upper`,
   companion `cv/test/impute/masked/impute_score/mean`; lower is better, 1.0 is baseline
   parity. It is the glossary's *paired single-seed CV protocol*. A launcher script in the
   shape of `experiment.ps1` runs the six studies with promotion and the twelve comparison
   runs.
   ([ticket 06](../wayfinder/imputation-optuna-reduced/issues/06-importance-and-comparison.md))

7. **The repair, done.** Commit `161fc92`: the task travels on the trial and retrain
   namespaces; the end-of-study write respects the task; the study parent and the
   pre-runner trial tags carry `task` and `is_optuna` through `execution_tags` (promoted
   from `_execution_tags`); the sampler is seeded; the log line names the ranking metric;
   the per-head width is sampled as `HEAD_DIM` and the trained configuration travels on
   the trial as a user attribute that the retrain, the saved file and the `best_` params
   read. `--retrain_best` was broken for both tasks between `356bcca` and `161fc92`; no
   run in the store used it. Two facts recorded for whoever reads the store: the decode
   stage logs its timing under `time/finetune_seconds` (no decode key exists), and at the
   defaults kr-vs-kp already scores 0.72 masked, well below parity.
   ([ticket 07](../wayfinder/imputation-optuna-reduced/issues/07-repair-optuna-path.md))

## How to compare

**Studies.** Filter `tags.is_optuna = 'true'`, `tags.task = 'imputation'`,
`tags.search_space = 'reduced'` and one `tags.lr_scheduler`. On the study parent,
`optuna/best_objective_value` is a **validation-split** score and is not comparable with
any `test/` or `cv/test/` number; `optuna/importance/<knob>` ranks the sampled knobs.
Every trial carries `optuna/objective_value` under the same rule.

**The comparison.** Filter `tags.run_role = 'parent'`, `tags.task = 'imputation'`,
`tags.is_optuna = 'false'`, `tags.lr_scheduler = 'cosine'` and `tags.dataset_variant`
per pair; `params.config_source` separates the promoted run from the default one.
Headline `cv/test/impute/induced/impute_score/mean` with `ci95_lower` / `ci95_upper`,
companion `cv/test/impute/masked/impute_score/mean`. Classification comparisons are
unchanged. Since [ADR 0006](0006-mirror-experiments-per-task.md) every run has a mirror
in `TRIDENT/mirror/<task>`; add `tags.is_mirror = 'false'` to any query that spans
experiments, or each run is counted twice.

## Considered options

- **Replace the imputation branch of `define_search_space` outright.** Recommended,
  because no Optuna run exists to keep comparable; rejected by the user in favour of the
  profile flag.
- **Score the search on the test split, as classification does.** Rejected for
  imputation: hyperparameters chosen on the test split leak into every number reported
  on it. Classification's study keeps its behaviour.
- **One study per base dataset on `_20nan`, shared across the ladder.** Rejected by the
  user: only `_20nan` and `_40nan`, each with its own study, which also shows whether the
  optimum shifts with missingness.
- **`vehicle` in place of spambase.** Offered as the cheaper numerical stand-in; the user
  kept spambase for its width and the existing baseline run.
- **A general `--hyperparams <path>` flag (backlog I1) instead of a task-keyed path.**
  Rejected: a run that cannot find its own configuration never uses it.
- **A dense `search_space` tag with `none` on every non-Optuna run, backfilled.**
  Rejected: it carries nothing `is_optuna` does not already carry.
- **A reduced classification profile by analogy.** Rejected: a space no one has asked to
  run; a parse-time rejection instead.
- **A full-profile pilot with fANOVA to rank the held knobs.** Deferred: fifteen
  dimensions at 50 trials rank noisily.
- **Leaving the `full` profile exactly as ADR 0004 decided it.** Rejected: the
  `LR_DECODE` boundary and the `LAMBDA_NUM` conditional are corrections, not reductions.

## Consequences

- Classification training behaviour is untouched; both regression fixtures stay unedited.
  Three classification-visible, non-behavioural changes, stated so nobody discovers them
  in a diff: a classification study's running best now goes to its study directory and
  the shared file is written only under `--promote_best` (decision 5); Optuna's storage
  names the sampled per-head width `HEAD_DIM` (decision 2); every run logs a
  `config_source` param (decision 5).
- New: `SEARCH_SPACE_PROFILES`, `TaskSpec.search_objective`, a programmatic request field
  for the validation score, a `validation/` metric family on imputation trials only, a
  column-mix helper in `src/training/data.py`, the `search_space` tag, the `config_source`
  param, the task-keyed configuration path, `--search_space` and `--promote_best`, the
  importance artifact, and a launcher script.
- Amended: ADR 0004 decision 13's ranges (`LR_DECODE`, conditional `LAMBDA_NUM`).
- Still fog on the map: pruning on the decode validation score; the full-profile pilot.
- Out of scope, recorded on the map: the other variants and the transfer of a promoted
  configuration to them, the other six datasets, a reduced classification profile,
  backlog I1, multi-seed robustness, other samplers, and tuning pre-training or the
  architecture for imputation.
