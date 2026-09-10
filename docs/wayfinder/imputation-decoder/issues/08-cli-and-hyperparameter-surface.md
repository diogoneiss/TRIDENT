# 08. CLI flag and hyperparameter schema for the imputation task

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 04

## Question

How is the task selected, and which hyperparameters does the decode stage add?

- **Flag**: `--task {classification,imputation}` (default `classification`) vs a boolean
  `--decode`. Validation rules in `validate_parsed_args` (`src/training/config.py`):
  interplay with `--all`, `--use_optuna`, `--cv_folds`, `--save_model`, `--plot_losses`.
- **JSON keys** in `datasets/hiperparams/<base>/<dataset>.json` and
  `Hyperparameters.from_mapping`: e.g. `EPOCHS_DECODE`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, an evaluation mask rate; defaults so every existing JSON
  (and the JSON `experiment.ps1` writes) stays valid; whether the task itself may be
  set from JSON or only from the CLI.
- **Batch mode**: what `run_all` in `main.py` prints in its summary table for an
  imputation run (today it reads `f1_macro`).
- **README**: the flag list and config schema sections to update.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

1. **The flag is `--task {classification,imputation}`, defaulting to `classification`,
   and it is command-line only.** An enum extends to a third task and reuses the tag
   vocabulary of ticket 06 exactly; a boolean `--decode` could not grow. It is **never**
   readable from `datasets/hiperparams/<base>/<dataset>.json`: a dataset's config
   describing *what to do with it* would be surprising, and under `--all` it would
   silently make different datasets train different tasks. The task is therefore a
   property of `TrainingRequest`, not of `Hyperparameters`, and it is logged as a run
   parameter beside `dataset_name`, `seed` and `cv_folds` in `_log_execution_params`.

2. **Five new hyperparameter keys**, all defaulted so every existing config keeps
   loading:

   | Key | Default | Source |
   |---|---|---|
   | `EPOCHS_DECODE` | 150 | mirrors `EPOCH_FINE` |
   | `LR_DECODE` | 0.001 | mirrors `LR_FINE` |
   | `WEIGHT_DECAY_DECODE` | 0.0019 | mirrors `WEIGHT_DECAY_FINE` |
   | `LAMBDA_NUM` | 1.0 | ticket 04 |
   | `EVAL_MASK_RATE` | 0.2 | ticket 03 |

   Backward compatibility is already proven rather than assumed:
   `datasets/hiperparams/vehicle/vehicle_00nan.json` holds 15 keys and **lacks
   `LR_SCHEDULER` entirely**, yet loads, because `Hyperparameters.from_mapping` reads
   every field through `values.get(KEY, default)`. The JSON that `experiment.ps1` writes
   is likewise short of that key.

   Naming notes: the existing convention suffixes the stage but is inconsistent
   (`EPOCHS_PRE` plural, `EPOCH_FINE` singular); decode takes the **plural**, the odd one
   out being the singular. `EVAL_MASK_RATE` is plain English rather than matching its
   Portuguese-named sibling `PROB_MASCARA`; new keys should not propagate a legacy quirk,
   and README documents the pairing so the evaluation counterpart is discoverable.

   **No new per-hyperparameter CLI flags.** That gap is backlog item I1 and deserves one
   general fix (`--hyperparams <path>`) rather than five specific flags. Recorded friction:
   until I1 lands, sweeping a decode hyperparameter needs the same JSON-writing dance
   `experiment.ps1` performs today.

3. **Only the parameters the run's task uses are logged.** An imputation run records the
   pre-training and decode keys and omits `EPOCH_FINE`, `LR_FINE`, `WEIGHT_DECAY_FINE`
   and `LABELS`, the last being meaningless without a classifier. Logging every key on
   every run would repeat backlog item C3, where a parameter is recorded but never used
   and therefore misleads. Differing parameter sets across run kinds is already normal in
   the store.

4. **Every flag combination stays legal.** Imputation composes with `--all`, `--cv_folds`,
   `--use_optuna`, `--save_model` and `--plot_losses`. Plots draw the decode stage's
   curves in place of fine-tuning's; what `--save_model` writes for a decoder run is the
   map's one remaining fog item and does not block this ticket. No new rejection is added
   to `validate_parsed_args`, which keeps it from growing a rule per task.

5. **The batch summary prints the task's ranking metric** with a matching column header,
   reusing the `(ranking_metric, direction)` contract from ticket 05 rather than adding a
   second lookup. `main.py` currently hardcodes `f1_macro` at lines 50, 52, 59 and 73 plus
   the `'F1 Macro'` header at line 70, so an imputation batch would otherwise print
   not-available in every row.

   This is the mildest of twelve hardcoded `f1_macro` sites across four modules. The other
   eight are ticket 09's work and fail far worse: the cross-validation summariser *raises*
   at `summary.py:124` after every fold has already trained, its fold ranking at lines 210
   and 214 assumes larger-is-better and would invert best and worst, the diagnostic
   manifest in `artifacts.py` would come out empty, and `opt.py:160` would raise a
   `KeyError` that the bare `except` swallows into a `0.0` score, reproducing the exact
   shape of backlog item B4.

Consequences:

- Unblocks ticket 13 (Optuna) and, with ticket 09, ticket 14 (regression fixture).
- Ticket 11 may change `EVAL_MASK_RATE` from a scalar to a list if it chooses within-run
  multi-rate evaluation; the key name survives either way.
- README needs its command-line flag list and its configuration-schema section updated,
  including the `PROB_MASCARA` / `EVAL_MASK_RATE` pairing.

## Comments

- 2026-09-10 (ticket 11): the open note above is resolved. `EVAL_MASK_RATE` stays a
  scalar. Multi-rate evaluation adds a separate optional list key,
  `EVAL_MASK_RATES_EXTRA`, defaulting to empty, rather than changing this key's type.

- 2026-09-10 (ticket 12): decision 4 above gains one deliberate exception. `--score_null_path`
  is **rejected at parse time** when passed without `--task imputation`, since there is no
  decoder under classification and nothing could be scored. Every other flag combination
  remains legal.
