# 04. Where a promoted configuration lives and how a run finds it

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, round 2)
Blocked by: none

## Question

Ticket 13 decision 5 keeps an imputation study's result inside its study directory and
says promotion is "a deliberate copy". But the only configuration path a run reads is
`datasets/hiperparams/<base>/<dataset>.json`, shared by both tasks, and the only other
route is the programmatic `hyperparams_override` (backlog I1: no CLI way to set
hyperparameters). So there is nowhere to copy *to* without clobbering classification, and
a configuration a run cannot find never gets used. Where does a promoted imputation
configuration live, and how does `--task imputation` find it?

## Answer

Resolved in one grilling round: **a task-keyed lookup with explicit promotion**.

- `--task imputation` reads `datasets/hiperparams/<base>/<dataset>.imputation.json`
  first, then falls back to the shared `<dataset>.json`, then to the defaults. The
  fallback preserves today's behaviour for a variant with no promoted configuration.
  Classification's lookup is unchanged: shared file, then defaults.
- A study writes its best configuration into its own timestamped study directory always,
  and outside it **only under an explicit `--promote_best` flag**. For imputation the
  promotion target is the task-keyed file above. For classification it is the shared
  file, which today is written unconditionally at every improvement and again at the end
  of the study; under the flag it is written only when asked. That is exactly the fix
  backlog **I2** proposes, it changes no training behaviour, and no classification study
  exists in the store to be affected, so I2 closes with this effort rather than staying
  open beside a flag that ignores one task.
- `--promote_best` and `--retrain_best` are independent: one publishes the configuration,
  the other trains a full parent run with it. Either may be given without the other.

Rejected: a general `--hyperparams <path>` flag (backlog I1) instead of a task-keyed path.
It is the more general tool, but it leaves the run unable to find its own configuration,
which is the failure this ticket exists to remove. I1 stays open and out of scope.

**Glossary**: *promoted configuration* is in `CONTEXT.md`.

**For the plan**: `_load_base_hyperparameters` in `src/training/config.py` gains the
task-keyed candidate ahead of the shared one; the run logs which file it loaded (a param,
not a tag; six pairs do not need a filter axis). The README's "Hyperparameter Configuration"
section documents the lookup order.

## Comments
