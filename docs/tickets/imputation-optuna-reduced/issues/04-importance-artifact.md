# 04. The importance artifact

Status: done (`6cc6873`) on `feat/imputation-task`, 2026-09-11
Blocked by: 03
Plan task: 4. ADR 0005 decision 6 (first half). Wayfinder ticket 06.

## Goal

A finished study ranks the knobs it sampled: `optuna.importance.get_param_importances`
(fANOVA) logged on the study parent as `optuna/importance/<knob>` metrics and an
`importance.json` artifact, guarded so a study too small to rank warns and finishes.

## Seams under test

- `run_hyperparameter_optimization` (stubbed three-trial study, temporary backend): an
  `optuna/importance/<knob>` metric per sampled knob, summing to about one, and
  `importance.json` among the study parent's artifacts.
- A one-trial study, and a study with tracking disabled: no importance metric, no
  exception.

## Acceptance criteria

- [x] Both, in `tests/unit/test_opt_tracking.py`.
- [x] `_DisabledMlflow` gains `log_dict` and `log_artifact` no-ops.
- [x] Unit suite green.
- [x] A real three-trial reduced study on `credit-g_20nan` (scratch store) shows the
      importances on the study parent.

## Constraints

- The evaluator ranks only sampled knobs; a full-profile pilot to rank the held ones is
  fog on the map, not this ticket.
- fANOVA needs scikit-learn, already a dependency.

## Comments

- 2026-09-11, two red-green slices, commit `6cc6873`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | A finished study ranks the knobs it sampled: one `optuna/importance/<knob>` metric per sampled knob, summing to one, and `importance.json` among the study parent's artifacts |
  | 2 | A one-trial study logs no importance and finishes; a study with tracking disabled still runs (`_DisabledMlflow.log_dict` / `log_artifact`) |

  `log_param_importances(study)` in `opt.py` wraps `optuna.importance.get_param_importances`
  in a guard that warns and returns an empty mapping, so a study too small to rank
  never fails after every trial has run. Unit suite 160 green. No training-package
  file changed, so the fixtures were not rerun; ticket 03's run stands.

  **Proof** (`credit-g_20nan --task imputation --use_optuna --n_trials 3 --lr_scheduler
  cosine`, scratch store `scratchpad/proof04/proof.db`): objectives 0.8533 / 0.9940 /
  0.8587; on the study parent, `optuna/importance/` DROPOUT 0.316,
  WEIGHT_DECAY_DECODE 0.211, LR_DECODE 0.193, LAMBDA_NUM 0.175, PROB_MASCARA 0.105
  (sum 1.0; five knobs because credit-g is mixed) and `importance.json` listed among its
  artifacts. Three trials rank nothing meaningful; the proof is the plumbing. The
  parent's artifact folder was removed from `mlruns/` afterwards.
