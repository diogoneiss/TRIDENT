# 04. The importance artifact

Status: ready-for-agent
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

- [ ] Both, in `tests/unit/test_opt_tracking.py`.
- [ ] `_DisabledMlflow` gains `log_dict` and `log_artifact` no-ops.
- [ ] Unit suite green.
- [ ] A real three-trial reduced study on `credit-g_20nan` (scratch store) shows the
      importances on the study parent.

## Constraints

- The evaluator ranks only sampled knobs; a full-profile pilot to rank the held ones is
  fog on the map, not this ticket.
- fANOVA needs scikit-learn, already a dependency.

## Comments
