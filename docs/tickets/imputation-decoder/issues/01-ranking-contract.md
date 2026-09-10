# 01. Per-task fold-ranking contract

Status: done (`cbf991c` on `feat/imputation-task`, 2026-09-10)
Blocked by: none
Plan task: 1. ADR 0004 decisions 6 and 12. Wayfinder tickets 05, 09.

## Goal

Make the fold-ranking metric and its direction a per-task property, threaded through the
cross-validation summariser and the diagnostic manifest, so that classification output is
byte-identical and the imputation task can rank folds lower-is-better.

## Seams under test

- `task_spec(name)` in `src/training/types.py`: name to `(ranking_metric, direction,
  stage)`; unknown names rejected.
- `summarize_cross_validation(records, task=CLASSIFICATION)`: diagnostic roles chosen in
  the task's direction; validation names the task's ranking metric; loss bands and timings
  from the task's stage.
- `ArtifactWriter.write_cv_tracking_artifacts(..., task=CLASSIFICATION)`: manifest ranking
  keys derived from the ranking metric's short name.

## Acceptance criteria

- [x] `task_spec("classification")` is `f1_macro` / `maximize` / `finetune`;
      `task_spec("imputation")` is `impute/masked/impute_score` / `minimize` / `decode`.
- [x] With the imputation task, folds scoring 0.9 and 0.4 make fold 2 `best_fold`.
- [x] Records lacking the task's ranking metric are rejected with a message naming it.
- [x] Decode loss bands and `decode_seconds` are summarised for imputation records.
- [x] The imputation manifest carries `impute_score_ranking` and `selected_folds[*].impute_score`.
- [x] The existing unit suite and the vehicle regression pass with no edit.

## Comments

- 2026-09-10: five red-green slices in `tests/unit/test_training_tasks.py`. One knock-on
  surfaced inside its own cycle: the summariser requires every loss key on every record,
  so loss keys had to become task-specific (`TaskSpec.loss_keys`) before a realistic
  imputation record could be summarised at all. Full unit suite 78 green; integration
  regression green, unedited.
