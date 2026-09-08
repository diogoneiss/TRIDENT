---
type: community
members: 25
---

# Cross-Validation Summary Stats

**Members:** 25 nodes

## Members
- [[Aggregate completed CV folds into comparable final and loss statistics.]] - rationale - src/training/summary.py
- [[FoldResult]] - code - src/training/types.py
- [[LossBand]] - code - src/training/types.py
- [[Metric aggregation helpers.]] - rationale - src/training/summary.py
- [[_diagnostic_roles()]] - code - src/training/summary.py
- [[_is_finite_number()]] - code - src/training/summary.py
- [[_loss_band()]] - code - src/training/summary.py
- [[_loss_events_by_key()]] - code - src/training/summary.py
- [[_record()_1]] - code - tests/unit/test_training_summary.py
- [[_student_t_interval()]] - code - src/training/summary.py
- [[_summarize_loss_bands()]] - code - src/training/summary.py
- [[_summarize_metric()]] - code - src/training/summary.py
- [[_validate_final_metrics()]] - code - src/training/summary.py
- [[ndarray_1]] - code
- [[summarize_cross_validation()]] - code - src/training/summary.py
- [[summary.py]] - code - src/training/summary.py
- [[test_summarize_cross_validation_logs_statistics_and_loss_bands()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_rejects_fewer_than_two_folds()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_rejects_mismatched_loss_epochs()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_rejects_missing_final_metric()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_rejects_non_finite_final_metric()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_uses_lowest_fold_for_a_maximum_tie()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_uses_lowest_fold_for_a_minimum_tie()]] - code - tests/unit/test_training_summary.py
- [[test_summarize_cross_validation_uses_one_role_for_a_tie()]] - code - tests/unit/test_training_summary.py
- [[test_training_summary.py]] - code - tests/unit/test_training_summary.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cross-Validation_Summary_Stats
SORT file.name ASC
```

## Connections to other communities
- 23 edges to [[_COMMUNITY_MLflow Fold Tracking]]
- 9 edges to [[_COMMUNITY_MLflow Tracking Tests]]
- 8 edges to [[_COMMUNITY_Model & Training Core]]
- 6 edges to [[_COMMUNITY_Training Config & Data Pipeline]]
- 6 edges to [[_COMMUNITY_Artifact Writing & Persistence]]
- 3 edges to [[_COMMUNITY_Fold Metrics Computation]]
- 1 edge to [[_COMMUNITY_CLI Entry & Hyperparameter Search]]

## Top bridge nodes
- [[summary.py]] - degree 24, connects to 7 communities
- [[FoldResult]] - degree 24, connects to 6 communities
- [[summarize_cross_validation()]] - degree 25, connects to 4 communities
- [[test_training_summary.py]] - degree 15, connects to 2 communities
- [[LossBand]] - degree 6, connects to 2 communities