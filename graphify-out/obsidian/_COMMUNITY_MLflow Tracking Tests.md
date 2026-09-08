---
type: community
members: 19
---

# MLflow Tracking Tests

**Members:** 19 nodes

## Members
- [[Disabled tracking must keep every MLflow API completely untouched.]] - rationale - tests/unit/test_training_runtime.py
- [[MlflowClient]] - code
- [[Path_1]] - code
- [[Return an MLflow-backed tracker only when tracking has been requested.]] - rationale - src/training/tracking.py
- [[_artifact_files()]] - code - tests/unit/test_training_tracking.py
- [[_artifact_paths()]] - code - tests/unit/test_training_tracking.py
- [[_buffered_record()]] - code - tests/unit/test_training_tracking.py
- [[_dataset()_1]] - code - tests/unit/test_training_tracking.py
- [[_experiment_runs()]] - code - tests/unit/test_training_tracking.py
- [[create_tracker()]] - code - src/training/tracking.py
- [[fixture]] - code
- [[mlflow_backend()]] - code - tests/unit/test_training_tracking.py
- [[test_disabled_tracker_buffers_records_without_creating_mlflow_runs()]] - code - tests/unit/test_training_tracking.py
- [[test_disabled_tracker_never_calls_mlflow_apis()]] - code - tests/unit/test_training_runtime.py
- [[test_finalize_cross_validation_logs_parent_artifacts_at_stable_paths()]] - code - tests/unit/test_training_tracking.py
- [[test_finalize_cross_validation_logs_parent_lineage_summary_and_selected_children()]] - code - tests/unit/test_training_tracking.py
- [[test_finalize_cross_validation_replays_one_diagnostic_child_for_a_tie()]] - code - tests/unit/test_training_tracking.py
- [[test_log_single_split_record_replays_into_parent_without_a_child()]] - code - tests/unit/test_training_tracking.py
- [[test_training_tracking.py]] - code - tests/unit/test_training_tracking.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/MLflow_Tracking_Tests
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Cross-Validation Summary Stats]]
- 6 edges to [[_COMMUNITY_MLflow Fold Tracking]]
- 5 edges to [[_COMMUNITY_Model & Training Core]]
- 2 edges to [[_COMMUNITY_Training Config & Data Pipeline]]

## Top bridge nodes
- [[test_training_tracking.py]] - degree 19, connects to 3 communities
- [[create_tracker()]] - degree 14, connects to 3 communities
- [[test_finalize_cross_validation_logs_parent_artifacts_at_stable_paths()]] - degree 9, connects to 1 community
- [[test_finalize_cross_validation_logs_parent_lineage_summary_and_selected_children()]] - degree 9, connects to 1 community
- [[test_finalize_cross_validation_replays_one_diagnostic_child_for_a_tie()]] - degree 8, connects to 1 community