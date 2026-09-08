---
type: community
members: 51
---

# MLflow Fold Tracking

**Members:** 51 nodes

## Members
- [[dot-__init__()_3]] - code - src/training/tracking.py
- [[dot-__init__()_4]] - code - src/training/tracking.py
- [[dot-__init__()_5]] - code - tests/unit/test_training_runtime.py
- [[dot-_log_diagnostic_children()]] - code - src/training/tracking.py
- [[dot-_log_parent_summary()]] - code - src/training/tracking.py
- [[dot-_replay_record()]] - code - src/training/tracking.py
- [[dot-finalize_cross_validation()]] - code - src/training/tracking.py
- [[dot-finalize_cross_validation()_1]] - code - src/training/tracking.py
- [[dot-finalize_cross_validation()_2]] - code - tests/unit/test_training_runtime.py
- [[dot-fold_run()]] - code - src/training/tracking.py
- [[dot-fold_run()_1]] - code - src/training/tracking.py
- [[dot-fold_run()_2]] - code - tests/unit/test_training_runtime.py
- [[dot-log_artifact()]] - code - src/training/tracking.py
- [[dot-log_artifact()_1]] - code - src/training/tracking.py
- [[dot-log_artifact()_2]] - code - src/training/tracking.py
- [[dot-log_artifact()_3]] - code - tests/unit/test_training_runtime.py
- [[dot-log_metrics()_1]] - code - src/training/tracking.py
- [[dot-log_metrics()_2]] - code - src/training/tracking.py
- [[dot-log_metrics()_3]] - code - src/training/tracking.py
- [[dot-log_prepared_dataset()]] - code - src/training/tracking.py
- [[dot-log_prepared_dataset()_1]] - code - src/training/tracking.py
- [[dot-log_prepared_dataset()_2]] - code - tests/unit/test_training_runtime.py
- [[dot-log_single_split_record()]] - code - src/training/tracking.py
- [[dot-log_single_split_record()_1]] - code - src/training/tracking.py
- [[dot-log_single_split_record()_2]] - code - tests/unit/test_training_runtime.py
- [[dot-parent_run()]] - code - src/training/tracking.py
- [[dot-parent_run()_1]] - code - tests/unit/test_training_runtime.py
- [[dot-to_record()]] - code - src/training/tracking.py
- [[A tracker whose methods deliberately avoid importing MLflow side effects.]] - rationale - src/training/tracking.py
- [[BufferedFoldTracker]] - code - src/training/tracking.py
- [[Build tags for a per-fold child run. Parameters ---------- fold_idx Zero-based…]] - rationale - src/mlflow_utils.py
- [[Collect one fold's MLflow events until its diagnostic role is known.]] - rationale - src/training/tracking.py
- [[CrossValidationSummary]] - code - src/training/types.py
- [[DisabledTracker]] - code - src/training/tracking.py
- [[FakeTracker]] - code - tests/unit/test_training_runtime.py
- [[FoldTrackingRecord]] - code - src/training/types.py
- [[Log one comparison parent and only selected diagnostic fold runs.]] - rationale - src/training/tracking.py
- [[LoggedArtifact]] - code - src/training/types.py
- [[LoggedMetric]] - code - src/training/types.py
- [[MLflow adapters for the training runtime.]] - rationale - src/training/tracking.py
- [[MetricSummary]] - code - src/training/types.py
- [[MlflowTracker]] - code - src/training/tracking.py
- [[TrackingArtifactPaths]] - code - src/training/types.py
- [[_dataset()]] - code - tests/unit/test_training_artifacts.py
- [[_record()]] - code - tests/unit/test_training_artifacts.py
- [[_summary()]] - code - tests/unit/test_training_artifacts.py
- [[build_fold_tags()]] - code - src/mlflow_utils.py
- [[test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations()]] - code - tests/unit/test_training_artifacts.py
- [[test_training_artifacts.py]] - code - tests/unit/test_training_artifacts.py
- [[test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values()]] - code - tests/unit/test_training_artifacts.py
- [[tracking.py]] - code - src/training/tracking.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/MLflow_Fold_Tracking
SORT file.name ASC
```

## Connections to other communities
- 27 edges to [[_COMMUNITY_Model & Training Core]]
- 23 edges to [[_COMMUNITY_Cross-Validation Summary Stats]]
- 14 edges to [[_COMMUNITY_Artifact Writing & Persistence]]
- 7 edges to [[_COMMUNITY_CLI Entry & Hyperparameter Search]]
- 6 edges to [[_COMMUNITY_MLflow Tracking Tests]]
- 1 edge to [[_COMMUNITY_Training Config & Data Pipeline]]

## Top bridge nodes
- [[tracking.py]] - degree 22, connects to 5 communities
- [[FoldTrackingRecord]] - degree 29, connects to 3 communities
- [[MlflowTracker]] - degree 18, connects to 3 communities
- [[CrossValidationSummary]] - degree 18, connects to 3 communities
- [[test_training_artifacts.py]] - degree 16, connects to 3 communities