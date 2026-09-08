---
source_file: "src/training/summary.py"
type: "code"
community: "Cross-Validation Summary Stats"
location: "L29"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Cross-Validation_Summary_Stats
---

# summarize_cross_validation()

## Connections
- [[Aggregate completed CV folds into comparable final and loss statistics.]] - `rationale_for` [EXTRACTED]
- [[CrossValidationSummary]] - `calls` [EXTRACTED]
- [[FoldTrackingRecord]] - `uses` [INFERRED]
- [[_diagnostic_roles()]] - `calls` [EXTRACTED]
- [[_summarize_loss_bands()]] - `calls` [EXTRACTED]
- [[_summarize_metric()]] - `calls` [EXTRACTED]
- [[_validate_final_metrics()]] - `calls` [EXTRACTED]
- [[final_metrics_for_tracking()]] - `calls` [EXTRACTED]
- [[run_training()]] - `calls` [EXTRACTED]
- [[runner.py]] - `imports` [EXTRACTED]
- [[summary.py]] - `contains` [EXTRACTED]
- [[test_disabled_tracker_buffers_records_without_creating_mlflow_runs()]] - `calls` [EXTRACTED]
- [[test_finalize_cross_validation_logs_parent_artifacts_at_stable_paths()]] - `calls` [EXTRACTED]
- [[test_finalize_cross_validation_logs_parent_lineage_summary_and_selected_children()]] - `calls` [EXTRACTED]
- [[test_finalize_cross_validation_replays_one_diagnostic_child_for_a_tie()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_logs_statistics_and_loss_bands()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_rejects_fewer_than_two_folds()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_rejects_mismatched_loss_epochs()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_rejects_missing_final_metric()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_rejects_non_finite_final_metric()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_uses_lowest_fold_for_a_maximum_tie()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_uses_lowest_fold_for_a_minimum_tie()]] - `calls` [EXTRACTED]
- [[test_summarize_cross_validation_uses_one_role_for_a_tie()]] - `calls` [EXTRACTED]
- [[test_training_summary.py]] - `imports` [EXTRACTED]
- [[test_training_tracking.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Cross-Validation_Summary_Stats