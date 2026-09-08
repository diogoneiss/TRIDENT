---
type: community
members: 13
---

# Fold Metrics Computation

**Members:** 13 nodes

## Members
- [[Build the legacy per-fold classification metric set.]] - rationale - src/training/finetuning.py
- [[DataFrame_2]] - code
- [[Path]] - code
- [[Return the legacy mean, sample standard deviation, and display string.]] - rationale - src/training/summary.py
- [[build_fold_result()]] - code - src/training/finetuning.py
- [[compute_cv_summary()]] - code - src/training/summary.py
- [[ndarray]] - code
- [[test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold()]] - code - tests/unit/test_training_metrics.py
- [[test_build_fold_result_omits_binary_confusion_fields_for_multiclass_dataset()]] - code - tests/unit/test_training_metrics.py
- [[test_build_fold_result_retains_classification_metrics()]] - code - tests/unit/test_training_metrics.py
- [[test_build_fold_result_uses_zero_precision_without_warning_for_unpredicted_class()]] - code - tests/unit/test_training_metrics.py
- [[test_compute_cv_summary_retains_legacy_mean_std_format()]] - code - tests/unit/test_training_metrics.py
- [[test_training_metrics.py]] - code - tests/unit/test_training_metrics.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Fold_Metrics_Computation
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Cross-Validation Summary Stats]]
- 3 edges to [[_COMMUNITY_Model & Training Core]]
- 2 edges to [[_COMMUNITY_Training Config & Data Pipeline]]
- 1 edge to [[_COMMUNITY_CLI Entry & Hyperparameter Search]]

## Top bridge nodes
- [[compute_cv_summary()]] - degree 9, connects to 3 communities
- [[build_fold_result()]] - degree 10, connects to 2 communities
- [[test_training_metrics.py]] - degree 9, connects to 2 communities