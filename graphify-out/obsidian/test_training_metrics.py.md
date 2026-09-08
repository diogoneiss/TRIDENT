---
source_file: "tests/unit/test_training_metrics.py"
type: "code"
community: "Fold Metrics Computation"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Fold_Metrics_Computation
---

# test_training_metrics.py

## Connections
- [[build_fold_result()]] - `imports` [EXTRACTED]
- [[compute_cv_summary()]] - `imports` [EXTRACTED]
- [[finetuning.py]] - `imports_from` [EXTRACTED]
- [[summary.py]] - `imports_from` [EXTRACTED]
- [[test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold()]] - `contains` [EXTRACTED]
- [[test_build_fold_result_omits_binary_confusion_fields_for_multiclass_dataset()]] - `contains` [EXTRACTED]
- [[test_build_fold_result_retains_classification_metrics()]] - `contains` [EXTRACTED]
- [[test_build_fold_result_uses_zero_precision_without_warning_for_unpredicted_class()]] - `contains` [EXTRACTED]
- [[test_compute_cv_summary_retains_legacy_mean_std_format()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Fold_Metrics_Computation