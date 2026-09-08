---
source_file: "src/training/finetuning.py"
type: "code"
community: "Fold Metrics Computation"
location: "L19"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Fold_Metrics_Computation
---

# build_fold_result()

## Connections
- [[Build the legacy per-fold classification metric set.]] - `rationale_for` [EXTRACTED]
- [[FoldResult]] - `calls` [EXTRACTED]
- [[finetuning.py]] - `contains` [EXTRACTED]
- [[ndarray]] - `references` [EXTRACTED]
- [[test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold()]] - `calls` [EXTRACTED]
- [[test_build_fold_result_omits_binary_confusion_fields_for_multiclass_dataset()]] - `calls` [EXTRACTED]
- [[test_build_fold_result_retains_classification_metrics()]] - `calls` [EXTRACTED]
- [[test_build_fold_result_uses_zero_precision_without_warning_for_unpredicted_class()]] - `calls` [EXTRACTED]
- [[test_training_metrics.py]] - `imports` [EXTRACTED]
- [[train_and_evaluate_classifier()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Fold_Metrics_Computation