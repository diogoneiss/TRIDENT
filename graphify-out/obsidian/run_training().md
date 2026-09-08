---
source_file: "src/training/runner.py"
type: "code"
community: "Training Config & Data Pipeline"
location: "L33"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Training_Config__Data_Pipeline
---

# run_training()

## Connections
- [[ArtifactWriter]] - `calls` [EXTRACTED]
- [[FoldResult]] - `calls` [EXTRACTED]
- [[Run all folds and return raw folds plus the legacy aggregate metrics.]] - `rationale_for` [EXTRACTED]
- [[TrainingRequest]] - `uses` [INFERRED]
- [[TrainingResult]] - `calls` [EXTRACTED]
- [[_mlflow_hyperparameters()]] - `calls` [EXTRACTED]
- [[build_folds()]] - `calls` [EXTRACTED]
- [[cli.py]] - `imports` [EXTRACTED]
- [[compute_cv_summary()]] - `calls` [EXTRACTED]
- [[create_tracker()]] - `calls` [EXTRACTED]
- [[prepare_dataset()]] - `calls` [EXTRACTED]
- [[run_from_namespace()]] - `calls` [EXTRACTED]
- [[runner.py]] - `contains` [EXTRACTED]
- [[set_global_seed()]] - `calls` [EXTRACTED]
- [[summarize_cross_validation()]] - `calls` [EXTRACTED]
- [[test_vehicle_regression.py]] - `imports` [EXTRACTED]
- [[test_vehicle_typed_runner_matches_regression_fixture()]] - `calls` [EXTRACTED]
- [[train.py]] - `imports` [EXTRACTED]
- [[train_and_evaluate_classifier()]] - `calls` [EXTRACTED]
- [[train_pretrainer()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Training_Config__Data_Pipeline