---
type: community
members: 44
---

# Training Config & Data Pipeline

**Members:** 44 nodes

## Members
- [[dot-from_mapping()]] - code - src/training/types.py
- [[dot-from_name()]] - code - src/training/types.py
- [[A deterministic short CPUCUDA run remains within recorded tolerances.]] - rationale - tests/integration/test_vehicle_regression.py
- [[Any_1]] - code
- [[ArgumentParser]] - code
- [[Construct the legacy predefined or cross-validation fold splits.]] - rationale - src/training/data.py
- [[DataFrame_3]] - code
- [[Dataset preparation and fold construction for the training workflow.]] - rationale - src/training/data.py
- [[DatasetSpec]] - code - src/training/types.py
- [[Load, encode, and scale a dataset before constructing its folds.]] - rationale - src/training/data.py
- [[Namespace_1]] - code
- [[Opt-in regression baseline for a short typed vehicle training run.]] - rationale - tests/integration/test_vehicle_regression.py
- [[Post-parse validation for mutually exclusive and dependent flags.]] - rationale - src/training/config.py
- [[Run all folds and return raw folds plus the legacy aggregate metrics.]] - rationale - src/training/runner.py
- [[RuntimeOptions]] - code - src/training/types.py
- [[Sets the global seed to ensure reproducibility]] - rationale - src/utils.py
- [[The compatibility facade returns Optuna's flat mean-metrics mapping.]] - rationale - tests/unit/test_training_runtime.py
- [[TrainingRequest]] - code - src/training/types.py
- [[TrainingResult]] - code - src/training/types.py
- [[Translation from legacy argparse namespaces to typed training requests.]] - rationale - src/training/config.py
- [[Typed orchestration for TRIDENT training.]] - rationale - src/training/runner.py
- [[_mlflow_hyperparameters()]] - code - src/training/runner.py
- [[build_folds()]] - code - src/training/data.py
- [[build_training_parser()]] - code - src/training/config.py
- [[config.py]] - code - src/training/config.py
- [[data.py]] - code - src/training/data.py
- [[integration]] - code
- [[load_hyperparameters()]] - code - src/training/config.py
- [[prepare_dataset()]] - code - src/training/data.py
- [[resolve_training_request()]] - code - src/training/config.py
- [[run_training()]] - code - src/training/runner.py
- [[runner.py]] - code - src/training/runner.py
- [[set_global_seed()]] - code - src/utils.py
- [[test_build_folds_creates_two_disjoint_test_splits()]] - code - tests/unit/test_training_data.py
- [[test_build_folds_raises_legacy_error_when_predefined_split_is_missing()]] - code - tests/unit/test_training_data.py
- [[test_legacy_namespace_accepts_hyperparameter_override()]] - code - tests/unit/test_training_config.py
- [[test_legacy_namespace_uses_legacy_runtime_defaults()]] - code - tests/unit/test_training_config.py
- [[test_parser_exposes_runtime_options()]] - code - tests/unit/test_training_config.py
- [[test_train_main_preserves_legacy_cross_validation_return_shape()]] - code - tests/unit/test_training_runtime.py
- [[test_training_config.py]] - code - tests/unit/test_training_config.py
- [[test_training_data.py]] - code - tests/unit/test_training_data.py
- [[test_vehicle_regression.py]] - code - tests/integration/test_vehicle_regression.py
- [[test_vehicle_typed_runner_matches_regression_fixture()]] - code - tests/integration/test_vehicle_regression.py
- [[validate_parsed_args()]] - code - src/training/config.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Training_Config__Data_Pipeline
SORT file.name ASC
```

## Connections to other communities
- 35 edges to [[_COMMUNITY_Model & Training Core]]
- 13 edges to [[_COMMUNITY_CLI Entry & Hyperparameter Search]]
- 6 edges to [[_COMMUNITY_Cross-Validation Summary Stats]]
- 4 edges to [[_COMMUNITY_Artifact Writing & Persistence]]
- 2 edges to [[_COMMUNITY_Fold Metrics Computation]]
- 2 edges to [[_COMMUNITY_MLflow Tracking Tests]]
- 1 edge to [[_COMMUNITY_MLflow Fold Tracking]]

## Top bridge nodes
- [[runner.py]] - degree 26, connects to 7 communities
- [[run_training()]] - degree 20, connects to 6 communities
- [[resolve_training_request()]] - degree 13, connects to 2 communities
- [[config.py]] - degree 12, connects to 2 communities
- [[set_global_seed()]] - degree 5, connects to 2 communities