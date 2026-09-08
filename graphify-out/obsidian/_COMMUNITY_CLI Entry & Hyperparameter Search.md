---
type: community
members: 45
---

# CLI Entry & Hyperparameter Search

**Members:** 45 nodes

## Members
- [[dot-__call__()]] - code - opt.py
- [[dot-__init__()_6]] - code - opt.py
- [[dot-parent_run()_2]] - code - src/training/tracking.py
- [[dot-save_best_params()]] - code - opt.py
- [[Build a flat dict of MLflow tags for a training run. Parameters ----------…]] - rationale - src/mlflow_utils.py
- [[Build a flat dict of all TRIDENT hyperparameters suitable for…]] - rationale - src/mlflow_utils.py
- [[Command-line entry point for TRIDENT.]] - rationale - main.py
- [[Compatibility entry points for command-line and programmatic training.]] - rationale - src/training/cli.py
- [[Configure the MLflow tracking URI. Priority order 1. ``tracking_uri`` argument…]] - rationale - src/mlflow_utils.py
- [[Define the hyperparameter search space for Optuna]] - rationale - opt.py
- [[Legacy public facade for TRIDENT training.]] - rationale - train.py
- [[Namespace]] - code
- [[ObjectiveFunctionWrapper]] - code - opt.py
- [[Retain the historical ``train.main`` programmatic entry point.]] - rationale - train.py
- [[Return a human-readable duration string.]] - rationale - main.py
- [[Return sorted dataset names from datasetsprocessed_datasets. Each…]] - rationale - src/training/discovery.py
- [[Return the experiment_id for the given dataset, creating it if absent.…]] - rationale - src/mlflow_utils.py
- [[Return the normalized percentage encoded by a ``_nnan`` suffix.]] - rationale - src/mlflow_utils.py
- [[Run a legacy namespace and retain the flat Optuna return contract.]] - rationale - src/training/cli.py
- [[Run hyperparameter optimization with Optuna]] - rationale - opt.py
- [[Run training sequentially on every discovered dataset.]] - rationale - main.py
- [[Runtime discovery of available datasets in datasetsprocessed_datasets.]] - rationale - src/training/discovery.py
- [[Save the best parameters found so far to the correct location]] - rationale - opt.py
- [[Wrapper class for the Optuna objective function to maintain state]] - rationale - opt.py
- [[_format_duration()]] - code - main.py
- [[build_hyperparams_dict()]] - code - src/mlflow_utils.py
- [[build_run_tags()]] - code - src/mlflow_utils.py
- [[cli.py]] - code - src/training/cli.py
- [[define_search_space()]] - code - opt.py
- [[discover_datasets()]] - code - src/training/discovery.py
- [[discovery.py]] - code - src/training/discovery.py
- [[get_or_create_experiment()]] - code - src/mlflow_utils.py
- [[main()_1]] - code - main.py
- [[main()_2]] - code - train.py
- [[main.py]] - code - main.py
- [[mlflow_utils.py]] - code - src/mlflow_utils.py
- [[opt.py]] - code - opt.py
- [[parse_args()]] - code - src/training/cli.py
- [[parse_missingness_percent()]] - code - src/mlflow_utils.py
- [[run_all()]] - code - main.py
- [[run_from_namespace()]] - code - src/training/cli.py
- [[run_hyperparameter_optimization()]] - code - opt.py
- [[setup_mlflow()]] - code - src/mlflow_utils.py
- [[srcmlflow_utils.py ------------------- Centralised MLflow utilities for…]] - rationale - src/mlflow_utils.py
- [[train.py]] - code - train.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/CLI_Entry__Hyperparameter_Search
SORT file.name ASC
```

## Connections to other communities
- 13 edges to [[_COMMUNITY_Training Config & Data Pipeline]]
- 7 edges to [[_COMMUNITY_MLflow Fold Tracking]]
- 2 edges to [[_COMMUNITY_Disabled MLflow Stub (Optuna)]]
- 1 edge to [[_COMMUNITY_Fold Metrics Computation]]
- 1 edge to [[_COMMUNITY_Model & Training Core]]
- 1 edge to [[_COMMUNITY_Cross-Validation Summary Stats]]

## Top bridge nodes
- [[opt.py]] - degree 12, connects to 3 communities
- [[train.py]] - degree 10, connects to 3 communities
- [[cli.py]] - degree 11, connects to 1 community
- [[run_hyperparameter_optimization()]] - degree 10, connects to 1 community
- [[run_from_namespace()]] - degree 10, connects to 1 community