# Graph Report - TRIDENT  (2026-09-07)

## Corpus Check
- 48 files · ~24,838 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 395 nodes · 890 edges · 15 communities (12 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d71c5c95`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- finetuning.py
- FoldTrackingRecord
- runner.py
- types.py
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- summarize_cross_validation
- TRIDENT README
- test_training_tracking.py
- TRIDENT Architecture
- generate_splits.py
- build_fold_result
- _DisabledMlflow
- __init__.py
- trident

## God Nodes (most connected - your core abstractions)
1. `FoldTrackingRecord` - 29 edges
2. `PreparedDataset` - 27 edges
3. `summarize_cross_validation()` - 25 edges
4. `FoldResult` - 24 edges
5. `run_training()` - 20 edges
6. `ArtifactWriter` - 19 edges
7. `MlflowTracker` - 18 edges
8. `CrossValidationSummary` - 18 edges
9. `TRIDENT README` - 16 edges
10. `train_and_evaluate_classifier()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `_stub_training_runtime()` --indirect_call--> `train_pretrainer()`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/pretraining.py
- `FakeTracker` --uses--> `BufferedFoldTracker`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/tracking.py
- `FakeTracker` --uses--> `PreparedDataset`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/types.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (15 total, 2 thin omitted)

### Community 0 - "finetuning.py"
Cohesion: 0.06
Nodes (39): Protocol, Class that encapsulates the creation of embeddings for tabular data: -…, For each row in the DataFrame, generates the resulting embedding: 1) Transforms…, TabularEmbedder, DataFrame, Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]…, Unified model for the classification task: 1) Generates tabular embeddings…, Parameters ---------- embedder : TabularEmbedder Responsible for generating… (+31 more)

### Community 1 - "FoldTrackingRecord"
Cohesion: 0.07
Nodes (32): build_fold_tags(), build_hyperparams_dict(), build_run_tags(), get_or_create_experiment(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, Build a flat dict of all TRIDENT hyperparameters suitable for…, Build a flat dict of MLflow tags for a training run. Parameters ----------… (+24 more)

### Community 2 - "runner.py"
Cohesion: 0.06
Nodes (41): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), define_search_space(), ObjectiveFunctionWrapper (+33 more)

### Community 3 - "types.py"
Cohesion: 0.12
Nodes (26): ArgumentParser, integration, build_training_parser(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests., resolve_training_request(), DatasetSpec (+18 more)

### Community 4 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 5 - "summarize_cross_validation"
Cohesion: 0.11
Nodes (34): DataFrame, _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray, Metric aggregation helpers., Aggregate completed CV folds into comparable final and loss statistics. (+26 more)

### Community 6 - "TRIDENT README"
Cohesion: 0.07
Nodes (43): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, Background & full reference, graphify (+35 more)

### Community 7 - "test_training_tracking.py"
Cohesion: 0.29
Nodes (18): fixture, MlflowClient, create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis(), _artifact_files(), _artifact_paths() (+10 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.09
Nodes (23): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+15 more)

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "_DisabledMlflow"
Cohesion: 0.22
Nodes (4): _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run

## Ambiguous Edges - Review These
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `TRIDENT README` → `src/training Package`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **30 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 141 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `TRIDENT README` and `src/training Package`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PreparedDataset` connect `FoldTrackingRecord` to `finetuning.py`, `runner.py`, `types.py`, `summarize_cross_validation`, `test_training_tracking.py`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `run_training()` connect `runner.py` to `finetuning.py`, `FoldTrackingRecord`, `types.py`, `summarize_cross_validation`, `test_training_tracking.py`, `build_fold_result`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `FoldResult` connect `summarize_cross_validation` to `finetuning.py`, `FoldTrackingRecord`, `runner.py`, `types.py`, `test_training_tracking.py`, `build_fold_result`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `train_and_evaluate_classifier()`) actually correct?**
  _`PreparedDataset` has 6 INFERRED edges - model-reasoned connections that need verification._