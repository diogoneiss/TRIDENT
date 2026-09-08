# Graph Report - TRIDENT  (2026-09-07)

## Corpus Check
- Corpus is ~22,113 words - fits in a single context window. You may not need a graph.

## Summary
- 367 nodes · 859 edges · 16 communities (13 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.92)
- Token cost: 136,858 input · 0 output

## Community Hubs (Navigation)
- Model & Training Core
- MLflow Fold Tracking
- CLI Entry & Hyperparameter Search
- Training Config & Data Pipeline
- Curated CV Docs & Glossary
- Cross-Validation Summary Stats
- Training Refactor Design Docs
- MLflow Tracking Tests
- Artifact Writing & Persistence
- Transformer Encoder Layers
- Dataset Split Generation
- Fold Metrics Computation
- Disabled MLflow Stub (Optuna)
- Training Package Init
- Trident Package Root

## God Nodes (most connected - your core abstractions)
1. `FoldTrackingRecord` - 29 edges
2. `PreparedDataset` - 27 edges
3. `summarize_cross_validation()` - 25 edges
4. `FoldResult` - 24 edges
5. `run_training()` - 20 edges
6. `ArtifactWriter` - 19 edges
7. `MlflowTracker` - 18 edges
8. `CrossValidationSummary` - 18 edges
9. `train_and_evaluate_classifier()` - 15 edges
10. `train_pretrainer()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `test_vehicle_typed_runner_matches_regression_fixture()` --uses--> `Hyperparameters`  [INFERRED]
  tests/integration/test_vehicle_regression.py → src/training/types.py
- `_stub_training_runtime()` --uses--> `DatasetSpec`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/types.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (16 total, 2 thin omitted)

### Community 0 - "Model & Training Core"
Cohesion: 0.08
Nodes (38): Protocol, Class that encapsulates the creation of embeddings for tabular data: -…, For each row in the DataFrame, generates the resulting embedding: 1) Transforms…, TabularEmbedder, DataFrame, Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]…, Unified model for the classification task: 1) Generates tabular embeddings…, Parameters ---------- embedder : TabularEmbedder Responsible for generating… (+30 more)

### Community 1 - "MLflow Fold Tracking"
Cohesion: 0.08
Nodes (21): build_fold_tags(), Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, BufferedFoldTracker, DisabledTracker, MlflowTracker, MLflow adapters for the training runtime., Collect one fold's MLflow events until its diagnostic role is known., A tracker whose methods deliberately avoid importing MLflow side effects. (+13 more)

### Community 2 - "CLI Entry & Hyperparameter Search"
Cohesion: 0.07
Nodes (35): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), define_search_space(), ObjectiveFunctionWrapper (+27 more)

### Community 3 - "Training Config & Data Pipeline"
Cohesion: 0.09
Nodes (36): ArgumentParser, integration, build_training_parser(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests., Post-parse validation for mutually exclusive and dependent flags., resolve_training_request() (+28 more)

### Community 4 - "Curated CV Docs & Glossary"
Cohesion: 0.12
Nodes (31): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+23 more)

### Community 5 - "Cross-Validation Summary Stats"
Cohesion: 0.19
Nodes (23): _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray, Metric aggregation helpers., Aggregate completed CV folds into comparable final and loss statistics., _student_t_interval() (+15 more)

### Community 6 - "Training Refactor Design Docs"
Cohesion: 0.13
Nodes (23): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern (+15 more)

### Community 7 - "MLflow Tracking Tests"
Cohesion: 0.29
Nodes (18): fixture, MlflowClient, create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis(), _artifact_files(), _artifact_paths() (+10 more)

### Community 8 - "Artifact Writing & Persistence"
Cohesion: 0.16
Nodes (11): ArtifactWriter, Any, DataFrame, Path, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run., Write the parent-run CSV, summary, diagnostic manifest, and lineage., _to_builtin() (+3 more)

### Community 9 - "Transformer Encoder Layers"
Cohesion: 0.17
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 10 - "Dataset Split Generation"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "Fold Metrics Computation"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "Disabled MLflow Stub (Optuna)"
Cohesion: 0.22
Nodes (4): _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run

## Ambiguous Edges - Review These
- `TRIDENT README` → `src/training Package`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `TRIDENT README` → `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **10 isolated node(s):** `info`, `trident`, `credit-g Categorical Columns`, `electricity Categorical Columns`, `kr-vs-kp Categorical Columns` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 121 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `TRIDENT README` and `src/training Package`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `TRIDENT README` and `kr-vs-kp Categorical Columns`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PreparedDataset` connect `Model & Training Core` to `Artifact Writing & Persistence`, `MLflow Fold Tracking`, `Training Config & Data Pipeline`, `MLflow Tracking Tests`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Training Config & Data Pipeline` to `Model & Training Core`, `CLI Entry & Hyperparameter Search`, `Cross-Validation Summary Stats`, `MLflow Tracking Tests`, `Artifact Writing & Persistence`, `Fold Metrics Computation`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `FoldResult` connect `Cross-Validation Summary Stats` to `Model & Training Core`, `MLflow Fold Tracking`, `Training Config & Data Pipeline`, `MLflow Tracking Tests`, `Artifact Writing & Persistence`, `Fold Metrics Computation`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `train_and_evaluate_classifier()`) actually correct?**
  _`PreparedDataset` has 6 INFERRED edges - model-reasoned connections that need verification._