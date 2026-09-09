# Graph Report - TRIDENT  (2026-09-08)

## Corpus Check
- 48 files · ~26,473 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 414 nodes · 914 edges · 20 communities (17 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a2371d8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EncoderLayer
- types.py
- opt.py
- runner.py
- pretraining.py
- summarize_cross_validation
- TRIDENT README
- test_training_tracking.py
- TRIDENT Architecture
- finetuning.py
- generate_splits.py
- build_fold_result
- _DisabledMlflow
- __init__.py
- trident
- TridentPretrainer
- Ticket 0003: Training Loop Performance (Behavior-Preserving)
- TridentModel
- .encode

## God Nodes (most connected - your core abstractions)
1. `FoldTrackingRecord` - 29 edges
2. `PreparedDataset` - 27 edges
3. `summarize_cross_validation()` - 25 edges
4. `FoldResult` - 23 edges
5. `run_training()` - 20 edges
6. `ArtifactWriter` - 19 edges
7. `MlflowTracker` - 18 edges
8. `CrossValidationSummary` - 18 edges
9. `TRIDENT README` - 16 edges
10. `train_pretrainer()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `_stub_training_runtime()` --indirect_call--> `train_pretrainer()`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/pretraining.py
- `Curated Diagnostic Child Runs Pattern` --conceptually_related_to--> `Dataset Provenance`  [INFERRED]
  docs/adr/0002-curated-cross-validation-mlflow-runs.md → CONTEXT.md
- `Curated Diagnostic Child Runs Pattern` --conceptually_related_to--> `Diagnostic Fold Run`  [INFERRED]
  docs/adr/0002-curated-cross-validation-mlflow-runs.md → CONTEXT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (20 total, 2 thin omitted)

### Community 0 - "EncoderLayer"
Cohesion: 0.17
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "types.py"
Cohesion: 0.07
Nodes (32): build_fold_tags(), Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, ArtifactWriter, Any, Path, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run., Write the parent-run CSV, summary, diagnostic manifest, and lineage. (+24 more)

### Community 2 - "opt.py"
Cohesion: 0.07
Nodes (37): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), define_search_space(), ObjectiveFunctionWrapper (+29 more)

### Community 3 - "runner.py"
Cohesion: 0.09
Nodes (40): ArgumentParser, integration, build_training_parser(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests., resolve_training_request(), build_folds() (+32 more)

### Community 4 - "pretraining.py"
Cohesion: 0.20
Nodes (10): Reduce a column name to a valid ModuleDict/ParameterDict key., Class that encapsulates the creation of embeddings for tabular data: -…, _sanitize(), TabularEmbedder, device, Masked reconstruction pre-training stage., Train the masked reconstruction model using the legacy optimization loop., train_pretrainer() (+2 more)

### Community 5 - "summarize_cross_validation"
Cohesion: 0.17
Nodes (25): DataFrame, _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray, Metric aggregation helpers., Aggregate completed CV folds into comparable final and loss statistics. (+17 more)

### Community 6 - "TRIDENT README"
Cohesion: 0.06
Nodes (57): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, Background & full reference, graphify (+49 more)

### Community 7 - "test_training_tracking.py"
Cohesion: 0.29
Nodes (18): fixture, MlflowClient, create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis(), _artifact_files(), _artifact_paths() (+10 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.09
Nodes (23): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+15 more)

### Community 9 - "finetuning.py"
Cohesion: 0.18
Nodes (12): Protocol, device, Classifier fine-tuning and fold metric calculation., Fine-tune and evaluate the classifier using the legacy optimization loop., train_and_evaluate_classifier(), FinetuningOutcome, Log metrics at an optional training step., TrainingTracker (+4 more)

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "_DisabledMlflow"
Cohesion: 0.22
Nodes (4): _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run

### Community 16 - "TridentPretrainer"
Cohesion: 0.24
Nodes (6): EncodedTable, A whole DataFrame converted to tensors once, then sliced per batch. The per-…, Select rows with a slice or an index tensor, keeping the layout., masked / original : pd.DataFrame or EncodedTable The corrupted view and the…, Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]…, TridentPretrainer

### Community 17 - "Ticket 0003: Training Loop Performance (Behavior-Preserving)"
Cohesion: 0.22
Nodes (8): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed

### Community 18 - "TridentModel"
Cohesion: 0.29
Nodes (4): data : pd.DataFrame or EncodedTable Input rows (possibly masked/null, but in…, Unified model for the classification task: 1) Generates tabular embeddings…, Parameters ---------- embedder : TabularEmbedder Responsible for generating…, TridentModel

### Community 19 - ".encode"
Cohesion: 0.33
Nodes (4): Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, For each row, generates the resulting embedding: 1) Transforms each categorical…, Separates real numerical values from special tokens `[MASK]` and `[NULL]`.…, split_numeric_and_special()

## Ambiguous Edges - Review These
- `src/training Package` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **37 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 153 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `src/training Package` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PreparedDataset` connect `types.py` to `finetuning.py`, `runner.py`, `pretraining.py`, `test_training_tracking.py`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `run_training()` connect `runner.py` to `types.py`, `opt.py`, `pretraining.py`, `summarize_cross_validation`, `test_training_tracking.py`, `finetuning.py`, `build_fold_result`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `FoldResult` connect `summarize_cross_validation` to `types.py`, `runner.py`, `test_training_tracking.py`, `finetuning.py`, `build_fold_result`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `train_and_evaluate_classifier()`) actually correct?**
  _`PreparedDataset` has 6 INFERRED edges - model-reasoned connections that need verification._