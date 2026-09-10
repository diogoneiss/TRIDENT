# Graph Report - TRIDENT  (2026-09-09)

## Corpus Check
- 55 files · ~31,653 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 506 nodes · 1071 edges · 31 communities (28 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 79 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6d49ed5f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- transformer.py
- FoldTrackingRecord
- cli.py
- types.py
- TabularEmbedder
- summarize_cross_validation
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- test_training_tracking.py
- TRIDENT Architecture
- pretraining.py
- generate_splits.py
- build_fold_result
- _DisabledMlflow
- __init__.py
- trident
- TridentPretrainer
- Ticket 0003: Training Loop Performance (Behavior-Preserving)
- finetuning.py
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- Training Refactor Implementation Plan
- opt.py
- backfill_lr_scheduler_tag.py
- ARCHITECTURE.md
- ADR 0002: Curated Cross-Validation MLflow Runs
- mlflow_utils.py
- TRIDENT README
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow
- test_backfill_lr_scheduler_tag.py
- TabularTransformerEncoder

## God Nodes (most connected - your core abstractions)
1. `FoldTrackingRecord` - 29 edges
2. `PreparedDataset` - 27 edges
3. `summarize_cross_validation()` - 25 edges
4. `FoldResult` - 23 edges
5. `run_training()` - 20 edges
6. `ArtifactWriter` - 19 edges
7. `Hyperparameters` - 19 edges
8. `MlflowTracker` - 18 edges
9. `CrossValidationSummary` - 18 edges
10. `train_pretrainer()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `_stub_training_runtime()` --indirect_call--> `train_pretrainer()`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/pretraining.py
- `TRIDENT README` --conceptually_related_to--> `vehicle_00nan Regression Fixture`  [INFERRED]
  README.md → AGENTS.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (31 total, 2 thin omitted)

### Community 0 - "transformer.py"
Cohesion: 0.18
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "FoldTrackingRecord"
Cohesion: 0.07
Nodes (32): build_fold_tags(), Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, ArtifactWriter, Any, DataFrame, Path, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run. (+24 more)

### Community 2 - "cli.py"
Cohesion: 0.17
Nodes (15): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), parse_args(), Namespace (+7 more)

### Community 3 - "types.py"
Cohesion: 0.07
Nodes (53): ArgumentParser, integration, build_training_parser(), _load_base_hyperparameters(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests., Post-parse validation for mutually exclusive and dependent flags. (+45 more)

### Community 4 - "TabularEmbedder"
Cohesion: 0.21
Nodes (8): Reduce a column name to a valid ModuleDict/ParameterDict key., Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, For each row, generates the resulting embedding: 1) Transforms each categorical…, Class that encapsulates the creation of embeddings for tabular data: -…, _sanitize(), TabularEmbedder, Separates real numerical values from special tokens `[MASK]` and `[NULL]`.…, split_numeric_and_special()

### Community 5 - "summarize_cross_validation"
Cohesion: 0.19
Nodes (23): _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray, Metric aggregation helpers., Aggregate completed CV folds into comparable final and loss statistics., _student_t_interval() (+15 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "test_training_tracking.py"
Cohesion: 0.29
Nodes (18): create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis(), _artifact_files(), _artifact_paths(), _buffered_record(), _dataset() (+10 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.09
Nodes (23): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+15 more)

### Community 9 - "pretraining.py"
Cohesion: 0.18
Nodes (12): Protocol, device, Masked reconstruction pre-training stage., Train the masked reconstruction model using the legacy optimization loop., train_pretrainer(), FoldSplit, Log metrics at an optional training step., TrainingTracker (+4 more)

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
Cohesion: 0.25
Nodes (8): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed

### Community 18 - "finetuning.py"
Cohesion: 0.22
Nodes (7): data : pd.DataFrame or EncodedTable Input rows (possibly masked/null, but in…, Unified model for the classification task: 1) Generates tabular embeddings…, TridentModel, Classifier fine-tuning and fold metric calculation., batches_per_epoch(), Selectable learning-rate schedules shared by both training stages. Every…, Number of optimizer steps one epoch performs over ``row_count`` rows.

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.11
Nodes (21): Optimizer, parametrize, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, One training stage's learning-rate schedule behind a uniform hook interface., StageScheduler, validate_lr_scheduler_name(), WarmupCosineMultiplier, _optimizer() (+13 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.09
Nodes (23): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used, C4. Pre-training re-initializes already-initialized layers — protected (+15 more)

### Community 21 - "Training Refactor Implementation Plan"
Cohesion: 0.22
Nodes (12): Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, FoldResult, PreparedDataset, resolve_training_request(), run_from_namespace() (+4 more)

### Community 22 - "opt.py"
Cohesion: 0.21
Nodes (9): define_search_space(), ObjectiveFunctionWrapper, Save the best parameters found so far to the correct location, Run hyperparameter optimization with Optuna, Define the hyperparameter search space for Optuna, Wrapper class for the Optuna objective function to maintain state, run_hyperparameter_optimization(), main() (+1 more)

### Community 23 - "backfill_lr_scheduler_tag.py"
Cohesion: 0.23
Nodes (9): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, Configure the MLflow tracking URI. Priority order: 1. ``tracking_uri`` argument… (+1 more)

### Community 24 - "ARCHITECTURE.md"
Cohesion: 0.23
Nodes (11): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern (+3 more)

### Community 25 - "ADR 0002: Curated Cross-Validation MLflow Runs"
Cohesion: 0.27
Nodes (10): Background & full reference, graphify, MLflow run analysis, ADR 0002: Curated Cross-Validation MLflow Runs, Curated MLflow Cross-Validation Implementation Plan, CrossValidationSummary, MlflowTracker Adapter, summarize_cross_validation() (+2 more)

### Community 26 - "mlflow_utils.py"
Cohesion: 0.22
Nodes (9): build_hyperparams_dict(), build_run_tags(), get_or_create_experiment(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, Build a flat dict of all TRIDENT hyperparameters suitable for…, Build a flat dict of MLflow tags for a training run. Parameters ----------…, Return the normalized percentage encoded by a ``_<n>nan`` suffix. (+1 more)

### Community 27 - "TRIDENT README"
Cohesion: 0.27
Nodes (10): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, TRIDENT README, compute_cv_summary(), TabularEmbedder, TabularTransformerEncoder, TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens (Paper) (+2 more)

### Community 28 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.20
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

### Community 29 - "test_backfill_lr_scheduler_tag.py"
Cohesion: 0.42
Nodes (9): client(), fixture, MlflowClient, _seed_runs(), test_apply_is_idempotent(), test_apply_tags_only_untagged_runs_and_marks_them_backfilled(), test_dry_run_reports_without_writing(), test_main_dry_run_prints_summary() (+1 more)

### Community 30 - "TabularTransformerEncoder"
Cohesion: 0.33
Nodes (3): Parameters ---------- embedder : TabularEmbedder Responsible for generating…, Complete transformer encoder for tabular data, TabularTransformerEncoder

## Ambiguous Edges - Review These
- `src/training Package` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **60 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+55 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 195 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `src/training Package` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PreparedDataset` connect `FoldTrackingRecord` to `pretraining.py`, `finetuning.py`, `types.py`, `test_training_tracking.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `run_training()` connect `types.py` to `FoldTrackingRecord`, `cli.py`, `summarize_cross_validation`, `test_training_tracking.py`, `pretraining.py`, `build_fold_result`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `train_pretrainer()` connect `pretraining.py` to `FoldTrackingRecord`, `types.py`, `TabularEmbedder`, `TridentPretrainer`, `finetuning.py`, `test_training_schedulers.py`, `TabularTransformerEncoder`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `train_and_evaluate_classifier()`) actually correct?**
  _`PreparedDataset` has 6 INFERRED edges - model-reasoned connections that need verification._