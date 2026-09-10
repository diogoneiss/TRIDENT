# Graph Report - TRIDENT  (2026-09-09)

## Corpus Check
- 59 files · ~35,423 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 570 nodes · 1252 edges · 24 communities (20 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e6f134f3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- TabularEmbedder
- FoldTrackingRecord
- opt.py
- types.py
- summarize_cross_validation
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- test_training_tracking.py
- TRIDENT Architecture
- runtime_environment_tags
- generate_splits.py
- build_fold_result
- __init__.py
- trident
- Ticket 0003: Training Loop Performance (Behavior-Preserving)
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- Training Refactor Implementation Plan
- backfill_lr_scheduler_tag.py
- ARCHITECTURE.md
- ADR 0002: Curated Cross-Validation MLflow Runs
- TRIDENT README
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow

## God Nodes (most connected - your core abstractions)
1. `FoldTrackingRecord` - 36 edges
2. `summarize_cross_validation()` - 31 edges
3. `PreparedDataset` - 30 edges
4. `FoldResult` - 24 edges
5. `MlflowTracker` - 23 edges
6. `run_training()` - 22 edges
7. `CrossValidationSummary` - 22 edges
8. `ArtifactWriter` - 19 edges
9. `create_tracker()` - 19 edges
10. `Hyperparameters` - 19 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `_timing_events()` --uses--> `FoldTrackingRecord`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/types.py
- `_without_timings()` --uses--> `FoldTrackingRecord`  [INFERRED]
  tests/unit/test_training_summary.py → src/training/types.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (24 total, 2 thin omitted)

### Community 0 - "TabularEmbedder"
Cohesion: 0.06
Nodes (28): EncodedTable, Reduce a column name to a valid ModuleDict/ParameterDict key., Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, A whole DataFrame converted to tensors once, then sliced per batch. The per-…, For each row, generates the resulting embedding: 1) Transforms each categorical…, Select rows with a slice or an index tensor, keeping the layout., Class that encapsulates the creation of embeddings for tabular data: -…, _sanitize() (+20 more)

### Community 1 - "FoldTrackingRecord"
Cohesion: 0.06
Nodes (39): build_fold_tags(), Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, ArtifactWriter, Any, DataFrame, Path, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run. (+31 more)

### Community 2 - "opt.py"
Cohesion: 0.05
Nodes (42): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), define_search_space(), _DisabledMlflow (+34 more)

### Community 3 - "types.py"
Cohesion: 0.05
Nodes (73): ArgumentParser, integration, Protocol, build_training_parser(), _load_base_hyperparameters(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests. (+65 more)

### Community 5 - "summarize_cross_validation"
Cohesion: 0.14
Nodes (34): _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray, Metric aggregation helpers., Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics. (+26 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "test_training_tracking.py"
Cohesion: 0.14
Nodes (33): build_hyperparams_dict(), get_or_create_experiment(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, Build a flat dict of all TRIDENT hyperparameters suitable for…, Return the normalized percentage encoded by a ``_<n>nan`` suffix., Configure the MLflow tracking URI. Priority order: 1. ``tracking_uri`` argument…, Return the experiment_id for the given dataset, creating it if absent.… (+25 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.09
Nodes (23): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+15 more)

### Community 9 - "runtime_environment_tags"
Cohesion: 0.36
Nodes (6): device, Hardware and library descriptors recorded as MLflow tags., Describe where a run trains so its timings are comparable across machines.…, runtime_environment_tags(), test_cpu_environment_tags_keep_every_key(), test_cuda_environment_tags_record_the_gpu()

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 17 - "Ticket 0003: Training Loop Performance (Behavior-Preserving)"
Cohesion: 0.25
Nodes (8): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.11
Nodes (21): Optimizer, parametrize, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, One training stage's learning-rate schedule behind a uniform hook interface., StageScheduler, validate_lr_scheduler_name(), WarmupCosineMultiplier, _optimizer() (+13 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.08
Nodes (24): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used (+16 more)

### Community 21 - "Training Refactor Implementation Plan"
Cohesion: 0.22
Nodes (12): Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, FoldResult, PreparedDataset, resolve_training_request(), run_from_namespace() (+4 more)

### Community 23 - "backfill_lr_scheduler_tag.py"
Cohesion: 0.17
Nodes (16): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, client() (+8 more)

### Community 24 - "ARCHITECTURE.md"
Cohesion: 0.23
Nodes (11): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern (+3 more)

### Community 25 - "ADR 0002: Curated Cross-Validation MLflow Runs"
Cohesion: 0.27
Nodes (10): Background & full reference, graphify, MLflow run analysis, ADR 0002: Curated Cross-Validation MLflow Runs, Curated MLflow Cross-Validation Implementation Plan, CrossValidationSummary, MlflowTracker Adapter, summarize_cross_validation() (+2 more)

### Community 27 - "TRIDENT README"
Cohesion: 0.27
Nodes (10): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, TRIDENT README, compute_cv_summary(), TabularEmbedder, TabularTransformerEncoder, TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens (Paper) (+2 more)

### Community 28 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.20
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

## Ambiguous Edges - Review These
- `src/training Package` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **61 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+56 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 212 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `src/training Package` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PreparedDataset` connect `FoldTrackingRecord` to `types.py`, `test_training_tracking.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `run_training()` connect `types.py` to `FoldTrackingRecord`, `opt.py`, `summarize_cross_validation`, `test_training_tracking.py`, `runtime_environment_tags`, `build_fold_result`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `FoldTrackingRecord` connect `FoldTrackingRecord` to `types.py`, `summarize_cross_validation`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `train_and_evaluate_classifier()`) actually correct?**
  _`PreparedDataset` has 7 INFERRED edges - model-reasoned connections that need verification._