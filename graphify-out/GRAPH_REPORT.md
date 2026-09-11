# Graph Report - TRIDENT  (2026-09-11)

## Corpus Check
- 137 files · ~113,708 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1340 nodes · 2558 edges · 104 communities (95 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 124 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d039c47c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- transformer.py
- BufferedFoldTracker
- test_training_config.py
- Answer
- decoding.py
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- tracking.py
- TRIDENT Architecture
- runner.py
- generate_splits.py
- build_fold_result
- test_training_data.py
- __init__.py
- trident
- TabularEmbedder
- score_cells
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- Training Refactor Implementation Plan
- test_training_tracking.py
- summarize_cross_validation
- FoldTrackingRecord
- test_training_tasks.py
- backfill
- TRIDENT README
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow
- Reduced Optuna Search for Imputation: Implementation Plan
- preprocess_table
- test_training_decoding.py
- test_opt_search_space.py
- Ticket 0003: Training Loop Performance (Behavior-Preserving)
- TridentDecoder
- ArtifactWriter
- run_hyperparameter_optimization
- Imputation Decoder Task Implementation Plan
- backfill_lr_scheduler_tag.py
- test_decoder_model.py
- embedder.py
- test_embedder.py
- TridentPretrainer
- spec.md
- run_training
- TRIDENT Agent Guide (AGENTS.md)
- 02. Request, hyperparameters and command line
- Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly
- Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow
- 03. Imputation metrics as pure functions
- 04. The decoder model
- 05. Data support for evaluation
- 06. The decode stage
- 07. Imputation artifacts
- 08. Runner, MLflow identity, batch summary
- 09. Tag backfill
- 10. Optuna for the imputation task
- 11. Regression fixture and documentation
- Map: Imputation decoder task
- 08. Record the design: ADR 0005 and the implementation plan
- Ticket 0004: Exact original units in the imputation preview
- 01. Research: how do imputation papers score mixed-type tabular imputation?
- 02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?
- 10. Record the design: ADR and implementation plan
- 03. What is the imputation ground truth, and on which cells is error scored?
- 04. Decoder architecture and the stage that trains it
- 05. Loss, error metrics and the imputation fold-ranking metric
- 06. How are imputation runs identified in MLflow?
- 07. Prototype: the human-friendly imputed-vs-actual preview
- 08. CLI flag and hyperparameter schema for the imputation task
- 09. How does the training package branch per task without touching classification?
- 11. Multi-rate evaluation: across runs or within one run?
- 12. Should the [NULL] path be scored as a diagnostic?
- 13. Optuna for the imputation task
- 14. A regression fixture for the imputation task
- 01. Search-space profiles
- Curated MLflow Cross-Validation Design Spec
- CLAUDE.md
- 02. The search objective on the validation split
- 04. The importance artifact
- 05. The launcher and the six studies
- 06. The comparison and the documentation
- run_from_namespace
- TridentModel
- Map: Reduced Optuna search for the imputation task
- Reduced Optuna search for imputation: implementation tickets
- 01. The held set and the ranges
- 02. The search objective is scored on the validation split
- 03. Study protocol: datasets, variants, budget, schedule, seed
- 04. Where a promoted configuration lives and how a run finds it
- 05. The search-space profile flag and its record
- 06. Checking the reduction and proving the tuning helped
- 07. Task: repair the imputation Optuna path and prove it with a two-trial study
- types.py
- Hyperparameters
- Research: third-party stub strategy for sklearn and scipy
- CrossValidationSummary
- 06. The hyperparameter JSON schema and the three naming layers
- 07. Enforcing the constants convention beyond what the checker catches
- 09. Record the design: ADR 0006 and the implementation plan

## God Nodes (most connected - your core abstractions)
1. `summarize_cross_validation()` - 39 edges
2. `PreparedDataset` - 39 edges
3. `FoldTrackingRecord` - 39 edges
4. `run_training()` - 35 edges
5. `run_hyperparameter_optimization()` - 30 edges
6. `Hyperparameters` - 30 edges
7. `FoldResult` - 29 edges
8. `ArtifactWriter` - 26 edges
9. `train_and_evaluate_decoder()` - 26 edges
10. `resolve_training_request()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `_stub_training_runtime()` --indirect_call--> `train_pretrainer()`  [INFERRED]
  tests/unit/test_training_runtime.py → src/training/pretraining.py
- `test_create_tracker_selects_the_tracker_by_role()` --uses--> `DisabledTracker`  [INFERRED]
  tests/unit/test_training_tracking.py → src/training/tracking.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (104 total, 2 thin omitted)

### Community 0 - "transformer.py"
Cohesion: 0.18
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "BufferedFoldTracker"
Cohesion: 0.15
Nodes (3): BufferedFoldTracker, Collect one fold's MLflow events until its diagnostic role is known., FakeTracker

### Community 3 - "test_training_config.py"
Cohesion: 0.09
Nodes (37): parse_args(), Compatibility entry points for command-line and programmatic training., build_training_parser(), logged_hyperparameters(), ArgumentParser, Path, Post-parse validation for mutually exclusive and dependent flags., The parameters this run actually used, under their config-file names. (+29 more)

### Community 4 - "Answer"
Cohesion: 0.04
Nodes (42): 01. The Python floor moves to 3.11, Acceptance for the bump, Answer, Question, What 3.11 buys, concretely, What was verified, and what was not, Why not further, Why the floor mattered more than it looks (+34 more)

### Community 5 - "decoding.py"
Cohesion: 0.17
Nodes (19): _already_missing(), _clean(), _exact(), DataFrame, device, Tensor, Decode stage: train a decoder to reconstruct hidden cells, then score what it…, Score the cells the dataset is actually missing, against the complete sibling.… (+11 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "tracking.py"
Cohesion: 0.09
Nodes (22): build_fold_tags(), build_hyperparams_dict(), build_run_tags(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, Build a flat dict of all TRIDENT hyperparameters suitable for…, Build a flat dict of MLflow tags for a training run. Parameters ----------…, Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based… (+14 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.08
Nodes (26): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+18 more)

### Community 9 - "runner.py"
Cohesion: 0.19
Nodes (11): device, Hardware and library descriptors recorded as MLflow tags., Describe where a run trains so its timings are comparable across machines.…, runtime_environment_tags(), _per_column_scores(), Typed orchestration for TRIDENT training., Each population's per-column errors, against this fold's own naive baseline., Sets the global seed to ensure reproducibility (+3 more)

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "test_training_data.py"
Cohesion: 0.11
Nodes (27): _assert_row_aligned(), build_folds(), declared_column_types(), load_complete_sibling(), prepare_dataset(), DataFrame, Path, Refuse a sibling that describes different rows. The variants are generated… (+19 more)

### Community 17 - "TabularEmbedder"
Cohesion: 0.21
Nodes (9): Class that encapsulates the creation of embeddings for tabular data: -…, TabularEmbedder, Parameters ---------- embedder : TabularEmbedder Responsible for generating…, Complete transformer encoder for tabular data, TabularTransformerEncoder, Everything pre-training learned survives the decoder being attached. Pre-…, Loss is the mean categorical surprise plus the weighted mean numerical error.…, test_attaching_the_decoder_leaves_the_pretrained_encoder_untouched() (+1 more)

### Community 18 - "score_cells"
Cohesion: 0.10
Nodes (32): _error_metrics(), ImputationScores, mean_mode_baselines(), DataFrame, _ratio(), Imputation error, and the score that ranks folds by it. Pure functions over a…, Model error as a share of the naive imputer's, guarding a flawless baseline. A…, What a naive imputer would fill each column's cells with. The mean of a… (+24 more)

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.11
Nodes (21): Optimizer, parametrize, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, One training stage's learning-rate schedule behind a uniform hook interface., StageScheduler, validate_lr_scheduler_name(), WarmupCosineMultiplier, _optimizer() (+13 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.08
Nodes (26): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used (+18 more)

### Community 21 - "Training Refactor Implementation Plan"
Cohesion: 0.22
Nodes (12): Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, FoldResult, PreparedDataset, resolve_training_request(), run_from_namespace() (+4 more)

### Community 22 - "test_training_tracking.py"
Cohesion: 0.19
Nodes (31): get_or_create_experiment(), Configure the MLflow tracking URI. Priority order: 1. ``tracking_uri`` argument…, Return the experiment_id for the given dataset, creating it if absent.…, setup_mlflow(), create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis() (+23 more)

### Community 23 - "summarize_cross_validation"
Cohesion: 0.22
Nodes (23): Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics.…, stage_timing_metrics(), summarize_cross_validation(), FoldResult, LoggedMetric, _record(), test_fold_timings_for_tracking_ignores_stepped_timing_events() (+15 more)

### Community 24 - "FoldTrackingRecord"
Cohesion: 0.15
Nodes (21): Write the parent-run CSV, summary, diagnostic manifest, and lineage. The…, _diagnostic_roles(), final_metrics_for_tracking(), fold_timings_for_tracking(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray (+13 more)

### Community 25 - "test_training_tasks.py"
Cohesion: 0.11
Nodes (23): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), discover_datasets(), Runtime discovery of available datasets in datasets/processed_datasets/. (+15 more)

### Community 26 - "backfill"
Cohesion: 0.13
Nodes (22): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Stamp a tag onto MLflow runs recorded before that tag existed. A tag that only…, Every run in every experiment, including deleted ones, as (experiment name,…, Stamp ``tag=value`` on every run that lacks it, optionally marking it as… (+14 more)

### Community 27 - "TRIDENT README"
Cohesion: 0.21
Nodes (12): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, ADR 0002: Curated Cross-Validation MLflow Runs, Ticket 0002: MLflow Logged-Model Lifecycle, TRIDENT README, compute_cv_summary(), TabularEmbedder (+4 more)

### Community 28 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

### Community 29 - "Reduced Optuna Search for Imputation: Implementation Plan"
Cohesion: 0.22
Nodes (9): File structure, Global Constraints, Reduced Optuna Search for Imputation: Implementation Plan, Task 1: Search-space profiles, Task 2: The search objective on the validation split, Task 3: Task-keyed lookup and explicit promotion, Task 4: The importance artifact, Task 5: The launcher and the six studies (+1 more)

### Community 30 - "preprocess_table"
Cohesion: 0.21
Nodes (13): create_pretrain_datasets(), preprocess_table(), Example function that: 1) Splits train/val (e.g., 90/10) 2) Generates masked DF…, Preprocesses the table to replace null values with the `[NULL]` token, applies…, _frame(), _mask_positions(), DataFrame, The masking primitive, whose draw sequence is protected behaviour. `AGENTS.md`… (+5 more)

### Community 31 - "test_training_decoding.py"
Cohesion: 0.14
Nodes (29): evaluation_mask(), Hide cells for scoring, the same way every time this fold is scored. Training…, _complete_frame(), _dataset(), _fold(), _hyperparameters(), _pretrained(), DataFrame (+21 more)

### Community 32 - "test_opt_search_space.py"
Cohesion: 0.08
Nodes (30): build_parser(), define_search_space(), ObjectiveFunctionWrapper, ArgumentParser, Path, Wrapper class for the Optuna objective function to maintain state, Save the best parameters found so far, inside the study's own directory. The…, The training parser, so a study launched here accepts every flag main.py… (+22 more)

### Community 33 - "Ticket 0003: Training Loop Performance (Behavior-Preserving)"
Cohesion: 0.11
Nodes (14): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed (+6 more)

### Community 34 - "TridentDecoder"
Cohesion: 0.18
Nodes (11): dtype, DecodedCells, Tensor, Contextual output for every column token, dropping the [CLS] position., Reconstruction loss over the cells hidden from the model. ``hidden`` is the…, Stack per-column tensors feature-major, keeping the shape when there are none., Fill every cell of the batch, in the column's own vocabulary., What the decoder would fill every cell of a batch with. ``categorical_ids`` are… (+3 more)

### Community 35 - "ArtifactWriter"
Cohesion: 0.07
Nodes (41): ArtifactWriter, Any, DataFrame, Path, Filesystem artifacts emitted by a training runtime., Write prepared-dataset lineage for a parent training run., A readable sample of what the model filled in, and the full record behind it.…, Every fold's per-column errors in one long-form table on the parent run. Pooled… (+33 more)

### Community 36 - "run_hyperparameter_optimization"
Cohesion: 0.08
Nodes (44): _DisabledMlflow, info, log_param_importances(), Rank the knobs the study sampled and record the ranking on the study parent.…, Run hyperparameter optimization with Optuna, Drop-in no-op used when an Optuna invocation disables tracking., _Run, run_hyperparameter_optimization() (+36 more)

### Community 38 - "Imputation Decoder Task Implementation Plan"
Cohesion: 0.14
Nodes (14): File structure, Global Constraints, Imputation Decoder Task Implementation Plan, Task 10: Optuna, Task 11: Regression fixture, documentation, complete verification, Task 1: Per-task ranking contract, classification byte-identical, Task 2: Request, hyperparameters and command line, Task 3: Imputation metrics as pure functions (+6 more)

### Community 39 - "backfill_lr_scheduler_tag.py"
Cohesion: 0.17
Nodes (16): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, client() (+8 more)

### Community 40 - "test_decoder_model.py"
Cohesion: 0.24
Nodes (13): _decoder(), _frame(), DataFrame, The decoder that reconstructs actual cell values (ADR 0004, decisions 3 and 4)., Each numerical column gets its own scalar guess, in the scaled space it lives…, Silencing one column's head leaves every other column's imputation untouched., A mixed frame whose categorical column has missing cells, as a real variant…, The decoder cannot answer "missing" or "masked" when asked to fill a cell.… (+5 more)

### Community 41 - "embedder.py"
Cohesion: 0.18
Nodes (9): as_category_strings(), ndarray, Reduce a column name to a valid ModuleDict/ParameterDict key., Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, Stringify a categorical column with every kind of missing value collapsed to…, For each row, generates the resulting embedding: 1) Transforms each categorical…, _sanitize(), Separates real numerical values from special tokens `[MASK]` and `[NULL]`.… (+1 more)

### Community 42 - "test_embedder.py"
Cohesion: 0.40
Nodes (5): _embedder(), DataFrame, Behaviour of the shared TabularEmbedder, used by both tasks., ``None`` and ``NaN`` are the same absence, so they share one vocabulary entry.…, test_a_missing_category_is_one_category_however_it_was_written()

### Community 43 - "TridentPretrainer"
Cohesion: 0.22
Nodes (6): EncodedTable, A whole DataFrame converted to tensors once, then sliced per batch. The per-…, Select rows with a slice or an index tensor, keeping the layout., Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]…, masked / original : pd.DataFrame or EncodedTable The corrupted view and the…, TridentPretrainer

### Community 44 - "spec.md"
Cohesion: 0.18
Nodes (9): 01. Per-task fold-ranking contract, Acceptance criteria, Comments, Goal, Seams under test, Global constraints (apply to every ticket), Imputation decoder task: implementation tickets, Status vocabulary (+1 more)

### Community 45 - "run_training"
Cohesion: 0.10
Nodes (35): Run all folds and return raw folds plus the legacy aggregate metrics., run_training(), DatasetSpec, RuntimeOptions, TrainingRequest, TrainingResult, integration, Opt-in regression baseline for a short credit-g imputation run. `credit-… (+27 more)

### Community 46 - "TRIDENT Agent Guide (AGENTS.md)"
Cohesion: 0.25
Nodes (11): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern (+3 more)

### Community 48 - "02. Request, hyperparameters and command line"
Cohesion: 0.25
Nodes (7): 02. Request, hyperparameters and command line, Acceptance criteria, Comments, Constraints, Files, Goal, Seams under test

### Community 49 - "Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly"
Cohesion: 0.29
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly, Status

### Community 50 - "Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow, Consequences, Considered options, Context, Decision, How to compare, Status

### Community 51 - "03. Imputation metrics as pure functions"
Cohesion: 0.29
Nodes (6): 03. Imputation metrics as pure functions, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 52 - "04. The decoder model"
Cohesion: 0.29
Nodes (6): 04. The decoder model, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 53 - "05. Data support for evaluation"
Cohesion: 0.29
Nodes (6): 05. Data support for evaluation, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 54 - "06. The decode stage"
Cohesion: 0.29
Nodes (6): 06. The decode stage, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 55 - "07. Imputation artifacts"
Cohesion: 0.29
Nodes (6): 07. Imputation artifacts, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 56 - "08. Runner, MLflow identity, batch summary"
Cohesion: 0.29
Nodes (6): 08. Runner, MLflow identity, batch summary, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 57 - "09. Tag backfill"
Cohesion: 0.29
Nodes (6): 09. Tag backfill, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 58 - "10. Optuna for the imputation task"
Cohesion: 0.29
Nodes (6): 10. Optuna for the imputation task, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 59 - "11. Regression fixture and documentation"
Cohesion: 0.29
Nodes (6): 11. Regression fixture and documentation, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 60 - "Map: Imputation decoder task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Imputation decoder task, Not yet specified, Notes, Out of scope

### Community 61 - "08. Record the design: ADR 0005 and the implementation plan"
Cohesion: 0.50
Nodes (4): 08. Record the design: ADR 0005 and the implementation plan, Answer, Comments, Question

### Community 62 - "Ticket 0004: Exact original units in the imputation preview"
Cohesion: 0.29
Nodes (7): Acceptance, Constraints, Decisions, Explicitly not in this ticket, Outcome, The reframing, Ticket 0004: Exact original units in the imputation preview

### Community 63 - "01. Research: how do imputation papers score mixed-type tabular imputation?"
Cohesion: 0.40
Nodes (4): 01. Research: how do imputation papers score mixed-type tabular imputation?, Answer, Comments, Question

### Community 64 - "02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?"
Cohesion: 0.40
Nodes (4): 02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?, Answer, Comments, Question

### Community 65 - "10. Record the design: ADR and implementation plan"
Cohesion: 0.40
Nodes (4): 10. Record the design: ADR and implementation plan, Answer, Comments, Question

### Community 66 - "03. What is the imputation ground truth, and on which cells is error scored?"
Cohesion: 0.50
Nodes (4): 03. What is the imputation ground truth, and on which cells is error scored?, Answer, Comments, Question

### Community 67 - "04. Decoder architecture and the stage that trains it"
Cohesion: 0.50
Nodes (4): 04. Decoder architecture and the stage that trains it, Answer, Comments, Question

### Community 68 - "05. Loss, error metrics and the imputation fold-ranking metric"
Cohesion: 0.50
Nodes (4): 05. Loss, error metrics and the imputation fold-ranking metric, Answer, Comments, Question

### Community 69 - "06. How are imputation runs identified in MLflow?"
Cohesion: 0.50
Nodes (4): 06. How are imputation runs identified in MLflow?, Answer, Comments, Question

### Community 70 - "07. Prototype: the human-friendly imputed-vs-actual preview"
Cohesion: 0.50
Nodes (4): 07. Prototype: the human-friendly imputed-vs-actual preview, Answer, Comments, Question

### Community 71 - "08. CLI flag and hyperparameter schema for the imputation task"
Cohesion: 0.50
Nodes (4): 08. CLI flag and hyperparameter schema for the imputation task, Answer, Comments, Question

### Community 72 - "09. How does the training package branch per task without touching classification?"
Cohesion: 0.50
Nodes (4): 09. How does the training package branch per task without touching classification?, Answer, Comments, Question

### Community 73 - "11. Multi-rate evaluation: across runs or within one run?"
Cohesion: 0.50
Nodes (4): 11. Multi-rate evaluation: across runs or within one run?, Answer, Comments, Question

### Community 74 - "12. Should the [NULL] path be scored as a diagnostic?"
Cohesion: 0.50
Nodes (4): 12. Should the [NULL] path be scored as a diagnostic?, Answer, Comments, Question

### Community 75 - "13. Optuna for the imputation task"
Cohesion: 0.50
Nodes (4): 13. Optuna for the imputation task, Answer, Comments, Question

### Community 76 - "14. A regression fixture for the imputation task"
Cohesion: 0.50
Nodes (4): 14. A regression fixture for the imputation task, Answer, Comments, Question

### Community 78 - "01. Search-space profiles"
Cohesion: 0.29
Nodes (6): 01. Search-space profiles, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 79 - "Curated MLflow Cross-Validation Design Spec"
Cohesion: 0.80
Nodes (5): Curated MLflow Cross-Validation Implementation Plan, CrossValidationSummary, MlflowTracker Adapter, summarize_cross_validation(), Curated MLflow Cross-Validation Design Spec

### Community 80 - "CLAUDE.md"
Cohesion: 0.50
Nodes (3): Background & full reference, graphify, MLflow run analysis

### Community 81 - "02. The search objective on the validation split"
Cohesion: 0.29
Nodes (6): 02. The search objective on the validation split, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 82 - "04. The importance artifact"
Cohesion: 0.29
Nodes (6): 04. The importance artifact, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 83 - "05. The launcher and the six studies"
Cohesion: 0.29
Nodes (6): 05. The launcher and the six studies, Acceptance criteria, Comments, Constraints, Goal, What the script runs

### Community 84 - "06. The comparison and the documentation"
Cohesion: 0.29
Nodes (6): 06. The comparison and the documentation, Acceptance criteria, Comments, Constraints, Goal, What runs

### Community 85 - "run_from_namespace"
Cohesion: 0.33
Nodes (6): Namespace, Run a legacy namespace and retain the flat Optuna return contract., run_from_namespace(), main(), Legacy public facade for TRIDENT training., Retain the historical ``train.main`` programmatic entry point.

### Community 86 - "TridentModel"
Cohesion: 0.50
Nodes (3): Unified model for the classification task: 1) Generates tabular embeddings…, data : pd.DataFrame or EncodedTable Input rows (possibly masked/null, but in…, TridentModel

### Community 88 - "Map: Reduced Optuna search for the imputation task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Reduced Optuna search for the imputation task, Not yet specified, Notes, Out of scope

### Community 91 - "Reduced Optuna search for imputation: implementation tickets"
Cohesion: 0.50
Nodes (4): Global constraints (apply to every ticket), Reduced Optuna search for imputation: implementation tickets, Status vocabulary, Tickets

### Community 92 - "01. The held set and the ranges"
Cohesion: 0.50
Nodes (4): 01. The held set and the ranges, Answer, Comments, Question

### Community 93 - "02. The search objective is scored on the validation split"
Cohesion: 0.50
Nodes (4): 02. The search objective is scored on the validation split, Answer, Comments, Question

### Community 94 - "03. Study protocol: datasets, variants, budget, schedule, seed"
Cohesion: 0.50
Nodes (4): 03. Study protocol: datasets, variants, budget, schedule, seed, Answer, Comments, Question

### Community 95 - "04. Where a promoted configuration lives and how a run finds it"
Cohesion: 0.50
Nodes (4): 04. Where a promoted configuration lives and how a run finds it, Answer, Comments, Question

### Community 96 - "05. The search-space profile flag and its record"
Cohesion: 0.50
Nodes (4): 05. The search-space profile flag and its record, Answer, Comments, Question

### Community 97 - "06. Checking the reduction and proving the tuning helped"
Cohesion: 0.50
Nodes (4): 06. Checking the reduction and proving the tuning helped, Answer, Comments, Question

### Community 98 - "07. Task: repair the imputation Optuna path and prove it with a two-trial study"
Cohesion: 0.50
Nodes (4): 07. Task: repair the imputation Optuna path and prove it with a two-trial study, Answer, Comments, Question

### Community 99 - "types.py"
Cohesion: 0.15
Nodes (22): Protocol, Dataset preparation and fold construction for the training workflow., device, Classifier fine-tuning and fold metric calculation., Fine-tune and evaluate the classifier using the legacy optimization loop., train_and_evaluate_classifier(), device, Masked reconstruction pre-training stage. (+14 more)

### Community 100 - "Hyperparameters"
Cohesion: 0.18
Nodes (16): promote_best_configuration(), Publish a study's winning configuration where the task's runs will find it. The…, complete_configuration(), hyperparameter_file(), _load_base_hyperparameters(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests. (+8 more)

### Community 101 - "Research: third-party stub strategy for sklearn and scipy"
Cohesion: 0.05
Nodes (35): 05. Third-party stubs for sklearn and scipy, Notes, Question, Resolution, The three candidate answers, What to find out, 10. Out of scope, but found, 11. What could not be verified (+27 more)

### Community 103 - "CrossValidationSummary"
Cohesion: 0.22
Nodes (9): DisabledTracker, A tracker whose methods deliberately avoid importing MLflow side effects., CrossValidationSummary, LoggedArtifact, _dataset(), _record(), _summary(), test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations() (+1 more)

### Community 104 - "06. The hyperparameter JSON schema and the three naming layers"
Cohesion: 0.33
Nodes (5): 06. The hyperparameter JSON schema and the three naming layers, Notes, Question, Scope, already fixed, What to decide

### Community 105 - "07. Enforcing the constants convention beyond what the checker catches"
Cohesion: 0.33
Nodes (5): 07. Enforcing the constants convention beyond what the checker catches, A prior worth stating, Notes, Question, What to decide

### Community 106 - "09. Record the design: ADR 0006 and the implementation plan"
Cohesion: 0.33
Nodes (5): 09. Record the design: ADR 0006 and the implementation plan, Acceptance, Notes, Question, The work

## Ambiguous Edges - Review These
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `TRIDENT README` → `src/training Package`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **341 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+336 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 663 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `TRIDENT README` and `src/training Package`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Hyperparameters` connect `Hyperparameters` to `types.py`, `ArtifactWriter`, `decoding.py`, `test_training_config.py`, `run_training`, `test_training_decoding.py`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `PreparedDataset` connect `types.py` to `BufferedFoldTracker`, `ArtifactWriter`, `decoding.py`, `tracking.py`, `CrossValidationSummary`, `test_training_data.py`, `run_training`, `test_training_tracking.py`, `FoldTrackingRecord`, `test_training_tasks.py`, `test_training_decoding.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `TRIDENT README` connect `TRIDENT README` to `0005-reduced-optuna-search-for-imputation.md`, `0004-imputation-decoder-task.md`, `TRIDENT Agent Guide (AGENTS.md)`, `Curated MLflow Cross-Validation Design Spec`, `Training Refactor Implementation Plan`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `summarize_cross_validation()` (e.g. with `FoldTrackingRecord` and `TaskSpec`) actually correct?**
  _`summarize_cross_validation()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `_score_induced_missing()`) actually correct?**
  _`PreparedDataset` has 10 INFERRED edges - model-reasoned connections that need verification._