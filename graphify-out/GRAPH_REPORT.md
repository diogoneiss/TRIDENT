# Graph Report - TRIDENT  (2026-09-10)

## Corpus Check
- 112 files · ~86,100 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1108 nodes · 2232 edges · 82 communities (75 shown, 3 thin omitted)
- Extraction: 94% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 122 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9adbd1b0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EncoderLayer
- FoldTrackingRecord
- imputation-optuna-reduced/map.md
- test_training_config.py
- types.py
- decoding.py
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- test_training_tracking.py
- TRIDENT Architecture
- runtime_environment_tags
- generate_splits.py
- build_fold_result
- DatasetSpec
- __init__.py
- trident
- cli.py
- score_cells
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- Training Refactor Implementation Plan
- ArtifactWriter
- summarize_cross_validation
- tracking.py
- BufferedFoldTracker
- backfill
- TRIDENT README
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow
- opt.py
- preprocess_table
- test_training_decoding.py
- test_opt_search_space.py
- Hyperparameters
- _stack
- test_imputation_artifacts.py
- run_hyperparameter_optimization
- Imputation Decoder Task Implementation Plan
- setup_mlflow
- TabularEmbedder
- pretraining.py
- test_embedder.py
- test_training_tasks.py
- spec.md
- finetuning.py
- TRIDENT Agent Guide (AGENTS.md)
- TridentPretrainer
- 02. Request, hyperparameters and command line
- _DisabledMlflow
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
- test_backfill_lr_scheduler_tag.py
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
- test_training_artifacts.py
- Curated MLflow Cross-Validation Design Spec
- CLAUDE.md
- mlflow_backend

## God Nodes (most connected - your core abstractions)
1. `PreparedDataset` - 39 edges
2. `FoldTrackingRecord` - 39 edges
3. `summarize_cross_validation()` - 38 edges
4. `run_training()` - 35 edges
5. `FoldResult` - 29 edges
6. `Hyperparameters` - 27 edges
7. `ArtifactWriter` - 26 edges
8. `train_and_evaluate_decoder()` - 26 edges
9. `CrossValidationSummary` - 24 edges
10. `MlflowTracker` - 23 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
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

## Communities (82 total, 3 thin omitted)

### Community 0 - "EncoderLayer"
Cohesion: 0.17
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "FoldTrackingRecord"
Cohesion: 0.14
Nodes (12): DisabledTracker, _log_summary_metrics(), MlflowTracker, OptunaTrialTracker, Log a lightweight trial record into the MLflow run the caller has active.…, Log the final CV statistics and timings; shared by parents and trials., A tracker whose methods deliberately avoid importing MLflow side effects., Log one comparison parent and only selected diagnostic fold runs. (+4 more)

### Community 2 - "imputation-optuna-reduced/map.md"
Cohesion: 0.05
Nodes (34): 01. The held set and the ranges, Answer, Comments, Question, 02. The search objective is scored on the validation split, Answer, Comments, Question (+26 more)

### Community 3 - "test_training_config.py"
Cohesion: 0.14
Nodes (25): ArgumentParser, build_training_parser(), logged_hyperparameters(), The parameters this run actually used, under their config-file names. Only the…, resolve_training_request(), RuntimeOptions, Classification is what every run did before, so it is what a run does by…, There is no decoder under classification, so there would be nothing to score.… (+17 more)

### Community 4 - "types.py"
Cohesion: 0.11
Nodes (34): _per_column_scores(), Typed orchestration for TRIDENT training., Each population's per-column errors, against this fold's own naive baseline., Run all folds and return raw folds plus the legacy aggregate metrics., run_training(), DecodingOutcome, FinetuningOutcome, FoldResult (+26 more)

### Community 5 - "decoding.py"
Cohesion: 0.19
Nodes (18): _already_missing(), _clean(), _exact(), DataFrame, device, Tensor, Decode stage: train a decoder to reconstruct hidden cells, then score what it…, Score the cells the dataset is actually missing, against the complete sibling.… (+10 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "test_training_tracking.py"
Cohesion: 0.24
Nodes (25): create_tracker(), Return an MLflow-backed tracker only when tracking has been requested., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis(), _artifact_files(), _artifact_paths(), _buffered_record(), _dataset() (+17 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.08
Nodes (26): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+18 more)

### Community 9 - "runtime_environment_tags"
Cohesion: 0.36
Nodes (6): device, Hardware and library descriptors recorded as MLflow tags., Describe where a run trains so its timings are comparable across machines.…, runtime_environment_tags(), test_cpu_environment_tags_keep_every_key(), test_cuda_environment_tags_record_the_gpu()

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "DatasetSpec"
Cohesion: 0.12
Nodes (27): _assert_row_aligned(), build_folds(), load_complete_sibling(), prepare_dataset(), DataFrame, Path, Dataset preparation and fold construction for the training workflow., Refuse a sibling that describes different rows. The variants are generated… (+19 more)

### Community 17 - "cli.py"
Cohesion: 0.17
Nodes (15): _format_duration(), main(), Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), parse_args(), Namespace (+7 more)

### Community 18 - "score_cells"
Cohesion: 0.10
Nodes (32): _error_metrics(), ImputationScores, mean_mode_baselines(), DataFrame, _ratio(), Imputation error, and the score that ranks folds by it. Pure functions over a…, Model error as a share of the naive imputer's, guarding a flawless baseline. A…, What a naive imputer would fill each column's cells with. The mean of a… (+24 more)

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.16
Nodes (17): parametrize, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, WarmupCosineMultiplier, _optimizer(), Learning rate seen by every batch, grouped per epoch., The period is ``epochs`` batches, so with 2 batches/epoch it completes a full…, _run(), test_batches_per_epoch_rounds_up() (+9 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.05
Nodes (40): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used (+32 more)

### Community 21 - "Training Refactor Implementation Plan"
Cohesion: 0.22
Nodes (12): Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, FoldResult, PreparedDataset, resolve_training_request(), run_from_namespace() (+4 more)

### Community 22 - "ArtifactWriter"
Cohesion: 0.12
Nodes (19): ArtifactWriter, Any, DataFrame, Path, Filesystem artifacts emitted by a training runtime., Write the parent-run CSV, summary, diagnostic manifest, and lineage. The…, Write prepared-dataset lineage for a parent training run., A readable sample of what the model filled in, and the full record behind it.… (+11 more)

### Community 23 - "summarize_cross_validation"
Cohesion: 0.23
Nodes (22): Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics.…, stage_timing_metrics(), summarize_cross_validation(), LoggedMetric, _record(), test_fold_timings_for_tracking_ignores_stepped_timing_events(), test_stage_timing_metrics_sums_the_two_stages() (+14 more)

### Community 24 - "tracking.py"
Cohesion: 0.13
Nodes (22): build_fold_tags(), Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, _diagnostic_roles(), fold_timings_for_tracking(), _is_finite_number(), _loss_band(), _loss_events_by_key(), ndarray (+14 more)

### Community 25 - "BufferedFoldTracker"
Cohesion: 0.15
Nodes (3): BufferedFoldTracker, Collect one fold's MLflow events until its diagnostic role is known., FakeTracker

### Community 26 - "backfill"
Cohesion: 0.13
Nodes (22): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Stamp a tag onto MLflow runs recorded before that tag existed. A tag that only…, Every run in every experiment, including deleted ones, as (experiment name,…, Stamp ``tag=value`` on every run that lacks it, optionally marking it as… (+14 more)

### Community 27 - "TRIDENT README"
Cohesion: 0.21
Nodes (12): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, ADR 0002: Curated Cross-Validation MLflow Runs, Ticket 0002: MLflow Logged-Model Lifecycle, TRIDENT README, compute_cv_summary(), TabularEmbedder (+4 more)

### Community 28 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

### Community 29 - "opt.py"
Cohesion: 0.11
Nodes (18): build_hyperparams_dict(), build_run_tags(), get_or_create_experiment(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for…, Build a flat dict of all TRIDENT hyperparameters suitable for…, Build a flat dict of MLflow tags for a training run. Parameters ----------…, Return the normalized percentage encoded by a ``_<n>nan`` suffix. (+10 more)

### Community 30 - "preprocess_table"
Cohesion: 0.21
Nodes (13): create_pretrain_datasets(), preprocess_table(), Example function that: 1) Splits train/val (e.g., 90/10) 2) Generates masked DF…, Preprocesses the table to replace null values with the `[NULL]` token, applies…, _frame(), _mask_positions(), DataFrame, The masking primitive, whose draw sequence is protected behaviour. `AGENTS.md`… (+5 more)

### Community 31 - "test_training_decoding.py"
Cohesion: 0.12
Nodes (31): evaluation_mask(), Hide cells for scoring, the same way every time this fold is scored. Training…, _maskable(), DataFrame, Which cells a fold is scored on is fixed, so its numbers can be compared at…, Evaluation must not consume randomness training was going to use. Every seeded…, test_drawing_an_evaluation_mask_leaves_the_training_draws_alone(), test_scoring_the_same_fold_twice_asks_the_same_question() (+23 more)

### Community 32 - "test_opt_search_space.py"
Cohesion: 0.13
Nodes (18): define_search_space(), ObjectiveFunctionWrapper, Wrapper class for the Optuna objective function to maintain state, Save the best parameters found so far, somewhere the other task cannot feel.…, Define the hyperparameter search space for Optuna, Optuna's search space and failure handling, per task (ADR 0004, decision 13)., Both tasks read datasets/hiperparams/<base>/<dataset>.json, and it names no…, Draw a spread of trials so a constraint is tested against many combinations. (+10 more)

### Community 33 - "Hyperparameters"
Cohesion: 0.18
Nodes (14): _load_base_hyperparameters(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests., Post-parse validation for mutually exclusive and dependent flags., validate_parsed_args(), Hyperparameters, Any (+6 more)

### Community 34 - "_stack"
Cohesion: 0.20
Nodes (9): dtype, DecodedCells, Tensor, Contextual output for every column token, dropping the [CLS] position., Reconstruction loss over the cells hidden from the model. ``hidden`` is the…, Stack per-column tensors feature-major, keeping the shape when there are none., Fill every cell of the batch, in the column's own vocabulary., What the decoder would fill every cell of a batch with. ``categorical_ids`` are… (+1 more)

### Community 35 - "test_imputation_artifacts.py"
Cohesion: 0.13
Nodes (25): StandardScaler, _identity_scaler(), DataFrame, The files an imputation run leaves behind (ADR 0004, decision 8)., A row can have a dozen cells filled in, and one table that wide reads as noise.…, Two rows' worth of scored cells, one of each kind and each population., Fitted so a scaled value and its original unit are the same number. Keeps a…, Ticket 0004: `1.144e-09` reads as a real measurement when it means zero.… (+17 more)

### Community 36 - "run_hyperparameter_optimization"
Cohesion: 0.18
Nodes (22): Run hyperparameter optimization with Optuna, run_hyperparameter_optimization(), _optuna_args(), MlflowClient, Namespace, MLflow structure of an Optuna study: one study run, one light run per trial., The objective builds each trial's namespace itself, so the task has to be…, ``task`` and ``is_optuna`` are dense on every other run kind (ADR 0004,… (+14 more)

### Community 38 - "Imputation Decoder Task Implementation Plan"
Cohesion: 0.14
Nodes (14): File structure, Global Constraints, Imputation Decoder Task Implementation Plan, Task 10: Optuna, Task 11: Regression fixture, documentation, complete verification, Task 1: Per-task ranking contract, classification byte-identical, Task 2: Request, hyperparameters and command line, Task 3: Imputation metrics as pure functions (+6 more)

### Community 39 - "setup_mlflow"
Cohesion: 0.23
Nodes (9): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, Configure the MLflow tracking URI. Priority order: 1. ``tracking_uri`` argument… (+1 more)

### Community 40 - "TabularEmbedder"
Cohesion: 0.13
Nodes (24): Class that encapsulates the creation of embeddings for tabular data: -…, TabularEmbedder, Parameters ---------- embedder : TabularEmbedder Responsible for generating…, Reconstructs actual cell values from the encoder's output at each column. One…, TridentDecoder, Complete transformer encoder for tabular data, TabularTransformerEncoder, _decoder() (+16 more)

### Community 41 - "pretraining.py"
Cohesion: 0.16
Nodes (10): as_category_strings(), ndarray, Reduce a column name to a valid ModuleDict/ParameterDict key., Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, Stringify a categorical column with every kind of missing value collapsed to…, For each row, generates the resulting embedding: 1) Transforms each categorical…, _sanitize(), Masked reconstruction pre-training stage. (+2 more)

### Community 42 - "test_embedder.py"
Cohesion: 0.40
Nodes (5): _embedder(), DataFrame, Behaviour of the shared TabularEmbedder, used by both tasks., ``None`` and ``NaN`` are the same absence, so they share one vocabulary entry.…, test_a_missing_category_is_one_category_however_it_was_written()

### Community 43 - "test_training_tasks.py"
Cohesion: 0.39
Nodes (7): The per-task fold-ranking contract (ADR 0004, decisions 6 and 12)., One cross-validation fold of an imputation run, logging decode-stage losses., _record(), test_imputation_manifest_ranks_folds_under_the_impute_score_name(), test_imputation_summary_bands_the_decode_stage_losses_instead_of_finetuning(), test_imputation_summary_marks_the_lowest_score_as_the_best_fold(), test_imputation_summary_rejects_folds_missing_the_impute_score()

### Community 44 - "spec.md"
Cohesion: 0.18
Nodes (9): 01. Per-task fold-ranking contract, Acceptance criteria, Comments, Goal, Seams under test, Global constraints (apply to every ticket), Imputation decoder task: implementation tickets, Status vocabulary (+1 more)

### Community 45 - "finetuning.py"
Cohesion: 0.09
Nodes (23): Optimizer, Protocol, Unified model for the classification task: 1) Generates tabular embeddings…, data : pd.DataFrame or EncodedTable Input rows (possibly masked/null, but in…, TridentModel, device, Classifier fine-tuning and fold metric calculation., Fine-tune and evaluate the classifier using the legacy optimization loop. (+15 more)

### Community 46 - "TRIDENT Agent Guide (AGENTS.md)"
Cohesion: 0.25
Nodes (11): Task 3 Report: Runtime Adapters and Regression Baseline, TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, vehicle_00nan Regression Fixture, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern (+3 more)

### Community 47 - "TridentPretrainer"
Cohesion: 0.22
Nodes (6): EncodedTable, A whole DataFrame converted to tensors once, then sliced per batch. The per-…, Select rows with a slice or an index tensor, keeping the layout., Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]…, masked / original : pd.DataFrame or EncodedTable The corrupted view and the…, TridentPretrainer

### Community 48 - "02. Request, hyperparameters and command line"
Cohesion: 0.25
Nodes (7): 02. Request, hyperparameters and command line, Acceptance criteria, Comments, Constraints, Files, Goal, Seams under test

### Community 49 - "_DisabledMlflow"
Cohesion: 0.22
Nodes (4): _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run

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
Nodes (3): 08. Record the design: ADR 0005 and the implementation plan, Comments, Question

### Community 62 - "test_backfill_lr_scheduler_tag.py"
Cohesion: 0.42
Nodes (9): client(), fixture, MlflowClient, _seed_runs(), test_apply_is_idempotent(), test_apply_tags_only_untagged_runs_and_marks_them_backfilled(), test_dry_run_reports_without_writing(), test_main_dry_run_prints_summary() (+1 more)

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

### Community 78 - "test_training_artifacts.py"
Cohesion: 0.57
Nodes (7): LoggedArtifact, LossBand, _dataset(), _record(), _summary(), test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations(), test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values()

### Community 79 - "Curated MLflow Cross-Validation Design Spec"
Cohesion: 0.80
Nodes (5): Curated MLflow Cross-Validation Implementation Plan, CrossValidationSummary, MlflowTracker Adapter, summarize_cross_validation(), Curated MLflow Cross-Validation Design Spec

### Community 80 - "CLAUDE.md"
Cohesion: 0.50
Nodes (3): Background & full reference, graphify, MLflow run analysis

## Ambiguous Edges - Review These
- `src/training Package` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **222 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+217 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 510 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `src/training Package` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `setup_mlflow()` connect `setup_mlflow` to `run_hyperparameter_optimization`, `test_training_tracking.py`, `tracking.py`, `backfill`, `opt.py`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `run_hyperparameter_optimization()` connect `run_hyperparameter_optimization` to `test_opt_search_space.py`, `setup_mlflow`, `cli.py`, `_DisabledMlflow`, `opt.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `PreparedDataset` connect `FoldTrackingRecord` to `types.py`, `decoding.py`, `test_training_tracking.py`, `pretraining.py`, `test_training_tasks.py`, `DatasetSpec`, `finetuning.py`, `test_training_artifacts.py`, `ArtifactWriter`, `tracking.py`, `BufferedFoldTracker`, `test_training_decoding.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `_score_induced_missing()`) actually correct?**
  _`PreparedDataset` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `FoldTrackingRecord` (e.g. with `ArtifactWriter` and `_diagnostic_roles()`) actually correct?**
  _`FoldTrackingRecord` has 15 INFERRED edges - model-reasoned connections that need verification._