# Graph Report - typing-and-constants  (2026-09-11)

## Corpus Check
- 136 files · ~120,046 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1368 nodes · 2599 edges · 114 communities (101 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 126 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3fa848d5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- transformer.py
- FoldTrackingRecord
- test_training_config.py
- Answer
- ArtifactWriter
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- tracking.py
- TRIDENT Architecture
- Answer
- generate_splits.py
- build_fold_result
- Answer
- training/__init__.py
- trident
- TabularTransformerEncoder
- decoding.py
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- Training Refactor Implementation Plan
- test_training_tracking.py
- summarize_cross_validation
- summary.py
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
- test_imputation_artifacts.py
- run_hyperparameter_optimization
- Imputation Decoder Task Implementation Plan
- Resolution
- TabularEmbedder
- Answer
- 05. Third-party stubs for sklearn and scipy
- imputation-decoder/spec.md
- types.py
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
- cli.py
- test_training_artifacts.py
- Map: Reduced Optuna search for the imputation task
- Correctness and methodology
- Code health
- Reduced Optuna search for imputation: implementation tickets
- 01. The held set and the ranges
- 02. The search objective is scored on the validation split
- 03. Study protocol: datasets, variants, budget, schedule, seed
- 04. Where a promoted configuration lives and how a run finds it
- 05. The search-space profile flag and its record
- 06. Checking the reduction and proving the tuning helped
- 07. Task: repair the imputation Optuna path and prove it with a two-trial study
- finetuning.py
- Hyperparameters
- Research: third-party stub strategy for sklearn and scipy
- 03. Task-keyed lookup and explicit promotion
- PreparedDataset
- 06. The hyperparameter JSON schema and the three naming layers
- 07. Enforcing the constants convention beyond what the checker catches
- 09. Record the design: ADR 0006 and the implementation plan
- 08. Configure the checker and make it green
- Map: Typing and naming standard for TRIDENT
- Bugs
- main
- test_typing_gate.py

## God Nodes (most connected - your core abstractions)
1. `PreparedDataset` - 41 edges
2. `summarize_cross_validation()` - 39 edges
3. `FoldTrackingRecord` - 39 edges
4. `run_training()` - 37 edges
5. `run_hyperparameter_optimization()` - 31 edges
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

## Communities (114 total, 3 thin omitted)

### Community 0 - "transformer.py"
Cohesion: 0.18
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "FoldTrackingRecord"
Cohesion: 0.12
Nodes (7): fold_timings_for_tracking(), Return a fold's step-less stage timings keyed without the ``time/`` prefix., BufferedFoldTracker, FoldKey, Collect one fold's MLflow events until its diagnostic role is known., FoldTrackingRecord, FakeTracker

### Community 3 - "test_training_config.py"
Cohesion: 0.11
Nodes (33): build_training_parser(), ArgumentParser, Path, Post-parse validation for mutually exclusive and dependent flags., resolve_training_request(), validate_parsed_args(), Classification is what every run did before, so it is what a run does by…, There is no decoder under classification, so there would be nothing to score.… (+25 more)

### Community 4 - "Answer"
Cohesion: 0.22
Nodes (8): 02. The checker, the strictness ramp and the gate, Answer, Checker: mypy, Question, Scope: all of `src/` plus the entry points, Strictness: `strict = true` with per-module overrides, The baseline this rests on, The gate: a unit test that shells out to mypy

### Community 5 - "ArtifactWriter"
Cohesion: 0.11
Nodes (20): ArtifactWriter, Any, DataFrame, Path, StandardScaler, Filesystem artifacts emitted by a training runtime., Write the parent-run CSV, summary, diagnostic manifest, and lineage. The…, Write prepared-dataset lineage for a parent training run. (+12 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "tracking.py"
Cohesion: 0.16
Nodes (14): build_fold_tags(), build_run_tags(), parse_missingness_percent(), Build a flat dict of MLflow tags for a training run. Parameters ----------…, Build tags for a per-fold child run. Parameters ---------- fold_idx: Zero-based…, Return the normalized percentage encoded by a ``_<n>nan`` suffix., execution_tags(), _flag() (+6 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.08
Nodes (26): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+18 more)

### Community 9 - "Answer"
Cohesion: 0.25
Nodes (8): 01. The Python floor moves to 3.11, Acceptance for the bump, Answer, Question, What 3.11 buys, concretely, What was verified, and what was not, Why not further, Why the floor mattered more than it looks

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 12 - "Answer"
Cohesion: 0.25
Nodes (8): 03. Five StrEnums, and where constants live, Answer, Decision: homes split keys from values, Decision: `StrEnum`, with the tuples derived, Domain-modeling note: no CONTEXT.md changes, Implementation cautions for the execution tickets, Question, What the diagnosis actually was

### Community 17 - "TabularTransformerEncoder"
Cohesion: 0.40
Nodes (4): Complete transformer encoder for tabular data, TabularTransformerEncoder, Loss is the mean categorical surprise plus the weighted mean numerical error.…, test_each_kind_of_cell_is_averaged_over_its_own_count()

### Community 18 - "decoding.py"
Cohesion: 0.06
Nodes (53): _already_missing(), _clean(), _exact(), DataFrame, device, Tensor, Decode stage: train a decoder to reconstruct hidden cells, then score what it…, Score the cells the dataset is actually missing, against the complete sibling.… (+45 more)

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.11
Nodes (21): Optimizer, parametrize, Linear warmup to the base rate, then cosine decay to zero, measured in steps.…, One training stage's learning-rate schedule behind a uniform hook interface., StageScheduler, validate_lr_scheduler_name(), WarmupCosineMultiplier, _optimizer() (+13 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.22
Nodes (9): Fixed already, I1. Hyperparameters cannot be set from the CLI, I2. Optuna overwrites the dataset's hyperparameter file, Improvements, Open housekeeping, P1. `preprocess_table` falls back to a Python loop per row, P2. Hand-rolled attention instead of fused SDPA — protected, Performance still on the table (+1 more)

### Community 21 - "Training Refactor Implementation Plan"
Cohesion: 0.22
Nodes (12): Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, FoldResult, PreparedDataset, resolve_training_request(), run_from_namespace() (+4 more)

### Community 22 - "test_training_tracking.py"
Cohesion: 0.09
Nodes (50): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, build_hyperparams_dict() (+42 more)

### Community 23 - "summarize_cross_validation"
Cohesion: 0.23
Nodes (22): Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics.…, stage_timing_metrics(), summarize_cross_validation(), LoggedMetric, _record(), test_fold_timings_for_tracking_ignores_stepped_timing_events(), test_stage_timing_metrics_sums_the_two_stages() (+14 more)

### Community 24 - "summary.py"
Cohesion: 0.19
Nodes (16): _diagnostic_roles(), _is_finite_number(), _loss_band(), _loss_events_by_key(), FoldKey, ndarray, Metric aggregation helpers., _student_t_interval() (+8 more)

### Community 25 - "test_training_tasks.py"
Cohesion: 0.21
Nodes (14): Return the ranking contract for a task name, rejecting unknown names., task_spec(), The per-task fold-ranking contract (ADR 0004, decisions 6 and 12)., A search ranks its trials by its own objective, which need not be the fold-…, One cross-validation fold of an imputation run, logging decode-stage losses., _record(), test_classification_ranks_folds_by_macro_f1_maximised(), test_each_task_declares_its_search_objective() (+6 more)

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
Cohesion: 0.11
Nodes (28): create_pretrain_datasets(), preprocess_table(), Example function that: 1) Splits train/val (e.g., 90/10) 2) Generates masked DF…, Preprocesses the table to replace null values with the `[NULL]` token, applies…, _decoder(), _frame(), DataFrame, The decoder that reconstructs actual cell values (ADR 0004, decisions 3 and 4). (+20 more)

### Community 31 - "test_training_decoding.py"
Cohesion: 0.11
Nodes (33): evaluation_mask(), Hide cells for scoring, the same way every time this fold is scored. Training…, _maskable(), DataFrame, Which cells a fold is scored on is fixed, so its numbers can be compared at…, Evaluation must not consume randomness training was going to use. Every seeded…, test_drawing_an_evaluation_mask_leaves_the_training_draws_alone(), test_scoring_the_same_fold_twice_asks_the_same_question() (+25 more)

### Community 32 - "test_opt_search_space.py"
Cohesion: 0.08
Nodes (30): build_parser(), define_search_space(), ObjectiveFunctionWrapper, ArgumentParser, Path, Wrapper class for the Optuna objective function to maintain state, Save the best parameters found so far, inside the study's own directory. The…, The training parser, so a study launched here accepts every flag main.py… (+22 more)

### Community 33 - "Ticket 0003: Training Loop Performance (Behavior-Preserving)"
Cohesion: 0.25
Nodes (8): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed

### Community 34 - "TridentDecoder"
Cohesion: 0.11
Nodes (17): dtype, EncodedTable, A whole DataFrame converted to tensors once, then sliced per batch. The per-…, Select rows with a slice or an index tensor, keeping the layout., DecodedCells, Tensor, Contextual output for every column token, dropping the [CLS] position., Reconstruction loss over the cells hidden from the model. ``hidden`` is the… (+9 more)

### Community 35 - "test_imputation_artifacts.py"
Cohesion: 0.13
Nodes (25): _identity_scaler(), DataFrame, StandardScaler, The files an imputation run leaves behind (ADR 0004, decision 8)., A row can have a dozen cells filled in, and one table that wide reads as noise.…, Two rows' worth of scored cells, one of each kind and each population., Fitted so a scaled value and its original unit are the same number. Keeps a…, Ticket 0004: `1.144e-09` reads as a real measurement when it means zero.… (+17 more)

### Community 36 - "run_hyperparameter_optimization"
Cohesion: 0.08
Nodes (45): _DisabledMlflow, info, log_param_importances(), Namespace, Rank the knobs the study sampled and record the ranking on the study parent.…, Run hyperparameter optimization with Optuna Annotated ahead of the rest of this…, Drop-in no-op used when an Optuna invocation disables tracking., _Run (+37 more)

### Community 38 - "Imputation Decoder Task Implementation Plan"
Cohesion: 0.14
Nodes (14): File structure, Global Constraints, Imputation Decoder Task Implementation Plan, Task 10: Optuna, Task 11: Regression fixture, documentation, complete verification, Task 1: Per-task ranking contract, classification byte-identical, Task 2: Request, hyperparameters and command line, Task 3: Imputation metrics as pure functions (+6 more)

### Community 39 - "Resolution"
Cohesion: 0.25
Nodes (8): Acceptance, Left for ticket 09, Resolution, The bump: verified, not assumed, The gate, The ramp, as measured, Two corrections to this ticket's own instructions, What the checker found on its first run

### Community 41 - "TabularEmbedder"
Cohesion: 0.11
Nodes (17): as_category_strings(), ndarray, Reduce a column name to a valid ModuleDict/ParameterDict key., Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, Stringify a categorical column with every kind of missing value collapsed to…, For each row, generates the resulting embedding: 1) Transforms each categorical…, Class that encapsulates the creation of embeddings for tabular data: -…, _sanitize() (+9 more)

### Community 42 - "Answer"
Cohesion: 0.29
Nodes (7): 04. The task-branch union and exhaustiveness, Answer, Question, Related, and deliberately not bundled, The fix, Why the `Protocol` is out of scope, Why this one error earns its own ticket

### Community 43 - "05. Third-party stubs for sklearn and scipy"
Cohesion: 0.29
Nodes (7): 05. Third-party stubs for sklearn and scipy, Notes, Question, Resolution, The three candidate answers, Three things ticket 08 must not rediscover the hard way, What to find out

### Community 44 - "imputation-decoder/spec.md"
Cohesion: 0.18
Nodes (9): 01. Per-task fold-ranking contract, Acceptance criteria, Comments, Goal, Seams under test, Global constraints (apply to every ticket), Imputation decoder task: implementation tickets, Status vocabulary (+1 more)

### Community 45 - "types.py"
Cohesion: 0.05
Nodes (75): logged_hyperparameters(), The parameters this run actually used, under their config-file names., _assert_row_aligned(), build_folds(), declared_column_types(), load_complete_sibling(), prepare_dataset(), DataFrame (+67 more)

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

### Community 85 - "cli.py"
Cohesion: 0.15
Nodes (17): _format_duration(), main(), Namespace, Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), parse_args() (+9 more)

### Community 86 - "test_training_artifacts.py"
Cohesion: 0.67
Nodes (6): LoggedArtifact, _dataset(), _record(), _summary(), test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations(), test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values()

### Community 88 - "Map: Reduced Optuna search for the imputation task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Reduced Optuna search for the imputation task, Not yet specified, Notes, Out of scope

### Community 89 - "Correctness and methodology"
Cohesion: 0.33
Nodes (6): C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used, C4. Pre-training re-initializes already-initialized layers — protected, C5. Numeric precision and scaling, Correctness and methodology

### Community 90 - "Code health"
Cohesion: 0.33
Nodes (6): Code health, H1. `create_pretrain_datasets` is dead code, H2. The padding-mask branch is unreachable, H3. `torch.save` pickles the whole model object, H4. `opt.py` rebinds the module-level `mlflow` name, H5. The decode stage logs its timing under `time/finetune_seconds`

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

### Community 99 - "finetuning.py"
Cohesion: 0.12
Nodes (20): Protocol, Unified model for the classification task: 1) Generates tabular embeddings…, data : pd.DataFrame or EncodedTable Input rows (possibly masked/null, but in…, TridentModel, device, Classifier fine-tuning and fold metric calculation., Fine-tune and evaluate the classifier using the legacy optimization loop., train_and_evaluate_classifier() (+12 more)

### Community 100 - "Hyperparameters"
Cohesion: 0.18
Nodes (16): promote_best_configuration(), Publish a study's winning configuration where the task's runs will find it. The…, complete_configuration(), hyperparameter_file(), _load_base_hyperparameters(), load_hyperparameters(), Namespace, Translation from legacy argparse namespaces to typed training requests. (+8 more)

### Community 101 - "Research: third-party stub strategy for sklearn and scipy"
Cohesion: 0.07
Nodes (29): 10. Out of scope, but found, 11. What could not be verified, 12. Sources, 1. Answer, 2.1 typeshed — no, for either library, 2.2 scipy — `scipy-stubs` is official and current, 2.3 sklearn — nothing credible exists, 2. Does a maintained stub package exist? (+21 more)

### Community 102 - "03. Task-keyed lookup and explicit promotion"
Cohesion: 0.33
Nodes (6): 03. Task-keyed lookup and explicit promotion, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 103 - "PreparedDataset"
Cohesion: 0.15
Nodes (9): DisabledTracker, MlflowTracker, OptunaTrialTracker, Log a lightweight trial record into the MLflow run the caller has active.…, A tracker whose methods deliberately avoid importing MLflow side effects., Log one comparison parent and only selected diagnostic fold runs., CrossValidationSummary, PreparedDataset (+1 more)

### Community 104 - "06. The hyperparameter JSON schema and the three naming layers"
Cohesion: 0.25
Nodes (7): 06. The hyperparameter JSON schema and the three naming layers, Answer, For the execution ticket, Notes, Question, Scope, already fixed, What to decide

### Community 105 - "07. Enforcing the constants convention beyond what the checker catches"
Cohesion: 0.22
Nodes (8): 07. Enforcing the constants convention beyond what the checker catches, A prior worth stating, Answer, Metric keys: same treatment where it is free, no new machinery, Notes, Question, The measurement this answer depends on, What to decide

### Community 106 - "09. Record the design: ADR 0006 and the implementation plan"
Cohesion: 0.33
Nodes (5): 09. Record the design: ADR 0006 and the implementation plan, Acceptance, Notes, Question, The work

### Community 107 - "08. Configure the checker and make it green"
Cohesion: 0.33
Nodes (6): 08. Configure the checker and make it green, Acceptance, Notes, Question, The work, Two corrections to the numbers above, made before execution

### Community 108 - "Map: Typing and naming standard for TRIDENT"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Typing and naming standard for TRIDENT, Not yet specified, Notes, Out of scope

### Community 109 - "Bugs"
Cohesion: 0.40
Nodes (5): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs

### Community 111 - "main"
Cohesion: 0.50
Nodes (4): main(), Namespace, OptunaMetrics, Retain the historical ``train.main`` programmatic entry point.

## Ambiguous Edges - Review These
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `TRIDENT README` → `src/training Package`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **352 isolated node(s):** `info`, `trident`, `graphify`, `MLflow run analysis`, `Background & full reference` (+347 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 683 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `TRIDENT README` and `src/training Package`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Hyperparameters` connect `Hyperparameters` to `test_training_config.py`, `finetuning.py`, `ArtifactWriter`, `types.py`, `decoding.py`, `test_training_decoding.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `PreparedDataset` connect `PreparedDataset` to `FoldTrackingRecord`, `finetuning.py`, `ArtifactWriter`, `tracking.py`, `types.py`, `decoding.py`, `test_training_artifacts.py`, `test_training_tracking.py`, `test_training_tasks.py`, `test_training_decoding.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `TRIDENT README` connect `TRIDENT README` to `0005-reduced-optuna-search-for-imputation.md`, `0004-imputation-decoder-task.md`, `TRIDENT Agent Guide (AGENTS.md)`, `Curated MLflow Cross-Validation Design Spec`, `Training Refactor Implementation Plan`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `PreparedDataset` (e.g. with `ArtifactWriter` and `_score_induced_missing()`) actually correct?**
  _`PreparedDataset` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `summarize_cross_validation()` (e.g. with `FoldTrackingRecord` and `TaskSpec`) actually correct?**
  _`summarize_cross_validation()` has 2 INFERRED edges - model-reasoned connections that need verification._