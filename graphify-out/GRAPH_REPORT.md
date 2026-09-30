# Graph Report - TRIDENT  (2026-09-30)

## Corpus Check
- 197 files · ~295,796 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2348 nodes · 4433 edges · 167 communities (153 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 174 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `76091c03`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EncoderLayer
- test_opt_search_space.py
- test_training_config.py
- Answer
- test_mirroring.py
- TRIDENT Experiment Tracking Context (CONTEXT.md)
- 3. Design and methodology critiques judged sound
- TRIDENT Architecture
- Answer
- generate_splits.py
- StoreError
- Answer
- training/__init__.py
- trident
- decoding.py
- DatasetSpec
- score_cells
- test_training_schedulers.py
- TRIDENT Backlog: Bugs, Pendencies and Improvements
- test_imputation_baselines.py
- test_training_tracking.py
- summarize_cross_validation
- test_training_tasks.py
- ArtifactWriter
- backfill
- Findings
- Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow
- Reduced Optuna Search for Imputation: Implementation Plan
- TabularEmbedder
- test_training_decoding.py
- setup_mlflow
- Ticket 0003: Training Loop Performance (Behavior-Preserving)
- BufferedFoldTracker
- FoldTrackingRecord
- run_hyperparameter_optimization
- Imputation Decoder Task Implementation Plan
- Resolution
- _stack
- Answer
- 05. Third-party stubs for sklearn and scipy
- imputation-decoder/spec.md
- runtime_environment_tags
- Findings
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
- TRIDENT Agent Guide (AGENTS.md)
- Map: Imputation decoder task
- 08. Record the design: ADR 0005 and the implementation plan
- BACKLOG.md
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
- Findings
- Verdicts
- 02. The search objective on the validation split
- 04. The importance artifact
- 05. The launcher and the six studies
- 06. The comparison and the documentation
- Verdicts
- Verdicts
- Map: Reduced Optuna search for the imputation task
- test_imputation_artifacts.py
- as_category_strings
- Reduced Optuna search for imputation: implementation tickets
- sync_remote.py
- 02. The search objective is scored on the validation split
- 03. Study protocol: datasets, variants, budget, schedule, seed
- 04. Where a promoted configuration lives and how a run finds it
- 05. The search-space profile flag and its record
- 06. Checking the reduction and proving the tuning helped
- test_sync_remote.py
- types.py
- config.py
- Research: third-party stub strategy for sklearn and scipy
- 0006-mirror-experiments-per-task.md
- CrossValidationSummary
- 06. The hyperparameter JSON schema and the three naming layers
- 07. Enforcing the constants convention beyond what the checker catches
- 09. Record the design: ADR 0006 and the implementation plan
- TRIDENT README
- Map: Typing and naming standard for TRIDENT
- Addendum — the Optuna results, 2026-09-15
- 2. Confirmed defects
- Findings
- test_typing_gate.py
- Findings
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Findings
- Findings
- Findings
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Verdicts
- Findings
- Verdicts
- Verdicts
- Verdicts
- backfill_baseline_imputers.py
- Fidelity audit of `CONSOLIDATED.md`
- test_backfill_baseline_imputers.py
- §8 — Experiments run 2026-09-17
- Training Refactor Implementation Plan
- Consolidated report — the imputation work on `feat/imputation-task`
- 4. What the experiments can and cannot support
- _step_frame
- CLAUDE.md
- preprocess_table
- Entries
- RsyncTransport
- build_fold_result
- DirectoryServer
- ObjectiveFunctionWrapper
- Transport
- test_backfill_search_objective_tag.py
- Hand the tracking store to a remote server whole, one writer at a time
- tracking.py
- fit_baseline_imputers
- Let an imputation study rank its trials by the validation split's induced gaps
- cli.py
- Score baseline imputers on exactly the cells the decoder is scored on
- test_backfill_lr_scheduler_tag.py
- .encode
- 8. The configuration for ticket 08
- _columns_holding
- _Spy
- TrainingTracker
- backfill_search_objective_tag.py
- 01. The held set and the ranges
- 07. Task: repair the imputation Optuna path and prove it with a two-trial study
- start_mlflow.sh script

## God Nodes (most connected - your core abstractions)
1. `summarize_cross_validation()` - 50 edges
2. `PreparedDataset` - 45 edges
3. `run_hyperparameter_optimization()` - 41 edges
4. `FoldTrackingRecord` - 40 edges
5. `run_training()` - 38 edges
6. `FoldResult` - 31 edges
7. `Hyperparameters` - 30 edges
8. `train_and_evaluate_decoder()` - 28 edges
9. `StoreError` - 27 edges
10. `push()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `compute_cv_summary()` --semantically_similar_to--> `CrossValidationSummary`  [INFERRED] [semantically similar]
  README.md → docs/superpowers/plans/2026-08-03-curated-mlflow-cross-validation.md
- `TRIDENT README` --conceptually_related_to--> `kr-vs-kp Categorical Columns`  [AMBIGUOUS]
  README.md → datasets/categorical_columns/kr-vs-kp.txt
- `TRIDENT README` --conceptually_related_to--> `src/training Package`  [AMBIGUOUS]
  README.md → AGENTS.md
- `promote_best_configuration()` --uses--> `Hyperparameters`  [INFERRED]
  opt.py → src/training/types.py
- `run_hyperparameter_optimization()` --uses--> `DatasetSpec`  [INFERRED]
  opt.py → src/training/types.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Curated MLflow Cross-Validation Documentation Trail** — docs_adr_0002_curated_cross_validation_mlflow_runs, docs_superpowers_specs_2026_08_03_curated_mlflow_cross_validation_design, docs_superpowers_plans_2026_08_03_curated_mlflow_cross_validation, docs_tickets_0002_mlflow_logged_model_lifecycle [INFERRED 0.85]
- **Training Refactor Documentation Trail** — docs_tickets_0001_training_refactor, docs_superpowers_specs_2026_08_02_training_refactor_design, docs_superpowers_plans_2026_08_02_training_refactor, docs_adr_0001_training_package_with_compatibility_facade, _superpowers_sdd_2026_08_02_training_refactor_task_3_report [INFERRED 0.85]
- **TRIDENT Model Architecture Pipeline** — readme_tabularembedder, readme_tabulartransformerencoder, readme_tridentpretrainer, readme_tridentclassifier [INFERRED 0.85]

## Communities (167 total, 5 thin omitted)

### Community 0 - "EncoderLayer"
Cohesion: 0.17
Nodes (8): EncoderLayer, FeedForwardNetwork, initialize_weight(), MultiHeadAttention, Simple feed-forward network with two linear layers and ReLU activation, Multi-head attention layer for transformers, Initializes the weight of a layer using Xavier uniform for weights and zeros…, Transformer encoder layer with self-attention and feed-forward

### Community 1 - "test_opt_search_space.py"
Cohesion: 0.12
Nodes (22): build_parser(), define_search_space(), ArgumentParser, The training parser, so a study launched here accepts every flag main.py…, Define the hyperparameter search space for Optuna. ``profile`` is the search-…, Optuna's search space and failure handling, per task (ADR 0004, decision 13)., A trial that could lower the evaluation mask rate would win by hiding less., Attention splits the model width across heads, so the width must divide evenly.… (+14 more)

### Community 3 - "test_training_config.py"
Cohesion: 0.10
Nodes (36): build_training_parser(), ArgumentParser, Post-parse validation for mutually exclusive and dependent flags., resolve_training_request(), validate_parsed_args(), ADR 0006, decision 4: default on by the user's explicit decision;…, Classification is what every run did before, so it is what a run does by…, There is no decoder under classification, so there would be nothing to score.… (+28 more)

### Community 4 - "Answer"
Cohesion: 0.22
Nodes (8): 02. The checker, the strictness ramp and the gate, Answer, Checker: mypy, Question, Scope: all of `src/` plus the entry points, Strictness: `strict = true` with per-module overrides, The baseline this rests on, The gate: a unit test that shells out to mypy

### Community 5 - "test_mirroring.py"
Cohesion: 0.05
Nodes (70): _Entry, _DisabledMlflow, info, Drop-in no-op used when an Optuna invocation disables tracking., _Run, RunTag, describe(), main() (+62 more)

### Community 6 - "TRIDENT Experiment Tracking Context (CONTEXT.md)"
Cohesion: 0.26
Nodes (14): TRIDENT Experiment Tracking Context (CONTEXT.md), Component-Ablation Benchmark, CV Summary, Dataset Provenance, Diagnostic Fold Run, Experiment Family, Fold-Ranking Metric, Full-Factorial Ablation (+6 more)

### Community 7 - "3. Design and methodology critiques judged sound"
Cohesion: 0.17
Nodes (12): 3.10 The documentation contract, 3.11 Residual leaks and units, 3.1 The evaluation mask is the training corruption helper, and one ADR sentence rests on it, 3.2 The two scored populations are not a pair, and the induced view is out of distribution, 3.3 The search cannot identify a winner, and the winner it publishes is selection-biased, 3.4 Checkpoint selection is a different objective from the one that ranks the fold, 3.5 What `impute_score` normalises, and what the record keeps of it, 3.6 Pre-training is not measurably contributing, and nothing in the record could say so (+4 more)

### Community 8 - "TRIDENT Architecture"
Cohesion: 0.08
Nodes (26): Building folds — `build_folds` (`src/training/data.py:69`), Contents, Data engineering pipeline, End-to-end training orchestration, Hyperparameter → code wiring reference, Implementation, Implementation, Implementation (+18 more)

### Community 9 - "Answer"
Cohesion: 0.25
Nodes (8): 01. The Python floor moves to 3.11, Acceptance for the bump, Answer, Question, What 3.11 buys, concretely, What was verified, and what was not, Why not further, Why the floor mattered more than it looks

### Community 10 - "generate_splits.py"
Cohesion: 0.23
Nodes (12): _create_split_dict(), inject_nans(), _json_key(), main(), _non_stratified_split(), Any, DataFrame, Perform a non-stratified split when stratification is not possible. (+4 more)

### Community 11 - "StoreError"
Cohesion: 0.12
Nodes (45): adopt(), copy_store(), default_master(), find_rsync(), journal_mode(), main(), open_store(), Connection (+37 more)

### Community 12 - "Answer"
Cohesion: 0.25
Nodes (8): 03. Five StrEnums, and where constants live, Answer, Decision: homes split keys from values, Decision: `StrEnum`, with the tuples derived, Domain-modeling note: no CONTEXT.md changes, Implementation cautions for the execution tickets, Question, What the diagnosis actually was

### Community 15 - "decoding.py"
Cohesion: 0.18
Nodes (19): _already_missing(), _clean(), _exact(), DataFrame, device, Tensor, Decode stage: train a decoder to reconstruct hidden cells, then score what it…, Score the cells the dataset is actually missing, against the complete sibling.… (+11 more)

### Community 17 - "DatasetSpec"
Cohesion: 0.09
Nodes (41): Every current baseline's metrics on each fold's populations, as the run scored…, recompute_folds(), _assert_row_aligned(), build_folds(), declared_column_types(), evaluation_mask(), has_complete_sibling(), in_variant_dtypes() (+33 more)

### Community 18 - "score_cells"
Cohesion: 0.10
Nodes (31): Baseline imputers, scored on exactly the cells the decoder is scored on. Four…, _error_metrics(), ImputationScores, DataFrame, _ratio(), Imputation error, and the score that ranks folds by it. Pure functions over a…, Model error as a share of the naive imputer's, guarding a flawless baseline. A…, Everything one population of scored cells says about a fold. ``metrics`` are… (+23 more)

### Community 19 - "test_training_schedulers.py"
Cohesion: 0.09
Nodes (27): Optimizer, parametrize, device, Fine-tune and evaluate the classifier using the legacy optimization loop., train_and_evaluate_classifier(), batches_per_epoch(), Selectable learning-rate schedules shared by both training stages. Every…, Number of optimizer steps one epoch performs over ``row_count`` rows. (+19 more)

### Community 20 - "TRIDENT Backlog: Bugs, Pendencies and Improvements"
Cohesion: 0.08
Nodes (26): B1. The learning-rate schedule completes a full cosine cycle — protected, B2. `--cv_folds 1` crashes, B3. Optuna proposes head counts that crash, and scores them as 0.0, B4. Optuna with MLflow enabled scored every trial 0.0, Bugs, C1. Scaler and encoders are fit before splitting — protected, C2. `"nan"` is a real category next to `[NULL]` — protected, C3. `LABELS` is logged but never used (+18 more)

### Community 21 - "test_imputation_baselines.py"
Cohesion: 0.11
Nodes (45): KnnImputer, The stronger baseline: each hidden cell filled from its nearest training rows.…, Each baseline imputer's error on the scored cells, keyed…, score_baselines(), mean_mode_baselines(), What a naive imputer would fill each column's cells with. The mean of a…, _cells(), _dependent_frame() (+37 more)

### Community 22 - "test_training_tracking.py"
Cohesion: 0.15
Nodes (33): create_tracker(), MlflowTracker, OptunaTrialTracker, Log a lightweight trial record into the MLflow run the caller has active.…, Return an MLflow-backed tracker only when tracking has been requested., Log one comparison parent and only selected diagnostic fold runs., Disabled tracking must keep every MLflow API completely untouched., test_disabled_tracker_never_calls_mlflow_apis() (+25 more)

### Community 23 - "summarize_cross_validation"
Cohesion: 0.18
Nodes (27): Build the per-fold timing events from the two measured stage durations., Aggregate completed CV folds into comparable final and loss statistics.…, stage_timing_metrics(), summarize_cross_validation(), LoggedMetric, A run may skip pre-training (zero epochs) to measure what it contributes. No…, Folds that disagree about whether a stage ran are not comparable, as before., _record() (+19 more)

### Community 24 - "test_training_tasks.py"
Cohesion: 0.15
Nodes (20): Return the ranking contract for a task name, rejecting unknown names., task_spec(), The per-task fold-ranking contract (ADR 0004, decisions 6 and 12)., One metric to read beside the model's: per population, the baseline whose mean…, How far the model is from the bar, as a number to sort and minimise: per fold,…, A search ranks its trials by its own objective, which need not be the fold-…, Nothing about baselines appears in a task that has none., One cross-validation fold of an imputation run, logging decode-stage losses. (+12 more)

### Community 25 - "ArtifactWriter"
Cohesion: 0.13
Nodes (16): ArtifactWriter, Any, DataFrame, Path, StandardScaler, Write prepared-dataset lineage for a parent training run., A readable sample of what the model filled in, and the full record behind it.…, Every fold's per-column errors in one long-form table on the parent run. Pooled… (+8 more)

### Community 26 - "backfill"
Cohesion: 0.13
Nodes (22): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Stamp a tag onto MLflow runs recorded before that tag existed. A tag that only…, Every run in every experiment, including deleted ones, as (experiment name,…, Stamp ``tag=value`` on every run that lacks it, optionally marking it as… (+14 more)

### Community 27 - "Findings"
Cohesion: 0.13
Nodes (14): 07 - The Optuna search for imputation, Checked and cleared, F-07-10 - The study's running-best file uses classification's promoted filename and holds only the sampled subset, F-07-1 - The search objective cannot resolve the configurations it is ranking; two of the four promoted files were decided by a tie-break, not by the objective, F-07-2 - The objective is measured on exactly the cells that chose the checkpoint, so `optuna/best_objective_value` is a minimum-of-minima, not a validation score, F-07-3 - There is no way to run the defaults arm of ADR 0005's own comparison protocol once a configuration has been promoted, F-07-4 - An imputation run with no promoted file of its own silently loads classification's tuned configuration, F-07-5 - fANOVA silently drops `HEAD_DIM` from the importance ranking, so the check on the reduction cannot see the model width (+6 more)

### Community 28 - "Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow"
Cohesion: 0.29
Nodes (7): Consequences, Considered options, Context, Decision, How to compare, Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow, Status

### Community 29 - "Reduced Optuna Search for Imputation: Implementation Plan"
Cohesion: 0.22
Nodes (9): File structure, Global Constraints, Reduced Optuna Search for Imputation: Implementation Plan, Task 1: Search-space profiles, Task 2: The search objective on the validation split, Task 3: Task-keyed lookup and explicit promotion, Task 4: The importance artifact, Task 5: The launcher and the six studies (+1 more)

### Community 30 - "TabularEmbedder"
Cohesion: 0.07
Nodes (34): EncodedTable, Reduce a column name to a valid ModuleDict/ParameterDict key., A whole DataFrame converted to tensors once, then sliced per batch. The per-…, Select rows with a slice or an index tensor, keeping the layout., Class that encapsulates the creation of embeddings for tabular data: -…, _sanitize(), TabularEmbedder, Given a masked DataFrame, asks the Transformer to reconstruct, only at [MASK]… (+26 more)

### Community 31 - "test_training_decoding.py"
Cohesion: 0.13
Nodes (35): _complete_frame(), _dataset(), _fold(), _hyperparameters(), DataFrame, The decode stage: training a decoder and scoring what it reconstructs., Its loss curves are what a reader plots, so every epoch has to be on them., A search ranks trials on the validation split, so the test split never chooses… (+27 more)

### Community 32 - "setup_mlflow"
Cohesion: 0.23
Nodes (9): backfill(), BackfillReport, iter_all_runs(), main(), MlflowClient, Backfill the ``lr_scheduler`` tag on MLflow runs recorded before ADR 0003.…, Every run in every experiment, including deleted ones, as (experiment name,…, Configure the MLflow tracking URI. Priority order: 1. ``tracking_uri`` argument… (+1 more)

### Community 33 - "Ticket 0003: Training Loop Performance (Behavior-Preserving)"
Cohesion: 0.25
Nodes (8): Exactness standard and result, Fixes included, Found and deliberately NOT fixed, Measured speedup, Problem, Status, Ticket 0003: Training Loop Performance (Behavior-Preserving), What changed

### Community 34 - "BufferedFoldTracker"
Cohesion: 0.24
Nodes (8): BufferedFoldTracker, Collect one fold's MLflow events until its diagnostic role is known., LoggedArtifact, _dataset(), _record(), _summary(), test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations(), test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values()

### Community 35 - "FoldTrackingRecord"
Cohesion: 0.11
Nodes (25): Write the parent-run CSV, summary, diagnostic manifest, and lineage. The…, _best_baseline_copies(), _best_baselines(), _diagnostic_roles(), final_metrics_for_tracking(), fold_timings_for_tracking(), _is_finite_number(), _loss_band() (+17 more)

### Community 36 - "run_hyperparameter_optimization"
Cohesion: 0.10
Nodes (52): Namespace, Run hyperparameter optimization with Optuna Annotated ahead of the rest of this…, run_hyperparameter_optimization(), RuntimeError, mlflow_backend(), _optuna_args(), fixture, MlflowClient (+44 more)

### Community 38 - "Imputation Decoder Task Implementation Plan"
Cohesion: 0.14
Nodes (14): File structure, Global Constraints, Imputation Decoder Task Implementation Plan, Task 10: Optuna, Task 11: Regression fixture, documentation, complete verification, Task 1: Per-task ranking contract, classification byte-identical, Task 2: Request, hyperparameters and command line, Task 3: Imputation metrics as pure functions (+6 more)

### Community 39 - "Resolution"
Cohesion: 0.14
Nodes (14): 08. Configure the checker and make it green, Acceptance, Acceptance, Left for ticket 09, Notes, Question, Resolution, The bump: verified, not assumed (+6 more)

### Community 41 - "_stack"
Cohesion: 0.20
Nodes (9): dtype, DecodedCells, Tensor, Contextual output for every column token, dropping the [CLS] position., Reconstruction loss over the cells hidden from the model. ``hidden`` is the…, Stack per-column tensors feature-major, keeping the shape when there are none., Fill every cell of the batch, in the column's own vocabulary., What the decoder would fill every cell of a batch with. ``categorical_ids`` are… (+1 more)

### Community 42 - "Answer"
Cohesion: 0.29
Nodes (7): 04. The task-branch union and exhaustiveness, Answer, Question, Related, and deliberately not bundled, The fix, Why the `Protocol` is out of scope, Why this one error earns its own ticket

### Community 43 - "05. Third-party stubs for sklearn and scipy"
Cohesion: 0.29
Nodes (7): 05. Third-party stubs for sklearn and scipy, Notes, Question, Resolution, The three candidate answers, Three things ticket 08 must not rediscover the hard way, What to find out

### Community 44 - "imputation-decoder/spec.md"
Cohesion: 0.18
Nodes (9): 01. Per-task fold-ranking contract, Acceptance criteria, Comments, Goal, Seams under test, Global constraints (apply to every ticket), Imputation decoder task: implementation tickets, Status vocabulary (+1 more)

### Community 45 - "runtime_environment_tags"
Cohesion: 0.36
Nodes (6): device, Hardware and library descriptors recorded as MLflow tags., Describe where a run trains so its timings are comparable across machines.…, runtime_environment_tags(), test_cpu_environment_tags_keep_every_key(), test_cuda_environment_tags_record_the_gpu()

### Community 46 - "Findings"
Cohesion: 0.13
Nodes (14): 08 - Experimental design of the launchers, Checked and cleared, F-08-10 - `experiment_imputation.ps1` takes its backups and writes the config outside the `try`, so a mid-setup failure leaves a clobbered file with no restore, F-08-1 - The committed `experiment_imputation.ps1` is a byte-for-byte copy of `experiment.ps1` and runs classification, not imputation, F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the experiment cannot test the decision it exists to test, F-08-3 - The studies never cross-validate: hyperparameters are chosen on one fixed validation split whose rows then make up 8.5-13% of every CV test fold they are evaluated on, F-08-4 - The twelve comparison runs cannot support a generalisation claim: one seed, one missingness draw, unpaired t-intervals over five correlated folds, six of them read at once, F-08-5 - The search optimises the masked proxy; the comparison is judged on the induced population, and the study's winner is not the best trial on it (+6 more)

### Community 48 - "02. Request, hyperparameters and command line"
Cohesion: 0.25
Nodes (7): 02. Request, hyperparameters and command line, Acceptance criteria, Comments, Constraints, Files, Goal, Seams under test

### Community 49 - "Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly"
Cohesion: 0.25
Nodes (8): Consequences, Considered options, Context, Decision, How to compare, Outcome (2026-09-12), Reduce the imputation search to the knobs that move it, score it on the validation split, and promote its result explicitly, Status

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

### Community 59 - "TRIDENT Agent Guide (AGENTS.md)"
Cohesion: 0.15
Nodes (13): TRIDENT Agent Guide (AGENTS.md), PyArrow <24 Pin Constraint, src/training Package, train.main(args, return_metrics=False) Compatibility Facade, ADR 0001: Training Package with Compatibility Facade, Compatibility Facade Pattern, Training Refactor Design Spec, 11. Regression fixture and documentation (+5 more)

### Community 60 - "Map: Imputation decoder task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Imputation decoder task, Not yet specified, Notes, Out of scope

### Community 61 - "08. Record the design: ADR 0005 and the implementation plan"
Cohesion: 0.50
Nodes (4): 08. Record the design: ADR 0005 and the implementation plan, Answer, Comments, Question

### Community 62 - "BACKLOG.md"
Cohesion: 0.12
Nodes (13): Acceptance, Constraints, Decisions, Explicitly not in this ticket, Outcome, The reframing, Ticket 0004: Exact original units in the imputation preview, 03. Task-keyed lookup and explicit promotion (+5 more)

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

### Community 79 - "Findings"
Cohesion: 0.13
Nodes (14): 09 - Test adequacy for the imputation task, Checked and cleared, F-09-10 - Tests that restate the implementation and could go, F-09-1 - The `[NULL]`-path diagnostic scores a different set of cells than it claims, and the only test that guards it asserts a permutation-invariant count, F-09-2 - No test pins `realised_rate`, and the metric it would have pinned shows `EVAL_MASK_RATE` is nearly inert above 40% missingness, F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder whose outputs are unrelated to the truth passes every test including the fixture, F-09-4 - The fixture's own justification for choosing `credit-g_20nan` is false, and the dataset it wrongly excluded is the one that would have caught a shipped crash, F-09-5 - The fixture pins six pooled averages and no structural invariant, so the cheapest and most decisive checks are absent (+6 more)

### Community 80 - "Verdicts"
Cohesion: 0.14
Nodes (13): 07 - Optuna search for imputation: verdicts, F-07-10 - The study's running-best file carries classification's promoted filename and holds only the sampled subset, F-07-1 - The validation objective is too coarse to separate the configurations it ranks, and two of the four promoted files were decided by trial order rather than by the objective, F-07-2 - The checkpoint is selected on exactly the cells the objective is then scored on, so `optuna/best_objective_value` is a best-of-`EPOCHS_DECODE` minimum, not a plain validation score, F-07-3 - Once a configuration is promoted there is no way to run the defaults arm of ADR 0005 decision 6's own comparison, F-07-4 - An imputation run with no promoted file of its own loads classification's tuned configuration wholesale, F-07-5 - fANOVA silently drops `HEAD_DIM`, so the importance ranking cannot see the model width, F-07-6 - The predefined validation rows the search selects on are scored again as test rows in the 5-fold comparison (+5 more)

### Community 81 - "02. The search objective on the validation split"
Cohesion: 0.29
Nodes (6): 02. The search objective on the validation split, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 82 - "04. The importance artifact"
Cohesion: 0.29
Nodes (6): 04. The importance artifact, Acceptance criteria, Comments, Constraints, Goal, Seams under test

### Community 83 - "05. The launcher and the six studies"
Cohesion: 0.33
Nodes (6): 05. The launcher and the six studies, Acceptance criteria, Comments, Constraints, Goal, What the script runs

### Community 84 - "06. The comparison and the documentation"
Cohesion: 0.29
Nodes (6): 06. The comparison and the documentation, Acceptance criteria, Comments, Constraints, Goal, What runs

### Community 85 - "Verdicts"
Cohesion: 0.14
Nodes (13): 08 - Experimental design of the launchers: verdicts, F-08-10 - Backups and the config write sit outside the `try`, so a mid-setup failure leaves a clobbered file with no restore, F-08-1 - The committed `experiment_imputation.ps1` is byte-identical to `experiment.ps1`; only an uncommitted edit makes it an imputation launcher, F-08-2 - `imputation_studies.ps1` hardcodes `--search_space reduced`, so the twelve comparison runs cannot test reduced-vs-full, F-08-3 - Optuna studies never cross-validate; the config is chosen on one fixed validation split whose rows then form 8.5-13% of every CV test fold, F-08-4 - Two overlapping unpaired single-arm t-intervals over five correlated folds, one seed, one missingness draw, six pairs read at once, F-08-5 - The search ranks on the masked proxy while the comparison headlines the induced population; the search winner is not the best trial on the headline, F-08-6 - The sweep writes `<dataset>.json`, which `--task imputation` reads only as a fallback, so on nan variants its config is silently shadowed while `[setup]` claims it was applied (+5 more)

### Community 86 - "Verdicts"
Cohesion: 0.14
Nodes (13): 09 - Test adequacy for the imputation task: verdicts, F-09-10 - Tests that restate the implementation and could go, F-09-1 - The `[NULL]`-path diagnostic builds its cell selection in frame order and it is read in embedder order, and the only guard asserts a count, F-09-2 - No test pins `realised_rate`, and the number it would pin shows `EVAL_MASK_RATE` is nearly inert at high missingness, F-09-3 - Nothing asserts the decoder ever beats the naive baseline, so a decoder wired to the wrong truth passes the whole suite, F-09-4 - The fixture's stated reason for choosing `credit-g_20nan` is false, and the variant it wrongly excluded is the one whose induced path crashes, F-09-5 - The fixture pins six pooled averages and no structural invariant, F-09-6 - `test_imputation_summary_carries_the_decode_stage_timing` fabricates a metric key no producer writes (+5 more)

### Community 88 - "Map: Reduced Optuna search for the imputation task"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Reduced Optuna search for the imputation task, Not yet specified, Notes, Out of scope

### Community 89 - "test_imputation_artifacts.py"
Cohesion: 0.13
Nodes (25): _identity_scaler(), DataFrame, StandardScaler, The files an imputation run leaves behind (ADR 0004, decision 8)., A row can have a dozen cells filled in, and one table that wide reads as noise.…, Two rows' worth of scored cells, one of each kind and each population., Fitted so a scaled value and its original unit are the same number. Keeps a…, Ticket 0004: `1.144e-09` reads as a real measurement when it means zero.… (+17 more)

### Community 90 - "as_category_strings"
Cohesion: 0.15
Nodes (17): bool_, _ColumnModel, float64, as_category_strings(), ndarray, Stringify a categorical column with every kind of missing value collapsed to…, _Constant, GradientBoostingImputer (+9 more)

### Community 91 - "Reduced Optuna search for imputation: implementation tickets"
Cohesion: 0.50
Nodes (4): Global constraints (apply to every ticket), Reduced Optuna search for imputation: implementation tickets, Status vocabulary, Tickets

### Community 92 - "sync_remote.py"
Cohesion: 0.09
Nodes (48): artifact_prefix(), branch_tips(), _branches(), code_files(), delete_pushed_files(), held_open_by(), _held_open_linux(), _held_open_windows() (+40 more)

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

### Community 98 - "test_sync_remote.py"
Cohesion: 0.19
Nodes (35): pull(), Rewrite every artifact root under ``source_prefix`` to ``target_prefix``, in…, Fetch outputs and the store; this checkout's store becomes the server's., translate_store(), Workspace, skipif, client_for(), git() (+27 more)

### Community 99 - "types.py"
Cohesion: 0.10
Nodes (43): Filesystem artifacts emitted by a training runtime., _per_column_scores(), DataFrame, Typed orchestration for TRIDENT training., Each population's per-column errors, against this fold's own naive baseline,…, Run all folds and return raw folds plus the legacy aggregate metrics., run_training(), DecodingOutcome (+35 more)

### Community 100 - "config.py"
Cohesion: 0.10
Nodes (24): promote_best_configuration(), Publish a study's winning configuration where the task's runs will find it. The…, complete_configuration(), hyperparameter_file(), _load_base_hyperparameters(), load_hyperparameters(), logged_hyperparameters(), Namespace (+16 more)

### Community 101 - "Research: third-party stub strategy for sklearn and scipy"
Cohesion: 0.08
Nodes (24): 10. Out of scope, but found, 11. What could not be verified, 12. Sources, 1. Answer, 2.1 typeshed — no, for either library, 2.2 scipy — `scipy-stubs` is official and current, 2.3 sklearn — nothing credible exists, 2. Does a maintained stub package exist? (+16 more)

### Community 102 - "0006-mirror-experiments-per-task.md"
Cohesion: 0.12
Nodes (15): Consequences, Considered options, Context, Decision, How to compare, Mirror every run into one derived MLflow experiment per task, Outcome (2026-09-17), Status (+7 more)

### Community 103 - "CrossValidationSummary"
Cohesion: 0.12
Nodes (7): DisabledTracker, _log_summary_metrics(), Log the final CV statistics and timings; shared by parents and trials., A tracker whose methods deliberately avoid importing MLflow side effects., CrossValidationSummary, TrackingArtifactPaths, FakeTracker

### Community 104 - "06. The hyperparameter JSON schema and the three naming layers"
Cohesion: 0.25
Nodes (7): 06. The hyperparameter JSON schema and the three naming layers, Answer, For the execution ticket, Notes, Question, Scope, already fixed, What to decide

### Community 105 - "07. Enforcing the constants convention beyond what the checker catches"
Cohesion: 0.22
Nodes (8): 07. Enforcing the constants convention beyond what the checker catches, A prior worth stating, Answer, Metric keys: same treatment where it is free, no new machinery, Notes, Question, The measurement this answer depends on, What to decide

### Community 106 - "09. Record the design: ADR 0006 and the implementation plan"
Cohesion: 0.33
Nodes (5): 09. Record the design: ADR 0006 and the implementation plan, Acceptance, Notes, Question, The work

### Community 107 - "TRIDENT README"
Cohesion: 0.18
Nodes (17): credit-g Categorical Columns, electricity Categorical Columns, kr-vs-kp Categorical Columns, ADR 0002: Curated Cross-Validation MLflow Runs, Curated MLflow Cross-Validation Implementation Plan, CrossValidationSummary, MlflowTracker Adapter, summarize_cross_validation() (+9 more)

### Community 108 - "Map: Typing and naming standard for TRIDENT"
Cohesion: 0.33
Nodes (6): Decisions so far, Destination, Map: Typing and naming standard for TRIDENT, Not yet specified, Notes, Out of scope

### Community 109 - "Addendum — the Optuna results, 2026-09-15"
Cohesion: 0.20
Nodes (8): 1. What ran, 2. The search objective, measured properly, 3. The decisive evidence: tuning does not beat defaults, 4. Finding-by-finding ledger, 5. Carried forward and new, 6. What is still unmeasured, 7. Verification record, Addendum — the Optuna results, 2026-09-15

### Community 110 - "2. Confirmed defects"
Cohesion: 0.20
Nodes (10): 2. Confirmed defects, D-1 — `_select` hands the decoder a frame-ordered mask where a token-ordered one is required (**high**), D-2 — `experiment_imputation.ps1` at HEAD is byte-identical to `experiment.ps1` and runs classification (**medium**), D-3 — Two shipped variants crash inside the shipped configuration space before scoring an induced cell (**medium**), D-4 — Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions (**medium**), D-5 — The flat per-dataset metrics file is task-blind and last-writer-wins (**medium**), D-6 — `--score_null_path` writes every induced cell twice and mislabels it in the preview (**low**), D-7 — Extra evaluation rates collide silently on a rounded percent key (**low**) (+2 more)

### Community 111 - "Findings"
Cohesion: 0.14
Nodes (13): 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code, Checked and cleared, F-11-1 - CONTEXT's "Search objective" promises an untouched test split; every trial scores it, logs it, and parks it one column from the objective, F-11-2 - ARCHITECTURE states a false universal about categorical vocabularies and the wrong head width; the code is right and the ADR is right, F-11-3 - The "Hyperparameter → code wiring reference" was updated for the schedule and not for the task, and three of its surviving rows are now wrong for imputation, F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, enforced in none, and the only pinned decode artifact is frozen under the schedule the documents warn against, F-11-5 - CONTEXT defines the `full` search-space profile as sampling everything; it holds the evaluation rate and the schedule, by design, F-11-6 - The stage-timing contract is false for the imputation task: the window is training plus the entire scoring pass, and it moves with flags that change no training (+5 more)

### Community 114 - "Findings"
Cohesion: 0.15
Nodes (12): 04 - Ground truth and the evaluation protocol, Checked and cleared, F-04-1 - `--score_null_path` scores the wrong cells on any table whose CSV interleaves categorical and numerical columns, F-04-2 - the self-masked population's sampling design changes character across the ladder, and the ADR asserts it is comparable anyway, F-04-3 - a growing share of the headline induced population has no observed cell to condition on, bounding `impute_score` toward 1.0, F-04-4 - the row-alignment guard passes on the one sibling incompatibility present in the shipped corpus, F-04-5 - the README's `EVAL_MASK_RATE` explanation is wrong in direction and points at a metric with a different denominator, F-04-6 - the validation and test evaluation masks are the same draw, not two independent ones (+4 more)

### Community 115 - "Findings"
Cohesion: 0.15
Nodes (12): 06 - Artifacts, metric keys and MLflow identity, Checked and cleared, F-06-1 - The shared per-dataset metrics file is task-blind, and imputation runs have already destroyed five classification baselines in the working tree, F-06-2 - The denominator of the headline `impute_score` is never recorded anywhere, and the per-column artifact reports raw error with no baseline beside it, F-06-3 - `--score_null_path` writes every induced cell into the ledger twice, and the preview prints the column twice in one block with a wrong cell count, F-06-4 - The repo's own MLflow filtering rule was edited by this branch and still omits `tags.task`, on the only two metric families both tasks share, F-06-5 - An imputation fold logs no step-less test loss, so the decoder's own objective is never scored on held-out data, F-06-6 - The imputation fold metric key set is data-dependent, and a fold that disagrees aborts the whole CV run after every fold has trained (+4 more)

### Community 116 - "Verdicts"
Cohesion: 0.15
Nodes (12): 11 - The documentation contract: CONTEXT.md, ARCHITECTURE.md and README against the code: verdicts, F-11-1 - CONTEXT's "Search objective" promises the test split "stays untouched"; every trial scores it and logs it, F-11-2 - ARCHITECTURE asserts every categorical vocabulary holds `"nan"` and the head is `Linear(d, V_col − 3)`; on `_00nan` variants it is `V_col − 2`, F-11-3 - The hyperparameter wiring table gained an `lr_scheduler` row, no decode rows, and three surviving rows are now task-inaccurate, F-11-4 - `--lr_scheduler cosine` is prescribed in two documents, defaulted in none, and the imputation fixture is pinned under `cosine_legacy`, F-11-5 - CONTEXT defines `full` as sampling every hyperparameter the task uses; it permanently holds three, F-11-6 - The stage-timing contract: the imputation window wraps the entire scoring pass and moves with flags that change no training, F-11-7 - "On the test fold only" in two documents, while a trial scores a validation population and `--score_null_path` adds a fourth (+4 more)

### Community 117 - "Verdicts"
Cohesion: 0.17
Nodes (11): 06 - Imputation artifacts, metric keys and MLflow identity: verdicts, F-06-1 - The root `metrics/<dataset>_metrics.csv` is keyed by dataset alone, so an imputation run overwrites a classification run's fold table with a different column set, F-06-2 - The naive denominator of `impute_score` is never persisted, and the per-column artifact reports raw error with no baseline beside it, F-06-3 - With `--score_null_path` the same cell is in the ledger under two populations, and the preview prints it twice in one block with a row-count header, F-06-4 - `CLAUDE.md`'s comparison rule was edited on this branch and still omits `tags.task`, on the only metric families both tasks share, F-06-5 - No step-less `test/loss` on an imputation fold, so the decoder's objective is allegedly never scored on held-out data, F-06-6 - The imputation fold metric key set is data-dependent, so a disagreeing fold aborts the CV run after every fold has trained, F-06-7 - Extra evaluation rates are keyed by a rounded integer percent, so nearby rates collide silently and a sub-0.5% rate becomes `rate_0` (+3 more)

### Community 118 - "Findings"
Cohesion: 0.18
Nodes (10): 03 - Imputation metrics and the composite score, Checked and cleared, F-03-1 - The masked `impute_score` is not comparable across the missingness ladder: the exam's realised mask rate is non-monotone, 0.209 → 0.155 → 0.250, F-03-2 - "1.0 is baseline parity" is false for the composite: a model that has collapsed to mode imputation on every categorical column scores 0.75, F-03-3 - The Optuna search objective is scored on the exact cells that selected the checkpoint, F-03-4 - The baseline normalisation is a per-fold scalar rescale, not a per-column normalisation: it removes no column heterogeneity and adds sampling noise to the CI, F-03-5 - `_ratio`'s zero guard protects the pooled error, not the constant column ADR 0004 says it protects, and inverts the ranking when it does fire, F-03-6 - The per-column artifact carries errors without their baselines, so the one artifact meant to localise a bad fold cannot (+2 more)

### Community 119 - "Verdicts"
Cohesion: 0.18
Nodes (10): 04 - Ground truth and the evaluation protocol: verdicts, F-04-1 - `--score_null_path` marks the selection tensor in CSV column order while the model's mask is in categorical-then-numerical order, so the diagnostic scores transposed cells, F-04-2 - the self-masked population's realised rate is non-monotone and floor-dominated across the ladder, contradicting the ADR's comparability claim, F-04-3 - a growing share of the induced population sits in rows with no observed cell, bounding those cells' contribution at baseline parity, F-04-4 - `_assert_row_aligned` checks only value equality through an object-dtype comparison, and certifies the corpus's one sibling incompatibility, F-04-5 - the README's `EVAL_MASK_RATE` paragraph omits the per-row floor and quotes a figure against a different denominator than the metric it names, F-04-6 - validation and test evaluation masks are drawn from one stream reseeded identically, so the two questions are not independent, F-04-7 - `rmse_num_z` / `mae_num_z` are in units of a sigma fit over all rows including the test fold, re-estimated at each rung (+2 more)

### Community 120 - "Findings"
Cohesion: 0.18
Nodes (10): 10 - Configuration plumbing and classification bit-identity, Checked and cleared, F-10-1 - The flat metrics aggregate is not task-keyed, so the two tasks overwrite each other's record for the same dataset variant, F-10-2 - The JSON/override loader's epoch fallbacks moved from 40 to 300/150, changing classification training length for any partial configuration, F-10-3 - The imputation launcher configures the decode stage through classification's keys, edits classification's config file, and tells the operator to compare a metric imputation never emits, F-10-4 - `config_source` over-claims: an imputation run reports a file that supplied none of its decode parameters, and silently inherits that file's schedule, F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is silently dropped by promotion, contradicting the documented "a promoted file is complete", F-10-6 - The agent-facing MLflow comparison rule was updated for the schedule but not for the task (+2 more)

### Community 121 - "Findings"
Cohesion: 0.18
Nodes (10): 12 - Reproducibility and provenance of an imputation run end to end, Checked and cleared, F-12-1 - The `_XXnan` missingness draw, which *is* the induced benchmark, is recorded nowhere and no check in the pipeline can detect its substitution, F-12-2 - Training corruption is drawn from one global numpy stream whose position at fold *k* is a function of every earlier fold's draws, so ADR 0005 decision 6's "the configuration is the only difference" is false from fold 2 and no single fold is reproducible on its own, F-12-3 - `config_source` is a mutable path into an untracked directory; the commit the run records does not contain the file, and re-running that commit silently trains the defaults instead, F-12-4 - The branch-new environment record names torch and CUDA only, omitting exactly the libraries whose behaviour the imputation path depends on, F-12-5 - A run's on-disk artifact tree and its MLflow run are joined only by a timestamp drawn twice, and both successful imputation parents in the store are off by one second, Findings (+2 more)

### Community 122 - "Findings"
Cohesion: 0.20
Nodes (9): 01 - The decode stage, Checked and cleared, F-01-1 - The masked population's realised mask rate is neither the configured rate nor comparable across the missingness ladder, and the code's stated explanation of the gap has the sign backwards, F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet are reported side by side as the same measurement at different difficulties, F-01-3 - The validation and test masks are drawn from one RNG stream, so the checkpoint is selected under the same masked-column configurations the test score is measured under, F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently, so the epoch chosen is not the epoch that best serves the reported score, F-01-5 - The extra-rate diagnostic ladder collapses at the low end and silently overwrites itself when two rates round to the same percent, Findings (+1 more)

### Community 123 - "Findings"
Cohesion: 0.20
Nodes (9): 02 - The decoder model and its heads, Checked and cleared, F-02-1 - Checkpoint selection runs on a λ-weighted composite, so `LAMBDA_NUM` decides which epoch is kept as well as how the gradient is mixed, F-02-2 - The per-kind loss terms are computed and detached specifically for reporting, and both callers throw them away, F-02-3 - Per-cell pooling inside the categorical term: a column that cannot be got wrong still takes its share of the denominator, F-02-4 - A categorical column with no real categories builds a zero-width head and kills `predict` after the whole stage has run, F-02-5 - Nothing in the suite asserts that decode gradients reach the embedder and the transformer, Findings (+1 more)

### Community 124 - "Verdicts"
Cohesion: 0.20
Nodes (9): 03 - Imputation metrics and the composite score: verdicts, F-03-1 - The masked exam is a different exam at each rung of the ladder, so the ADR's cross-ladder comparability claim is unsupported, F-03-2 - "1.0 is baseline parity" allegedly false for the composite, F-03-3 - The Optuna trial score is read off the same cells that selected the checkpoint, F-03-4 - The baseline normalisation is a per-fold scalar rescale that allegedly injects noise into the promotion CI, F-03-5 - `_ratio`'s zero guard allegedly inverts the ranking and guards the wrong thing, F-03-6 - The per-column artifact carries errors without their baselines, Verdicts (+1 more)

### Community 125 - "Verdicts"
Cohesion: 0.20
Nodes (9): 10 - Configuration plumbing and classification bit-identity: verdicts, F-10-1 - The flat `metrics/<dataset>_metrics.csv` carries no task key, so the two tasks silently overwrite each other's record for one dataset variant, F-10-2 - `from_mapping`'s epoch fallbacks moved 40/40 → 300/150, changing classification training length for partial configs, F-10-3 - `experiment_imputation.ps1` drives the decode stage through classification's keys, writes the shared classification file, and closes with an f1_macro query, F-10-4 - An imputation run falling back to the shared file logs a `config_source` naming a file that supplied none of its decode parameters, and silently takes its `LR_SCHEDULER`, F-10-5 - `EVAL_MASK_RATES_EXTRA` is in no run record and is deleted by `--promote_best`, against README's "a promoted file is complete", F-10-6 - `CLAUDE.md`'s comparison rule gained `tags.lr_scheduler` but not `tags.task`, Verdicts (+1 more)

### Community 126 - "Findings"
Cohesion: 0.20
Nodes (9): 13 - What the shared pre-training stage contributes to imputation, Checked and cleared, F-13-1 - 300 epochs of pre-training end 4.2x worse than a 20-vector constant lookup on pre-training's own objective; the 98.5% loss drop is the encoder's output norm shrinking, not learning, F-13-2 - A paired ablation cannot detect any benefit from 298 of the 300 pre-training epochs: the arms sit inside fold-to-fold noise with the sign flipping between seeds, and nothing in the branch has ever run this comparison, F-13-3 - Nothing in the branch logs, tests or asserts any scale-free property of the pretrained representation, so "pre-training worked" is unfalsifiable from a run's record, F-13-4 - The reduced profile spends two thirds of every trial's epoch budget on a held stage whose contribution is unmeasured, and its one cross-stage knob makes the two stages' corruption inseparable, F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions: measured 11 halvings to 1.66e-07 on spambase, from a signal with a 9% noise floor, Findings (+1 more)

### Community 127 - "Verdicts"
Cohesion: 0.22
Nodes (8): 01 - The decode stage: verdicts, F-01-1 - The realised eval-mask rate is neither the configured rate nor comparable across the ladder, and `decoding.py:152-153` names a mechanism that is already netted out, F-01-2 - The masked and induced populations are drawn from systematically different rows and shown different context, yet reported as a pair, F-01-3 - Validation and test masks come from one RNG stream, so the checkpoint is selected under the configurations the test score is measured under, F-01-4 - The checkpoint criterion and the fold-ranking metric weight the two column kinds differently, F-01-5 - The `rate_*` family collides when two configured rates round to the same percent and overwrites silently; the low end of the ladder is degenerate, Verdicts, What this critique missed

### Community 128 - "Verdicts"
Cohesion: 0.22
Nodes (8): 02 - The decoder model and its heads: verdicts, F-02-1 - The checkpoint is chosen by minimising the same λ-weighted composite the decoder trains on, so `LAMBDA_NUM` also decides which epoch survives, and that criterion is only loosely aligned with the `impute_score` that ranks the fold, F-02-2 - The per-kind loss terms are built and detached for reporting and every caller discards them, so no run can decompose the decode loss curve, F-02-3 - The categorical term pools per cell across columns, so a column that cannot be got wrong contributes zero loss while still inflating the denominator, F-02-4 - An all-missing categorical column builds `nn.Linear(d, 0)` and crashes `predict` after the whole decode stage has run; `local_of` maps excluded ids to `-1` with no `ignore_index` guard, F-02-5 - No test asserts that decode-stage gradients reach the embedder and transformer, so the ADR's "the encoder is fine-tuned, not frozen" is unguarded, Verdicts, What this critique missed

### Community 129 - "Findings"
Cohesion: 0.22
Nodes (8): 05 - The embedder, [MASK] and [NULL] semantics, Checked and cleared, F-05-1 - `_select` hands the decoder a frame-ordered mask where a token-ordered one is required, so the `[NULL]`-path diagnostic scores cells that were never hidden, F-05-2 - `encode` has a third representation of a missing cell - literal `NaN` and the `'nan'` category - and pre-training feeds exactly that to its clean targets, F-05-3 - the induced population is scored on a row shape the decoder never trained on: every gap becomes `[MASK]` and no `[NULL]` survives, F-05-4 - `rmse_num_z`'s unit is fit on the whole table, evaluation rows included, Findings, Open questions

### Community 130 - "Verdicts"
Cohesion: 0.22
Nodes (8): 12 - Reproducibility and provenance of an imputation run end to end: verdicts, F-12-1 - The induced-missing benchmark rests on an unrecorded, ungoverned missingness draw that nothing can detect the substitution of, F-12-2 - One global numpy stream means folds >= 2 enter at an offset set by earlier folds, so decision 6's "the configuration is the only difference" is not what it sounds like, F-12-3 - `config_source` is a mutable path into an untracked directory, and the fallback is silent, F-12-4 - The environment record omits the scikit-learn version, F-12-5 - The results directory and the MLflow run are joined only by a timestamp drawn twice, and both finished parents are off by a second, Verdicts, What this critique missed

### Community 131 - "Verdicts"
Cohesion: 0.22
Nodes (8): 13 - What the shared pre-training stage contributes to imputation: verdicts, F-13-1 - Pre-training's own loss falls 97-99% while the encoder ends worse than a constant-per-column lookup, and carries no per-cell information about the hidden value, F-13-2 - The "shared and protected" `EPOCHS_PRE = 300` has no measurement behind it anywhere in the branch, and the critic's first ablation cannot detect a benefit, F-13-3 - The stage logs three scalars and nothing scale-free, and no unit test ever executes it, F-13-4 - The reduced profile holds two thirds of each trial's epoch budget on a stage the search cannot touch, and no sampled dimension moves pre-training alone, F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions, Verdicts, What this critique missed

### Community 132 - "Verdicts"
Cohesion: 0.25
Nodes (7): 05 - The embedder, [MASK] and [NULL] semantics: verdicts, F-05-1 - `_score_induced_missing` overrides `masked_positions` with a DataFrame-ordered `isna()` matrix, while every reader of that field indexes it categorical-then-numerical, so the `[NULL]` diagnostic scores the wrong cells, F-05-2 - `encode` on a frame that never went through `preprocess_table` represents a missing cell as raw `NaN` (numerical) or the `'nan'` category, and pre-training's clean targets are built that way, F-05-3 - the induced population is scored on rows where every gap has become `[MASK]` and no `[NULL]` survives, a missingness pattern the decoder essentially never trained on, F-05-4 - the `StandardScaler` that defines `rmse_num_z`'s unit is fitted on the whole table, including the rows the metric scores, Verdicts, What this critique missed

### Community 133 - "backfill_baseline_imputers.py"
Cohesion: 0.07
Nodes (48): Exception, FoldMetrics, Metric, apply_plan(), _arguments(), _BudgetSpent, candidate_runs(), _cells() (+40 more)

### Community 134 - "Fidelity audit of `CONSOLIDATED.md`"
Cohesion: 0.29
Nodes (6): Fidelity audit of `CONSOLIDATED.md`, Judged faithful, recorded rather than changed, Per-source confirmation, What this audit did not check, What was checked, and how, What was found and changed

### Community 135 - "test_backfill_baseline_imputers.py"
Cohesion: 0.11
Nodes (21): client(), _folds(), _gap_rows(), fixture, MlflowClient, Backfilling the baseline imputers of ADR 0007 onto runs recorded before the…, A run recorded before ``baseline/best`` existed gets it from its own logged…, Two folds as ``raw_fold_metrics.csv`` holds them for a run of 2026-09-24: the… (+13 more)

### Community 136 - "§8 — Experiments run 2026-09-17"
Cohesion: 0.13
Nodes (15): 8.10 The one separation in twelve, replicated: it does not hold, 8.11 The committed `credit-g_20nan` promoted file loses to the defaults; `credit-g_40nan` leans the same way, 8.12 The sign of tuned-versus-defaults follows the winner's rank, 8.13 Verification record for §8, 8.14 The control: what the best trial is worth once it is scored the way the comparison is, 8.1 The typing refactor preserved imputation behaviour exactly, 8.2 The noise floor N-1 asked for, 8.3 Mis-selection on the study's own split — and, in §8.14, how little of it survives five-fold (+7 more)

### Community 137 - "Training Refactor Implementation Plan"
Cohesion: 0.18
Nodes (16): Task 3 Report: Runtime Adapters and Regression Baseline, vehicle_00nan Regression Fixture, Plan: Add --all CLI Flag for Batch Dataset Training, discover_datasets(), run_all(args), Training Refactor Implementation Plan, create_tracker(), FoldResult (+8 more)

### Community 138 - "Consolidated report — the imputation work on `feat/imputation-task`"
Cohesion: 0.29
Nodes (7): 1. Verdict, 5. Checked and cleared, 6. Coverage map, Consolidated report — the imputation work on `feat/imputation-task`, Refuted, and severity-none, Source files the branch touched that **no dimension opened**, Suspected and disproved — do not spend time here

### Community 139 - "4. What the experiments can and cannot support"
Cohesion: 0.33
Nodes (6): 4. What the experiments can and cannot support, Cross-ladder reads (`_20nan` → `_80nan`), `experiment_imputation.ps1` — do not run it as it stands, `imputation_studies.ps1 -Compare` — the twelve comparison runs (none has been run), `imputation_studies.ps1` — the study branch (five studies run, one in flight, one not started), Two crashes to expect

### Community 140 - "_step_frame"
Cohesion: 0.25
Nodes (8): ``level`` jumps from -3 to 3 where ``x`` passes 10, and ``side`` follows the…, The third baseline trains one model per column on the training rows where that…, Boosting subsamples a validation split on large folds; with a fixed seed of its…, A column the training fold holds in one category only, or not at all, gives a…, _step_frame(), test_gradient_boosting_falls_back_to_a_constant_where_there_is_nothing_to_learn(), test_gradient_boosting_fills_the_same_way_every_time(), test_gradient_boosting_learns_each_column_from_the_others()

### Community 141 - "CLAUDE.md"
Cohesion: 0.11
Nodes (15): Background & full reference, Experimentation log, gorgona8: the store lives on the SSD, graphify, MLflow run analysis, Times, Consequences, Context (+7 more)

### Community 142 - "preprocess_table"
Cohesion: 0.11
Nodes (27): Classifier fine-tuning and fold metric calculation., create_pretrain_datasets(), preprocess_table(), Example function that: 1) Splits train/val (e.g., 90/10) 2) Generates masked DF…, Preprocesses the table to replace null values with the `[NULL]` token, applies…, _decoder(), _frame(), DataFrame (+19 more)

### Community 143 - "Entries"
Cohesion: 0.08
Nodes (25): E01 · `[MASK]` vs `[NULL]` token path for induced cells (2026-09-10; re-checked 2026-09-11), E02 · Reduced Optuna search: six studies under cosine (2026-09-11), E03 · Promoted configuration vs defaults, cosine, seed 42 (2026-09-11 – 2026-09-12), E04 · Does the pre-training loss drop mean learning? (critique probe, 2026-09-11), E05 · Pre-training epochs 300 vs 2 (critique ablation, 2026-09-11), E06 · Decode checkpoint criterion vs ranking metric (critique probe, 2026-09-11), E07 · Plateau halvings driven by pre-training's re-rolled validation mask (critique read, 2026-09-11), E08 · Resolution and fidelity of the masked search objective (critique reads, 2026-09-11) (+17 more)

### Community 144 - "RsyncTransport"
Cohesion: 0.30
Nodes (5): CompletedProcess, A Windows path as WSL mounts it: ``C:/Users/x`` becomes ``/mnt/c/Users/x``., rsync and ssh; through WSL on Windows, where their keys and known_hosts live., RsyncTransport, to_wsl()

### Community 145 - "build_fold_result"
Cohesion: 0.22
Nodes (12): build_fold_result(), ndarray, Build the legacy per-fold classification metric set., compute_cv_summary(), DataFrame, Path, Return the legacy mean, sample standard deviation, and display string., test_build_fold_result_keeps_binary_confusion_fields_for_single_class_fold() (+4 more)

### Community 146 - "DirectoryServer"
Cohesion: 0.21
Nodes (5): _copy(), _copy_tree(), DirectoryServer, _make_writable_and_retry(), The server as a directory on this machine: copies stand in for rsync and the…

### Community 147 - "ObjectiveFunctionWrapper"
Cohesion: 0.22
Nodes (8): ObjectiveFunctionWrapper, Path, Wrapper class for the Optuna objective function to maintain state, Save the best parameters found so far, inside the study's own directory. The…, Scoring a crash teaches the search to seek crashes when lower is better.…, Both tasks read datasets/hiperparams/<base>/<dataset>.json, and it names no…, test_a_trial_that_crashes_is_discarded_rather_than_scored(), test_an_imputation_study_never_overwrites_a_datasets_shared_config()

### Community 148 - "Transport"
Cohesion: 0.22
Nodes (3): Protocol, How files and commands reach the server. Paths are relative to each side's root., Transport

### Community 149 - "test_backfill_search_objective_tag.py"
Cohesion: 0.31
Nodes (10): client(), fixture, MlflowClient, Stamping the search objective on Optuna runs recorded before ADR 0008., Every study before ADR 0008 ranked imputation trials by the masked validation…, Mirrors copy their source's tags (ADR 0006), so a stamped tree is re-mirrored;…, _seed_runs(), test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing() (+2 more)

### Community 150 - "Hand the tracking store to a remote server whole, one writer at a time"
Cohesion: 0.29
Nodes (7): Consequences, Context, Decision, Hand the tracking store to a remote server whole, one writer at a time, How to use, Outcome (2026-09-29, rehearsal), Status

### Community 151 - "tracking.py"
Cohesion: 0.11
Nodes (24): log_param_importances(), Rank the knobs the study sampled and record the ranking on the study parent.…, build_fold_tags(), build_hyperparams_dict(), build_run_tags(), get_or_create_experiment(), parse_missingness_percent(), src/mlflow_utils.py ------------------- Centralised MLflow utilities for… (+16 more)

### Community 152 - "fit_baseline_imputers"
Cohesion: 0.20
Nodes (7): BaselineImputer, fit_baseline_imputers(), MeanModeImputer, Protocol, The baseline imputers, fit on this fold's training split. Once per fold., A baseline imputer: learns from the training fold, fills whatever is hidden., The naive baseline: each column's training mean or mode, whatever was hidden.

### Community 153 - "Let an imputation study rank its trials by the validation split's induced gaps"
Cohesion: 0.05
Nodes (35): Consequences, Considered options, Context, Decision, Let an imputation study rank its trials by the validation split's induced gaps, Status, Verification, Design (fixed before launch) (+27 more)

### Community 154 - "cli.py"
Cohesion: 0.15
Nodes (17): _format_duration(), main(), Namespace, Return a human-readable duration string., Command-line entry point for TRIDENT., Run training sequentially on every discovered dataset., run_all(), parse_args() (+9 more)

### Community 155 - "Score baseline imputers on exactly the cells the decoder is scored on"
Cohesion: 0.22
Nodes (8): Consequences, Considered options, Context, Decision, How to compare, Outcome (2026-09-24/25 and 27: 22 five-fold runs with the change, seed 42, cosine), Score baseline imputers on exactly the cells the decoder is scored on, Status

### Community 156 - "test_backfill_lr_scheduler_tag.py"
Cohesion: 0.42
Nodes (9): client(), fixture, MlflowClient, _seed_runs(), test_apply_is_idempotent(), test_apply_tags_only_untagged_runs_and_marks_them_backfilled(), test_dry_run_reports_without_writing(), test_main_dry_run_prints_summary() (+1 more)

### Community 157 - ".encode"
Cohesion: 0.33
Nodes (4): Convert a whole DataFrame into the tensors ``forward`` consumes. Doing this…, For each row, generates the resulting embedding: 1) Transforms each categorical…, Separates real numerical values from special tokens `[MASK]` and `[NULL]`.…, split_numeric_and_special()

### Community 158 - "8. The configuration for ticket 08"
Cohesion: 0.40
Nodes (5): 8.1 `pyproject.toml`, 8.2 Where the stubs live, 8.3 Resolver impact — verified against copies of the real `pyproject.toml` + `uv.lock`, 8.4 Honest cost to ticket 08, 8. The configuration for ticket 08

### Community 159 - "_columns_holding"
Cohesion: 0.40
Nodes (5): _columns_holding(), _foreign_roots(), Connection, Artifact roots under neither prefix; ``mlflow-artifacts:`` roots belong to no…, Every ``table.column`` of the store in which some value contains ``text``.

### Community 161 - "TrainingTracker"
Cohesion: 0.50
Nodes (3): Protocol, Log metrics at an optional training step., TrainingTracker

### Community 164 - "backfill_search_objective_tag.py"
Cohesion: 0.24
Nodes (11): backfill(), BackfillReport, default_objective(), main(), _optuna_runs(), MlflowClient, Stamp the ``search_objective`` tag on Optuna runs recorded before ADR 0008.…, What a study of this task ranked by before the objective could be chosen. (+3 more)

### Community 166 - "01. The held set and the ranges"
Cohesion: 0.50
Nodes (4): 01. The held set and the ranges, Answer, Comments, Question

### Community 167 - "07. Task: repair the imputation Optuna path and prove it with a two-trial study"
Cohesion: 0.50
Nodes (4): 07. Task: repair the imputation Optuna path and prove it with a two-trial study, Answer, Comments, Question

## Ambiguous Edges - Review These
- `kr-vs-kp Categorical Columns` → `TRIDENT README`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `TRIDENT README` → `src/training Package`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **716 isolated node(s):** `info`, `trident`, `start_mlflow.sh script`, `graphify`, `MLflow run analysis` (+711 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1263 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `kr-vs-kp Categorical Columns` and `TRIDENT README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `TRIDENT README` and `src/training Package`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `SyncError` connect `StoreError` to `RsyncTransport`, `test_sync_remote.py`, `sync_remote.py`, `run_hyperparameter_optimization`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `test_a_run_that_failed_is_left_to_the_script()` connect `test_mirroring.py` to `run_hyperparameter_optimization`, `test_training_tracking.py`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `run_hyperparameter_optimization()` connect `run_hyperparameter_optimization` to `setup_mlflow`, `config.py`, `test_mirroring.py`, `backfill_search_objective_tag.py`, `DatasetSpec`, `ObjectiveFunctionWrapper`, `tracking.py`, `test_training_tasks.py`, `cli.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `summarize_cross_validation()` (e.g. with `FoldTrackingRecord` and `TaskSpec`) actually correct?**
  _`summarize_cross_validation()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `PreparedDataset` (e.g. with `_frame_values()` and `ArtifactWriter`) actually correct?**
  _`PreparedDataset` has 12 INFERRED edges - model-reasoned connections that need verification._