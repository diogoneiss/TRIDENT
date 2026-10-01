# TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens

This repository accompanies the paper **"TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens"**, a Transformer-based framework for tabular data classification designed for mixed datasets with categorical and numerical features, with a strong emphasis on learning from missing values rather than simply imputing them.

## Paper

- **Title**: TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens
- **Publication**: In *Intelligent Systems*, BRACIS 2025
- **Series**: Lecture Notes in Computer Science (LNCS)
- **Volume**: 16180
- **Publisher**: Springer, Cham
- **Published**: January 30, 2026
- **DOI**: [10.1007/978-3-032-15984-7_39](https://doi.org/10.1007/978-3-032-15984-7_39)
- **Springer Chapter**: [Read the paper on Springer](https://link.springer.com/chapter/10.1007/978-3-032-15984-7_39)
- **Print ISBN**: 978-3-032-15983-0
- **Online ISBN**: 978-3-032-15984-7
- **eBook Package**: Computer Science / Springer Nature Proceedings Computer Science

## Overview

TRIDENT addresses the challenge of applying Transformer architectures to tabular data, especially in scenarios with high missingness, through a two-stage training paradigm:

1. **Pre-training Stage**: Self-supervised learning where the model learns to reconstruct the original feature embeddings of artificially masked positions, capturing latent structure and inter-feature dependencies.
2. **Fine-tuning Stage**: The pre-trained model is fine-tuned end-to-end for downstream classification using a classification head over the `[CLS]` token representation.

The framework introduces specialized components for heterogeneous tabular data and treats missing values as **informative, learnable signals** rather than noise to be imputed away.

## Key Ideas

- Unified embedding pipeline for **categorical** and **numerical** features
- Dedicated handling of **missing values** through learnable `[NULL]` representations
- Transformer-based modeling of **inter-feature dependencies**
- Self-supervised pre-training adapted to the tabular setting
- End-to-end fine-tuning for downstream classification tasks
- Support for experiments under different levels of artificial missingness

## Architecture

For a layer-by-layer walkthrough (tensor shapes, data pipeline, mermaid diagrams for both the theoretical model and the PyTorch implementation), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This section stays a high-level summary.

### Core Components

- **TabularEmbedder**: Converts mixed tabular data into unified embeddings
  - Categorical features: learnable embeddings with special tokens `[MASK]` and `[NULL]`
  - Numerical features: MLP-based transformations with special token embeddings
  - Positional encoding for feature order awareness
  - `[CLS]` token for downstream classification

- **TabularTransformerEncoder**: Multi-layer Transformer encoder
  - Multi-head self-attention
  - Layer normalization and residual connections
  - Feed-forward networks with configurable dimensions

- **TridentPretrainer**: Self-supervised pre-training model
  - Masked modeling adapted for tabular data
  - Dynamic masking based on missing-value density
  - MSE loss for embedding reconstruction

- **TridentModel**: Supervised classification model
  - Reuses pre-trained encoder representations
  - Multi-layer classification head with dropout
  - Support for class-weighted loss functions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TRIDENT

# Install uv (if not already installed)
# On Windows:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# On macOS/Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize the virtual environment and dependencies
uv sync
```

### Data Preparation

1. Place CSV files in `datasets/datasets_raw/`
2. Ensure the target column is named `class` (or specify it with `--label_column`)
3. Generate processed datasets and train/validation/test splits:

```bash
cd datasets
uv run generate_splits.py
```

This creates:

- Processed datasets with different NaN levels (20%, 40%, 60%, 80%)
- Train/validation/test splits in `processed_datasets/splits/`
- Categorical column definitions in `categorical_columns/`

### Basic Usage

```bash
# Standard training
uv run main.py --dataset_name vehicle_00nan

# With visualization and model saving
uv run main.py --dataset_name vehicle_00nan --plot_losses --save_model

# Running with 5-fold cross-validation
uv run main.py --dataset_name vehicle_00nan --cv_folds 5

# Hyperparameter optimization
uv run main.py --dataset_name vehicle_00nan --use_optuna --n_trials 100 --retrain_best

# Keep project-level metrics outside the default metrics/ directory
uv run main.py --dataset_name vehicle_00nan --metrics_dir ./experiments/vehicle/metrics

# Run without MLflow tracking (useful for isolated experiments and tests)
uv run main.py --dataset_name vehicle_00nan --disable_mlflow
```

### Batch Training (All Datasets)

```bash
# Run all available datasets with default settings (0% missingness)
uv run main.py --all

# Run all datasets with 3-fold cross-validation
uv run main.py --all --cv_folds 3

# Run only the first 3 datasets (alphabetical order)
uv run main.py --all --limit 3 --cv_folds 3

# Run all datasets at 40% missingness level
uv run main.py --all --nan_level 40

# Combine limit, nan_level, and other flags
uv run main.py --all --limit 5 --nan_level 20 --cv_folds 5 --disable_mlflow
```

The `--all` flag discovers datasets from `datasets/processed_datasets/` at runtime, runs them sequentially, and prints a summary table at the end showing each dataset's status, F1 macro score, and runtime. Failed datasets are logged and skipped without aborting the remaining runs.

## Command Line Arguments

### Dataset Selection

Exactly one of `--dataset_name` or `--all` is required.

- `--dataset_name`: Dataset name without the `.csv` extension (mutually exclusive with `--all`)
- `--all`: Run training on all available datasets sequentially (mutually exclusive with `--dataset_name` and `--use_optuna`)
- `--limit`: When used with `--all`, run only the first N datasets in alphabetical order
- `--nan_level`: When used with `--all`, select the missingness level — one of `0`, `20`, `40`, `60`, `80` (default: `0`)

### Main Parameters

- `--label_column`: Target column name (default: `class`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--output_dir`: Results directory (default: `results`)
- `--metrics_dir`: Project-level raw-fold metrics directory (default: `metrics`)
- `--disable_mlflow`: Disable all MLflow setup, runs, logs, and artifacts (enabled by default)
- `--disable_mirror`: Do not mirror the finished run into `TRIDENT/mirror/<task>` (mirroring is on by default; implied by `--disable_mlflow`)

### Task Selection

- `--task`: What to train, one of `classification` (default) or `imputation`. Classification predicts the label from the `[CLS]` token and is what every run did before [ADR 0004](docs/adr/0004-imputation-decoder-task.md); imputation replaces the classifier with a decoder that reconstructs hidden cell values. Command-line only, never from a hyperparameter file, so an `--all` batch cannot end up training a different task per dataset
- `--score_null_path`: Also score the dataset's own missing cells with the model seeing `[NULL]` rather than `[MASK]`, as a diagnostic that never ranks folds. Requires `--task imputation` and is refused at parse time otherwise
- `--decode_patience`: Stop the decode stage once its validation loss has not improved for this many epochs, keeping the best epoch; the schedule still spans the configured epochs. Overrides `DECODE_PATIENCE`. Requires `--task imputation`. Default: `0`, every epoch trains
- `--decoder_heads`: How the decoder applies its per-column heads: `batched` (default since 2026-10-01, [ADR 0012](docs/adr/0012-batched-decoder-heads-by-default.md): the same arithmetic in a few operations, 1.4x to 1.6x faster a cell, equal up to rounding, so a different training draw) or `per_column` (one operation per column, which reproduces every earlier run to the last digit). Overrides `DECODER_HEADS`
- `--no_baseline_cache`: Compute the baseline imputers' scores every time instead of reusing the ones cached under `results/baseline_cache/` for the same fold and cells ([ADR 0007](docs/adr/0007-baseline-imputers-scored-on-the-decoders-cells.md) decision 10; a hit is identical to the computation). `python3 scripts/prune_cache.py [--max_mb N] [--max_age_days D] [--apply]` keeps the cache under a size or age limit, least recently used first

### Training Options

- `--plot_losses`: Generate training loss visualizations
- `--save_model`: Save trained model checkpoints
- `--cv_folds`: Number of folds for cross-validation (default: `None`, uses pre-defined split)
- `--lr_scheduler`: Learning-rate schedule for both stages, one of `cosine_legacy`, `cosine`, `warmup_cosine`, `constant`, `plateau`. Overrides `LR_SCHEDULER` from the hyperparameter file and applies to every dataset of an `--all` batch and every Optuna trial. Default: `cosine_legacy`, the schedule of every run before [ADR 0003](docs/adr/0003-selectable-learning-rate-schedule.md)
- `--pretrain_objective`: What pre-training reconstructs at a masked cell, one of `embedding` (the cell's detached clean embedding), `value` (the value itself, through fresh decoder heads) or `embedding_normalized` (a layer-normalised embedding target read through a predictor head). Overrides `PRETRAIN_OBJECTIVE`. Default: `embedding`, the objective of every run before [ADR 0011](docs/adr/0011-selectable-pretraining-objective.md)

### Optuna Optimization

- `--use_optuna`: Enable hyperparameter optimization (mutually exclusive with `--all`)
- `--n_trials`: Number of optimization trials (default: `50`)
- `--retrain_best`: Retrain using the best parameters found during search
- `--promote_best`: Publish the winning configuration where the task's runs read it (see "Configuration System"); independent of `--retrain_best`
- `--search_objective`: Which validation population an imputation study ranks its trials by, `masked` (default, every study before [ADR 0008](docs/adr/0008-selectable-search-objective-population.md)) or `induced` (the validation rows' own gaps scored against the complete `_00nan` sibling); `induced` needs a variant with gaps, and the flag requires `--task imputation`
- `--search_space`: Which knobs a study samples, `full` or `reduced` ([ADR 0005](docs/adr/0005-reduced-optuna-search-for-imputation.md)); `reduced` exists for `--task imputation` only, and unspecified follows the task (`reduced` for imputation, `full` for classification)

## Tests

Use the project wrapper with Python 3.11:

```bash
# Fast unit tests
uv run --python 3.11 pytest -m "not integration"

# Short end-to-end regression test: two folds of vehicle_00nan
uv run --python 3.11 pytest -m integration

# Entire suite
uv run --python 3.11 pytest
```

The integration test disables MLflow and writes all run artifacts to temporary directories. Its checked-in baseline validates per-fold and mean accuracy, micro F1, and macro F1 against a small two-epoch pre-training/fine-tuning run.

## Training on a Remote Server

Training can run on a Linux server while the Windows checkout remains where syncs start. `scripts/sync_remote.py` (run from Windows, through WSL's rsync and ssh) sends the code and hands the whole MLflow store back and forth, one writing side at a time; [ADR 0009](docs/adr/0009-store-hand-off-to-a-remote-server.md) has the protocol, what it refuses, and the commands. Commits made on the server come back through GitHub: push them there, and pull them on Windows before the next `code` or `push`.

## Configuration System

A run resolves its hyperparameters in this order and logs the source it used as the
`config_source` param (`override`, the repository-relative path of the file, or `defaults`):

1. **Programmatic override**: the mapping an Optuna trial or `--retrain_best` passes (`config_source = override`).
2. **Task-keyed file** (`--task imputation` only): `datasets/hiperparams/{base_dataset}/{dataset_name}.imputation.json`.
3. **Shared file**: `datasets/hiperparams/{base_dataset}/{dataset_name}.json`, read by both tasks.
4. **Default values**: the `Hyperparameters` defaults when no file exists.

`--lr_scheduler` overrides `LR_SCHEDULER` from any of these. An Optuna study never writes
these files on its own: its running best and final best stay inside its results directory
(`results/{dataset_name}/optuna_{timestamp}/`), and it publishes the winning configuration
only under `--promote_best`, to the task-keyed file for imputation and the shared file for
classification. A promoted file is complete: every key the task uses, held values included,
and `LR_SCHEDULER` set to the schedule the study ran under
([ADR 0005](docs/adr/0005-reduced-optuna-search-for-imputation.md)).

### Example Configuration

```json
{
  "DIM": 128,
  "HIDDEN_DIM": 16,
  "HEADS": 16,
  "LAYERS": 2,
  "DIM_FEED": 32,
  "DROPOUT": 0.2,
  "EPOCHS_PRE": 300,
  "BATCH": 256,
  "LR_PRE": 0.00034,
  "WEIGHT_DECAY_PRE": 0.005,
  "PROB_MASCARA": 0.5,
  "EPOCH_FINE": 150,
  "LR_FINE": 0.001,
  "WEIGHT_DECAY_FINE": 0.0019
}
```

Every key is optional and defaulted, so a file written before a key existed keeps loading. The decode keys below are read only by `--task imputation`, and the fine-tuning keys only by `--task classification`; each run records just the ones it used.

### Hyperparameter Descriptions

#### Model Architecture

- `DIM`: Embedding dimension for features
- `HIDDEN_DIM`: Hidden dimension for numerical MLPs
- `HEADS`: Number of attention heads
- `LAYERS`: Number of Transformer layers
- `DIM_FEED`: Feed-forward network dimension

#### Training Configuration

- `EPOCHS_PRE`: Number of pre-training epochs
- `EPOCH_FINE`: Number of fine-tuning epochs
- `BATCH`: Batch size
- `DROPOUT`: Dropout probability
- `PROB_MASCARA`: Masking probability during pre-training

#### Optimization

- `LR_PRE`: Pre-training learning rate
- `LR_FINE`: Fine-tuning learning rate
- `WEIGHT_DECAY_PRE`: Pre-training weight decay
- `WEIGHT_DECAY_FINE`: Fine-tuning weight decay
- `LR_SCHEDULER`: Learning-rate schedule name, see `--lr_scheduler` (default: `cosine_legacy`)
- `PRETRAIN_OBJECTIVE`: Pre-training objective, see `--pretrain_objective` (default: `embedding`)

#### Imputation (`--task imputation` only)

- `EPOCHS_DECODE`: Number of decode-stage epochs (default: `150`)
- `LR_DECODE`: Decode-stage learning rate (default: `0.001`)
- `DECODE_PATIENCE`: Decode-stage early-stopping patience in epochs, see `--decode_patience` (default: `0`, off)
- `DECODER_HEADS`: `batched` or `per_column`, see `--decoder_heads` (default: `batched`)
- `WEIGHT_DECAY_DECODE`: Decode-stage weight decay (default: `0.0019`)
- `LAMBDA_NUM`: Weight on the numerical reconstruction term, each term already averaged over its own hidden cells (default: `1.0`)
- `EVAL_MASK_RATE`: **Nominal** share of cells hidden for scoring (default: `0.2`). Nominal because the masking helper scales it down by each row's null density and never hides an already-missing cell, so asking for `0.2` hides about 20% of a `_00nan` variant but about 5% of an `_80nan` one. Each run logs the realised share as `impute/masked/realised_rate`. This is the evaluation counterpart of `PROB_MASCARA`, which governs training corruption
- `EVAL_MASK_RATES_EXTRA`: Extra nominal rates to score the test fold at, as diagnostics only (default: `[]`). They reuse the same checkpoint, so each costs one forward pass rather than another training run; the primary rate above is what validation selects on and what ranks folds

## Project Structure

```text
TRIDENT/
├── main.py                    # Main entry point and argument parsing
├── train.py                   # Core training logic and evaluation
├── opt.py                     # Optuna hyperparameter optimization
├── src/
│   ├── embedder.py            # TabularEmbedder implementation
│   ├── transformer.py         # TabularTransformerEncoder
│   ├── models.py              # TridentPretrainer and TridentModel
│   └── utils.py               # Utility functions and preprocessing
└── datasets/
    ├── datasets_raw/          # Original CSV files
    ├── processed_datasets/    # Processed data with splits
    ├── categorical_columns/   # Feature type definitions
    └── generate_splits.py     # Data preprocessing pipeline
```

## The Imputation Task

`--task imputation` keeps pre-training exactly as it is and replaces classifier fine-tuning with a **decode stage**: per-column heads reconstruct the actual value of every hidden cell. A categorical head scores only the column's real categories, so it can never answer `[MASK]`, `[NULL]` or the placeholder a missing cell stringifies to; a numerical head predicts a scalar in scaled space.

```bash
# Imputation on a variant with injected gaps, scored against its complete sibling
uv run main.py --dataset_name credit-g_20nan --task imputation --cv_folds 3 --lr_scheduler cosine

# Add the diagnostic that shows the same gaps through the [NULL] token instead
uv run main.py --dataset_name credit-g_20nan --task imputation --cv_folds 3 --score_null_path
```

The decode stage has no published runs to preserve, so prefer `--lr_scheduler cosine` over the legacy default.

### What gets scored

Two populations, on the test fold only:

- **Self-masked cells** — observed cells hidden for scoring. They exist on every variant, and they rank the fold.
- **Induced-missing cells** — cells a `_XXnan` variant is missing whose true value the row-aligned `_00nan` sibling holds. They are the real imputation benchmark and the headline number wherever the sibling exists, and they reach the model as `[MASK]`, the token the decoder was trained to fill.

Metrics per population: `rmse_num_z` and `mae_num_z` pooled over numerical cells in scaled space, `acc_cat` pooled over categorical cells, `macro_f1_cat` averaged per column, and the counts of each. Folds are ranked by

```
impute_score = w_num * (rmse_num_z / mean-imputation rmse) + w_cat * (err_cat / mode-imputation err)
```

with the baselines learned from the **training** fold and applied to the same scored cells, and `w_*` the share of scored cells of each kind. **Lower is better**; `1.0` means no better than filling the column mean or mode. It degrades correctly on the six all-numerical datasets and on all-categorical `kr-vs-kp`.

Beside those numbers, every scored population also carries what four **baseline imputers** scored on exactly the same cells ([ADR 0007](docs/adr/0007-baseline-imputers-scored-on-the-decoders-cells.md)): `baseline/mean_mode/*`, the column mean or mode the score divides by, whose `impute_score` is `1.0` by construction; `baseline/knn5/*` and `baseline/knn10/*`, a k-nearest-neighbours imputer at five and at ten neighbours (uniform weights, numerical columns as scaled, categorical columns one-hot); and `baseline/hgb/*`, one gradient-boosting model per column that fills a hidden cell from the rest of its row. All are fit on the training fold alone. Each carries `rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat` and `impute_score`. The **baseline bar** of a run is the lowest `impute_score` any of them reached on a population, logged as `baseline/best/*` (every statistic of that one baseline) with the tag `best_baseline/<population>` naming it: `cv/test/impute/induced/baseline/best/impute_score/mean` beside `cv/test/impute/induced/impute_score/mean` says whether the model clears every baseline and not only the naive one. The same comparison comes as one number to sort or minimise: `cv/test/impute/<population>/gap_to_best_baseline/impute_score/*` is the model's score minus the bar's, fold by fold (negative where the model beats it, the interval a paired one), and `gap_to_best_baseline_pct/impute_score/*` is that gap in percent of the bar's cross-validated mean, so its mean equals `100 * (model mean - bar mean) / bar mean`. Both live on cross-validated parents only. `metrics/per_column_imputation.csv` carries every baseline's errors per column beside the model's.

### Artifacts

| File | What it holds |
|---|---|
| `imputation_fold_N_preview.md` | A fixed-seed sample of rows, each as three lines: what was true, what the model saw, what it filled in. Original units, three decimals, `*` on any number the display had to shorten. A truth comes from the frame before scaling, so it is exact wherever the dataset carries three decimals or fewer, and marked where it carries more (`electricity`, `biodeg`); an imputation is the model's float32 output, so it is nearly always marked |
| `imputation_fold_N_cells.csv` | Every scored cell, a superset of the preview, with both scalings, the model's confidence and an `in_preview` flag |
| `metrics/per_column_imputation.csv` | Per-column errors for every fold, long form, on the parent run |

Every fold writes its files; only the `best_fold` and `worst_fold` children upload them, exactly as loss plots already behave.

### Tuning it with Optuna

`--use_optuna --task imputation` samples the `reduced` search-space profile by default ([ADR 0005](docs/adr/0005-reduced-optuna-search-for-imputation.md)): `PROB_MASCARA` (0.2 to 0.6), `LR_DECODE` (1e-4 to 1e-2), `WEIGHT_DECAY_DECODE` (1e-5 to 1e-2), `DROPOUT` (0.1 to 0.5) and, only when the table has both column types, `LAMBDA_NUM` (0.1 to 10). Every other hyperparameter is held at the task default, so a trial costs what the promoted configuration will cost to run; `--search_space full` samples every knob instead. Trials are ranked on `validation/impute/masked/impute_score`, computed on the validation split's fixed mask, so the test split never chooses hyperparameters; `--search_objective induced` ranks them on `validation/impute/induced/impute_score`, the validation rows' own gaps, instead ([ADR 0008](docs/adr/0008-selectable-search-objective-population.md)). Every trial on a variant with gaps logs both. The study ranks the sampled knobs with fANOVA (`optuna/importance/<knob>` on the study run, `importance.json` beside it), and `--promote_best` writes the winner to `datasets/hiperparams/<base>/<dataset>.imputation.json`, which later `--task imputation` runs read first (see "Configuration System"). `imputation_studies.ps1` runs the six studies of ADR 0005 and, with `-Compare`, the paired 5-fold comparison of promoted against default configurations.

## Training Methodology

### Pre-training Phase

1. **Dynamic Masking**: Randomly masks features with probability adjusted by existing null density
2. **Embedding Reconstruction**: Predicts the original embeddings for masked positions
3. **Self-Supervised Learning**: Learns the structure of the data without labels

### Fine-tuning Phase

1. **Pre-trained Initialization**: The model is initialized with pre-trained weights and fine-tuned end-to-end
2. **Classification Head**: Learns task-specific predictions from the `[CLS]` representation
3. **Class Balancing**: Optional class weights for imbalanced datasets

## Evaluation Metrics

The framework reports standard classification metrics, including:

- **Accuracy**
- **F1-score** (micro and macro)
- **Precision** (micro and macro)
- **Recall** (micro and macro)
- **Confusion Matrix** for detailed error analysis in binary tasks

#### Cross-Validation Summarization
When cross-validation is enabled (`--cv_folds <K>`), individual metrics are computed for each fold:
- The generated `metrics.csv` files (saved under both the run results directory and the root `metrics/` folder) contain only the raw per-fold metrics (one row per fold) to facilitate programmatic loading and parsing.
- A summary containing the `mean` and standard deviation (`mean ± std`) is computed and printed as a confidence interval table directly to the console at the end of the run.
- You can also programmatically compute these summary stats using the `compute_cv_summary(df_or_path)` function in `train.py`.

#### MLflow Cross-Validation Comparisons

Each training invocation is a top-level MLflow run. Filter `tags.run_role = parent` and `tags.task` (`classification` or `imputation`) before comparing executions, and compare runs that share `tags.lr_scheduler`: the schedule changes results, so it is tagged on every parent, diagnostic child and Optuna run. Runs recorded before the tag existed were backfilled with `lr_scheduler = cosine_legacy` and also carry `lr_scheduler_backfilled = true` ([ADR 0003](docs/adr/0003-selectable-learning-rate-schedule.md); re-run `scripts/backfill_lr_scheduler_tag.py --apply` against any other tracking store). For a cross-validation run, compare `cv/test/f1_macro/mean` with `cv/test/f1_macro/ci95_lower`, `cv/test/f1_macro/ci95_upper`, and `cv/test/f1_macro/std`; the same `mean`, `ci95_lower`, `ci95_upper`, `std`, `min`, `max`, and `fold_count` fields are logged under `cv/test/<metric>/...` for every numeric final fold metric.

The 95% bounds are internal CV uncertainty, not an independent-test guarantee: the folds share training data. Parent loss charts use the readable mean/lower/upper bands `cv/pretrain/train_loss/{mean,ci95_lower,ci95_upper}`, `cv/pretrain/val_loss/{mean,ci95_lower,ci95_upper}`, `cv/finetune/train_loss/{mean,ci95_lower,ci95_upper}`, and `cv/finetune/val_loss/{mean,ci95_lower,ci95_upper}`.

Open `best_fold`/`worst_fold` children only for diagnosis. They retain raw fold histories and optional plot/model artifacts; a tied best/worst selection produces one `best_and_worst` child. The parent remains the comparison record and retains `metrics/raw_fold_metrics.csv`, `metrics/cv_summary.json`, `tracking/diagnostic_manifest.json`, and `data/provenance.json`. MLflow input lineage records the prepared dataset and its processed CSV source. Encoding, scaling, split strategy, fold count, seed, and the prepared schema are recorded in `data/provenance.json`.

A predefined single split is one top-level `parent` run with its raw histories, final metrics, lineage, and optional artifacts logged directly there. It creates no nested `single_split` run and no CV summary. With `--disable_mlflow`, tracking remains inert and no MLflow calls are made.

Diagnostic children also log `pretrain/learning_rate` and `finetune/learning_rate` per epoch so the schedule that actually ran can be inspected.

Every fold records wall-clock stage timings as step-less metrics: `time/pretrain_seconds`, `time/finetune_seconds`, and `time/total_seconds` (their sum). They cover the two training stages only, not data preparation, plotting, or artifact writes. A cross-validation parent summarizes them as `cv/time/<name>/{mean,ci95_lower,ci95_upper,std,min,max,fold_count}` and logs `time/training_seconds`, the sum over folds; a single-split parent carries the raw `time/*` values and the same `time/training_seconds`, so one column sorts every parent by training cost. `metrics/raw_fold_metrics.csv` and `metrics/cv_summary.json` keep every fold's timing, not only the replayed diagnostic children. Timings never enter `metrics.csv` or the regression fixture. Every parent is tagged `device`, `gpu_name`, `torch_version`, and `cuda_version`; compare timings only between runs that share them.

Optuna studies live in the same experiment without crowding the comparison table. The study is one top-level run tagged `run_role = optuna_study`, with `optuna/best_objective_value`, `optuna/best_trial_number`, and `best_<param>` params. Each trial is one nested run under it tagged `run_role = optuna_trial` and `trial_number`, and the trainer logs into that run rather than opening a second top-level run: the full parameter set, the same dataset, schedule, and environment tags as a parent, the final metrics (`cv/test/*` and `cv/time/*`, or `test/*` and `time/*` for the predefined split that trials use), `time/training_seconds`, `optuna/objective_value`, and `trial_status`. The winner is tagged `best_trial = true` once the study finishes. Trials deliberately skip loss histories, artifacts, dataset lineage, and diagnostic children; `--retrain_best` records the chosen configuration as a normal `run_role = parent` run tagged `optuna_study_run_id`. The study run and its trials also carry `tags.search_space` (`full` or `reduced`, [ADR 0005](docs/adr/0005-reduced-optuna-search-for-imputation.md)), and the study logs one `optuna/importance/<knob>` metric per sampled knob (fANOVA) with an `importance.json` artifact. Both run kinds carry `tags.search_objective`, the metric key the study ranked by. For `--task imputation` the objective, and so `optuna/best_objective_value`, is `validation/impute/masked/impute_score` unless the study asked for `induced`, a validation-split score that is not comparable with any `test/` metric; classification's objective stays `f1_macro` on the test split. Every parent and trial run logs a `config_source` param (`defaults`, `override`, or the configuration file it loaded), so tuned and default runs are told apart without a tag.

Two tags identify a run's kind, and both sit on every run kind so either can be filtered on safely. `tags.task` says which task trained; runs recorded before the task existed are backfilled as `classification` and also carry `tags.task_backfilled = true`. `tags.is_optuna` is `true` only on an Optuna study parent and its trials, so `tags.is_optuna = 'false'` selects the runs meant for comparison; a `--retrain_best` run stays `false` and links back through `optuna_study_run_id`. It needs no backfilled marker, because the store itself proves the value. Both are stamped by `scripts/backfill_run_tags.py` ([ADR 0004](docs/adr/0004-imputation-decoder-task.md)).

Every run in every `TRIDENT/<base_dataset>` family also has a **mirror run** in `TRIDENT/mirror/<task>` ([ADR 0006](docs/adr/0006-mirror-experiments-per-task.md)), so one task can be read across every dataset in one table. A mirror run carries its source's params, tags, complete metric histories, dataset lineage, run name, status and timestamps, but no artifacts; `mlflow.parentRunId` is rewritten to the mirror of the parent, so the UI nests mirrors as it nests sources. Two tags tell the sides apart: `tags.is_mirror` is dense (`true` on every mirror run, `false` on every other run, backfilled onto the runs that predate it), and `tags.source_run_id` on a mirror names the run it copies. **Any query that spans experiments (`search_all_experiments=True`) must add `tags.is_mirror = 'false'`**, or every source run is counted twice; inside one family or one mirror experiment the filters above are unchanged. The trainer mirrors a run tree when its root closes normally (a comparison parent with its diagnostic children; an Optuna study with its trials); `--disable_mirror` turns that off, and a mirroring failure is a warning, never a failed training. `scripts/mirror_runs.py` (a dry run by default, `--apply` to write) backfills the history, repairs whatever the live path missed, refreshes mirrors whose source changed, and deletes mirrors whose source is gone; it is an upsert, so running it twice is the same as once.

An imputation parent is named `impute_<dataset>_<timestamp>` and reports `cv/test/impute/masked/impute_score/mean` alongside `cv/test/impute/induced/...`; lower is better.

The tracking layout is specified in [ADR 0002](docs/adr/0002-curated-cross-validation-mlflow-runs.md). Deployable logged-model lifecycle work is intentionally deferred to [Ticket 0002](docs/tickets/0002-mlflow-logged-model-lifecycle.md).

## Example Workflows

### Basic Training

```bash
# Train on a clean dataset
uv run main.py --dataset_name spambase_00nan

# Train on a dataset with 20% missing values
uv run main.py --dataset_name spambase_20nan --plot_losses
```

### Batch Training

```bash
# Run all datasets with 3-fold cross-validation, no MLflow
uv run main.py --all --cv_folds 3 --disable_mlflow

# Run first 3 datasets at 60% missingness
uv run main.py --all --limit 3 --nan_level 60 --cv_folds 5
```

### Hyperparameter Optimization

```bash
# Quick optimization (50 trials)
uv run main.py --dataset_name credit-g_00nan --use_optuna --n_trials 50

# Thorough optimization with retraining
uv run main.py --dataset_name vehicle_40nan --use_optuna --n_trials 200 --retrain_best

# Publish the winner so later runs of the same task and variant pick it up
uv run main.py --dataset_name credit-g_20nan --task imputation --use_optuna --n_trials 40 --lr_scheduler cosine --promote_best
```

### Custom Configuration

```bash
# Use a custom target column
uv run main.py --dataset_name custom_data --label_column target

# Save results to a specific directory
uv run main.py --dataset_name biodeg_00nan --output_dir ./experiments/biodeg
```

## Supported Datasets

TRIDENT has been evaluated on a variety of tabular classification benchmarks, including:

- **Vehicle Classification**: Multi-class vehicle type recognition
- **Credit Risk Assessment**: Binary credit approval prediction
- **Spam Detection**: Binary email spam classification
- **Biodegradation**: Binary molecular biodegradability prediction
- **Letter Recognition**: Multi-class character recognition
- **Electrical Grid Stability**: Binary stability prediction

## Citation

If you use this repository, the TRIDENT method, or results derived from this implementation, please cite the paper below.

### Plain Text Citation

Rigueira, P.B. et al. (2026). *TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens*. In: de Freitas, R., Furtado, D. (eds) *Intelligent Systems*. BRACIS 2025. Lecture Notes in Computer Science, vol 16180. Springer, Cham. https://doi.org/10.1007/978-3-032-15984-7_39

### BibTeX

```bibtex
@incollection{rigueira2026trident,
  author    = {Rigueira, P. B. and others},
  title     = {TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens},
  booktitle = {Intelligent Systems},
  editor    = {de Freitas, R. and Furtado, D.},
  series    = {Lecture Notes in Computer Science},
  volume    = {16180},
  year      = {2026},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-032-15984-7_39},
  url       = {https://doi.org/10.1007/978-3-032-15984-7_39}
}
```

## Paper Link

- Springer page: [https://link.springer.com/chapter/10.1007/978-3-032-15984-7_39](https://link.springer.com/chapter/10.1007/978-3-032-15984-7_39)
- DOI: [https://doi.org/10.1007/978-3-032-15984-7_39](https://doi.org/10.1007/978-3-032-15984-7_39)

## Notes

- This README documents the method, repository structure, training flow, and publication metadata currently available from the paper information provided.
- If you later want, the README can also be extended with sections such as **Results**, **Reproducibility**, **Environment Setup**, **Dataset Sources**, or **Acknowledgements**.
