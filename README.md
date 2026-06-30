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

- **TridentClassifier**: Supervised classification model
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

# Hyperparameter optimization
uv run main.py --dataset_name vehicle_00nan --use_optuna --n_trials 100 --retrain_best
```

## Command Line Arguments

### Main Parameters

- `--dataset_name`: Dataset name without the `.csv` extension (**required**)
- `--label_column`: Target column name (default: `class`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--output_dir`: Results directory (default: `results`)

### Training Options

- `--plot_losses`: Generate training loss visualizations
- `--save_model`: Save trained model checkpoints

### Optuna Optimization

- `--use_optuna`: Enable hyperparameter optimization
- `--n_trials`: Number of optimization trials (default: `50`)
- `--retrain_best`: Retrain using the best parameters found during search

## Configuration System

TRIDENT supports three configuration modes:

1. **Automatic Optimization**: Hyperparameter search with Optuna
2. **JSON Configuration**: Load from `datasets/hiperparams/{base_dataset}/{dataset_name}.json`
3. **Default Values**: Fallback configuration when no custom settings exist

### Example Configuration

```json
{
  "DIM": 128,
  "HIDDEN_DIM": 16,
  "HEADS": 16,
  "LAYERS": 2,
  "DIM_FEED": 32,
  "DROPOUT": 0.2,
  "EPOCHS_PRE": 40,
  "BATCH": 256,
  "LR_PRE": 0.00034,
  "WEIGHT_DECAY_PRE": 0.005,
  "PROB_MASCARA": 0.5,
  "EPOCH_FINE": 40,
  "LR_FINE": 0.001,
  "WEIGHT_DECAY_FINE": 0.0019
}
```

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

## Project Structure

```text
TRIDENT/
├── main.py                    # Main entry point and argument parsing
├── train.py                   # Core training logic and evaluation
├── opt.py                     # Optuna hyperparameter optimization
├── src/
│   ├── embedder.py            # TabularEmbedder implementation
│   ├── transformer.py         # TabularTransformerEncoder
│   ├── models.py              # TridentPretrainer and TridentClassifier
│   └── utils.py               # Utility functions and preprocessing
└── datasets/
    ├── datasets_raw/          # Original CSV files
    ├── processed_datasets/    # Processed data with splits
    ├── categorical_columns/   # Feature type definitions
    └── generate_splits.py     # Data preprocessing pipeline
```

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

## Example Workflows

### Basic Training

```bash
# Train on a clean dataset
uv run main.py --dataset_name spambase_00nan

# Train on a dataset with 20% missing values
uv run main.py --dataset_name spambase_20nan --plot_losses
```

### Hyperparameter Optimization

```bash
# Quick optimization (50 trials)
uv run main.py --dataset_name credit-g_00nan --use_optuna --n_trials 50

# Thorough optimization with retraining
uv run main.py --dataset_name vehicle_40nan --use_optuna --n_trials 200 --retrain_best
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
