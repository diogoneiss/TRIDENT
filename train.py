import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import datetime
import argparse
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from collections import Counter
import mlflow

from src.mlflow_utils import (
    setup_mlflow,
    get_or_create_experiment,
    build_hyperparams_dict,
    build_run_tags,
    build_fold_tags,
)

from src.utils import set_global_seed, preprocess_table
from src.embedder import TabularEmbedder
from src.transformer import TabularTransformerEncoder
from src.models import TridentPretrainer, TridentModel

mlflow.autolog()

def main(args, return_metrics=False):
    """
    Main function for TRIDENT training with pre-training and fine-tuning
    
    Parameters:
    -----------
    args : argparse.Namespace
        Arguments for the training process
    return_metrics : bool, default=False
        If True, return metrics instead of None (for hyperparameter optimization)
    
    Returns:
    --------
    dict or None
        If return_metrics is True, returns a dictionary with evaluation metrics
    """
    # -----------------------------------------------------------------
    # Load data and configurations
    # -----------------------------------------------------------------
    set_global_seed(args.seed)
    
    # Extract the base name of the dataset (before the first underscore, if any)
    base_dataset_name = args.dataset_name.split('_')[0]
    dataset_name = args.dataset_name
    
    # Define paths based on the base dataset name and padding setup
    dataset_path = Path('datasets/processed_datasets') / base_dataset_name / f"{dataset_name}.csv"
    splits_path = Path('datasets/processed_datasets/splits') / f"{base_dataset_name}_split.json"
    cat_cols_path = Path('datasets/categorical_columns') / f"{base_dataset_name}.txt"
    
    # Define path for hyperparameters - sempre usa essa estrutura padrão
    hyperparams_path = Path('datasets/hiperparams') / base_dataset_name / f"{dataset_name}.json"
    
    # Use hyperparameters overrides if provided (for Optuna)
    if hasattr(args, 'hyperparams_override') and args.hyperparams_override is not None:
        print("Using hyperparameters provided by Optuna")
        hyperparams = args.hyperparams_override
        
        # Extract hyperparameters directly from the override dict
        DIM = hyperparams.get('DIM', 128)
        HIDDEN_DIM = hyperparams.get('HIDDEN_DIM', 16)
        HEADS = hyperparams.get('HEADS', 16)
        LAYERS = hyperparams.get('LAYERS', 2)
        DIM_FEED = hyperparams.get('DIM_FEED', 32)
        DROPOUT = hyperparams.get('DROPOUT', 0.2)
        EPOCHS_PRE = hyperparams.get('EPOCHS_PRE', 40)
        BATCH = hyperparams.get('BATCH', 256)
        LR_PRE = hyperparams.get('LR_PRE', 0.00034)
        WEIGHT_DECAY_PRE = hyperparams.get('WEIGHT_DECAY_PRE', 0.005)
        PROB_MASCARA = hyperparams.get('PROB_MASCARA', 0.5)
        EPOCH_FINE = hyperparams.get('EPOCH_FINE', 40)
        LR_FINE = hyperparams.get('LR_FINE', 0.001)
        WEIGHT_DECAY_FINE = hyperparams.get('WEIGHT_DECAY_FINE', 0.0019)
        LABELS = hyperparams.get('LABELS', 4)
    # Load hyperparameters from JSON if it exists
    elif hyperparams_path.exists():
        print(f"Loading hyperparameters from {hyperparams_path}")
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        
        # Extract hyperparameters from JSON
        DIM = hyperparams.get('DIM', 128)
        HIDDEN_DIM = hyperparams.get('HIDDEN_DIM', 16)
        HEADS = hyperparams.get('HEADS', 16)
        LAYERS = hyperparams.get('LAYERS', 2)
        DIM_FEED = hyperparams.get('DIM_FEED', 32)
        DROPOUT = hyperparams.get('DROPOUT', 0.2)
        EPOCHS_PRE = hyperparams.get('EPOCHS_PRE', 40)
        BATCH = hyperparams.get('BATCH', 256)
        LR_PRE = hyperparams.get('LR_PRE', 0.00034)
        WEIGHT_DECAY_PRE = hyperparams.get('WEIGHT_DECAY_PRE', 0.005)
        PROB_MASCARA = hyperparams.get('PROB_MASCARA', 0.5)
        EPOCH_FINE = hyperparams.get('EPOCH_FINE', 40)
        LR_FINE = hyperparams.get('LR_FINE', 0.001)
        WEIGHT_DECAY_FINE = hyperparams.get('WEIGHT_DECAY_FINE', 0.0019)
        LABELS = hyperparams.get('LABELS', 4)
    else:
        print(f"Hyperparameter file not found at {hyperparams_path}. Using default values.")
        # Default values if no JSON was provided
        DIM = 128
        HIDDEN_DIM = 16
        HEADS = 16
        LAYERS = 2
        DIM_FEED = 32
        DROPOUT = 0.2
        EPOCHS_PRE = 40
        #EPOCHS_PRE = 500

        BATCH = 256
        LR_PRE = 0.00034
        WEIGHT_DECAY_PRE = 0.005
        PROB_MASCARA = 0.5
        EPOCH_FINE = 40
        #EPOCH_FINE = 200

        LR_FINE = 0.001
        WEIGHT_DECAY_FINE = 0.0019
        LABELS = 4
    
    # Check if dataset file exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Define target column (default is 'class')
    label_column = args.label_column if args.label_column else 'class'
    
    # Lê colunas categóricas do arquivo - padrão fixo
    categorical_columns = []
    if cat_cols_path.exists() and cat_cols_path.stat().st_size > 0:
        with open(cat_cols_path, 'r') as f:
            content = f.read().strip()
            if content:  # Check if the file is not empty
                categorical_columns = content.split(',')
                print(f"Loaded categorical columns from file: {categorical_columns}")
    else:
        print("Warning: Categorical columns file not found. All columns will be treated as numerical.")
    
    # ===============================================================
    # 0) General configurations
    # ===============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Encode target column
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    # Determine the number of labels automatically
    LABELS = len(label_encoder.classes_)

    # Display label mapping
    print("=== Label Mapping ===")
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{idx} → {cls}")
    print("====================")

    # Determine numerical columns as those that are neither categorical nor label
    all_cols = df.columns.tolist()
    all_cols.remove(label_column)
    numerical_columns = [c for c in all_cols if c not in categorical_columns]
    
    # Normalize numerical attributes
    if numerical_columns:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        scaler = None

    # Configuration for saving results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cria a estrutura de diretórios para resultados:
    # output_dir/dataset_name/timestamp/
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)
    results_dir = dataset_dir / timestamp
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Cria a pasta de plots apenas se o argumento plot_losses for passado
    if args.plot_losses:
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

    # ===============================================================
    # 1) Dataset for PRE-TRAINING (features only, no label)
    # Load or generate splits/folds
    # ===============================================================
    cv_folds = getattr(args, 'cv_folds', None)
    if cv_folds is None:
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            folds = [{
                "train_indices": np.array(splits["train_indices"]),
                "val_indices": np.array(splits["val_indices"]),
                "test_indices": np.array(splits["test_indices"])
            }]
        else:
            raise ValueError(f"Splits file not found: {splits_path}")
    else:
        y_enc = df[label_column].values
        class_counts = Counter(y_enc)
        min_samples = min(class_counts.values()) if len(class_counts) > 0 else 0
        
        if min_samples < cv_folds or len(class_counts) < 2:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
            raw_folds = list(kf.split(df))
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
            raw_folds = list(skf.split(df, y_enc))
            
        folds = []
        for train_val_idx, test_idx in raw_folds:
            val_ratio = 0.1 / (1.0 - 1.0 / cv_folds)
            val_ratio = max(0.01, min(0.5, val_ratio))
            
            y_train_val = y_enc[train_val_idx]
            train_val_class_counts = Counter(y_train_val)
            tv_min_samples = min(train_val_class_counts.values()) if len(train_val_class_counts) > 0 else 0
            
            if tv_min_samples >= 2 and len(train_val_class_counts) >= 2:
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
                    inner_train_idx, inner_val_idx = next(sss.split(df.iloc[train_val_idx], y_train_val))
                    train_idx = train_val_idx[inner_train_idx]
                    val_idx = train_val_idx[inner_val_idx]
                except Exception:
                    ss = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
                    inner_train_idx, inner_val_idx = next(ss.split(df.iloc[train_val_idx]))
                    train_idx = train_val_idx[inner_train_idx]
                    val_idx = train_val_idx[inner_val_idx]
            else:
                ss = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
                inner_train_idx, inner_val_idx = next(ss.split(df.iloc[train_val_idx]))
                train_idx = train_val_idx[inner_train_idx]
                val_idx = train_val_idx[inner_val_idx]
                
            folds.append({
                "train_indices": train_idx,
                "val_indices": val_idx,
                "test_indices": test_idx
            })

    all_fold_metrics = []

    # ===============================================================
    # MLflow — configure tracking and open the parent run
    # ===============================================================
    setup_mlflow()  # Honours MLFLOW_TRACKING_URI env var, falls back to sqlite:///mlflow.db
    experiment_id = get_or_create_experiment(dataset_name)

    # Build the flat hyperparameter dict (available after the if/elif/else block above)
    hyperparameters_mlflow = build_hyperparams_dict(
        DIM=DIM, HIDDEN_DIM=HIDDEN_DIM, HEADS=HEADS, LAYERS=LAYERS,
        DIM_FEED=DIM_FEED, DROPOUT=DROPOUT, EPOCHS_PRE=EPOCHS_PRE,
        BATCH=BATCH, LR_PRE=LR_PRE, WEIGHT_DECAY_PRE=WEIGHT_DECAY_PRE,
        PROB_MASCARA=PROB_MASCARA, EPOCH_FINE=EPOCH_FINE,
        LR_FINE=LR_FINE, WEIGHT_DECAY_FINE=WEIGHT_DECAY_FINE, LABELS=LABELS,
    )

    timestamp_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_run_name = f"train_{dataset_name}_{timestamp_run}"
    parent_tags = build_run_tags(
        dataset_name=dataset_name,
        run_type="train",
        seed=args.seed,
        cv_folds=cv_folds,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
        tags=parent_tags,
    ) as parent_run:
        # Log all hyperparameters once on the parent run
        mlflow.log_params(hyperparameters_mlflow)
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("seed", args.seed)
        if cv_folds is not None:
            mlflow.log_param("cv_folds", cv_folds)

        # ===========================================================
        # Fold Loop (inside the MLflow parent run)
        # ===========================================================
        for fold_idx, fold_split in enumerate(folds):
            # Open a nested (child) run for this fold
            fold_run_name = (
                f"fold_{fold_idx + 1}_of_{cv_folds}"
                if cv_folds is not None
                else "single_split"
            )
            fold_tags = build_fold_tags(fold_idx, cv_folds)
            fold_tags["dataset"] = dataset_name

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=fold_run_name,
                tags=fold_tags,
                nested=True,
            ):
                if cv_folds is not None:
                    print(f"\n==================== Running Fold {fold_idx + 1}/{cv_folds} ====================")

                train_idx = fold_split["train_indices"]
                val_idx   = fold_split["val_indices"]
                test_idx  = fold_split["test_indices"]
                
                df_pretrain = df.drop(columns=[label_column]).copy()

                df_train_orig = df_pretrain.iloc[train_idx].reset_index(drop=True)
                df_val_orig   = df_pretrain.iloc[val_idx].reset_index(drop=True)

                # ===============================================================
                # 2) Pre-training model
                # ===============================================================
                embedder = TabularEmbedder(
                    df=df_pretrain,
                    categorical_columns=categorical_columns,
                    numerical_columns=numerical_columns,
                    dimensao=DIM,
                    hidden_dim=HIDDEN_DIM
                )

                transformer = TabularTransformerEncoder(
                    d_model=embedder.dimensao,
                    nhead=HEADS,
                    num_layers=LAYERS,
                    dim_feedforward=DIM_FEED,
                    dropout=DROPOUT
                )

                pretrain_model = TridentPretrainer(embedder, transformer).to(device)

                # Weight initialization
                def _init(m):
                    if isinstance(m, (nn.Embedding, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)

                pretrain_model.apply(_init)

                # Configure optimizer and scheduler
                optimizer = optim.AdamW(
                    pretrain_model.parameters(), lr=LR_PRE, weight_decay=WEIGHT_DECAY_PRE)
                num_epochs = EPOCHS_PRE
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs)

                batch_size = BATCH

                train_losses_history = []
                val_losses_history = []

                # ===============================================================
                # 3) PRE-TRAINING LOOP with DYNAMIC MASKS
                # ===============================================================
                print("\n=== Starting Pre-Training (new masks each epoch) ===")
                
                for epoch in tqdm(range(num_epochs), desc="Pre train epochs"):
                    # Generate new masks for the entire base each epoch
                    df_train_masked = preprocess_table(
                        df_train_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)
                    df_val_masked = preprocess_table(
                        df_val_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)

                    df_train_size = len(df_train_masked)
                    df_val_size = len(df_val_masked)

                    # Train
                    pretrain_model.train()
                    indices = torch.randperm(df_train_size)
                    train_loss_sum = 0.0
                    train_steps = 0

                    for start in tqdm(range(0, df_train_size, batch_size),
                                      desc="Batch", unit="batch", leave=False):
                        end = start + batch_size
                        batch_idx = indices[start:end]

                        batch_masked = df_train_masked.iloc[batch_idx].reset_index(drop=True)
                        batch_orig   = df_train_orig.iloc[batch_idx].reset_index(drop=True)

                        optimizer.zero_grad()
                        total_loss, _ = pretrain_model(batch_masked, batch_orig)
                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()

                        train_loss_sum += total_loss.item()
                        train_steps += 1

                    avg_train_loss = train_loss_sum / train_steps
                    train_losses_history.append(avg_train_loss)

                    # Validation
                    pretrain_model.eval()
                    with torch.no_grad():
                        val_loss_sum = 0.0
                        val_steps = 0
                        for start in range(0, df_val_size, batch_size):
                            end = start + batch_size
                            batch_masked = df_val_masked.iloc[start:end].reset_index(drop=True)
                            batch_orig   = df_val_orig.iloc[start:end].reset_index(drop=True)

                            total_loss, _ = pretrain_model(batch_masked, batch_orig)
                            val_loss_sum += total_loss.item()
                            val_steps += 1

                        avg_val_loss = val_loss_sum / val_steps
                        val_losses_history.append(avg_val_loss)

                    # Log pre-training losses to MLflow (child run), one point per epoch
                    mlflow.log_metrics(
                        {
                            "pretrain/train_loss": float(avg_train_loss),
                            "pretrain/val_loss": float(avg_val_loss),
                        },
                        step=epoch,
                    )

                # print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

                # Plot loss curves
                if args.plot_losses:
                    plt.figure(figsize=(8, 6))
                    plt.plot(train_losses_history, label='Train Loss')
                    plt.plot(val_losses_history, label='Val Loss')
                    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('MLM Pre-Training')
                    plt.legend()
                    
                    # Save plot to the "plots" folder
                    plot_name = f"pretrain_losses_fold_{fold_idx+1}.png" if cv_folds is not None else "pretrain_losses.png"
                    plt.savefig(plots_dir / plot_name)
                    plt.close()
                    mlflow.log_artifact(str(plots_dir / plot_name), artifact_path="plots")

                # =================================================
                # 4) FINE TUNING (Classification)
                # =================================================
                print("\n=== Starting Fine-Tuning (Classification) ===")
            
                df_train = df.iloc[train_idx].reset_index(drop=True)
                df_val   = df.iloc[val_idx].reset_index(drop=True)
                df_test  = df.iloc[test_idx].reset_index(drop=True)

                y_train = torch.tensor(df_train[label_column].values, dtype=torch.long, device=device)
                y_val   = torch.tensor(df_val[label_column].values,   dtype=torch.long, device=device)
                y_test  = torch.tensor(df_test[label_column].values,  dtype=torch.long, device=device)

                # Preparation of features for fine-tuning
                df_train_features = df_train.drop(columns=[label_column])
                df_val_features   = df_val.drop(columns=[label_column])
                df_test_features  = df_test.drop(columns=[label_column])

                # Pre-processing without masks for fine-tuning (just replaces null values)
                processed_train = preprocess_table(df_train_features, p_base=0.0, fine_tunning=True)
                processed_val   = preprocess_table(df_val_features,   p_base=0.0, fine_tunning=True)
                processed_test  = preprocess_table(df_test_features,  p_base=0.0, fine_tunning=True)

                # Reuse pre-trained model components
                finetune_embedder = pretrain_model.embedder
                finetune_transformer = pretrain_model.transformer

                # Calculate class weights for balancing (optional)
                unique_classes_in_train = np.unique(y_train.cpu().numpy())
                class_weights_np = np.ones(LABELS, dtype=np.float32)
                if len(unique_classes_in_train) > 0:
                    computed_weights = compute_class_weight(
                        class_weight='balanced',
                        classes=unique_classes_in_train,
                        y=y_train.cpu().numpy()
                    )
                    for cls, w in zip(unique_classes_in_train, computed_weights):
                        class_weights_np[cls] = w
                class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

                # Build the classification model
                classifier_model = TridentModel(
                    embedder=finetune_embedder,
                    transformer=finetune_transformer,
                    num_labels=LABELS,
                    class_weights=class_weights
                ).to(device)

                # Optimizer for fine-tuning
                optimizer_ft = optim.AdamW(classifier_model.parameters(), lr=LR_FINE, weight_decay=WEIGHT_DECAY_FINE)
                num_epochs_ft = EPOCH_FINE
                batch_size_ft = BATCH
                scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_ft, T_max=num_epochs_ft)
            
                # Loss history
                ft_train_losses_history = []
                ft_val_losses_history = []

                # Best model state
                best_val_loss = float('inf')
                best_model_state = None

                # Fine-tuning loop
                for epoch in tqdm(range(num_epochs_ft), desc="Fine-tuning epochs"):
                    classifier_model.train()
                    indices = torch.randperm(len(processed_train))
                    train_loss_sum = 0.0
                    train_batch_count = 0

                    # Train
                    for start in tqdm(range(0, len(processed_train), batch_size_ft),
                                      desc=f"FineTune Epoch {epoch+1}/{num_epochs_ft}",
                                      unit="batch", leave=False):
                        end = start + batch_size_ft
                        batch_idx = indices[start:end]

                        batch_df = processed_train.iloc[batch_idx]
                        batch_labels = y_train[batch_idx]

                        logits_batch, loss_batch = classifier_model(batch_df, labels=batch_labels)

                        optimizer_ft.zero_grad()
                        loss_batch.backward()
                        optimizer_ft.step()
                        scheduler_ft.step()
                        
                        train_loss_sum += loss_batch.item()
                        train_batch_count += 1

                    train_loss_epoch = train_loss_sum / train_batch_count
                    ft_train_losses_history.append(train_loss_epoch)

                    # Validation
                    classifier_model.eval()
                    with torch.no_grad():
                        logits_val, val_loss = classifier_model(processed_val, labels=y_val)
                        preds_val = torch.argmax(logits_val, dim=1)

                    val_loss_value = val_loss.item()
                    ft_val_losses_history.append(val_loss_value)

                    # Calculation of validation metrics
                    preds_val_np = preds_val.cpu().numpy()
                    y_val_np = y_val.cpu().numpy()

                    f1_val_micro = f1_score(y_val_np, preds_val_np, average='micro')
                    f1_val_macro = f1_score(y_val_np, preds_val_np, average='macro')

                    # Display validation metrics
                    # print(f"\n[Epoch {epoch+1}/{num_epochs_ft} Summary]")
                    # print(f"  Train Loss = {train_loss_epoch:.4f}")
                    # print(f"  Val   Loss = {val_loss_value:.4f}")
                    # print(f"  Val   F1 (micro) = {f1_val_micro:.4f}")
                    # print(f"  Val   F1 (macro) = {f1_val_macro:.4f}")

                    # Log fine-tuning losses and val metrics to MLflow, one point per epoch
                    mlflow.log_metrics(
                        {
                            "finetune/train_loss": float(train_loss_epoch),
                            "finetune/val_loss": float(val_loss_value),
                            "finetune/val_f1_micro": float(f1_val_micro),
                            "finetune/val_f1_macro": float(f1_val_macro),
                        },
                        step=epoch,
                    )
                    
                    # Store best model
                    if val_loss_value < best_val_loss:
                        best_val_loss = val_loss_value
                        best_model_state = copy.deepcopy(classifier_model.state_dict())

                    # print(f"[Epoch {epoch+1}] train={train_loss_epoch:.4f} "
                    #       f"(best={best_val_loss:.4f}) | val={val_loss_value:.4f}")

                # Load the best model found
                if best_model_state is not None:
                    classifier_model.load_state_dict(best_model_state)

                # Evaluation on test set
                classifier_model.eval()
                with torch.no_grad():
                    logits_test, test_loss = classifier_model(processed_test, labels=y_test)
                    preds_test = torch.argmax(logits_test, dim=1)

                # Calculation of final metrics
                preds_test_np = preds_test.cpu().numpy()
                y_test_np     = y_test.cpu().numpy()

                acc_test = accuracy_score(y_test_np, preds_test_np)
                f1_test_micro = f1_score(y_test_np, preds_test_np, average='micro')
                f1_test_macro = f1_score(y_test_np, preds_test_np, average='macro')
                
                precision_test_micro = precision_score(y_test_np, preds_test_np, average='micro')
                precision_test_macro = precision_score(y_test_np, preds_test_np, average='macro')
                
                recall_test_micro = recall_score(y_test_np, preds_test_np, average='micro')
                recall_test_macro = recall_score(y_test_np, preds_test_np, average='macro')
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test_np, preds_test_np)

                # Display final results
                print(f"\n=== Fold {fold_idx + 1} Test Set Results ===")
                print(f"Overall accuracy: {acc_test:.4f}")
                print(f"F1 micro:         {f1_test_micro:.4f}")
                print(f"F1 macro:         {f1_test_macro:.4f}")
                print(f"Precision micro:  {precision_test_micro:.4f}")
                print(f"Precision macro:  {precision_test_macro:.4f}")
                print(f"Recall micro:     {recall_test_micro:.4f}")
                print(f"Recall macro:     {recall_test_macro:.4f}")
                
                # Show confusion matrix for two classes
                if len(label_encoder.classes_) == 2:
                    print("\n--- Confusion Matrix ---")
                    print(conf_matrix)
                    print("------------------------")

                # Create metrics dictionary for the fold
                fold_metrics = {
                    'fold': fold_idx + 1 if cv_folds is not None else 'single_split',
                    'dataset': dataset_name,
                    'accuracy': acc_test,
                    'f1_micro': f1_test_micro,
                    'f1_macro': f1_test_macro,
                    'precision_micro': precision_test_micro,
                    'precision_macro': precision_test_macro,
                    'recall_micro': recall_test_micro,
                    'recall_macro': recall_test_macro
                }
                
                # Add confusion matrix for two classes
                if len(label_encoder.classes_) == 2:
                    fold_metrics['confusion_matrix_tn'] = conf_matrix[0, 0]
                    fold_metrics['confusion_matrix_fp'] = conf_matrix[0, 1]
                    fold_metrics['confusion_matrix_fn'] = conf_matrix[1, 0]
                    fold_metrics['confusion_matrix_tp'] = conf_matrix[1, 1]
                    
                all_fold_metrics.append(fold_metrics)

                # Log final test metrics for this fold to its MLflow child run
                test_metrics_mlflow = {
                    "test/accuracy": float(acc_test),
                    "test/f1_micro": float(f1_test_micro),
                    "test/f1_macro": float(f1_test_macro),
                    "test/precision_micro": float(precision_test_micro),
                    "test/precision_macro": float(precision_test_macro),
                    "test/recall_micro": float(recall_test_micro),
                    "test/recall_macro": float(recall_test_macro),
                    "test/loss": float(test_loss.item()),
                }
                if len(label_encoder.classes_) == 2:
                    test_metrics_mlflow["test/confusion_matrix_tn"] = int(conf_matrix[0, 0])
                    test_metrics_mlflow["test/confusion_matrix_fp"] = int(conf_matrix[0, 1])
                    test_metrics_mlflow["test/confusion_matrix_fn"] = int(conf_matrix[1, 0])
                    test_metrics_mlflow["test/confusion_matrix_tp"] = int(conf_matrix[1, 1])
                mlflow.log_metrics(test_metrics_mlflow)

                # Plot and save fine-tuning loss curves
                if args.plot_losses:
                    plt.figure(figsize=(8, 6))
                    plt.plot(ft_train_losses_history, label='Train Loss')
                    plt.plot(ft_val_losses_history, label='Val Loss')
                    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Fine-Tuning')
                    plt.legend()
                    
                    # Save the plot to the "plots" folder
                    plot_name = f"finetune_losses_fold_{fold_idx+1}.png" if cv_folds is not None else "finetune_losses.png"
                    plt.savefig(plots_dir / plot_name)
                    plt.close()
                    mlflow.log_artifact(str(plots_dir / plot_name), artifact_path="plots")

                # Save the final model if specified
                if args.save_model:
                    model_name = f"final_model_fold_{fold_idx+1}.pt" if cv_folds is not None else "final_model.pt"
                    model_path = results_dir / model_name
                    torch.save(classifier_model, model_path)
                    print(f"Final model saved to: {model_path}")
                    mlflow.log_artifact(str(model_path), artifact_path="models")

                # ---- end of per-fold MLflow child run (context manager closes it here) ----

        # ---- end of MLflow parent run (folds are done) ----

        # ===========================================================
        # Save results for the entire run
        # ===========================================================
        # Save the hyperparameters used (sem timestamp no nome do arquivo)
        hyperparameters = {
            'DIM': DIM,
            'HIDDEN_DIM': HIDDEN_DIM,
            'HEADS': HEADS,
            'LAYERS': LAYERS,
            'DIM_FEED': DIM_FEED,
            'DROPOUT': DROPOUT,
            'EPOCHS_PRE': EPOCHS_PRE,
            'BATCH': BATCH,
            'LR_PRE': LR_PRE,
            'WEIGHT_DECAY_PRE': WEIGHT_DECAY_PRE,
            'PROB_MASCARA': PROB_MASCARA,
            'EPOCH_FINE': EPOCH_FINE,
            'LR_FINE': LR_FINE,
            'WEIGHT_DECAY_FINE': WEIGHT_DECAY_FINE,
            'LABELS': LABELS
        }
        
        # Save hyperparameters to JSON
        json_path = results_dir / "hyperparameters.json"
        with open(json_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        print(f"Hyperparameters saved to: {json_path}")
        
        # Create metrics dataframe
        df_metrics = pd.DataFrame(all_fold_metrics)
        
        # Save metrics to CSV in the results directory (raw folds only!)
        csv_path = results_dir / "metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        print(f"Metrics saved to: {csv_path}")

        # Save metrics to project-level metrics directory (raw folds only!)
        # TODO improve this later saving somewhere in output folder, like a latest
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True, parents=True)
        root_csv_path = metrics_dir / f"{dataset_name}_metrics.csv"
        df_metrics.to_csv(root_csv_path, index=False)
        print(f"Metrics also saved to: {root_csv_path}")
        
        # If cross-validation was run, calculate and print statistics
        if cv_folds is not None and len(folds) > 1:
            summary = compute_cv_summary(df_metrics)
            
            # Print confidence interval summary table to terminal
            print("\n=== Cross-Validation Results Summary ===")
            for col, stats in summary.items():
                print(f"{col:<20}: {stats['mean_std']}")
            print("=========================================\n")
            
        # Return metrics if requested (for Optuna)
        if return_metrics:
            if cv_folds is not None and len(folds) > 1:
                summary = compute_cv_summary(df_metrics)
                mean_metrics = {'dataset': dataset_name}
                for col, stats in summary.items():
                    mean_metrics[col] = stats['mean']
                return mean_metrics
            else:
                return all_fold_metrics[0]
        return None


    return None


def compute_cv_summary(df_or_path):
    """
    Computes summary statistics (mean, std, confidence interval) from cross-validation fold metrics.

    Parameters:
    -----------
    df_or_path : pd.DataFrame or str or Path
        The DataFrame or path to the CSV file containing the fold metrics.

    Returns:
    --------
    dict
        A dictionary containing the summary statistics (mean, std, and mean_std string) for each metric.
    """
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # Exclude non-numeric columns like fold and dataset
    numeric_cols = [col for col in df.columns if col not in ["fold", "dataset"]]

    # Calculate statistics
    mean_series = df[numeric_cols].mean()
    std_series = df[numeric_cols].std()

    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": mean_series[col],
            "std": std_series[col],
            "mean_std": f"{mean_series[col]:.4f} ± {std_series[col]:.4f}",
        }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRIDENT: Training and evaluation of the model')
    
    # Required parameters (now just the dataset name)
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (without the .csv extension)')
    
    # Optional parameters
    parser.add_argument('--label_column', type=str, default=None,
                        help='Name of the label/target column in the dataset (default: "class")')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results and models')
    parser.add_argument('--plot_losses', action='store_true',
                        help='Generate loss plots during training')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the final model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generation')
    parser.add_argument('--cv_folds', type=int, default=None,
                        help='Number of folds for cross-validation (default: None, uses pre-defined split)')
    
    args = parser.parse_args()
    main(args)