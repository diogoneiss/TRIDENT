"""Dataset preparation and fold construction for the training workflow."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .types import DatasetSpec, FoldSplit, PreparedDataset


def prepare_dataset(spec: DatasetSpec) -> PreparedDataset:
    """Load, encode, and scale a dataset before constructing its folds."""
    dataset_path = (
        Path("datasets/processed_datasets")
        / spec.base_dataset_name
        / f"{spec.dataset_name}.csv"
    )
    splits_path = (
        Path("datasets/processed_datasets/splits") / f"{spec.base_dataset_name}_split.json"
    )
    categorical_columns_path = (
        Path("datasets/categorical_columns") / f"{spec.base_dataset_name}.txt"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    categorical_columns: list[str] = []
    if categorical_columns_path.exists() and categorical_columns_path.stat().st_size > 0:
        content = categorical_columns_path.read_text().strip()
        if content:
            categorical_columns = content.split(",")
            print(f"Loaded categorical columns from file: {categorical_columns}")
    else:
        print("Warning: Categorical columns file not found. All columns will be treated as numerical.")

    label_encoder = LabelEncoder()
    frame[spec.label_column] = label_encoder.fit_transform(frame[spec.label_column])
    print("=== Label Mapping ===")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{index} → {label}")
    print("====================")

    feature_columns = frame.columns.tolist()
    feature_columns.remove(spec.label_column)
    numerical_columns = [column for column in feature_columns if column not in categorical_columns]
    if numerical_columns:
        scaler = StandardScaler()
        frame[numerical_columns] = scaler.fit_transform(frame[numerical_columns])

    frame.attrs["dataset_name"] = spec.dataset_name
    frame.attrs["splits_path"] = splits_path
    return PreparedDataset(
        frame=frame,
        label_column=spec.label_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        label_classes=label_encoder.classes_,
    )


def build_folds(
    frame: pd.DataFrame, label_column: str, cv_folds: int | None, seed: int
) -> list[FoldSplit]:
    """Construct the legacy predefined or cross-validation fold splits."""
    if cv_folds is None:
        splits_path = frame.attrs.get("splits_path")
        if splits_path is None:
            raise ValueError("Predefined splits require frame.attrs['splits_path'].")
        with Path(splits_path).open() as stream:
            splits = json.load(stream)
        return [
            FoldSplit(
                train_indices=np.array(splits["train_indices"]),
                validation_indices=np.array(splits["val_indices"]),
                test_indices=np.array(splits["test_indices"]),
            )
        ]

    encoded_labels = frame[label_column].values
    class_counts = Counter(encoded_labels)
    min_samples = min(class_counts.values()) if class_counts else 0
    if min_samples < cv_folds or len(class_counts) < 2:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        raw_folds = list(splitter.split(frame))
    else:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        raw_folds = list(splitter.split(frame, encoded_labels))

    folds: list[FoldSplit] = []
    for train_validation_indices, test_indices in raw_folds:
        validation_ratio = 0.1 / (1.0 - 1.0 / cv_folds)
        validation_ratio = max(0.01, min(0.5, validation_ratio))

        train_validation_labels = encoded_labels[train_validation_indices]
        train_validation_class_counts = Counter(train_validation_labels)
        minimum_train_validation_samples = (
            min(train_validation_class_counts.values()) if train_validation_class_counts else 0
        )
        if minimum_train_validation_samples >= 2 and len(train_validation_class_counts) >= 2:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=validation_ratio, random_state=seed
                )
                train_local_indices, validation_local_indices = next(
                    splitter.split(frame.iloc[train_validation_indices], train_validation_labels)
                )
            except Exception:
                splitter = ShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=seed)
                train_local_indices, validation_local_indices = next(
                    splitter.split(frame.iloc[train_validation_indices])
                )
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=seed)
            train_local_indices, validation_local_indices = next(
                splitter.split(frame.iloc[train_validation_indices])
            )

        folds.append(
            FoldSplit(
                train_indices=train_validation_indices[train_local_indices],
                validation_indices=train_validation_indices[validation_local_indices],
                test_indices=test_indices,
            )
        )
    return folds
