import re

import numpy as np
import pandas as pd
import pytest

from src.training import data
from src.training.data import (
    build_folds,
    evaluation_mask,
    load_complete_sibling,
    prepare_dataset,
)
from src.training.types import DatasetSpec


def test_build_folds_creates_two_disjoint_test_splits() -> None:
    frame = pd.DataFrame({"feature": range(12), "class": [0, 1] * 6})

    folds = build_folds(frame, label_column="class", cv_folds=2, seed=42)

    assert len(folds) == 2
    assert set(folds[0].test_indices).isdisjoint(set(folds[1].test_indices))


def test_build_folds_raises_legacy_error_when_predefined_split_is_missing(tmp_path) -> None:
    frame = pd.DataFrame({"feature": range(2), "class": [0, 1]})
    missing_split_path = tmp_path / "missing_split.json"
    frame.attrs["splits_path"] = missing_split_path

    with pytest.raises(ValueError, match=re.escape(f"Splits file not found: {missing_split_path}")):
        build_folds(frame, label_column="class", cv_folds=None, seed=42)


def test_a_prepared_dataset_can_return_its_numbers_to_original_units() -> None:
    """Scaled numbers are unreadable, so a preview has to be able to undo the scaling.

    Preparation scales the numerical columns and, until now, forgot how, leaving nothing
    downstream able to say that a credit amount was 1620 rather than -0.41.
    """
    dataset = prepare_dataset(DatasetSpec.from_name("vehicle_00nan", "class"))
    columns = list(dataset.numerical_columns)

    original_units = dataset.scaler.inverse_transform(dataset.frame[columns].to_numpy())

    on_disk = pd.read_csv(dataset.source_path)[columns].to_numpy()
    assert original_units == pytest.approx(on_disk)


def test_a_variant_with_injected_gaps_can_find_the_table_they_were_cut_from() -> None:
    """Every gap in a _XXnan variant has a known true value in its _00nan sibling.

    That sibling is the only ground truth an imputer can be scored against, and a
    complete variant has none, being the sibling itself.
    """
    from_gapped = load_complete_sibling(DatasetSpec.from_name("credit-g_20nan", "class"))
    from_complete = load_complete_sibling(DatasetSpec.from_name("credit-g_00nan", "class"))

    assert from_gapped is not None
    assert not from_gapped.isna().to_numpy().any()
    assert from_complete is None


def test_a_sibling_that_does_not_line_up_is_refused(tmp_path, monkeypatch) -> None:
    """Ground truth is only ground truth if it describes the very same rows.

    The variants are generated together, so they line up today. Regenerating one of them
    alone would silently score every imputation against the wrong row, which is worse
    than having no sibling at all.
    """
    monkeypatch.setattr(data, "PROCESSED_DATASETS", tmp_path)
    (tmp_path / "toy").mkdir()
    pd.DataFrame({"amount": [1.0, None], "class": [0, 1]}).to_csv(
        tmp_path / "toy" / "toy_20nan.csv", index=False
    )
    pd.DataFrame({"amount": [999.0, 2.0], "class": [0, 1]}).to_csv(
        tmp_path / "toy" / "toy_00nan.csv", index=False
    )

    with pytest.raises(ValueError, match="toy_20nan"):
        load_complete_sibling(DatasetSpec.from_name("toy_20nan", "class"))


def _maskable() -> pd.DataFrame:
    return pd.DataFrame({"amount": [float(index) for index in range(20)], "fee": [1.0] * 20})


def test_scoring_the_same_fold_twice_asks_the_same_question() -> None:
    """Which cells a fold is scored on is fixed, so its numbers can be compared at all.

    Training re-rolls its masks every epoch. Evaluation must not: a validation loss that
    moved because the question changed would make checkpoint selection meaningless, and
    two runs of the same fold would not be comparable.
    """
    frame = _maskable()

    first = evaluation_mask(frame, rate=0.2, seed=42, fold=1)
    again = evaluation_mask(frame, rate=0.2, seed=42, fold=1)
    another_fold = evaluation_mask(frame, rate=0.2, seed=42, fold=2)

    assert first.equals(again)
    assert not first.equals(another_fold)


def test_drawing_an_evaluation_mask_leaves_the_training_draws_alone() -> None:
    """Evaluation must not consume randomness training was going to use.

    Every seeded classification result depends on the exact sequence of draws, which
    AGENTS.md protects. An evaluation mask drawn from the global stream would shift that
    sequence and quietly move every published number.
    """
    frame = _maskable()
    np.random.seed(0)
    undisturbed = np.random.rand(3).tolist()

    np.random.seed(0)
    evaluation_mask(frame, rate=0.2, seed=42, fold=1)
    after_drawing_a_mask = np.random.rand(3).tolist()

    assert after_drawing_a_mask == undisturbed
