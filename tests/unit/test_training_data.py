import re

import pandas as pd
import pytest

from src.training.data import build_folds


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
