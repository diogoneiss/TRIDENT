import pandas as pd

from src.training.data import build_folds


def test_build_folds_creates_two_disjoint_test_splits() -> None:
    frame = pd.DataFrame({"feature": range(12), "class": [0, 1] * 6})

    folds = build_folds(frame, label_column="class", cv_folds=2, seed=42)

    assert len(folds) == 2
    assert set(folds[0].test_indices).isdisjoint(set(folds[1].test_indices))
