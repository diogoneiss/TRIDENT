import numpy as np
import pandas as pd

from src.training.finetuning import build_fold_result
from src.training.summary import compute_cv_summary


def test_build_fold_result_retains_classification_metrics() -> None:
    result = build_fold_result(1, "vehicle_00nan", np.array([0, 1]), np.array([0, 1]), 0.2)

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["f1_micro"] == 1.0
    assert result.metrics["f1_macro"] == 1.0


def test_compute_cv_summary_retains_legacy_mean_std_format() -> None:
    summary = compute_cv_summary(
        pd.DataFrame(
            {
                "fold": [1, 2],
                "dataset": ["vehicle_00nan", "vehicle_00nan"],
                "accuracy": [0.5, 1.0],
            }
        )
    )

    assert summary["accuracy"] == {
        "mean": 0.75,
        "std": 0.3535533905932738,
        "mean_std": "0.7500 ± 0.3536",
    }
