"""Opt-in regression baseline for a short credit-g imputation run.

`credit-g_20nan` is the only variant that exercises both column types and both scored
populations: 13 categorical and 7 numerical columns, with a row-aligned `_00nan` sibling
holding the truth of every gap. The scores it pins are poor, because two decode epochs
cannot beat mean-and-mode imputation; the fixture pins determinism, not quality.

Regenerate with the block in `docs/tickets/imputation-decoder/issues/11-fixture-and-docs.md`,
and only for an intentional, documented behaviour change (see `AGENTS.md`).
"""

import json
from pathlib import Path

import pytest

from src.training.runner import run_training
from src.training.types import DatasetSpec, Hyperparameters, RuntimeOptions, TrainingRequest

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "credit-g_20nan_imputation_regression.json"

PINNED = (
    "impute/masked/impute_score",
    "impute/masked/rmse_num_z",
    "impute/masked/acc_cat",
    "impute/induced/impute_score",
    "impute/induced/rmse_num_z",
    "impute/induced/acc_cat",
)


@pytest.mark.integration
def test_credit_g_imputation_matches_regression_fixture(tmp_path) -> None:
    """A deterministic short imputation run stays within the recorded tolerance."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    request = TrainingRequest(
        dataset=DatasetSpec.from_name("credit-g_20nan", "class"),
        hyperparameters=Hyperparameters.from_mapping(fixture["hyperparameters"]),
        runtime=RuntimeOptions(
            output_dir=tmp_path / "results",
            metrics_dir=tmp_path / "metrics",
            tracking_enabled=False,
        ),
        seed=fixture["environment"]["seed"],
        cv_folds=fixture["environment"]["cv_folds"],
        plot_losses=False,
        save_model=False,
        task="imputation",
    )

    result = run_training(request)

    for observed, expected in zip(result.fold_results, fixture["folds"], strict=True):
        for metric in PINNED:
            assert observed.metrics[metric] == pytest.approx(
                expected[metric], abs=fixture["tolerance"]
            )
    for metric in PINNED:
        assert result.mean_metrics[metric] == pytest.approx(
            fixture["mean"][metric], abs=fixture["tolerance"]
        )
