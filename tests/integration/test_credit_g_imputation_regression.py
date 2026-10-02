"""Opt-in regression baseline for a short credit-g imputation run.

`credit-g_20nan` is the only variant that exercises both column types and both scored
populations: 13 categorical and 7 numerical columns, with a row-aligned `_00nan` sibling
holding the truth of every gap. The scores it pins are poor, because two decode epochs
cannot beat mean-and-mode imputation; the fixture pins determinism, not quality.

The run is pinned to the CPU because CUDA dropout masks depend on the GPU model. The dropout
kernel hands random numbers out per thread and caps its threads by the card's SM count, so
above SMs x 6 x 256 x 4 elements two cards draw different masks from the same seed.
Attention dropout here covers 64 x 4 x 21 x 21 = 112,896 elements, over the RTX 3050
Laptop's 98,304, so that card and gorgona8's RTX 3090 Ti end up up to 0.21 apart. On the
CPU both machines agree to 2e-9. The vehicle fixture's 92,416 elements stay under the
limit, which is why that test still runs on the GPU. Evidence: ticket 11, comment of
2026-09-29.

Regenerate on the CPU with the block in that comment of
`docs/tickets/imputation-decoder/issues/11-fixture-and-docs.md`, and only for an
intentional, documented behaviour change (see `AGENTS.md`).
"""

import json
from pathlib import Path

import pytest
import torch

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
def test_credit_g_imputation_matches_regression_fixture(tmp_path, monkeypatch) -> None:
    """A deterministic short imputation run on the CPU stays within the recorded tolerance."""
    # The runner picks its device from this call, so hiding CUDA runs every fold on the CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fixture = json.loads(FIXTURE_PATH.read_text())
    hyperparameters = Hyperparameters.from_mapping(fixture["hyperparameters"])
    # The fixture predates ADR 0013 and stays on the configuration it was recorded with: read
    # without a task, the mapping takes the dataclass defaults, not imputation's new ones.
    assert (hyperparameters.pretraining_objective, hyperparameters.lr_scheduler) == (
        "embedding", "cosine_legacy",
    )
    request = TrainingRequest(
        dataset=DatasetSpec.from_name("credit-g_20nan", "class"),
        hyperparameters=hyperparameters,
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
