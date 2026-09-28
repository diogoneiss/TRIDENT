"""Backfilling the baseline imputers of ADR 0007 onto runs recorded before the current set."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import BASELINES_BACKFILLED_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_baseline_imputers as backfill  # noqa: E402

MASKED = "impute/masked/baseline"


def _folds() -> dict[int, dict[str, float]]:
    """What recomputing two folds gives: mean/mode at parity, KNN at both sizes."""
    return {
        1: {f"{MASKED}/mean_mode/impute_score": 1.0, f"{MASKED}/knn5/impute_score": 0.8, f"{MASKED}/knn10/impute_score": 0.7},
        2: {f"{MASKED}/mean_mode/impute_score": 1.0, f"{MASKED}/knn5/impute_score": 1.0, f"{MASKED}/knn10/impute_score": 0.9},
    }


def test_a_missing_baseline_gets_every_statistic_the_summariser_would_have_logged() -> None:
    """A run from before the amendment logged the k = 5 baseline as ``knn``. Its missing
    baselines are written as a live run would have written them: seven statistics on
    the parent, the fold's own value on each diagnostic child.

    Worked by hand for ``knn10``, folds 0.7 and 0.9: mean 0.8, sd 0.1414, and a 95% margin
    of t(0.975, 1) = 12.7062 times 0.1414 / sqrt(2) = 1.2706.
    """
    logged = {
        f"cv/test/{MASKED}/mean_mode/impute_score/mean": 1.0,
        f"cv/test/{MASKED}/knn/impute_score/mean": 0.9,
    }

    plan = backfill.plan_run("run", "impute_toy_20nan_x", logged, _folds(), {"child": 2})

    assert plan.refused is None
    assert plan.checked == 2
    knn10 = f"cv/test/{MASKED}/knn10/impute_score"
    assert plan.parent[f"{knn10}/mean"] == pytest.approx(0.8)
    assert plan.parent[f"{knn10}/min"] == pytest.approx(0.7)
    assert plan.parent[f"{knn10}/max"] == pytest.approx(0.9)
    assert plan.parent[f"{knn10}/std"] == pytest.approx(0.141421, abs=1e-6)
    assert plan.parent[f"{knn10}/fold_count"] == 2.0
    assert plan.parent[f"{knn10}/ci95_upper"] == pytest.approx(0.8 + 1.270620, abs=1e-5)
    assert plan.parent[f"cv/test/{MASKED}/knn5/impute_score/mean"] == pytest.approx(0.9)
    # Nothing the run already logged is written again, not even under its old name.
    assert not any("/mean_mode/" in key or "/knn/" in key for key in plan.parent)
    assert plan.children == {
        "child": {f"test/{MASKED}/knn5/impute_score": 1.0, f"test/{MASKED}/knn10/impute_score": 0.9}
    }


def test_a_run_whose_recomputed_baseline_disagrees_with_its_own_is_refused() -> None:
    """The recomputation is trusted only where it reproduces what the run logged. The
    electricity run of 2026-09-25 logged a mode accuracy of 0.0 on its induced cells,
    scored before the spelling fix; recomputed, it is 0.145, so nothing is written."""
    logged = {
        "cv/test/impute/induced/baseline/mean_mode/acc_cat/mean": 0.0,
    }
    folds = {
        1: {"impute/induced/baseline/mean_mode/acc_cat": 0.14, "impute/induced/baseline/knn10/acc_cat": 0.19},
        2: {"impute/induced/baseline/mean_mode/acc_cat": 0.15, "impute/induced/baseline/knn10/acc_cat": 0.19},
    }

    plan = backfill.plan_run("run", "impute_electricity_20nan_x", logged, folds, {"child": 1})

    assert plan.refused is not None and "mean_mode/acc_cat" in plan.refused
    assert plan.parent == {} and plan.children == {}


def test_a_run_with_no_baseline_to_reproduce_is_refused() -> None:
    """With nothing logged to check the recomputation against, there is no evidence it
    scored the run's own cells, so it is not written."""
    plan = backfill.plan_run("run", "impute_toy_20nan_x", {"cv/test/impute/masked/impute_score/mean": 0.9}, _folds(), {})

    assert plan.refused is not None
    assert plan.parent == {}


@pytest.fixture
def client(tmp_path, monkeypatch) -> MlflowClient:
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield MlflowClient(tracking_uri=tracking_uri)
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


def test_applying_writes_at_step_zero_marks_the_runs_and_a_second_pass_writes_nothing(client) -> None:
    """The written numbers sit where a live run puts them (step 0, parent and child), both
    runs say they were backfilled, and planning again from what is now logged writes
    nothing, so the script can be rerun safely."""
    experiment = client.create_experiment("TRIDENT/toy")
    parent = client.create_run(experiment, tags={"run_role": "parent"}).info.run_id
    child = client.create_run(
        experiment, tags={"run_role": "best_fold", "fold": "2", "mlflow.parentRunId": parent}
    ).info.run_id
    logged = {
        f"cv/test/{MASKED}/mean_mode/impute_score/mean": 1.0,
        f"cv/test/{MASKED}/knn/impute_score/mean": 0.9,
    }
    for key, value in logged.items():
        client.log_metric(parent, key, value, step=0)

    plan = backfill.plan_run(parent, "impute_toy_20nan_x", logged, _folds(), {child: 2})
    backfill.apply_plan(client, plan)

    parent_run, child_run = client.get_run(parent), client.get_run(child)
    history = client.get_metric_history(parent, f"cv/test/{MASKED}/knn10/impute_score/mean")
    assert [(entry.value, entry.step) for entry in history] == [(pytest.approx(0.8), 0)]
    assert child_run.data.metrics[f"test/{MASKED}/knn10/impute_score"] == pytest.approx(0.9)
    assert parent_run.data.tags[BASELINES_BACKFILLED_TAG] == "true"
    assert child_run.data.tags[BASELINES_BACKFILLED_TAG] == "true"

    again = backfill.plan_run(parent, "impute_toy_20nan_x", parent_run.data.metrics, _folds(), {child: 2})
    assert again.refused is None
    assert again.parent == {} and again.children == {}
