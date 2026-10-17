"""Backfilling ADR 0021 onto runs recorded before it: the linear arms and the strong tier's best."""

import sys
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.mlflow_utils import STRONG_TIER_BACKFILLED_TAG
from src.training.mirroring import mirror_run_tree

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_strong_tier as backfill  # noqa: E402

INDUCED = "impute/induced"


def _stats(prefix: str, folds: list[float]) -> dict[str, float]:
    mean = sum(folds) / len(folds)
    return {f"cv/test/{prefix}/mean": mean, f"cv/test/{prefix}/min": min(folds), f"cv/test/{prefix}/max": max(folds)}


def _logged() -> dict[str, float]:
    """A run of two folds as ADR 0017 left it: model 0.9 and 0.7, mean/mode at parity, hgb its
    bar, missForest 0.7 and 0.8."""
    return {
        **_stats(f"{INDUCED}/impute_score", [0.9, 0.7]),
        **_stats(f"{INDUCED}/baseline/mean_mode/impute_score", [1.0, 1.0]),
        **_stats(f"{INDUCED}/baseline/hgb/impute_score", [0.8, 1.0]),
        **_stats(f"{INDUCED}/baseline/missforest/impute_score", [0.7, 0.8]),
        **_stats(f"{INDUCED}/baseline/best/impute_score", [0.8, 1.0]),
    }


ROWS = {
    1: {f"{INDUCED}/impute_score": 0.9, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0},
    2: {f"{INDUCED}/impute_score": 0.7, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0},
}


def _recomputed(condexp: tuple[float, float], mean_mode: float = 1.0):
    return {
        fold: {
            f"{INDUCED}/baseline/mean_mode/impute_score": mean_mode,
            f"{INDUCED}/baseline/hgb/impute_score": (0.8, 1.0)[fold - 1],
            f"{INDUCED}/baseline/missforest/impute_score": (0.7, 0.8)[fold - 1],
            f"{INDUCED}/baseline/condexp/impute_score": condexp[fold - 1],
            f"{INDUCED}/baseline/softimpute/impute_score": 0.95,
            f"{INDUCED}/baseline/ice/impute_score": 0.9,
        }
        for fold in (1, 2)
    }


def _plan(condexp: tuple[float, float], mean_mode: float = 1.0, children=None, run_id="run", rows=ROWS):
    return backfill.plan_strong_tier(
        run_id, "impute_toy_60nan_x", backfill._as_planned_from(_logged()), _recomputed(condexp, mean_mode),
        children or {}, rows,
    )


def test_the_arms_are_written_and_the_strong_tier_takes_the_lowest_of_them_and_missforest() -> None:
    """condexp 0.5 and 0.6 (mean 0.55) beats missForest (0.75): it is the strong tier's best, and
    the model's gap to it is 0.4 and 0.1, mean 0.25. missForest keeps its own logged numbers."""
    plan = _plan((0.5, 0.6), children={"child": 2})

    assert plan.refused is None
    assert plan.tags == {"best_strong_baseline/impute/induced": "condexp"}
    assert plan.parent[f"cv/test/{INDUCED}/baseline/condexp/impute_score/mean"] == pytest.approx(0.55)
    assert plan.parent[f"cv/test/{INDUCED}/baseline/best_strong/impute_score/mean"] == pytest.approx(0.55)
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_best_strong_baseline/impute_score/mean"] == pytest.approx(0.25)
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_best_strong_baseline_pct/impute_score/mean"] == pytest.approx(100 * 0.25 / 0.55)
    assert not any("/baseline/missforest/" in key or "/baseline/best/" in key for key in plan.parent)
    assert plan.children == {"child": {
        f"test/{INDUCED}/baseline/condexp/impute_score": 0.6,
        f"test/{INDUCED}/baseline/softimpute/impute_score": 0.95,
        f"test/{INDUCED}/baseline/ice/impute_score": 0.9,
    }}


def test_where_the_arms_lose_missforest_is_the_strong_tiers_best() -> None:
    plan = _plan((0.9, 0.9))
    assert plan.tags == {"best_strong_baseline/impute/induced": "missforest"}
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_best_strong_baseline/impute_score/mean"] == pytest.approx(0.05)


def test_a_run_whose_cells_are_not_reproduced_or_whose_folds_are_missing_is_refused() -> None:
    assert _plan((0.5, 0.6), mean_mode=0.99).refused is not None
    assert _plan((0.5, 0.6), rows={}).refused is not None


@pytest.fixture
def client(tmp_path, monkeypatch):
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    try:
        yield MlflowClient(tracking_uri=uri)
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def test_the_strong_tier_is_what_the_run_and_its_mirror_read_and_the_tree_is_marked(client) -> None:
    experiment = client.create_experiment("TRIDENT/toy")
    base = {"task": "imputation", "is_mirror": "false"}
    parent = client.create_run(experiment, tags={**base, "run_role": "parent"}).info.run_id
    child = client.create_run(
        experiment, tags={**base, "run_role": "best_fold", "fold": "2", "mlflow.parentRunId": parent}
    ).info.run_id
    for key, value in _logged().items():
        client.log_metric(parent, key, value, step=0)
    for run_id in (parent, child):
        client.set_terminated(run_id)
    mirror_run_tree(client, parent)

    backfill.apply_strong_tier_plan(client, _plan((0.5, 0.6), children={child: 2}, run_id=parent))

    mirrors = client.search_runs(
        [client.get_experiment_by_name("TRIDENT/mirror/imputation").experiment_id],
        filter_string=f"tags.source_run_id = '{parent}'",
    )
    for run in (client.get_run(parent), client.get_run(mirrors[0].info.run_id)):
        assert run.data.metrics[f"cv/test/{INDUCED}/baseline/best_strong/impute_score/mean"] == pytest.approx(0.55)
        assert run.data.tags["best_strong_baseline/impute/induced"] == "condexp"
        assert run.data.tags[STRONG_TIER_BACKFILLED_TAG] == "true"
    assert client.get_run(child).data.metrics[f"test/{INDUCED}/baseline/condexp/impute_score"] == pytest.approx(0.6)
    assert not backfill._pending(client.get_run(parent))
