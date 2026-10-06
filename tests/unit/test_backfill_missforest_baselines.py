"""Backfilling the missForest baselines of ADR 0016 onto runs recorded before them."""

from pathlib import Path
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import MISSFOREST_BACKFILLED_TAG
from src.training.mirroring import mirror_run_tree

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_missforest_baselines as backfill  # noqa: E402

INDUCED = "impute/induced"


def _stats(prefix: str, folds: list[float]) -> dict[str, float]:
    """The parent statistics a live summary logs, enough of them for the plan to read."""
    mean = sum(folds) / len(folds)
    return {f"cv/test/{prefix}/mean": mean, f"cv/test/{prefix}/min": min(folds), f"cv/test/{prefix}/max": max(folds)}


def _logged() -> dict[str, float]:
    """A run of two folds: model 0.9 and 0.7, mean/mode at parity, hgb 0.8 and 1.0 as its
    best baseline, and the gap to it (-0.1) already logged."""
    return {
        **_stats(f"{INDUCED}/impute_score", [0.9, 0.7]),
        **_stats(f"{INDUCED}/baseline/mean_mode/impute_score", [1.0, 1.0]),
        **_stats(f"{INDUCED}/baseline/hgb/impute_score", [0.8, 1.0]),
        **_stats(f"{INDUCED}/baseline/best/impute_score", [0.8, 1.0]),
        **_stats(f"{INDUCED}/gap_to_best_baseline/impute_score", [0.1, -0.3]),
    }


TAGS = {"best_baseline/impute/induced": "hgb"}
# The model's and hgb's fold values, as the run's raw_fold_metrics.csv keeps them.
ROWS = {
    1: {f"{INDUCED}/impute_score": 0.9, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0,
        f"{INDUCED}/baseline/hgb/impute_score": 0.8},
    2: {f"{INDUCED}/impute_score": 0.7, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0,
        f"{INDUCED}/baseline/hgb/impute_score": 1.0},
}


def _recomputed(forest: tuple[float, float], lightgbm: tuple[float, float], naive: float = 1.0):
    return {
        fold: {
            f"{INDUCED}/baseline/mean_mode/impute_score": naive,
            f"{INDUCED}/baseline/missforest/impute_score": forest[fold - 1],
            f"{INDUCED}/baseline/missforest_lgbm/impute_score": lightgbm[fold - 1],
        }
        for fold in (1, 2)
    }


def test_where_missforest_beats_the_bar_the_best_baseline_and_the_gap_move_with_it() -> None:
    """Worked by hand. missForest scores 0.7 and 0.8 (mean 0.75), below hgb's 0.9, so it
    becomes the best baseline: every statistic under ``baseline/best`` becomes its own,
    the tag names it, and the gap is taken again fold by fold against it: 0.9 - 0.7 =
    0.2 and 0.7 - 0.8 = -0.1, mean 0.05, which is 6.67% of 0.75. Kept at hgb's -0.1, the
    run would claim the model beats a bar it no longer clears."""
    plan = backfill.plan_missforest(
        "run", "impute_toy_20nan_x", _logged(), TAGS, _recomputed((0.7, 0.8), (0.95, 0.95)), {"child": 2}, ROWS
    )

    assert plan.refused is None
    assert plan.moved == {INDUCED: ("hgb", "missforest")}
    assert plan.tags == {"best_baseline/impute/induced": "missforest"}
    parent = plan.parent
    assert parent[f"cv/test/{INDUCED}/baseline/missforest/impute_score/mean"] == pytest.approx(0.75)
    assert parent[f"cv/test/{INDUCED}/baseline/missforest_lgbm/impute_score/mean"] == pytest.approx(0.95)
    assert parent[f"cv/test/{INDUCED}/baseline/best/impute_score/mean"] == pytest.approx(0.75)
    assert parent[f"cv/test/{INDUCED}/baseline/best/impute_score/min"] == pytest.approx(0.7)
    assert parent[f"cv/test/{INDUCED}/gap_to_best_baseline/impute_score/mean"] == pytest.approx(0.05)
    assert parent[f"cv/test/{INDUCED}/gap_to_best_baseline_pct/impute_score/mean"] == pytest.approx(100 * 0.05 / 0.75)
    # The children get the new baselines' fold values, and nothing a live run would not log.
    assert plan.children == {
        "child": {
            f"test/{INDUCED}/baseline/missforest/impute_score": 0.8,
            f"test/{INDUCED}/baseline/missforest_lgbm/impute_score": 0.95,
        }
    }


def test_where_missforest_does_not_beat_the_bar_only_its_own_numbers_are_written() -> None:
    plan = backfill.plan_missforest(
        "run", "impute_toy_20nan_x", _logged(), TAGS, _recomputed((0.95, 0.95), (0.92, 0.92)), {}, ROWS
    )

    assert plan.refused is None
    assert plan.moved == {}
    assert plan.tags == {}
    assert not any("/best/" in key or "gap_to_best_baseline" in key for key in plan.parent)
    assert f"cv/test/{INDUCED}/baseline/missforest_lgbm/impute_score/mean" in plan.parent


def test_a_run_whose_mean_mode_is_not_reproduced_is_refused() -> None:
    """The mean/mode baseline is what says these are the run's own cells and truths; the
    other baselines are not recomputed, so they cannot be checked."""
    plan = backfill.plan_missforest(
        "run", "impute_toy_20nan_x", _logged(), TAGS, _recomputed((0.7, 0.8), (0.9, 0.9), naive=0.99), {}, ROWS
    )

    assert plan.refused is not None and "mean_mode" in plan.refused
    assert plan.parent == {} and plan.tags == {}


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


def test_a_moved_best_baseline_is_what_the_run_and_its_mirror_now_read(client) -> None:
    """The best baseline and the gap are rewritten at the step a live run logs them, 0,
    beside the values logged there days before: MLflow's latest value must be the new
    one on the run and on its mirror, or every comparison would keep reading the old bar
    while the history looked updated. Every run of the tree says it was backfilled."""
    experiment = client.create_experiment("TRIDENT/toy")
    base = {"task": "imputation", "is_mirror": "false"}
    parent = client.create_run(experiment, tags={**base, "run_role": "parent", **TAGS}).info.run_id
    child = client.create_run(
        experiment, tags={**base, "run_role": "best_fold", "fold": "2", "mlflow.parentRunId": parent}
    ).info.run_id
    days_ago = int(time.time() * 1000) - 3 * 86_400_000
    for key, value in _logged().items():
        client.log_metric(parent, key, value, timestamp=days_ago, step=0)
    for run_id in (parent, child):
        client.set_terminated(run_id)
    mirror_run_tree(client, parent)

    plan = backfill.plan_missforest(
        parent, "impute_toy_20nan_x", _logged(), TAGS, _recomputed((0.7, 0.8), (0.95, 0.95)), {child: 2}, ROWS
    )
    backfill.apply_missforest_plan(client, plan)

    best = f"cv/test/{INDUCED}/baseline/best/impute_score/mean"
    gap = f"cv/test/{INDUCED}/gap_to_best_baseline/impute_score/mean"
    mirrors = client.search_runs(
        [client.get_experiment_by_name("TRIDENT/mirror/imputation").experiment_id],
        filter_string=f"tags.source_run_id = '{parent}'",
    )
    for run in (client.get_run(parent), client.get_run(mirrors[0].info.run_id)):
        assert run.data.metrics[best] == pytest.approx(0.75)
        assert run.data.metrics[gap] == pytest.approx(0.05)
        assert run.data.tags["best_baseline/impute/induced"] == "missforest"
        assert run.data.tags[MISSFOREST_BACKFILLED_TAG] == "true"
    assert client.get_run(child).data.tags[MISSFOREST_BACKFILLED_TAG] == "true"
    assert [entry.value for entry in client.get_metric_history(parent, best)] == pytest.approx([0.9, 0.75])


def test_a_pass_cut_short_is_planned_again_and_a_live_run_is_left_alone(client) -> None:
    """The parent's mark is the last write of a tree. A run that predates ADR 0016 and
    carries missForest numbers without it was cut short, so it is planned again from its
    numbers without them; a run that logged its missForests live never needs the pass."""
    experiment = client.create_experiment("TRIDENT/toy")
    before, after = backfill._LIVE_SINCE_MS - 60_000, backfill._LIVE_SINCE_MS + 60_000
    forest = f"cv/test/{INDUCED}/baseline/missforest/impute_score/mean"
    lightgbm = f"cv/test/{INDUCED}/baseline/missforest_lgbm/impute_score/mean"

    def run(start: int, metrics: dict[str, float], tags: dict[str, str] | None = None):
        run_id = client.create_run(experiment, start_time=start, tags=tags or {}).info.run_id
        for key, value in metrics.items():
            client.log_metric(run_id, key, value, step=0)
        return client.get_run(run_id)

    untouched = run(before, _logged())
    cut_short = run(before, {**_logged(), forest: 0.75, lightgbm: 0.95})
    finished = run(before, {**_logged(), forest: 0.75, lightgbm: 0.95}, {MISSFOREST_BACKFILLED_TAG: "true"})
    live = run(after, {**_logged(), forest: 0.75, lightgbm: 0.95})
    no_baselines = run(before, {f"cv/test/{INDUCED}/impute_score/mean": 0.8})

    assert [backfill._pending(entry) for entry in (untouched, cut_short, finished, live, no_baselines)] == [
        True, True, False, False, False,
    ]
    plan = backfill.plan_missforest(
        "run", "impute_toy_20nan_x", backfill._without_missforest(cut_short.data.metrics), TAGS,
        _recomputed((0.7, 0.8), (0.95, 0.95)), {}, ROWS,
    )
    assert plan.refused is None and plan.parent[forest] == pytest.approx(0.75)
