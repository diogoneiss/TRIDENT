"""Backfilling ADR 0017 onto runs recorded before it: ``lgbm`` joins the bar, missForest
leaves it for a gap of its own."""

from pathlib import Path
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import LGBM_BACKFILLED_TAG
from src.training.mirroring import mirror_run_tree

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_lgbm_baseline as backfill  # noqa: E402

INDUCED = "impute/induced"


def _stats(prefix: str, folds: list[float]) -> dict[str, float]:
    """The parent statistics a live summary logs, enough of them for the plan to read."""
    mean = sum(folds) / len(folds)
    return {f"cv/test/{prefix}/mean": mean, f"cv/test/{prefix}/min": min(folds), f"cv/test/{prefix}/max": max(folds)}


def _logged() -> dict[str, float]:
    """A run of two folds as ADR 0016's backfill left it: model 0.9 and 0.7, mean/mode at
    parity, hgb 0.8 and 1.0, missForest 0.7 and 0.8 as its best baseline, the gap to it
    (+0.05) logged, and the withdrawn missforest_lgbm, lower than everything."""
    return {
        **_stats(f"{INDUCED}/impute_score", [0.9, 0.7]),
        **_stats(f"{INDUCED}/baseline/mean_mode/impute_score", [1.0, 1.0]),
        **_stats(f"{INDUCED}/baseline/hgb/impute_score", [0.8, 1.0]),
        **_stats(f"{INDUCED}/baseline/missforest/impute_score", [0.7, 0.8]),
        **_stats(f"{INDUCED}/baseline/missforest_lgbm/impute_score", [0.1, 0.1]),
        **_stats(f"{INDUCED}/baseline/best/impute_score", [0.7, 0.8]),
        **_stats(f"{INDUCED}/gap_to_best_baseline/impute_score", [0.2, -0.1]),
    }


TAGS = {"best_baseline/impute/induced": "missforest"}
# The model's, mean/mode's and hgb's fold values, as raw_fold_metrics.csv keeps them; the
# backfilled missForest is not there.
ROWS = {
    1: {f"{INDUCED}/impute_score": 0.9, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0,
        f"{INDUCED}/baseline/hgb/impute_score": 0.8},
    2: {f"{INDUCED}/impute_score": 0.7, f"{INDUCED}/baseline/mean_mode/impute_score": 1.0,
        f"{INDUCED}/baseline/hgb/impute_score": 1.0},
}


def _recomputed(lgbm: tuple[float, float], forest: tuple[float, float] = (0.7, 0.8)):
    return {
        fold: {
            f"{INDUCED}/baseline/mean_mode/impute_score": 1.0,
            f"{INDUCED}/baseline/hgb/impute_score": (0.8, 1.0)[fold - 1],
            f"{INDUCED}/baseline/lgbm/impute_score": lgbm[fold - 1],
            f"{INDUCED}/baseline/missforest/impute_score": forest[fold - 1],
        }
        for fold in (1, 2)
    }


def _plan(lgbm: tuple[float, float], forest: tuple[float, float] = (0.7, 0.8), children=None, run_id="run"):
    return backfill.plan_lgbm(
        run_id, "impute_toy_20nan_x", backfill._as_planned_from(_logged()), TAGS,
        _recomputed(lgbm, forest), children or {}, ROWS,
    )


def test_where_lgbm_beats_hgb_the_bar_moves_from_missforest_to_lgbm() -> None:
    """Worked by hand. lgbm scores 0.85 twice, below hgb's 0.9, so it is the bar; the
    missForest's 0.75 is lower still but outside it, and the withdrawn missforest_lgbm's
    0.1 is ignored. Gap to the bar: 0.9 - 0.85 = 0.05 and 0.7 - 0.85 = -0.15, mean -0.05.
    Gap to missForest: 0.2 and -0.1, mean 0.05, which is 6.67% of 0.75."""
    plan = _plan((0.85, 0.85), children={"child": 2})

    assert plan.refused is None
    assert plan.moved == {INDUCED: ("missforest", "lgbm")}
    assert plan.tags == {"best_baseline/impute/induced": "lgbm"}
    parent = plan.parent
    assert parent[f"cv/test/{INDUCED}/baseline/lgbm/impute_score/mean"] == pytest.approx(0.85)
    assert parent[f"cv/test/{INDUCED}/baseline/best/impute_score/mean"] == pytest.approx(0.85)
    assert parent[f"cv/test/{INDUCED}/gap_to_best_baseline/impute_score/mean"] == pytest.approx(-0.05)
    assert parent[f"cv/test/{INDUCED}/gap_to_best_baseline_pct/impute_score/mean"] == pytest.approx(-100 * 0.05 / 0.85)
    assert parent[f"cv/test/{INDUCED}/gap_to_missforest/impute_score/mean"] == pytest.approx(0.05)
    assert parent[f"cv/test/{INDUCED}/gap_to_missforest_pct/impute_score/mean"] == pytest.approx(100 * 0.05 / 0.75)
    assert plan.children == {"child": {f"test/{INDUCED}/baseline/lgbm/impute_score": 0.85}}


def test_where_lgbm_does_not_beat_hgb_the_bar_moves_back_to_hgb() -> None:
    """The bar ADR 0016 moved to missForest returns to hgb, every statistic hgb's own;
    gap to it 0.9 - 0.8 = 0.1 and 0.7 - 1.0 = -0.3, mean -0.1, from hgb's fold values in
    raw_fold_metrics.csv."""
    plan = _plan((0.95, 0.95))

    assert plan.refused is None
    assert plan.moved == {INDUCED: ("missforest", "hgb")}
    assert plan.parent[f"cv/test/{INDUCED}/baseline/best/impute_score/mean"] == pytest.approx(0.9)
    assert plan.parent[f"cv/test/{INDUCED}/baseline/best/impute_score/min"] == pytest.approx(0.8)
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_best_baseline/impute_score/mean"] == pytest.approx(-0.1)
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_missforest/impute_score/mean"] == pytest.approx(0.05)


def test_a_missforest_logged_from_a_threaded_prediction_is_corrected() -> None:
    """ADR 0016 logged missForest from forests that predicted on several threads, whose fills
    could differ between passes. Where the single-threaded recomputation differs, its numbers
    replace the logged ones (parent statistics and the child's fold value) and the gap to
    missForest is taken from them: fold 2 at 0.81 instead of 0.8 gives a mean of 0.755 and
    gaps of 0.2 and -0.11, mean 0.045. Where it agrees, nothing of missForest is written."""
    corrected = _plan((0.85, 0.85), forest=(0.7, 0.81), children={"child": 2})

    assert corrected.refused is None
    assert corrected.parent[f"cv/test/{INDUCED}/baseline/missforest/impute_score/mean"] == pytest.approx(0.755)
    assert corrected.parent[f"cv/test/{INDUCED}/gap_to_missforest/impute_score/mean"] == pytest.approx(0.045)
    assert corrected.children["child"][f"test/{INDUCED}/baseline/missforest/impute_score"] == pytest.approx(0.81)
    assert corrected.corrected == {f"{INDUCED}/impute_score": pytest.approx(0.005)}

    agreed = _plan((0.85, 0.85), children={"child": 2})
    assert not any("/baseline/missforest/" in key for key in agreed.parent)
    assert not any("/baseline/missforest/" in key for key in agreed.children["child"])
    assert agreed.corrected == {}


def test_a_knn_or_hgb_that_differs_in_the_last_digits_is_not_a_refusal() -> None:
    """Runs recorded on the Windows PC carry KNN and hgb numbers this machine reproduces
    only to about 1e-4 (KNN breaks distance ties differently). They are not checked; the
    gap to a bar that moves to hgb reads hgb's fold values from the run's own artifact."""
    logged = {**backfill._as_planned_from(_logged()), **_stats(f"{INDUCED}/baseline/knn5/impute_score", [0.95, 0.95])}
    recomputed = _recomputed((0.95, 0.95))
    for fold in recomputed.values():
        fold[f"{INDUCED}/baseline/knn5/impute_score"] = 0.95034
        fold[f"{INDUCED}/baseline/hgb/impute_score"] += 3e-4

    plan = backfill.plan_lgbm("run", "impute_toy_20nan_x", logged, TAGS, recomputed, {}, ROWS)

    assert plan.refused is None
    assert plan.moved == {INDUCED: ("missforest", "hgb")}
    assert plan.parent[f"cv/test/{INDUCED}/gap_to_best_baseline/impute_score/mean"] == pytest.approx(-0.1)


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


def test_the_moved_bar_is_what_the_run_and_its_mirror_now_read(client) -> None:
    """The bar is rewritten at the step a live run logs it, 0, beside values logged there
    before: MLflow's latest value must be the new one on the run and on its mirror, and
    every run of the tree says it was brought to ADR 0017."""
    experiment = client.create_experiment("TRIDENT/toy")
    base = {"task": "imputation", "is_mirror": "false"}
    parent = client.create_run(experiment, tags={**base, "run_role": "parent", **TAGS}).info.run_id
    child = client.create_run(
        experiment, tags={**base, "run_role": "best_fold", "fold": "2", "mlflow.parentRunId": parent}
    ).info.run_id
    hours_ago = int(time.time() * 1000) - 3 * 3_600_000
    for key, value in _logged().items():
        client.log_metric(parent, key, value, timestamp=hours_ago, step=0)
    for run_id in (parent, child):
        client.set_terminated(run_id)
    mirror_run_tree(client, parent)

    backfill.apply_lgbm_plan(client, _plan((0.85, 0.85), children={child: 2}, run_id=parent))

    best = f"cv/test/{INDUCED}/baseline/best/impute_score/mean"
    mirrors = client.search_runs(
        [client.get_experiment_by_name("TRIDENT/mirror/imputation").experiment_id],
        filter_string=f"tags.source_run_id = '{parent}'",
    )
    for run in (client.get_run(parent), client.get_run(mirrors[0].info.run_id)):
        assert run.data.metrics[best] == pytest.approx(0.85)
        assert run.data.metrics[f"cv/test/{INDUCED}/gap_to_missforest/impute_score/mean"] == pytest.approx(0.05)
        assert run.data.tags["best_baseline/impute/induced"] == "lgbm"
        assert run.data.tags[LGBM_BACKFILLED_TAG] == "true"
    assert client.get_run(child).data.tags[LGBM_BACKFILLED_TAG] == "true"
    assert [entry.value for entry in client.get_metric_history(parent, best)] == pytest.approx([0.75, 0.85])


def test_a_pass_cut_short_is_planned_again_and_a_live_run_is_left_alone(client, monkeypatch) -> None:
    """The parent's mark is the last write of a tree. A run that predates ADR 0017 and
    carries lgbm numbers without it was cut short; a run that logged lgbm live needs
    nothing; a run with no baseline at all has nothing to check against."""
    landed = int(time.time() * 1000)
    monkeypatch.setattr(backfill, "_LIVE_SINCE_MS", landed)
    experiment = client.create_experiment("TRIDENT/toy")
    lgbm = f"cv/test/{INDUCED}/baseline/lgbm/impute_score/mean"

    def run(start: int, metrics: dict[str, float], tags: dict[str, str] | None = None):
        run_id = client.create_run(experiment, start_time=start, tags=tags or {}).info.run_id
        for key, value in metrics.items():
            client.log_metric(run_id, key, value, step=0)
        return client.get_run(run_id)

    untouched = run(landed - 60_000, _logged())
    cut_short = run(landed - 60_000, {**_logged(), lgbm: 0.85})
    finished = run(landed - 60_000, {**_logged(), lgbm: 0.85}, {LGBM_BACKFILLED_TAG: "true"})
    live = run(landed + 60_000, {**_logged(), lgbm: 0.85})
    no_baselines = run(landed - 60_000, {f"cv/test/{INDUCED}/impute_score/mean": 0.8})

    assert [backfill._pending(entry) for entry in (untouched, cut_short, finished, live, no_baselines)] == [
        True, True, False, False, False,
    ]
    assert lgbm not in backfill._as_planned_from(cut_short.data.metrics)
