"""Stamping ``decode_metric`` on imputation runs recorded before the flag (all ranked by ``impute_score``)."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import DECODE_METRIC_BACKFILLED_TAG, DECODE_METRIC_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_decode_metric_tag as backfill_script  # noqa: E402


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


def _seed_runs(client: MlflowClient) -> dict[str, str]:
    credit = client.create_experiment("TRIDENT/credit-g")

    def run(task: str = "imputation", **tags: str) -> str:
        return client.create_run(credit, tags={"is_mirror": "false", "task": task, **tags}).info.run_id

    parent = run(run_role="parent")
    study = run(run_role="optuna_study", is_optuna="true")
    runs = {
        "parent": parent,
        "child": run(run_role="best_fold", **{"mlflow.parentRunId": parent}),
        "study": study,
        "trial": run(run_role="optuna_trial", is_optuna="true", **{"mlflow.parentRunId": study}),
        "recorded": run(run_role="parent", decode_metric="dist_score"),
        "classification": run(task="classification", run_role="parent"),
    }
    for run_id in runs.values():
        client.set_terminated(run_id)
    return runs


def test_every_imputation_run_gets_impute_score_and_nothing_else_is_touched(client: MlflowClient) -> None:
    """A parent and an Optuna trial each chose a checkpoint, so they get the stamp, marked as inferred;
    a diagnostic child, a study or a classification run did not, and a recorded value stays."""
    runs = _seed_runs(client)

    assert backfill_script.backfill(client, apply=False).tagged_count == 2
    assert DECODE_METRIC_TAG not in client.get_run(runs["parent"]).data.tags

    assert backfill_script.backfill(client, apply=True).tagged_count == 2
    for name in ("parent", "trial"):
        tags = client.get_run(runs[name]).data.tags
        assert (tags[DECODE_METRIC_TAG], tags[DECODE_METRIC_BACKFILLED_TAG]) == ("impute_score", "true")
    for name in ("child", "study", "classification"):
        assert DECODE_METRIC_TAG not in client.get_run(runs[name]).data.tags
    recorded = client.get_run(runs["recorded"]).data.tags
    assert recorded[DECODE_METRIC_TAG] == "dist_score"
    assert DECODE_METRIC_BACKFILLED_TAG not in recorded
    assert backfill_script.backfill(client, apply=True).tagged_count == 0


def test_an_existing_mirror_is_stamped_in_place_and_a_missing_one_is_made(client: MlflowClient) -> None:
    from src.training.mirroring import mirror_run_tree

    runs = _seed_runs(client)
    mirror_run_tree(client, runs["parent"])

    backfill_script.backfill(client, apply=True)

    mirror_experiment = client.get_experiment_by_name("TRIDENT/mirror/imputation")
    assert mirror_experiment is not None
    for name in ("parent", "trial"):
        mirrors = client.search_runs([mirror_experiment.experiment_id], filter_string=f"tags.source_run_id = '{runs[name]}'")
        assert len(mirrors) == 1
        assert (mirrors[0].data.tags[DECODE_METRIC_TAG], mirrors[0].data.tags[DECODE_METRIC_BACKFILLED_TAG]) == ("impute_score", "true")
