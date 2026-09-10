"""Stamping a tag onto runs recorded before that tag existed."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from scripts.backfill_run_tags import backfill


@pytest.fixture()
def store(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        yield tracking_uri
    finally:
        mlflow.set_tracking_uri(previous)


def _runs(store, *tag_sets) -> MlflowClient:
    experiment_id = mlflow.create_experiment("TRIDENT/toy")
    for tags in tag_sets:
        with mlflow.start_run(experiment_id=experiment_id, tags=tags):
            pass
    return MlflowClient(tracking_uri=store)


def test_a_run_that_already_says_it_keeps_what_it_said(store) -> None:
    """Only runs missing the tag are stamped, so re-running the script changes nothing.

    A recorded value is the truth; an inferred one must never overwrite it.
    """
    client = _runs(store, {"task": "imputation"}, {})

    first = backfill(client, tag="task", value="classification", marker="task_backfilled", apply=True)
    second = backfill(client, tag="task", value="classification", marker="task_backfilled", apply=True)

    assert first.tagged_count == 1
    assert first.skipped_count == 1
    assert second.tagged_count == 0  # idempotent
    tasks = {run.data.tags["task"] for run in client.search_runs([_experiment(client)])}
    assert tasks == {"imputation", "classification"}


def test_an_inferred_value_says_that_it_was_inferred(store) -> None:
    """A reader has to be able to tell a recorded value from one guessed afterwards."""
    client = _runs(store, {})

    backfill(client, tag="task", value="classification", marker="task_backfilled", apply=True)

    tags = client.search_runs([_experiment(client)])[0].data.tags
    assert tags["task"] == "classification"
    assert tags["task_backfilled"] == "true"


def test_a_value_the_store_itself_proves_needs_no_marker(store) -> None:
    """No run carries an Optuna role, so `is_optuna=false` is checkable, not inferred."""
    client = _runs(store, {})

    backfill(client, tag="is_optuna", value="false", marker=None, apply=True)

    tags = client.search_runs([_experiment(client)])[0].data.tags
    assert tags["is_optuna"] == "false"
    assert "is_optuna_backfilled" not in tags


def test_a_dry_run_reports_without_writing(store) -> None:
    """The default is to say what would happen, because 251 runs is not a rehearsal."""
    client = _runs(store, {})

    report = backfill(client, tag="task", value="classification", marker=None, apply=False)

    assert report.tagged_count == 1
    assert "task" not in client.search_runs([_experiment(client)])[0].data.tags


def _experiment(client: MlflowClient) -> str:
    return client.get_experiment_by_name("TRIDENT/toy").experiment_id
