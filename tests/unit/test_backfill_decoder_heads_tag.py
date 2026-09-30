"""Stamping the decoder heads on imputation runs recorded before they could be batched."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import DECODER_HEADS_BACKFILLED_TAG, DECODER_HEADS_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_decoder_heads_tag as backfill_script  # noqa: E402


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

    def run(**tags: str) -> str:
        return client.create_run(credit, tags={"is_mirror": "false", "task": "imputation", **tags}).info.run_id

    parent = run(run_role="parent")
    child = run(run_role="best_fold", **{"mlflow.parentRunId": parent})
    study = run(run_role="optuna_study", is_optuna="true")
    trial = run(run_role="optuna_trial", is_optuna="true", **{"mlflow.parentRunId": study})
    recorded = run(run_role="parent", decoder_heads="batched")
    classification = client.create_run(credit, tags={"is_mirror": "false", "task": "classification", "run_role": "parent"}).info.run_id
    for run_id in (parent, child, study, trial, recorded, classification):
        client.set_terminated(run_id)
    return {"parent": parent, "child": child, "study": study, "trial": trial, "recorded": recorded, "classification": classification}


def test_every_imputation_run_gets_per_column_heads_and_nothing_else_is_touched(
    client: MlflowClient,
) -> None:
    """Every imputation run before the flag applied its heads per column. A parent and an
    Optuna trial each trained a decode stage, so they get the stamp, marked as inferred; a
    diagnostic child, a study or a classification run did not, and a recorded value stays."""
    runs = _seed_runs(client)

    report = backfill_script.backfill(client, apply=True)

    assert report.tagged_count == 2
    for name in ("parent", "trial"):
        tags = client.get_run(runs[name]).data.tags
        assert (tags[DECODER_HEADS_TAG], tags[DECODER_HEADS_BACKFILLED_TAG]) == ("per_column", "true")
    for name in ("child", "study", "classification"):
        assert DECODER_HEADS_TAG not in client.get_run(runs[name]).data.tags
    recorded = client.get_run(runs["recorded"]).data.tags
    assert recorded[DECODER_HEADS_TAG] == "batched"
    assert DECODER_HEADS_BACKFILLED_TAG not in recorded


def test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    assert backfill_script.backfill(client, apply=False).tagged_count == 2
    assert DECODER_HEADS_TAG not in client.get_run(runs["parent"]).data.tags

    backfill_script.backfill(client, apply=True)
    assert backfill_script.backfill(client, apply=True).tagged_count == 0


def test_the_mirror_of_a_stamped_run_carries_the_stamp(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    backfill_script.backfill(client, apply=True)

    mirror_experiment = client.get_experiment_by_name("TRIDENT/mirror/imputation")
    assert mirror_experiment is not None
    mirrors = client.search_runs(
        [mirror_experiment.experiment_id], filter_string=f"tags.source_run_id = '{runs['parent']}'"
    )
    assert len(mirrors) == 1
    assert mirrors[0].data.tags[DECODER_HEADS_TAG] == "per_column"
