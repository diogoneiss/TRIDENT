"""Stamping the decode mask weighting on imputation runs recorded before E65 (none applied)."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import DECODE_MASK_WEIGHTING_BACKFILLED_TAG, DECODE_MASK_WEIGHTING_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_decode_mask_weighting_tag as backfill_script  # noqa: E402


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
    recorded = run(run_role="parent", decode_mask_weighting="propensity")
    classification = client.create_run(credit, tags={"is_mirror": "false", "task": "classification", "run_role": "parent"}).info.run_id
    for run_id in (parent, child, study, trial, recorded, classification):
        client.set_terminated(run_id)
    return {"parent": parent, "child": child, "study": study, "trial": trial, "recorded": recorded, "classification": classification}


def test_every_imputation_run_gets_uniform_and_nothing_else_is_touched(
    client: MlflowClient,
) -> None:
    """Every imputation run before E65 hid its decode training cells uniformly. A parent and an
    Optuna trial each trained a decode stage, so they get the stamp, marked as inferred; a
    diagnostic child, a study or a classification run did not, and a recorded value stays."""
    runs = _seed_runs(client)

    report = backfill_script.backfill(client, apply=True)

    assert report.tagged_count == 2
    for name in ("parent", "trial"):
        tags = client.get_run(runs[name]).data.tags
        assert (tags[DECODE_MASK_WEIGHTING_TAG], tags[DECODE_MASK_WEIGHTING_BACKFILLED_TAG]) == ("uniform", "true")
    for name in ("child", "study", "classification"):
        assert DECODE_MASK_WEIGHTING_TAG not in client.get_run(runs[name]).data.tags
    recorded = client.get_run(runs["recorded"]).data.tags
    assert recorded[DECODE_MASK_WEIGHTING_TAG] == "propensity"
    assert DECODE_MASK_WEIGHTING_BACKFILLED_TAG not in recorded


def test_a_dry_run_writes_nothing_and_a_second_apply_finds_nothing(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    assert backfill_script.backfill(client, apply=False).tagged_count == 2
    assert DECODE_MASK_WEIGHTING_TAG not in client.get_run(runs["parent"]).data.tags

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
    assert mirrors[0].data.tags[DECODE_MASK_WEIGHTING_TAG] == "uniform"


def test_an_existing_mirror_is_stamped_in_place_and_never_duplicated(client: MlflowClient) -> None:
    """The fast path: a stamped run that already has a mirror gets the tags on that mirror,
    with no second mirror made."""
    from src.training.mirroring import mirror_run_tree

    runs = _seed_runs(client)
    mirror_run_tree(client, runs["parent"])

    backfill_script.backfill(client, apply=True)

    mirror_experiment = client.get_experiment_by_name("TRIDENT/mirror/imputation")
    assert mirror_experiment is not None
    mirrors = client.search_runs(
        [mirror_experiment.experiment_id], filter_string=f"tags.source_run_id = '{runs['parent']}'"
    )
    assert len(mirrors) == 1
    tags = mirrors[0].data.tags
    assert (tags[DECODE_MASK_WEIGHTING_TAG], tags[DECODE_MASK_WEIGHTING_BACKFILLED_TAG]) == ("uniform", "true")
