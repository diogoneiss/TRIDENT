from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import LR_SCHEDULER_BACKFILLED_TAG, LR_SCHEDULER_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_lr_scheduler_tag as backfill_script  # noqa: E402


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
    vehicle = client.create_experiment("TRIDENT/vehicle")
    letter = client.create_experiment("TRIDENT/letter")
    untagged_parent = client.create_run(vehicle, tags={"run_role": "parent"}).info.run_id
    untagged_child = client.create_run(
        vehicle, tags={"run_role": "best_fold", "mlflow.parentRunId": untagged_parent}
    ).info.run_id
    already_tagged = client.create_run(
        letter, tags={"run_role": "parent", LR_SCHEDULER_TAG: "cosine"}
    ).info.run_id
    deleted = client.create_run(letter, tags={"run_role": "parent"}).info.run_id
    client.delete_run(deleted)
    return {
        "untagged_parent": untagged_parent,
        "untagged_child": untagged_child,
        "already_tagged": already_tagged,
        "deleted": deleted,
    }


def test_dry_run_reports_without_writing(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    report = backfill_script.backfill(client, "cosine_legacy", apply=False)

    assert report.tagged_count == 2
    assert report.skipped == {"TRIDENT/letter": [runs["already_tagged"]]}
    assert report.deleted == {"TRIDENT/letter": [runs["deleted"]]}
    for run_id in (runs["untagged_parent"], runs["untagged_child"], runs["deleted"]):
        assert LR_SCHEDULER_TAG not in client.get_run(run_id).data.tags


def test_apply_tags_only_untagged_runs_and_marks_them_backfilled(client: MlflowClient) -> None:
    runs = _seed_runs(client)

    report = backfill_script.backfill(client, "cosine_legacy", apply=True)

    assert report.tagged_count == 2
    for run_id in (runs["untagged_parent"], runs["untagged_child"]):
        tags = client.get_run(run_id).data.tags
        assert tags[LR_SCHEDULER_TAG] == "cosine_legacy"
        assert tags[LR_SCHEDULER_BACKFILLED_TAG] == "true"
    tags = client.get_run(runs["already_tagged"]).data.tags
    assert tags[LR_SCHEDULER_TAG] == "cosine"
    assert LR_SCHEDULER_BACKFILLED_TAG not in tags
    # Deleted runs cannot be tagged through the MLflow API, so they are reported only.
    assert LR_SCHEDULER_TAG not in client.get_run(runs["deleted"]).data.tags


def test_apply_is_idempotent(client: MlflowClient) -> None:
    _seed_runs(client)
    backfill_script.backfill(client, "cosine_legacy", apply=True)

    second = backfill_script.backfill(client, "cosine_legacy", apply=True)

    assert second.tagged_count == 0
    assert second.skipped_count == 3
    assert second.deleted_count == 1


def test_rejects_unknown_schedule_names(client: MlflowClient) -> None:
    with pytest.raises(ValueError):
        backfill_script.backfill(client, "not_a_schedule", apply=False)


def test_main_dry_run_prints_summary(client: MlflowClient, capsys) -> None:
    _seed_runs(client)

    assert backfill_script.main([]) == 0

    output = capsys.readouterr().out
    assert "[DRY RUN]" in output
    assert "lr_scheduler=cosine_legacy on 2 run(s); 1 already tagged; 1 deleted" in output
    assert "Re-run with --apply" in output
