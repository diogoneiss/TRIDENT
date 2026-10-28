"""Stamping the host on runs recorded before the tag, from their GPU (ADR 0034)."""

from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.mlflow_utils import HOST_BACKFILLED_TAG, HOST_TAG

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import backfill_host_tag as backfill_script  # noqa: E402


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


def test_stamps_source_and_mirror_by_gpu_and_leaves_the_rest(client: MlflowClient) -> None:
    source = client.create_experiment("TRIDENT/credit-g")
    mirror = client.create_experiment("TRIDENT/mirror/imputation")
    laptop = client.create_run(source, tags={"gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"}).info.run_id
    server = client.create_run(source, tags={"gpu_name": "NVIDIA GeForce RTX 3090 Ti"}).info.run_id
    mirrored = client.create_run(mirror, tags={"gpu_name": "NVIDIA GeForce RTX 3090 Ti", "is_mirror": "true"}).info.run_id
    recorded = client.create_run(source, tags={"gpu_name": "NVIDIA GeForce RTX 3090 Ti", HOST_TAG: "elsewhere"}).info.run_id
    unknown = client.create_run(source, tags={"gpu_name": "none"}).info.run_id
    old = client.create_run(source).info.run_id

    assert sum(backfill_script.backfill(client, apply=False).values()) == 3
    assert HOST_TAG not in client.get_run(laptop).data.tags

    stamped = backfill_script.backfill(client, apply=True)

    assert stamped == {"windows": 1, "gorgona8": 2}
    for run_id, host in ((laptop, "windows"), (server, "gorgona8"), (mirrored, "gorgona8")):
        tags = client.get_run(run_id).data.tags
        assert (tags[HOST_TAG], tags[HOST_BACKFILLED_TAG]) == (host, "true")
    assert client.get_run(recorded).data.tags[HOST_TAG] == "elsewhere"
    for run_id in (recorded, unknown, old):
        assert HOST_BACKFILLED_TAG not in client.get_run(run_id).data.tags
    assert not backfill_script.backfill(client, apply=True)
