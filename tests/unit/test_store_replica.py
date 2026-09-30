"""gorgona8's store on the SSD, behind a link, with a replica on the hard disk (ADR 0010).

Stores are built and read with MLflow, as training does, through the link in the checkout.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from scripts.store_replica import REPLICA_FILE, adopt, find_rsync, journal_mode, sync
from scripts.sync_remote import StoreError, SyncError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the layout is gorgona8's: a symlink, flock and /proc"
)


@pytest.fixture(autouse=True)
def release_store_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "NullPool")


@pytest.fixture(autouse=True)
def rsync_installed() -> None:
    try:
        find_rsync()
    except SyncError:
        pytest.skip("sqlite3_rsync is not installed")


def client_for(store: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{store.as_posix()}")


def run_ids(store: Path) -> set[str]:
    connection = sqlite3.connect(f"{store.as_uri()}?mode=rw", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT run_uuid FROM runs")}
    finally:
        connection.close()


def quiet(_: str) -> None:
    pass


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A checkout on the "hard disk" whose store holds one run, before ``adopt``."""
    checkout = tmp_path / "hdd" / "TRIDENT"
    checkout.mkdir(parents=True)
    client = client_for(checkout / "mlflow.db")
    client.create_run(client.create_experiment("TRIDENT/credit-g"))
    return checkout


@pytest.fixture
def master(tmp_path: Path) -> Path:
    return tmp_path / "ssd" / "TRIDENT" / "mlflow.db"


def test_adopting_moves_the_store_to_the_ssd_behind_a_link(root: Path, master: Path) -> None:
    (run,) = run_ids(root / "mlflow.db")

    adopt(root, master, quiet)

    store = root / "mlflow.db"
    assert store.is_symlink() and store.resolve() == master
    assert journal_mode(master) == "wal"
    assert client_for(store).get_run(run).info.run_id == run
    assert run_ids(root / REPLICA_FILE) == {run}


def test_a_sync_carries_what_training_wrote_to_the_replica(root: Path, master: Path) -> None:
    adopt(root, master, quiet)
    client = client_for(root / "mlflow.db")
    experiment = client.get_experiment_by_name("TRIDENT/credit-g")
    assert experiment is not None
    trained = client.create_run(experiment.experiment_id)

    sync(root, quiet)

    assert trained.info.run_id in run_ids(root / REPLICA_FILE)
    # Left behind, they would read as an unfinished write and stop the next hand-off.
    assert not Path(f"{master}-wal").exists() and not Path(f"{master}-shm").exists()


def test_a_sync_restores_a_deleted_ssd_store_from_the_replica(root: Path, master: Path) -> None:
    adopt(root, master, quiet)
    runs = run_ids(master)
    master.unlink()
    master.parent.rmdir()

    sync(root, quiet)

    assert run_ids(master) == runs
    assert journal_mode(master) == "wal"
    assert (root / "mlflow.db").resolve() == master


def test_a_deleted_store_still_held_open_is_not_restored_over(root: Path, master: Path) -> None:
    # A training holding the deleted file keeps writing into it; restoring would hide those
    # runs behind an older copy.
    adopt(root, master, quiet)
    trainer = sqlite3.connect(master)
    try:
        trainer.execute("SELECT count(*) FROM runs").fetchone()
        master.unlink()
        with pytest.raises(StoreError, match="still holds it open"):
            sync(root, quiet)
    finally:
        trainer.close()

    assert not master.exists()


def test_a_damaged_ssd_store_never_reaches_the_replica(root: Path, master: Path) -> None:
    adopt(root, master, quiet)
    runs = run_ids(root / REPLICA_FILE)
    master.write_bytes(b"not a database" * 512)

    with pytest.raises(StoreError, match="quick_check"):
        sync(root, quiet)

    assert run_ids(root / REPLICA_FILE) == runs


def test_a_store_that_is_not_a_link_is_left_alone(root: Path) -> None:
    runs = run_ids(root / "mlflow.db")

    with pytest.raises(StoreError, match="not a link"):
        sync(root, quiet)

    assert run_ids(root / "mlflow.db") == runs
    assert not (root / REPLICA_FILE).exists()
