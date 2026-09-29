"""The hand-off between this checkout and a Linux server (ADR 0009).

Stores are built with MLflow itself, so the schema the translation scans is the one the
installed MLflow writes, and they are read back through ``MlflowClient``, the way every
analysis reads them.
"""

import os
import shutil
import sqlite3
import stat
import subprocess
from pathlib import Path
from typing import Callable

import pytest
from mlflow.tracking import MlflowClient

from scripts.sync_remote import (
    StoreError,
    SyncConfig,
    Workspace,
    pull,
    push_code,
    push,
    remote_command,
    translate_store,
)

WINDOWS_PREFIX = "file:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlruns/"
SERVER_PREFIX = "file:///scratch/diogo/TRIDENT/mlruns/"
SCRIPT_PATH = r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT\main.py"
OPTUNA_STORAGE = "sqlite:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/results/x/optuna_study.db"


@pytest.fixture(autouse=True)
def release_store_files(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pooled connection keeps the file open, and Windows refuses to replace an open file.
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "NullPool")


def client_for(store: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{store.as_posix()}")


def make_store(store: Path, prefix: str) -> tuple[str, str]:
    """One experiment family with one finished run, its artifact root under ``prefix``."""
    client = client_for(store)
    experiment_id = client.create_experiment("TRIDENT/credit-g", artifact_location=f"{prefix}1")
    # MLflow roots the Default experiment at the working directory, which sits under
    # ``prefix`` only when the suite runs from the Windows checkout; pin it there.
    connection = sqlite3.connect(store)
    with connection:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = 0", (f"{prefix}0",)
        )
    connection.close()
    run = client.create_run(
        experiment_id, tags={"mlflow.source.name": SCRIPT_PATH, "mlflow.user": "Diogo Neiss"}
    )
    client.log_param(run.info.run_id, "optuna_storage", OPTUNA_STORAGE)
    client.set_terminated(run.info.run_id)
    return experiment_id, run.info.run_id


def test_translation_moves_artifact_roots_and_keeps_provenance(tmp_path: Path) -> None:
    store = tmp_path / "mlflow.db"
    experiment_id, run_id = make_store(store, WINDOWS_PREFIX)

    translate_store(store, WINDOWS_PREFIX, SERVER_PREFIX)

    client = client_for(store)
    assert client.get_experiment(experiment_id).artifact_location == (
        "file:///scratch/diogo/TRIDENT/mlruns/1"
    )
    run = client.get_run(run_id)
    assert run.info.artifact_uri == f"file:///scratch/diogo/TRIDENT/mlruns/1/{run_id}/artifacts"
    assert run.data.tags["mlflow.source.name"] == SCRIPT_PATH
    assert run.data.tags["mlflow.user"] == "Diogo Neiss"
    assert run.data.params["optuna_storage"] == OPTUNA_STORAGE

    translate_store(store, SERVER_PREFIX, WINDOWS_PREFIX)

    assert client.get_run(run_id).info.artifact_uri == (
        f"file:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlruns/1/{run_id}/artifacts"
    )


def test_an_artifact_root_under_neither_prefix_stops_the_translation(tmp_path: Path) -> None:
    # A run trained from another working directory gets its root there; translating the
    # rest would leave that run pointing at a path that exists on neither machine.
    store = tmp_path / "mlflow.db"
    experiment_id, run_id = make_store(store, WINDOWS_PREFIX)
    client_for(store).create_experiment(
        "TRIDENT/vehicle", artifact_location="file:///C:/Users/Diogo Neiss/scratch/mlruns/2"
    )

    with pytest.raises(StoreError, match="scratch/mlruns/2"):
        translate_store(store, WINDOWS_PREFIX, SERVER_PREFIX)

    client = client_for(store)
    assert client.get_experiment(experiment_id).artifact_location == f"{WINDOWS_PREFIX}1"
    assert client.get_run(run_id).info.artifact_uri.startswith(WINDOWS_PREFIX)


def test_an_artifact_root_stored_anywhere_else_stops_the_translation(tmp_path: Path) -> None:
    # Only two columns are rewritten. A root that a later MLflow (or a tag) keeps somewhere
    # else would survive the hand-off pointing at the wrong machine, so it must stop it.
    store = tmp_path / "mlflow.db"
    experiment_id, run_id = make_store(store, WINDOWS_PREFIX)
    client_for(store).set_tag(run_id, "copied_root", f"{WINDOWS_PREFIX}1/{run_id}/artifacts")

    with pytest.raises(StoreError, match="tags.value"):
        translate_store(store, WINDOWS_PREFIX, SERVER_PREFIX)

    client = client_for(store)
    assert client.get_experiment(experiment_id).artifact_location == f"{WINDOWS_PREFIX}1"
    assert client.get_run(run_id).info.artifact_uri.startswith(WINDOWS_PREFIX)


class DirectoryServer:
    """The server as a directory on this machine: copies stand in for rsync and the
    remote commands run in-process, so the protocol runs end to end without a network."""

    def __init__(self, local_root: Path, remote_root: Path) -> None:
        self.local_root = local_root
        self.remote_root = remote_root

    def remote(self, *args: str) -> dict[str, object]:
        return remote_command(list(args))

    def send_files(self, files: list[str], *, dry_run: bool) -> int:
        if not dry_run:
            for relative in files:
                _copy(self.local_root / relative, self.remote_root / relative)
        return len(files)

    def send_dir(self, relative: str, *, dry_run: bool, delete: bool = False) -> int:
        return _copy_tree(self.local_root / relative, self.remote_root / relative, dry_run, delete)

    def send_file(self, local: str, remote: str, *, dry_run: bool) -> int:
        if not dry_run:
            _copy(self.local_root / local, self.remote_root / remote)
        return 1

    def fetch_dir(self, relative: str, *, dry_run: bool) -> int:
        return _copy_tree(self.remote_root / relative, self.local_root / relative, dry_run, False)

    def fetch_file(self, remote: str, local: str, *, dry_run: bool) -> int:
        if not dry_run:
            _copy(self.remote_root / remote, self.local_root / local)
        return 1


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path, dry_run: bool, delete: bool) -> int:
    if not source.is_dir() or dry_run:
        return 0
    if delete and target.exists():
        shutil.rmtree(target, onerror=_make_writable_and_retry)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return sum(1 for path in source.rglob("*") if path.is_file())


def _make_writable_and_retry(remove: Callable[[str], object], path: str, _: object) -> None:
    # git writes its objects read-only, and Windows will not delete a read-only file.
    os.chmod(path, stat.S_IWRITE)
    remove(path)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def make_checkout(root: Path) -> None:
    """A git checkout with the repository's ignore rules for the store and the outputs."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "model.py").write_text("VERSION = 1\n")
    (root / ".gitignore").write_text("mlflow.db\nmlruns/\nresults/\nmetrics/\nsync/\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    local_root, remote_root = tmp_path / "windows", tmp_path / "server"
    local_root.mkdir()
    remote_root.mkdir()
    make_checkout(local_root)
    config = SyncConfig(
        host="gpu",
        remote_root=str(remote_root),
        local_prefix=WINDOWS_PREFIX,
        remote_prefix=SERVER_PREFIX,
    )
    return Workspace(local_root, config, DirectoryServer(local_root, remote_root), say=lambda _: None)


def server_root(workspace: Workspace) -> Path:
    return Path(workspace.config.remote_root)


def test_a_push_then_a_pull_carries_runs_and_artifacts_both_ways(workspace: Workspace) -> None:
    local, server = workspace.root, server_root(workspace)
    experiment_id, run_id = make_store(local / "mlflow.db", WINDOWS_PREFIX)
    artifact = local / "mlruns" / "1" / run_id / "artifacts" / "metrics.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("fold,f1\n1,0.9\n")

    push(workspace, apply=True)

    assert (server / "src" / "model.py").read_text() == "VERSION = 1\n"
    assert (server / "mlruns" / "1" / run_id / "artifacts" / "metrics.csv").exists()
    on_server = client_for(server / "mlflow.db")
    assert on_server.get_run(run_id).info.artifact_uri == (
        f"file:///scratch/diogo/TRIDENT/mlruns/1/{run_id}/artifacts"
    )

    # The server trains: a new run with an artifact of its own.
    trained = on_server.create_run(experiment_id)
    on_server.log_metric(trained.info.run_id, "impute_score", 0.25)
    on_server.set_terminated(trained.info.run_id)
    produced = server / "mlruns" / "1" / trained.info.run_id / "artifacts" / "ledger.csv"
    produced.parent.mkdir(parents=True)
    produced.write_text("cell,score\n")

    pull(workspace, apply=True)

    here = client_for(local / "mlflow.db").get_run(trained.info.run_id)
    assert here.info.artifact_uri == (
        "file:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlruns/1/"
        f"{trained.info.run_id}/artifacts"
    )
    assert here.data.metrics["impute_score"] == 0.25
    assert (local / "mlruns" / "1" / trained.info.run_id / "artifacts" / "ledger.csv").exists()
    assert artifact.read_text() == "fold,f1\n1,0.9\n"


def test_after_both_sides_wrote_neither_direction_overwrites_the_other(
    workspace: Workspace,
) -> None:
    # One writer per sync interval is the rule; when it is broken, the next sync in either
    # direction would erase the other side's runs, so both must refuse.
    local, server = workspace.root, server_root(workspace)
    experiment_id, _ = make_store(local / "mlflow.db", WINDOWS_PREFIX)
    push(workspace, apply=True)
    on_server = client_for(server / "mlflow.db").create_run(experiment_id).info.run_id
    here = client_for(local / "mlflow.db").create_run(experiment_id).info.run_id

    with pytest.raises(StoreError, match="changed since the last sync"):
        pull(workspace, apply=True)
    with pytest.raises(StoreError, match="changed since the last sync"):
        push(workspace, apply=True)

    assert client_for(local / "mlflow.db").get_run(here).info.run_id == here
    assert client_for(server / "mlflow.db").get_run(on_server).info.run_id == on_server


def test_a_first_pull_does_not_replace_a_store_no_sync_has_recorded(
    workspace: Workspace,
) -> None:
    # The server trained into a store of its own before any push: pulling it would erase
    # every run this checkout holds.
    local, server = workspace.root, server_root(workspace)
    _, run_id = make_store(local / "mlflow.db", WINDOWS_PREFIX)
    make_store(server / "mlflow.db", SERVER_PREFIX)

    with pytest.raises(StoreError, match="no sync has recorded"):
        pull(workspace, apply=True)

    assert client_for(local / "mlflow.db").get_run(run_id).info.run_id == run_id


def test_a_push_waits_for_the_processes_writing_to_the_store(workspace: Workspace) -> None:
    # A training still running here would keep writing after the hand-off, into a store
    # that is no longer the one the server trains into.
    local = workspace.root
    make_store(local / "mlflow.db", WINDOWS_PREFIX)
    trainer = sqlite3.connect(local / "mlflow.db")
    try:
        with pytest.raises(StoreError, match="open"):
            push(workspace, apply=True)
    finally:
        trainer.close()

    assert not (server_root(workspace) / "mlflow.db").exists()


def test_the_code_push_mirrors_the_checkout_and_spares_the_servers_own_files(
    workspace: Workspace,
) -> None:
    local, server = workspace.root, server_root(workspace)
    (local / "src" / "old.py").write_text("")
    git(local, "add", ".")
    git(local, "commit", "-q", "-m", "old")
    (local / "src" / "wip.py").write_text("")  # untracked but not ignored: code on disk
    (local / "mlflow.db.bak").write_bytes(b"a store backup")
    (local / "results").mkdir()
    (local / "results" / "local.csv").write_text("")
    inputs = local / "datasets" / "processed_datasets"
    inputs.mkdir(parents=True)
    (inputs / "vehicle_00nan.csv").write_text("a,b\n")
    # A nested directory named like an output is code (the repository's stub for
    # sklearn.metrics once went missing to exactly that confusion).
    stub = local / "stubs" / "sklearn" / "metrics" / "__init__.pyi"
    stub.parent.mkdir(parents=True)
    stub.write_text("")
    with (local / ".gitignore").open("a") as ignore:
        ignore.write("datasets/processed_datasets/\n!stubs/**/metrics/\n")
    (server / "train.log").write_text("the server's own")
    (server / "results").mkdir()
    (server / "results" / "run.csv").write_text("")

    push_code(workspace, apply=True)

    for sent in ("src/old.py", "src/wip.py", "datasets/processed_datasets/vehicle_00nan.csv",
                 "stubs/sklearn/metrics/__init__.pyi", ".git/HEAD"):
        assert (server / sent).exists(), sent
    assert not (server / "mlflow.db.bak").exists()
    assert not (server / "results" / "local.csv").exists()

    (local / "src" / "old.py").unlink()
    push_code(workspace, apply=True)

    assert not (server / "src" / "old.py").exists()
    assert (server / "train.log").read_text() == "the server's own"
    assert (server / "results" / "run.csv").exists()


def test_moving_a_diverged_store_aside_lets_the_other_side_in(workspace: Workspace) -> None:
    # The documented way out of a divergence: keep one side, rename the other's store.
    local, server = workspace.root, server_root(workspace)
    experiment_id, _ = make_store(local / "mlflow.db", WINDOWS_PREFIX)
    push(workspace, apply=True)
    kept = client_for(server / "mlflow.db").create_run(experiment_id).info.run_id
    client_for(local / "mlflow.db").create_run(experiment_id)
    (local / "mlflow.db").rename(local / "mlflow.db.diverged")

    pull(workspace, apply=True)

    assert client_for(local / "mlflow.db").get_run(kept).info.run_id == kept
    assert (local / "mlflow.db.diverged").exists()
