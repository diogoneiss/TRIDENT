"""Leader and follower over one store (ADR 0033), with the peer on a local path instead of ssh."""

import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from scripts import leadership
from scripts.leadership import (
    CANONICAL_ROOT,
    FORK_BASE_AFTER,
    PROMOTE_AFTER,
    Peer,
    Services,
    guard_roots,
    init,
    load_role,
    promote,
    switchover,
    tick,
)
from scripts.store_replica import find_rsync
from scripts.sync_remote import StoreError, SyncError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="flock, chmod and sqlite3_rsync")

T0 = 1_800_000_000.0


@pytest.fixture(autouse=True)
def release_store_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "NullPool")


@pytest.fixture(autouse=True)
def rsync_installed() -> None:
    try:
        find_rsync()
    except SyncError:
        pytest.skip("sqlite3_rsync is not installed")


class LocalPeer(Peer):
    def __init__(self, root: Path) -> None:
        super().__init__("peer", str(root))
        self.path, self.answers, self.copies_fail = root, True, False

    def read_role(self) -> tuple[bool, dict[str, Any] | None]:
        return (True, load_role(self.path)) if self.answers else (False, None)

    def pull_store(self, target: Path) -> str:
        if not self.answers or self.copies_fail:
            raise SyncError("unreachable")
        subprocess.run([find_rsync(), str(self.path / "mlflow.db"), str(target)], check=True)
        return "sent"

    def exchange_outputs(self, root: Path, backup: str) -> None:
        pass

    def yield_leadership(self, epoch: int) -> None:
        leadership.yield_leadership(self.path, FakeServices(self.path), epoch, T0)


class FakeServices(Services):
    def __init__(self, root: Path) -> None:
        super().__init__(root, 47999)
        self.running = False

    def server_running(self) -> bool:
        return self.running

    def start_server(self) -> None:
        self.running = True

    def stop_server(self) -> None:
        self.running = False

    def ensure_tunnel(self, alias: str) -> None:
        self.running = False


def store_with_run(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    found = client.get_experiment_by_name("TRIDENT/credit-g")
    experiment = found.experiment_id if found else client.create_experiment("TRIDENT/credit-g")
    return str(client.create_run(experiment).info.run_id)


def run_ids(store: Path) -> set[str]:
    connection = sqlite3.connect(f"{store.as_uri()}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT run_uuid FROM runs")}
    finally:
        connection.close()


def events(root: Path) -> list[str]:
    return [json.loads(line)["event"] for line in (root / "sync/leadership.log").read_text().splitlines()]


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path, LocalPeer, FakeServices]:
    leader, follower = tmp_path / "leader", tmp_path / "follower"
    store_with_run(leader)
    connection = sqlite3.connect(leader / "mlflow.db")
    connection.execute("PRAGMA journal_mode=WAL").fetchone()  # as gorgona8's store
    connection.close()
    follower.mkdir()
    init(leader, "leader", "follower", str(follower), 47999, T0)
    init(follower, "follower", "leader", str(leader), 47999, T0)
    return leader, follower, LocalPeer(leader), FakeServices(follower)


def follow(follower: Path, peer: LocalPeer, services: FakeServices, now: float) -> None:
    tick(follower, now, lambda _: peer, lambda _: services)


def test_tick_without_a_role_does_nothing(tmp_path: Path) -> None:
    tick(tmp_path, T0, lambda _: pytest.fail("no peer"), lambda _: pytest.fail("no services"))
    assert not (tmp_path / "sync").exists()


def test_the_follower_pulls_the_store_and_keeps_it_read_only(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    assert run_ids(follower / "mlflow.db") == run_ids(leader / "mlflow.db")
    assert stat.S_IMODE((follower / "mlflow.db").stat().st_mode) == 0o444
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connection = sqlite3.connect(follower / "mlflow.db")
        connection.execute("DELETE FROM runs")
        connection.commit()
    reader = sqlite3.connect(f"{(follower / 'mlflow.db').as_uri()}?mode=rw", uri=True)
    reader.execute("PRAGMA quick_check").fetchone()
    reader.close()
    assert stat.S_IMODE((follower / "mlflow.db-shm").stat().st_mode) == 0o444  # a reader leaves these behind
    added = store_with_run(leader)
    follow(follower, peer, services, T0 + 120)  # within ten minutes: no copy
    assert added not in run_ids(follower / "mlflow.db")
    follow(follower, peer, services, T0 + 60 + 600)
    assert added in run_ids(follower / "mlflow.db")


def test_a_silent_leader_gets_a_fork_base_that_contact_removes(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    peer.answers = False
    follow(follower, peer, services, T0 + 60 + FORK_BASE_AFTER)
    base = load_role(follower)["fork_base"]  # type: ignore[index]
    assert Path(base).exists() and run_ids(Path(base)) == run_ids(leader / "mlflow.db")
    assert events(follower)[-2:] == ["peer_unreachable", "backup_created"]
    peer.answers = True
    follow(follower, peer, services, T0 + 60 + FORK_BASE_AFTER + 600)
    assert not Path(base).exists() and load_role(follower)["fork_base"] is None  # type: ignore[index]
    assert "contact_regained" in events(follower)


def test_after_24_hours_of_silence_the_follower_promotes_itself(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    peer.answers = False
    follow(follower, peer, services, T0 + 60 + PROMOTE_AFTER)
    role = load_role(follower)
    assert role is not None and role["role"] == "leader" and role["epoch"] == 2 and services.running
    assert stat.S_IMODE((follower / "mlflow.db").stat().st_mode) == 0o644
    client = MlflowClient(tracking_uri=f"sqlite:///{(follower / 'mlflow.db').as_posix()}")
    assert int(client.create_experiment("TRIDENT/after-promotion")) == 200_001
    assert events(follower)[-1] == "promoted"


def test_a_stalled_copy_with_a_live_leader_never_promotes(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    peer.copies_fail = True
    follow(follower, peer, services, T0 + 60 + PROMOTE_AFTER)
    assert load_role(follower)["role"] == "follower"  # type: ignore[index]
    assert "replication_stalled" in events(follower)
    with pytest.raises(StoreError, match="switchover"):
        promote(follower, load_role(follower), peer, services, T0 + 60 + PROMOTE_AFTER, "manual")  # type: ignore[arg-type]


def test_a_leader_that_meets_a_higher_epoch_is_deposed(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    peer.answers = False
    follow(follower, peer, services, T0 + 60 + PROMOTE_AFTER)
    old = FakeServices(leader)
    old.running = True
    tick(leader, T0 + 2 * PROMOTE_AFTER, lambda _: LocalPeer(follower), lambda _: old)
    assert load_role(leader)["role"] == "deposed" and not old.running  # type: ignore[index]
    tick(leader, T0 + 2 * PROMOTE_AFTER + 60, lambda _: LocalPeer(follower), lambda _: old)
    assert stat.S_IMODE((leader / "mlflow.db").stat().st_mode) == 0o444


def test_switchover_swaps_roles_and_carries_the_last_writes(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    last = store_with_run(leader)
    switchover(follower, peer, services, T0 + 120)
    assert last in run_ids(follower / "mlflow.db")
    assert (load_role(follower)["role"], load_role(follower)["epoch"]) == ("leader", 2)  # type: ignore[index]
    assert (load_role(leader)["role"], load_role(leader)["epoch"]) == ("follower", 2)  # type: ignore[index]
    assert stat.S_IMODE((leader / "mlflow.db").stat().st_mode) == 0o444 and services.running


def test_the_guard_moves_roots_under_the_physical_path_to_the_canonical_prefix(tmp_path: Path) -> None:
    root = tmp_path / "TRIDENT"
    store_with_run(root)
    connection = sqlite3.connect(root / "mlflow.db")
    with connection:
        connection.execute("UPDATE experiments SET artifact_location = ? || experiment_id", (f"{root}/mlruns/",))
        connection.execute("UPDATE runs SET artifact_uri = ? || run_uuid", (f"file://{root}/mlruns/1/",))
    connection.close()
    role: dict[str, Any] = {}
    guard_roots(root, role, T0)
    connection = sqlite3.connect(root / "mlflow.db")
    roots = [r for (r,) in connection.execute("SELECT artifact_location FROM experiments UNION ALL SELECT artifact_uri FROM runs")]
    connection.close()
    assert roots and all(r.startswith(CANONICAL_ROOT) for r in roots)
    assert role.get("foreign_roots", 0) == 0 and os.path.exists(root / "sync/leadership.log")


def test_a_follower_that_never_copied_has_nothing_to_fork(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    peer.answers = False
    follow(follower, peer, services, T0 + PROMOTE_AFTER)
    assert load_role(follower)["role"] == "follower" and load_role(follower).get("fork_base") is None  # type: ignore[index,union-attr]


def test_a_switchover_cut_short_after_the_yield_resumes_at_the_same_epoch(pair: tuple[Path, Path, LocalPeer, FakeServices]) -> None:
    leader, follower, peer, services = pair
    follow(follower, peer, services, T0 + 60)
    peer.yield_leadership(2)
    switchover(follower, peer, services, T0 + 120)
    assert (load_role(follower)["role"], load_role(follower)["epoch"]) == ("leader", 2)  # type: ignore[index]
