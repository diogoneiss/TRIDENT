#!/usr/bin/env python3
"""Two servers sharing one tracking store: a leader and a follower (ADR 0033).

The leader's ``mlflow.db`` is the only one written; it runs an MLflow server on port 47731
for the follower's trainings, which reach it through an ssh tunnel on the same port, so
``MLFLOW_TRACKING_URI=http://127.0.0.1:47731`` works on either machine whatever its role.
The follower pulls the store every ten minutes with ``sqlite3_rsync`` and keeps it mode 444,
and both sides' ``mlruns/``, ``results/`` and ``metrics/`` are copied additively each way.
After 30 minutes without a copy the follower backs its store up as the fork base; after 24
hours, if the leader does not even answer ssh, it promotes itself (decision 6).

The role lives in ``sync/role.json``; every step appends a JSON line (UTC) to
``sync/leadership.log``. ``tick`` reads the role, so the cron line is the same on both
machines and a switchover never touches it. Standard library only: cron's python3 runs it.

    python3 scripts/leadership.py init --role leader|follower --peer <alias> --peer-root <path>
    python3 scripts/leadership.py status
    python3 scripts/leadership.py tick          # cron, every minute, both machines
    python3 scripts/leadership.py switchover    # on the follower; the leader must answer
    python3 scripts/leadership.py promote       # on the follower; refused while the leader answers

Cron entry (``crontab -e``), the same on both machines but for the checkout's path:

    * * * * * cd /scratch2/diogoneiss/TRIDENT && python3 scripts/leadership.py tick >> sync/leadership.cron.log 2>&1

``reconcile`` (decision 8) is not written yet: after a promotion, the old leader stays
``deposed`` until it is.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.store_replica import copy_store, find_rsync, quick_check  # noqa: E402
from scripts.sync_remote import (  # noqa: E402
    ARTIFACT_ROOT_COLUMNS,
    EXIT_REFUSED,
    OUTPUT_DIRS,
    STORE_NAME,
    SYNC_DIR,
    StoreError,
    SyncError,
    refuse_if_busy,
    store_lock,
)

ROLE_FILE = f"{SYNC_DIR}/role.json"
LOG_FILE = f"{SYNC_DIR}/leadership.log"
TICK_LOCK = f"{SYNC_DIR}/leadership.lock"
SERVER_LOG = f"{SYNC_DIR}/mlflow_server.log"
BACKUP_DIR = f"{SYNC_DIR}/backups"
PORT = 47731  # unusual on purpose: 5000 is the default of many tools
CANONICAL_ROOT = "file:///scratch/diogoneiss/TRIDENT/mlruns/"
COPY_EVERY = 10 * 60
FORK_BASE_AFTER = 30 * 60
PROMOTE_AFTER = 24 * 60 * 60
EPOCH_ID_STRIDE = 100_000
REMOTE_RSYNC = ".local/bin/sqlite3_rsync"  # relative to the peer's home, where ssh starts
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")


def utc(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(root: Path, event: str, now: float, **fields: Any) -> None:
    line = json.dumps({"ts": utc(now), "event": event, **fields})
    (root / SYNC_DIR).mkdir(parents=True, exist_ok=True)
    with open(root / LOG_FILE, "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_role(root: Path) -> dict[str, Any] | None:
    path = root / ROLE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def save_role(root: Path, role: dict[str, Any]) -> None:
    path = root / ROLE_FILE
    staged = path.with_suffix(".json.staged")
    staged.write_text(json.dumps(role, indent=1) + "\n")
    os.replace(staged, path)


def live_store(root: Path) -> Path:
    """The file SQLite writes: on gorgona8 ``mlflow.db`` links to the SSD (ADR 0010)."""
    return (root / STORE_NAME).resolve()


def set_writable(root: Path, writable: bool) -> None:
    store = live_store(root)
    if store.exists():
        store.chmod(0o644 if writable else 0o444)


class Peer:
    """The other server, over ssh. Tests replace it with one on a local path."""

    def __init__(self, alias: str, root: str) -> None:
        self.alias, self.root = alias, root

    def read_role(self) -> tuple[bool, dict[str, Any] | None]:
        """(answers ssh, its role.json or None when it has none)."""
        result = subprocess.run(
            [*SSH, self.alias, f"cat {self.root}/{ROLE_FILE} 2>/dev/null || echo null"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, None
        return True, json.loads(result.stdout or "null")

    def pull_store(self, target: Path) -> str:
        result = subprocess.run(
            [find_rsync(), "-v", "--exe", REMOTE_RSYNC, f"{self.alias}:{self.root}/{STORE_NAME}", str(target)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SyncError(f"sqlite3_rsync failed: {(result.stderr or result.stdout).strip()}")
        return next((line for line in result.stdout.splitlines() if line.startswith("sent")), "")

    def exchange_outputs(self, root: Path, backup: str) -> None:
        """Additive both ways; a file a copy replaces goes to ``backup`` (decision 5)."""
        for source, target in (
            ([str(root / name) for name in OUTPUT_DIRS], f"{self.alias}:{self.root}/"),
            ([f"{self.alias}:{self.root}/{name}" for name in OUTPUT_DIRS], f"{root}/"),
        ):
            result = subprocess.run(
                ["rsync", "-a", "--ignore-missing-args", "-e", " ".join(SSH), "--backup", f"--backup-dir={backup}", *source, target],
                capture_output=True, text=True,
            )
            if result.returncode not in (0, 24):  # 24: a file vanished while being sent
                raise SyncError(f"rsync of the outputs failed: {result.stderr.strip()}")

    def yield_leadership(self, epoch: int) -> None:
        result = subprocess.run(
            [*SSH, self.alias, f"cd {self.root} && python3 scripts/leadership.py yield --epoch {epoch}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise StoreError(f"{self.alias} did not yield: {(result.stdout + result.stderr).strip()}")


class Services:
    """The MLflow server and the tunnel, both bound to 127.0.0.1:<port>; never both on one side."""

    def __init__(self, root: Path, port: int) -> None:
        self.root, self.port = root, port

    def _pids(self, pattern: str) -> list[int]:
        result = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", pattern], capture_output=True, text=True)
        return [int(pid) for pid in result.stdout.split() if int(pid) != os.getpid()]

    def _server(self) -> list[int]:
        return self._pids(f"[m]lflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port {self.port}")

    def _tunnel(self) -> list[int]:
        return self._pids(f"[s]sh .*-L 127.0.0.1:{self.port}:127.0.0.1:{self.port} ")

    def server_running(self) -> bool:
        return bool(self._server())

    def start_server(self) -> None:
        self.close_tunnel()
        env = dict(os.environ, PATH=f"{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}")
        env.setdefault("_MLFLOW_SERVER_ARTIFACT_ROOT", CANONICAL_ROOT.rstrip("/"))
        with open(self.root / SERVER_LOG, "a") as out:
            subprocess.Popen(
                ["./start_mlflow.sh", str(self.port)], cwd=self.root, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True,
            )

    def stop_server(self) -> None:
        """Stop the whole process group: uv, mlflow and its uvicorn workers (ADR 0009)."""
        groups = {os.getpgid(pid) for pid in self._server()}
        for group in groups:
            os.killpg(group, signal.SIGTERM)
        for _ in range(30):
            if not self._server():
                return
            time.sleep(1)
        for group in groups:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)

    def ensure_tunnel(self, alias: str) -> None:
        if self.server_running():
            self.stop_server()
        if self._tunnel():
            return
        subprocess.run(
            [*SSH, "-f", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
             "-o", "ServerAliveCountMax=3", "-L", f"127.0.0.1:{self.port}:127.0.0.1:{self.port}", alias],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=60,
        )

    def close_tunnel(self) -> None:
        for pid in self._tunnel():
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)


def checked_backup(root: Path, kind: str, now: float) -> str:
    """A full copy of the local store on the checkout's disk, checked (decision 7)."""
    store = live_store(root)
    folder = root / BACKUP_DIR
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < 2 * store.stat().st_size:
        raise StoreError(f"{folder} has {free / 1e9:.1f} GB free, less than twice the store; no backup taken")
    target = folder / f"{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%dT%H%M}-{kind}.db"
    copy_store(store, target)  # backup API, quick_check before it takes its name
    connection = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True)
    try:
        runs = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
    finally:
        connection.close()
    log(root, "backup_created", now, kind=kind, path=str(target), bytes=target.stat().st_size, runs=runs, check="ok")
    return str(target)


def guard_roots(root: Path, role: dict[str, Any], now: float) -> None:
    """Rewrite roots under this checkout's physical path to the canonical prefix (decision 1)."""
    if not live_store(root).exists():
        return  # sqlite3.connect would create an empty store
    physical = str(root.resolve())
    prefixes = [p for p in (f"file://{physical}/mlruns/", f"{physical}/mlruns/") if not p.startswith(CANONICAL_ROOT)]
    connection = sqlite3.connect(live_store(root), timeout=30)
    try:
        foreign = 0
        for table, column in ARTIFACT_ROOT_COLUMNS:
            for prefix in prefixes:
                with connection:
                    cursor = connection.execute(
                        f'UPDATE "{table}" SET "{column}" = ? || substr("{column}", ?) WHERE substr("{column}", 1, ?) = ?',
                        (CANONICAL_ROOT, len(prefix) + 1, len(prefix), prefix),
                    )
                if cursor.rowcount:
                    log(root, "root_normalized", now, column=f"{table}.{column}", prefix=prefix, rows=cursor.rowcount)
            foreign += connection.execute(
                f'SELECT count(*) FROM "{table}" WHERE "{column}" IS NULL OR substr("{column}", 1, ?) <> ?',
                (len(CANONICAL_ROOT), CANONICAL_ROOT),
            ).fetchone()[0]
    finally:
        connection.close()
    if foreign != role.get("foreign_roots", 0):
        log(root, "root_foreign", now, rows=foreign)
        role["foreign_roots"] = foreign


def replicate(root: Path, role: dict[str, Any], peer: Peer, now: float) -> None:
    """Pull the leader's store and swap outputs; the store is writable only during the copy."""
    with store_lock(root):
        set_writable(root, True)
        try:
            sent = peer.pull_store(live_store(root))
        finally:
            set_writable(root, False)
    peer.exchange_outputs(root, f"{BACKUP_DIR}/{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d}-outputs")
    role["last_contact"] = now
    log(root, "replicated", now, transfer=sent)


def reserve_ids(root: Path, epoch: int, now: float) -> None:
    """A deleted placeholder at ``epoch × 100000``: the new leader's experiments take ids above it."""
    reserved = epoch * EPOCH_ID_STRIDE
    connection = sqlite3.connect(live_store(root), timeout=30)
    try:
        with connection:
            connection.execute(
                "INSERT INTO experiments (experiment_id, name, artifact_location, lifecycle_stage, "
                "creation_time, last_update_time) VALUES (?, ?, ?, 'deleted', ?, ?)",
                (reserved, f"_epoch_{epoch}_reserved", f"{CANONICAL_ROOT}{reserved}", int(now * 1000), int(now * 1000)),
            )
    finally:
        connection.close()


def promote(root: Path, role: dict[str, Any], peer: Peer, services: Services, now: float, reason: str) -> None:
    if role["role"] != "follower":
        raise StoreError(f"this side is {role['role']}, not a follower")
    answers, peer_role = peer.read_role()
    if answers and peer_role and peer_role.get("role") == "leader":
        if reason == "manual":
            raise StoreError(f"{peer.alias} answers and leads at epoch {peer_role['epoch']}: use `switchover`")
        if not role.get("stalled_logged"):
            log(root, "replication_stalled", now, peer=peer.alias, last_contact=utc(role["last_contact"]))
            role["stalled_logged"] = True
        return
    if not role.get("fork_base"):
        role["fork_base"] = checked_backup(root, "fork-base", now)
        role["fork_since"] = now
    epoch = role["epoch"] + 1
    with store_lock(root):
        set_writable(root, True)
        reserve_ids(root, epoch, now)
    role.update(role="leader", epoch=epoch, promoted_at=now, promoted_from=peer.alias, stalled_logged=False)
    save_role(root, role)
    services.start_server()
    log(root, "promoted", now, epoch=epoch, reason=reason, base=role["fork_base"])


def follower_tick(root: Path, role: dict[str, Any], peer: Peer, services: Services, now: float) -> None:
    services.ensure_tunnel(peer.alias)
    answers, peer_role = peer.read_role()
    if answers and peer_role and peer_role.get("role") == "leader" and peer_role["epoch"] >= role["epoch"]:
        role["epoch"] = peer_role["epoch"]
        if now - (role.get("last_copy_try") or 0) >= COPY_EVERY:
            role["last_copy_try"] = now
            try:
                replicate(root, role, peer, now)
            except SyncError as error:
                log(root, "replication_failed", now, error=str(error))
            if role.get("last_contact") == now and role.get("fork_base"):
                Path(role["fork_base"]).unlink(missing_ok=True)
                log(root, "contact_regained", now, removed=role["fork_base"])
                role.update(fork_base=None, fork_since=None, stalled_logged=False)
    elif answers and peer_role and not role.get("peer_odd_logged"):
        log(root, "peer_not_leading", now, peer_role=peer_role.get("role"), peer_epoch=peer_role.get("epoch"))
        role["peer_odd_logged"] = True
    silent = now - (role.get("last_contact") or role["initialized_at"])
    if silent >= FORK_BASE_AFTER and not role.get("fork_base"):
        log(root, "peer_unreachable", now, answers_ssh=answers, silent_seconds=int(silent))
        role["fork_base"] = checked_backup(root, "fork-base", now)
        role["fork_since"] = now
    if silent >= PROMOTE_AFTER:
        promote(root, role, peer, services, now, "timeout")


def leader_tick(root: Path, role: dict[str, Any], peer: Peer, services: Services, now: float) -> None:
    answers, peer_role = peer.read_role()
    if answers and peer_role and peer_role.get("epoch", 0) > role["epoch"]:
        # The peer was promoted while this side was cut off (decision 8).
        role.update(role="deposed", deposed_at=now, deposed_by_epoch=peer_role["epoch"])
        services.stop_server()
        log(root, "deposed", now, epoch=role["epoch"], peer_epoch=peer_role["epoch"])
        return
    if answers and peer_role and peer_role.get("role") == "deposed" and not role.get("peer_odd_logged"):
        log(root, "peer_awaits_reconcile", now, peer=peer.alias)
        role["peer_odd_logged"] = True
    if not services.server_running():
        services.start_server()
        log(root, "server_started", now, port=services.port)
    guard_roots(root, role, now)


def deposed_tick(root: Path, role: dict[str, Any], services: Services, now: float) -> None:
    if services.server_running():
        services.stop_server()
    if not role.get("read_only"):
        with store_lock(root):
            try:
                refuse_if_busy(root / STORE_NAME)
            except StoreError:
                return  # its trainings still write; read-only once they end
            set_writable(root, False)
        role["read_only"] = True
        log(root, "read_only", now)


@contextlib.contextmanager
def tick_lock(root: Path) -> Iterator[bool]:
    """Non-blocking: a tick still copying 15 GB makes the next ones skip."""
    import fcntl

    with open(root / TICK_LOCK, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def tick(root: Path, now: float, peer_for: Callable[[dict[str, Any]], Peer], services_for: Callable[[dict[str, Any]], Services]) -> None:
    role = load_role(root)
    if role is None:
        return  # not initialised: the cron line can go in before `init` (ADR Consequences)
    with tick_lock(root) as mine:
        if not mine:
            return
        role = load_role(root) or role
        peer, services = peer_for(role), services_for(role)
        if role["role"] == "leader":
            leader_tick(root, role, peer, services, now)
        elif role["role"] == "follower":
            follower_tick(root, role, peer, services, now)
        else:
            deposed_tick(root, role, services, now)
        save_role(root, role)


def yield_leadership(root: Path, services: Services, epoch: int, now: float) -> None:
    """The leader's half of a switchover: stop serving, go read-only, follow at ``epoch``."""
    role = load_role(root)
    if role is None or role["role"] != "leader" or epoch != role["epoch"] + 1:
        raise StoreError(f"cannot yield at epoch {epoch}: role is {role}")
    services.stop_server()
    with store_lock(root):
        try:
            refuse_if_busy(root / STORE_NAME)
        except StoreError:
            services.start_server()
            raise
        set_writable(root, False)
    role.update(role="follower", epoch=epoch, last_contact=now, last_copy_try=now)
    save_role(root, role)
    log(root, "yielded", now, epoch=epoch)


def switchover(root: Path, peer: Peer, services: Services, now: float) -> None:
    """The follower's half: the leader yields, its last pages come over, this side leads."""
    role = load_role(root)
    if role is None or role["role"] != "follower":
        raise StoreError(f"switchover runs on the follower; this side is {role and role['role']}")
    answers, peer_role = peer.read_role()
    if not answers or not peer_role:
        raise StoreError(f"{peer.alias} does not answer; a switchover needs both sides (see `promote`)")
    epoch = max(role["epoch"], peer_role["epoch"]) + 1
    if not (peer_role["role"] == "follower" and peer_role["epoch"] == epoch - 1 and role["epoch"] < epoch - 1):
        peer.yield_leadership(epoch)  # skipped only when a cut-short switchover already got this far
    with tick_lock(root):
        replicate(root, role, peer, now)
        with store_lock(root):
            set_writable(root, True)
        role.update(role="leader", epoch=epoch, fork_base=None, fork_since=None)
        save_role(root, role)
    services.start_server()
    log(root, "switchover", now, epoch=epoch, previous_leader=peer.alias)


def status(root: Path, now: float) -> None:
    role = load_role(root)
    if role is None:
        print(f"no {ROLE_FILE}: this checkout is not part of a pair (run `init`)")
        return
    contact = role.get("last_contact")
    print(f"role {role['role']}, epoch {role['epoch']}, peer {role['peer']}:{role['peer_root']}, port {role['port']}")
    print(f"last contact: {utc(contact) + f' ({(now - contact) / 60:.0f} min ago)' if contact else 'never'}")
    if role.get("fork_since"):
        print(f"fork open for {(now - role['fork_since']) / 3600:.1f} h, base {role['fork_base']}")
    if role["role"] == "deposed":
        print("deposed: waiting for `reconcile` (not written yet), which merges this store into the leader's")
    backups = root / BACKUP_DIR
    if backups.is_dir():
        used = sum(path.stat().st_size for path in backups.rglob("*") if path.is_file())
        print(f"{BACKUP_DIR}: {used / 1e9:.1f} GB")
    print(f"store {live_store(root)}: quick_check {quick_check(live_store(root)) if live_store(root).exists() else 'absent'}")
    lines = (root / LOG_FILE).read_text().splitlines()[-5:] if (root / LOG_FILE).exists() else []
    for line in lines:
        print("  " + line)


def init(root: Path, role_name: str, peer: str, peer_root: str, port: int, now: float) -> None:
    if load_role(root):
        raise StoreError(f"{ROLE_FILE} already exists; this checkout is already part of a pair")
    (root / SYNC_DIR).mkdir(exist_ok=True)
    role = dict(role=role_name, epoch=1, peer=peer, peer_root=peer_root, port=port, initialized_at=now, last_contact=None)
    save_role(root, role)
    if role_name == "follower":
        set_writable(root, False)
    log(root, "initialized", now, role=role_name, peer=peer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="the checkout (default: this one)")
    commands = parser.add_subparsers(dest="command", required=True)
    init_command = commands.add_parser("init", help="make this checkout one side of the pair")
    init_command.add_argument("--role", choices=("leader", "follower"), required=True)
    init_command.add_argument("--peer", required=True, help="the other server's ssh alias")
    init_command.add_argument("--peer-root", required=True, help="the other server's checkout")
    init_command.add_argument("--port", type=int, default=PORT)
    commands.add_parser("status")
    commands.add_parser("tick", help="cron, every minute")
    commands.add_parser("switchover", help="on the follower: swap roles with a leader that answers")
    commands.add_parser("promote", help="on the follower: lead without the leader, which must not answer")
    yield_command = commands.add_parser("yield", help="internal: the leader's half of a switchover")
    yield_command.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parents[1]).absolute()
    now = time.time()

    def peer_for(role: dict[str, Any]) -> Peer:
        return Peer(role["peer"], role["peer_root"])

    def services_for(role: dict[str, Any]) -> Services:
        return Services(root, role["port"])

    try:
        if args.command == "init":
            init(root, args.role, args.peer, args.peer_root, args.port, now)
        elif args.command == "status":
            status(root, now)
        elif args.command == "tick":
            tick(root, now, peer_for, services_for)
        else:
            role = load_role(root)
            if role is None:
                raise StoreError(f"no {ROLE_FILE}; run `init` first")
            if args.command == "switchover":
                switchover(root, peer_for(role), services_for(role), now)
            elif args.command == "yield":
                yield_leadership(root, services_for(role), args.epoch, now)
            else:
                promote(root, role, peer_for(role), services_for(role), now, "manual")
                save_role(root, role)
    except StoreError as error:
        log(root, "refused", now, command=args.command, reason=str(error))
        return EXIT_REFUSED
    except SyncError as error:
        log(root, "failed", now, command=args.command, error=str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
