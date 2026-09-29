#!/usr/bin/env python3
"""Hand the tracking store, artifacts and outputs between this checkout and a Linux server.

One side writes the MLflow store at a time. A push sends the code, the outputs and a
snapshot of the store to the server; a pull brings the server's outputs and store back.
The store travels whole, with its artifact roots rewritten for the machine that receives
it, and a sync overwrites only a store that is unchanged since the last sync. See
ADR 0009 for the protocol and what it refuses.

Usage (a dry run by default, run from Windows; rsync and ssh are called through WSL):

    uv run --python 3.11 python scripts/sync_remote.py init --host gpu --remote-root /scratch/me/TRIDENT
    uv run --python 3.11 python scripts/sync_remote.py status
    uv run --python 3.11 python scripts/sync_remote.py code --apply   # code only
    uv run --python 3.11 python scripts/sync_remote.py push --apply   # code, outputs, store
    uv run --python 3.11 python scripts/sync_remote.py pull --apply   # outputs, store

An empty ``--host`` rehearses against a directory in this machine's WSL instead of a
server. The server needs Python 3.9+, rsync and key-based ssh from WSL; nothing is
installed there: each server-side step runs this file through ``ssh <host> python3 -``.

Commits made on the server come back through GitHub: pull them here before ``code`` or
``push``, which refuse while a server branch holds a commit no local branch contains.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

STORE_NAME = "mlflow.db"
SYNC_DIR = "sync"
STATE_FILE = f"{SYNC_DIR}/state.json"
CONFIG_FILE = f"{SYNC_DIR}/config.json"
# argparse already exits with 2 on a usage error.
EXIT_REFUSED = 3
# Plain files, each written once by one run: copied additively in both directions.
OUTPUT_DIRS = ("mlruns", "results", "metrics")
# Inputs git ignores but training reads.
UNTRACKED_INPUTS = ("datasets/processed_datasets",)
# Top-level names the code push never sends (``.git`` travels on its own).
NEVER_PUSHED = frozenset({".git", ".venv", SYNC_DIR, *OUTPUT_DIRS})

# The two columns MLflow resolves artifact files from. Everything else that names a path
# (``mlflow.source.name``, ``mlflow.user``, the ``optuna_storage`` param) records where a
# run happened and is never rewritten.
ARTIFACT_ROOT_COLUMNS = (("experiments", "artifact_location"), ("runs", "artifact_uri"))


class SyncError(RuntimeError):
    """The hand-off stopped before changing anything it could not undo."""


class StoreError(SyncError):
    """A store the hand-off refuses to move, translate or overwrite."""


# --- the store --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    def _held_open_windows(path: Path) -> list[str]:
        """Windows names no holder cheaply, but refuses an exclusive open while one exists."""
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        generic_read, no_sharing, open_existing, normal = 0x80000000, 0, 3, 0x80
        handle = create_file(str(path), generic_read, no_sharing, None, open_existing, normal, None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            error = ctypes.get_last_error()
            if error == 32:  # ERROR_SHARING_VIOLATION
                return ["another process (close it, or stop the MLflow UI)"]
            if error == 2:  # ERROR_FILE_NOT_FOUND
                return []
            raise OSError(error, ctypes.FormatError(error), str(path))
        kernel32.CloseHandle(handle)
        return []


def _held_open_linux(path: Path) -> list[str]:
    target = str(path.resolve())
    holders: list[str] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
            if any(os.readlink(descriptor) == target for descriptor in descriptors):
                command = (process / "comm").read_text(encoding="utf-8").strip()
                holders.append(f"pid {process.name} ({command})")
        except OSError:  # another user's process, or one that just exited
            continue
    return holders


def held_open_by(path: Path) -> list[str]:
    """The processes holding ``path`` open: a training, the MLflow UI, a notebook."""
    if not path.exists():
        return []
    if sys.platform == "win32":
        return _held_open_windows(path)
    if sys.platform.startswith("linux"):
        return _held_open_linux(path)
    raise SyncError(f"cannot tell which processes hold files open on {sys.platform}")


def refuse_if_busy(store: Path) -> None:
    """A store is handed over only when nothing is writing it and no write was cut short."""
    holders = held_open_by(store)
    if holders:
        raise StoreError(f"{store} is open by {', '.join(holders)}; wait for it or stop it")
    for suffix in ("-journal", "-wal"):
        if Path(f"{store}{suffix}").exists():
            raise StoreError(
                f"{store}{suffix} exists: a write to the store was interrupted. Open the store "
                "once with MLflow (or sqlite3) so SQLite recovers it, then sync again."
            )


def _foreign_roots(
    connection: sqlite3.Connection, source_prefix: str, target_prefix: str
) -> list[str]:
    """Artifact roots under neither prefix; ``mlflow-artifacts:`` roots belong to no machine."""
    foreign: list[str] = []
    for table, column in ARTIFACT_ROOT_COLUMNS:
        rows = connection.execute(
            f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NULL OR ('
            f'substr("{column}", 1, ?) <> ? AND substr("{column}", 1, ?) <> ? '
            f"AND \"{column}\" NOT LIKE 'mlflow-artifacts:%')",
            (len(source_prefix), source_prefix, len(target_prefix), target_prefix),
        ).fetchall()
        foreign.extend(f"{table}.{column}: {value!r}" for (value,) in rows)
    return foreign


def _columns_holding(connection: sqlite3.Connection, text: str) -> list[str]:
    """Every ``table.column`` of the store in which some value contains ``text``."""
    holding: list[str] = []
    tables = [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    ]
    for table in tables:
        for _, column, declared, *_ in connection.execute(f'PRAGMA table_info("{table}")'):
            affinity = str(declared).upper()
            if affinity and not any(kind in affinity for kind in ("CHAR", "TEXT", "CLOB")):
                continue
            found = connection.execute(
                f'SELECT 1 FROM "{table}" WHERE instr("{column}", ?) > 0 LIMIT 1', (text,)
            ).fetchone()
            if found:
                holding.append(f"{table}.{column}")
    return holding


def translate_store(store: Path, source_prefix: str, target_prefix: str) -> dict[str, int]:
    """Rewrite every artifact root under ``source_prefix`` to ``target_prefix``, in place.

    Prefixes are compared with ``substr``, never ``LIKE``: ``_`` and ``%`` are ordinary
    characters in a path.
    """
    connection = sqlite3.connect(store)
    try:
        foreign = _foreign_roots(connection, source_prefix, target_prefix)
        if foreign:
            listed = "\n  ".join(foreign[:10])
            raise StoreError(
                f"{len(foreign)} artifact root(s) sit under neither {source_prefix!r} nor "
                f"{target_prefix!r}; nothing was translated:\n  {listed}"
            )
        counts: dict[str, int] = {}
        with connection:
            for table, column in ARTIFACT_ROOT_COLUMNS:
                cursor = connection.execute(
                    f'UPDATE "{table}" SET "{column}" = ? || substr("{column}", ?) '
                    f'WHERE substr("{column}", 1, ?) = ?',
                    (target_prefix, len(source_prefix) + 1, len(source_prefix), source_prefix),
                )
                counts[f"{table}.{column}"] = cursor.rowcount
            leftovers = _columns_holding(connection, source_prefix)
            if leftovers:
                raise StoreError(
                    f"{source_prefix!r} is still stored outside the translated columns, in "
                    f"{', '.join(leftovers)}; the translation was rolled back"
                )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise StoreError(f"integrity_check failed after translating: {integrity}")
        return counts
    finally:
        connection.close()


def snapshot_store(store: Path, destination: Path) -> str:
    """Copy ``store`` consistently to ``destination``; return the digest of ``store``.

    The digest is taken before the copy: a write slipping in between leaves a recorded
    digest that no longer matches, which the next sync reports instead of overwriting.
    """
    refuse_if_busy(store)
    digest = sha256_file(store)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    source = sqlite3.connect(f"{store.resolve().as_uri()}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    return digest


def refuse_if_changed(where: str, recorded: str | None, current: str | None) -> None:
    """The one rule of the hand-off: a sync overwrites only a store it left unchanged.

    A side with no store has nothing to lose, which is also how a divergence is resolved:
    move one side's store aside and sync the other in.
    """
    if current == recorded or current is None:
        return
    if recorded is None:
        raise StoreError(
            f"{where} holds a store no sync has recorded (sha256 {current}); move it aside "
            "if it should be replaced, or sync it the other way first"
        )
    raise StoreError(
        f"{where} changed since the last sync (recorded {recorded[:12]}, now {current[:12]}): "
        "something wrote to it, a training, a backfill or an edit in the UI. Sync that side "
        "the other way first, or move its store aside if its changes should be dropped."
    )


def install_store(
    root: Path, incoming: Path, expected_sha256: str | None, incoming_sha256: str
) -> str:
    """Put ``incoming`` in place as ``root``'s store, keeping the one it replaces.

    Refuses unless the store in place is the one the last sync left there.
    """
    store = root / STORE_NAME
    refuse_if_busy(store)
    refuse_if_changed(str(root), expected_sha256, sha256_file(store) if store.exists() else None)
    received = sha256_file(incoming)
    if received != incoming_sha256:
        raise StoreError(
            f"the store arrived damaged: sha256 {received}, expected {incoming_sha256}"
        )
    if store.exists():
        os.replace(store, root / SYNC_DIR / f"{STORE_NAME}.prev")
    os.replace(incoming, store)
    return received


def _never_pushed(relative: str) -> bool:
    top = relative.split("/", 1)[0]
    return top in NEVER_PUSHED or top.startswith(STORE_NAME)


def delete_pushed_files(root: Path, relatives: list[str]) -> list[str]:
    """Delete files an earlier code push sent and the checkout no longer has.

    Only such files: every path is checked before any is deleted, so a list naming the
    store, an output or anything outside ``root`` deletes nothing.
    """
    base = root.resolve()
    for relative in relatives:
        if _never_pushed(relative) or base not in (root / relative).resolve().parents:
            raise SyncError(f"refusing to delete {relative!r}: the code push never sends it")
    deleted: list[str] = []
    for relative in relatives:
        target = root / relative
        if target.is_file():
            target.unlink()
            deleted.append(relative)
    return deleted


def branch_tips(root: Path) -> dict[str, str]:
    """The commit each branch of ``root``'s repository points at; none without a ``.git``."""
    if not (root / ".git").exists():
        return {}
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "for-each-ref", "--format=%(refname:short) %(objectname)",
             "refs/heads"],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise SyncError(f"cannot list the branches of {root}: {error}") from error
    return {name: commit for name, commit in (line.rsplit(" ", 1) for line in listed.splitlines())}


# --- the server's side: run there as ``python3 - remote <command>`` ----------------------


def remote_command(argv: list[str]) -> dict[str, object]:
    import argparse

    parser = argparse.ArgumentParser(prog="sync_remote.py remote")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "snapshot", "install", "delete"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        if name == "delete":
            command.add_argument("--list", required=True, help="JSON list, relative to --root")
        if name == "install":
            command.add_argument("--expected-sha256", default="")
            command.add_argument("--incoming-sha256", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    store = root / STORE_NAME
    (root / SYNC_DIR).mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        return {
            "store_sha256": sha256_file(store) if store.exists() else None,
            "holders": held_open_by(store),
            "dirs": [name for name in OUTPUT_DIRS if (root / name).is_dir()],
            "branches": branch_tips(root),
        }
    if args.command == "snapshot":
        digest = snapshot_store(store, root / SYNC_DIR / "outgoing.db")
        return {
            "store_sha256": digest,
            "dirs": [name for name in OUTPUT_DIRS if (root / name).is_dir()],
        }
    if args.command == "delete":
        listed = json.loads((root / args.list).read_text(encoding="utf-8"))
        return {"deleted": delete_pushed_files(root, [str(item) for item in listed])}
    installed = install_store(
        root,
        root / SYNC_DIR / "incoming.db",
        args.expected_sha256 or None,
        args.incoming_sha256,
    )
    return {"store_sha256": installed}


# --- this checkout's side ----------------------------------------------------------------


def artifact_prefix(root: str) -> str:
    """The artifact-root prefix MLflow gives experiments created from checkout ``root``."""
    return f"file:///{root.strip('/')}/mlruns/"


@dataclass(frozen=True)
class SyncConfig:
    host: str
    remote_root: str
    local_prefix: str
    remote_prefix: str
    remote_python: str = "python3"
    use_wsl: bool = False

    @classmethod
    def load(cls, root: Path) -> SyncConfig:
        path = root / CONFIG_FILE
        if not path.exists():
            raise SyncError(f"{path} does not exist; run `sync_remote.py init` first")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, root: Path) -> None:
        path = root / CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class SyncState:
    local_store_sha256: str | None = None
    remote_store_sha256: str | None = None
    pushed_files: list[str] = field(default_factory=list)
    last_sync: str | None = None

    @classmethod
    def load(cls, root: Path) -> SyncState:
        path = root / STATE_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, root: Path) -> None:
        path = root / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(temporary, path)


class Transport(Protocol):
    """How files and commands reach the server. Paths are relative to each side's root."""

    def remote(self, *args: str) -> dict[str, object]: ...

    def send_files(self, files: list[str], *, dry_run: bool) -> int: ...

    def send_dir(self, relative: str, *, dry_run: bool, delete: bool = False) -> int: ...

    def send_file(self, local: str, remote: str, *, dry_run: bool) -> int: ...

    def fetch_dir(self, relative: str, *, dry_run: bool) -> int: ...

    def fetch_file(self, remote: str, local: str, *, dry_run: bool) -> int: ...


@dataclass(frozen=True)
class Workspace:
    root: Path
    config: SyncConfig
    transport: Transport
    say: Callable[[str], None] = print


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _names(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _branches(value: object) -> dict[str, str]:
    return {str(name): str(commit) for name, commit in value.items()} if isinstance(value, dict) else {}


def refuse_if_server_has_commits(root: Path, branches: dict[str, str]) -> None:
    """The code push mirrors ``.git``, which would roll back a commit made on the server.

    Every branch of the server must point at a commit some branch here contains, so merely
    fetched is not enough: the push would still send the older files. A commit made there
    reaches this checkout through GitHub (or any pull) before the next push.
    """

    def contained_here(commit: str) -> bool:
        listed = subprocess.run(
            ["git", "-C", str(root), "for-each-ref", "--contains", commit, "--count=1",
             "--format=%(refname)", "refs/heads"],
            capture_output=True,
        )
        return listed.returncode == 0 and bool(listed.stdout.strip())

    missing = [
        f"{name} ({commit[:7]})"
        for name, commit in sorted(branches.items())
        if not contained_here(commit)
    ]
    if missing:
        raise SyncError(
            f"the server has commits no branch here contains, on {', '.join(missing)}. The code "
            "push mirrors .git and would roll them back: push them from the server (to GitHub) "
            "and pull them here first. A branch deleted here on purpose: delete it there too."
        )


def code_files(root: Path) -> list[str]:
    """The code as it is on disk: tracked and untracked-but-not-ignored files, plus inputs."""
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    files = {relative for relative in listed.split("\0") if relative and (root / relative).is_file()}
    for inputs in UNTRACKED_INPUTS:
        if (root / inputs).is_dir():
            files.update(
                path.relative_to(root).as_posix()
                for path in (root / inputs).rglob("*")
                if path.is_file()
            )
    return sorted(relative for relative in files if not _never_pushed(relative))


def _uncommitted(root: Path) -> str | None:
    """The commit a server run will record, when the tracked code differs from it."""
    changed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not changed:
        return None
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, check=True
    )
    return head.stdout.decode().strip()


def _mode(apply: bool) -> str:
    return "APPLIED" if apply else "DRY RUN"


def _send_code(workspace: Workspace, state: SyncState, apply: bool) -> None:
    files = code_files(workspace.root)
    sent = workspace.transport.send_files(files, dry_run=not apply)
    history = workspace.transport.send_dir(".git", dry_run=not apply, delete=True)
    workspace.say(f"[{_mode(apply)}] code: {len(files)} files, {sent} changed; .git: {history} changed")
    stale = sorted(set(state.pushed_files) - set(files))
    if stale and apply:
        listing = f"{SYNC_DIR}/stale-files.json"
        (workspace.root / listing).write_text(json.dumps(stale), encoding="utf-8")
        workspace.transport.send_file(listing, listing, dry_run=False)
        workspace.transport.remote(
            "delete", "--root", workspace.config.remote_root, "--list", listing
        )
    if stale:
        workspace.say(f"[{_mode(apply)}] code: {len(stale)} files gone from the checkout removed")
    head = _uncommitted(workspace.root)
    if head:
        workspace.say(
            f"note: tracked files differ from {head}; runs on the server will record {head} "
            "but run this working tree"
        )
    if apply:
        state.pushed_files = files
        state.save(workspace.root)


def push_code(workspace: Workspace, *, apply: bool, while_busy: bool = False) -> None:
    """Send the code only; the stores stay where they are."""
    state = SyncState.load(workspace.root)
    server = workspace.transport.remote("prepare", "--root", workspace.config.remote_root)
    holders = _names(server["holders"])
    if holders and not while_busy:
        raise SyncError(
            f"the server's store is open by {', '.join(holders)}: a batch still starting "
            "processes would run the new code for its remaining runs. Wait, or pass --while-busy."
        )
    refuse_if_server_has_commits(workspace.root, _branches(server["branches"]))
    _send_code(workspace, state, apply)


def push(workspace: Workspace, *, apply: bool) -> None:
    """Send code, outputs and the store; the server's store becomes this checkout's."""
    config, transport, mode = workspace.config, workspace.transport, _mode(apply)
    state = SyncState.load(workspace.root)
    refuse_if_busy(workspace.root / STORE_NAME)
    server = transport.remote("prepare", "--root", config.remote_root)
    if _names(server["holders"]):
        raise StoreError(f"the server's store is open by {', '.join(_names(server['holders']))}")
    refuse_if_server_has_commits(workspace.root, _branches(server["branches"]))
    refuse_if_changed("the server", state.remote_store_sha256, _text(server["store_sha256"]))
    _send_code(workspace, state, apply)
    for name in OUTPUT_DIRS:
        changed = transport.send_dir(name, dry_run=not apply)
        workspace.say(f"[{mode}] {name}/: {changed} files sent")
    outgoing = workspace.root / SYNC_DIR / "outgoing.db"
    local_digest = snapshot_store(workspace.root / STORE_NAME, outgoing)
    counts = translate_store(outgoing, config.local_prefix, config.remote_prefix)
    workspace.say(f"[{mode}] store translated for the server: {counts}")
    if not apply:
        outgoing.unlink()
        workspace.say(f"[{mode}] the server's store was left as it is")
        return
    outgoing_digest = sha256_file(outgoing)
    transport.send_file(f"{SYNC_DIR}/outgoing.db", f"{SYNC_DIR}/incoming.db", dry_run=False)
    transport.remote(
        "install",
        "--root",
        config.remote_root,
        "--expected-sha256",
        state.remote_store_sha256 or "",
        "--incoming-sha256",
        outgoing_digest,
    )
    state.local_store_sha256 = local_digest
    state.remote_store_sha256 = outgoing_digest
    state.last_sync = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.save(workspace.root)
    outgoing.unlink()
    workspace.say(f"[{mode}] the server's store is now this checkout's ({outgoing_digest[:12]})")


def pull(workspace: Workspace, *, apply: bool) -> None:
    """Fetch outputs and the store; this checkout's store becomes the server's."""
    config, transport, mode = workspace.config, workspace.transport, _mode(apply)
    state = SyncState.load(workspace.root)
    store = workspace.root / STORE_NAME
    refuse_if_busy(store)
    current = sha256_file(store) if store.exists() else None
    refuse_if_changed("this checkout", state.local_store_sha256, current)
    snapshot = transport.remote("snapshot", "--root", config.remote_root)
    remote_digest = _text(snapshot["store_sha256"])
    for name in _names(snapshot["dirs"]):
        changed = transport.fetch_dir(name, dry_run=not apply)
        workspace.say(f"[{mode}] {name}/: {changed} files fetched")
    (workspace.root / SYNC_DIR).mkdir(parents=True, exist_ok=True)
    transport.fetch_file(f"{SYNC_DIR}/outgoing.db", f"{SYNC_DIR}/incoming.db", dry_run=False)
    incoming = workspace.root / SYNC_DIR / "incoming.db"
    counts = translate_store(incoming, config.remote_prefix, config.local_prefix)
    workspace.say(f"[{mode}] store translated for this checkout: {counts}")
    if not apply:
        incoming.unlink()
        workspace.say(f"[{mode}] this checkout's store was left as it is")
        return
    installed = install_store(
        workspace.root, incoming, state.local_store_sha256, sha256_file(incoming)
    )
    state.local_store_sha256 = installed
    state.remote_store_sha256 = remote_digest
    state.last_sync = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.save(workspace.root)
    workspace.say(f"[{mode}] this checkout's store is now the server's ({installed[:12]})")


def status(workspace: Workspace) -> None:
    """Say which direction may sync now, without changing anything."""
    state = SyncState.load(workspace.root)
    store = workspace.root / STORE_NAME
    here = sha256_file(store) if store.exists() else None
    server = workspace.transport.remote("prepare", "--root", workspace.config.remote_root)
    there = _text(server["store_sha256"])
    # The predicate refuse_if_changed applies: a side without a store accepts one.
    local_changed = here is not None and here != state.local_store_sha256
    remote_changed = there is not None and there != state.remote_store_sha256
    workspace.say(f"last sync: {state.last_sync or 'never'}")
    for side, now, recorded in (
        ("this checkout", here, state.local_store_sha256),
        ("the server", there, state.remote_store_sha256),
    ):
        if now is None:
            verdict = "accepts one"
        elif now == recorded:
            verdict = "unchanged since the last sync"
        elif recorded is None:
            verdict = "no sync has recorded it"
        else:
            verdict = "changed since the last sync"
        workspace.say(f"{side}: {now[:12] if now else 'no store'} ({verdict})")
    holders = _names(server["holders"])
    if holders:
        workspace.say(f"the server's store is open by {', '.join(holders)}")
    try:
        refuse_if_server_has_commits(workspace.root, _branches(server["branches"]))
    except SyncError as error:
        workspace.say(f"code and push would be refused: {error}")
    if local_changed and remote_changed:
        workspace.say(
            "DIVERGED: both stores changed, so neither push nor pull will run. Decide which "
            "side's new runs to keep and move the other store aside."
        )
    elif local_changed:
        workspace.say("push is allowed; pull would be refused")
    elif remote_changed:
        workspace.say("pull is allowed; push would be refused")
    else:
        workspace.say("both directions are allowed")


def to_wsl(path: Path) -> str:
    """A Windows path as WSL mounts it: ``C:/Users/x`` becomes ``/mnt/c/Users/x``."""
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) != 2 or drive[1] != ":":
        raise SyncError(f"{resolved} is not on a drive letter WSL mounts under /mnt")
    return f"/mnt/{drive[0].lower()}{resolved.as_posix()[2:]}"


# Per call, so a host entry written for interactive logins (``RequestTTY``,
# ``RemoteCommand``, ``LocalForward``) neither refuses a command, mangles rsync's stream nor
# fights an open login for its forwarded ports, and a missing key fails instead of prompting.
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "RequestTTY=no",
    "-o", "RemoteCommand=none",
    "-o", "ClearAllForwardings=yes",
)


class RsyncTransport:
    """rsync and ssh; through WSL on Windows, where their keys and known_hosts live."""

    def __init__(self, root: Path, config: SyncConfig) -> None:
        self.root = root
        self.config = config

    def _run(
        self, command: list[str], stdin: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        prefix = ["wsl", "-e"] if self.config.use_wsl else []
        return subprocess.run([*prefix, *command], input=stdin, capture_output=True)

    def _local(self, relative: str = "") -> str:
        path = self.root / relative
        return to_wsl(path) if self.config.use_wsl else str(path)

    def _remote(self, relative: str = "") -> str:
        path = self.config.remote_root.rstrip("/")
        if relative:
            path = f"{path}/{relative}"
        return f"{self.config.host}:{path}" if self.config.host else path

    def _rsync(self, arguments: list[str], dry_run: bool) -> int:
        command = ["rsync", "-rtz", "--protect-args", "--itemize-changes"]
        if self.config.host:
            command += ["-e", " ".join(("ssh", *SSH_OPTIONS))]
        if dry_run:
            command.append("--dry-run")
        result = self._run([*command, *arguments])
        if result.returncode:
            error = result.stderr.decode(errors="replace").strip()
            raise SyncError(f"rsync exited with {result.returncode}: {error}")
        lines = result.stdout.decode(errors="replace").splitlines()
        return sum(1 for line in lines if line[:2] in ("<f", ">f") or line.startswith("*deleting"))

    def remote(self, *args: str) -> dict[str, object]:
        script = Path(__file__).read_bytes()
        if self.config.host:
            line = f"{self.config.remote_python} - remote {shlex.join(args)}"
            command = ["ssh", *SSH_OPTIONS, self.config.host, line]
        else:
            command = [*shlex.split(self.config.remote_python), "-", "remote", *args]
        result = self._run(command, stdin=script)
        message = result.stderr.decode(errors="replace").strip()
        if result.returncode == EXIT_REFUSED:
            raise StoreError(f"the server refused: {message}")
        if result.returncode:
            raise SyncError(f"`remote {args[0]}` exited with {result.returncode}: {message}")
        answer: dict[str, object] = json.loads(result.stdout.decode().strip().splitlines()[-1])
        return answer

    def send_files(self, files: list[str], *, dry_run: bool) -> int:
        listing = self.root / SYNC_DIR / "code-files.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_bytes(b"".join(name.encode("utf-8") + b"\0" for name in files))
        arguments = ["--files-from", self._local(f"{SYNC_DIR}/code-files.txt"), "--from0"]
        return self._rsync([*arguments, f"{self._local()}/", f"{self._remote()}/"], dry_run)

    def send_dir(self, relative: str, *, dry_run: bool, delete: bool = False) -> int:
        if not (self.root / relative).is_dir():
            return 0
        arguments = ["--delete"] if delete else []
        return self._rsync(
            [*arguments, f"{self._local(relative)}/", f"{self._remote(relative)}/"], dry_run
        )

    def send_file(self, local: str, remote: str, *, dry_run: bool) -> int:
        return self._rsync([self._local(local), self._remote(remote)], dry_run)

    def fetch_dir(self, relative: str, *, dry_run: bool) -> int:
        return self._rsync([f"{self._remote(relative)}/", f"{self._local(relative)}/"], dry_run)

    def fetch_file(self, remote: str, local: str, *, dry_run: bool) -> int:
        return self._rsync([self._remote(remote), self._local(local)], dry_run)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["remote"]:
        # The server's side. It never reads ``__file__``: it runs from stdin.
        try:
            print(json.dumps(remote_command(arguments[1:])))
        except StoreError as error:
            print(error, file=sys.stderr)
            return EXIT_REFUSED
        except SyncError as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="the checkout (default: this one)")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help=f"write {CONFIG_FILE}")
    init.add_argument("--host", required=True, help="ssh host or alias; empty rehearses in WSL")
    init.add_argument("--remote-root", required=True, help="absolute path of the server checkout")
    init.add_argument("--remote-python", default="python3")
    init.add_argument("--local-prefix", default=None, help="default: derived from --root")
    init.add_argument("--remote-prefix", default=None, help="default: derived from --remote-root")
    commands.add_parser("status", help="which direction may sync now")
    for name, summary in (
        ("code", "send the code only"),
        ("push", "send code, outputs and the store"),
        ("pull", "fetch outputs and the store"),
    ):
        command = commands.add_parser(name, help=summary)
        command.add_argument("--apply", action="store_true", help="write (default: a dry run)")
        if name == "code":
            command.add_argument("--while-busy", action="store_true")
    args = parser.parse_args(arguments)
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()

    try:
        if args.command == "init":
            if not args.remote_root.startswith("/"):
                raise SyncError("--remote-root must be an absolute Linux path")
            config = SyncConfig(
                host=args.host,
                remote_root=args.remote_root.rstrip("/"),
                local_prefix=args.local_prefix or artifact_prefix(root.as_posix()),
                remote_prefix=args.remote_prefix or artifact_prefix(args.remote_root),
                remote_python=args.remote_python,
                use_wsl=sys.platform == "win32",
            )
            config.save(root)
            print(json.dumps(asdict(config), indent=2))
            return 0
        config = SyncConfig.load(root)
        workspace = Workspace(root, config, RsyncTransport(root, config))
        if args.command == "status":
            status(workspace)
        elif args.command == "code":
            push_code(workspace, apply=args.apply, while_busy=args.while_busy)
        elif args.command == "push":
            push(workspace, apply=args.apply)
        else:
            pull(workspace, apply=args.apply)
    except StoreError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except SyncError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
