#!/usr/bin/env python3
"""Keep gorgona8's tracking store on the SSD, with a replica on the hard disk (ADR 0010).

The checkout sits on /scratch2, a hard disk where each SQLite commit costs ~225 ms, so the
store lives on the SSD, under /var/tmp, and ``mlflow.db`` in the checkout is a link to it:
``sqlite:///mlflow.db`` keeps working everywhere. Only that SSD file is ever written.
``sync`` copies it to ``sync/mlflow.replica.db`` on the hard disk, sending only the pages
that changed; cron runs it every 10 minutes, and it can be run by hand at any time. When the
SSD file is gone, ``sync`` restores it from the replica instead.

Usage (gorgona8 only, standard library, so the system python3 runs it; ``sqlite3_rsync``
from SQLite 3.50+ must be on PATH or in ~/.local/bin):

    python3 scripts/store_replica.py adopt   # once: move the store to the SSD and link it
    python3 scripts/store_replica.py sync    # replica <- SSD, or SSD <- replica when gone

Cron entry (``crontab -e``), logging in UTC like the rest of the server:

    */10 * * * * cd /scratch2/diogoneiss/TRIDENT && python3 scripts/store_replica.py sync >> sync/replica.log 2>&1
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sync_remote import (  # noqa: E402
    EXIT_REFUSED,
    STORE_NAME,
    SYNC_DIR,
    StoreError,
    SyncError,
    processes_holding,
    refuse_if_busy,
    store_lock,
)

REPLICA_FILE = f"{SYNC_DIR}/mlflow.replica.db"


def default_master(root: Path) -> Path:
    """Where ``adopt`` puts the store: /var/tmp, on the SSD, never cleaned by the system."""
    return Path("/var/tmp") / getpass.getuser() / root.name / STORE_NAME


def find_rsync() -> str:
    found = shutil.which("sqlite3_rsync") or shutil.which(
        "sqlite3_rsync", path=str(Path.home() / ".local" / "bin")
    )
    if found is None:
        raise SyncError(
            "sqlite3_rsync is not on PATH nor in ~/.local/bin; take it from the "
            "sqlite-tools-linux-x64 zip at https://sqlite.org/download.html (3.50 or later)"
        )
    return found


def open_store(store: Path) -> sqlite3.Connection:
    """Open an existing store read-write, never creating one.

    A read-only connection cannot fold a WAL store's -wal back when it closes, and the
    -wal and -shm files it leaves behind make a hand-off refuse (ADR 0009); a read-write one
    closing last removes them.
    """
    return sqlite3.connect(f"{store.as_uri()}?mode=rw", uri=True)


def quick_check(store: Path) -> str:
    """``PRAGMA quick_check``'s verdict, or why SQLite could not read the file at all."""
    try:
        connection = open_store(store)
        try:
            return str(connection.execute("PRAGMA quick_check").fetchone()[0])
        finally:
            connection.close()
    except sqlite3.DatabaseError as error:
        return str(error)


def journal_mode(store: Path) -> str:
    connection = open_store(store)
    try:
        return str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        connection.close()


def copy_store(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target`` whole, in WAL mode, through SQLite's backup API.

    The copy is checked before it takes ``target``'s place, in one rename.
    """
    staged = target.with_name(f"{target.name}.staged")
    staged.unlink(missing_ok=True)
    reader = open_store(source)
    try:
        writer = sqlite3.connect(staged)
        try:
            reader.backup(writer)
            # Training writes while the replica is copied; in WAL mode it never waits for it.
            writer.execute("PRAGMA journal_mode=WAL")
        finally:
            writer.close()
    finally:
        reader.close()
    verdict = quick_check(staged)
    if verdict != "ok":
        staged.unlink()
        raise StoreError(f"the copy of {source} fails quick_check ({verdict}); nothing was replaced")
    os.replace(staged, target)


def adopt(root: Path, master: Path, say: Callable[[str], None]) -> None:
    """Move the checkout's store to ``master``, leave a link in its place, keep a replica."""
    store, replica = root / STORE_NAME, root / REPLICA_FILE
    if store.is_symlink():
        raise StoreError(f"{store} already links to {os.readlink(store)}")
    if not store.exists():
        raise StoreError(f"{store} does not exist: there is no store to adopt")
    for taken in (master, replica):
        if taken.exists():
            raise StoreError(f"{taken} already exists; move it aside if it should be replaced")
    with store_lock(root):
        refuse_if_busy(store)
        master.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        copy_store(store, master)
        # The file on the hard disk becomes the first replica; the sync below refreshes it.
        os.replace(store, replica)
        store.symlink_to(master)
    say(f"{store} now links to {master}")
    sync(root, say)


def sync(root: Path, say: Callable[[str], None]) -> None:
    """Refresh the replica from the SSD store, or restore the SSD store when it is gone."""
    store, replica = root / STORE_NAME, root / REPLICA_FILE
    if not store.is_symlink():
        raise StoreError(
            f"{store} is not a link to the SSD; run `adopt` once "
            "(or a hand-off replaced the link: see ADR 0010)"
        )
    master = store.resolve()
    with store_lock(root):
        if master.exists():
            _refresh_replica(master, replica, say)
        else:
            _restore(master, replica, say)


def _refresh_replica(master: Path, replica: Path, say: Callable[[str], None]) -> None:
    verdict = quick_check(master)
    if verdict != "ok":
        raise StoreError(
            f"{master} fails quick_check ({verdict}); the replica keeps the last good copy"
        )
    mode = journal_mode(master)
    if mode != "wal":
        say(
            f"warning: {master} is in {mode} mode, so copying it holds training's writes back; "
            f'while nothing holds it, run: sqlite3 {master} "PRAGMA journal_mode=WAL"'
        )
    result = subprocess.run(
        [find_rsync(), "-v", str(master), str(replica)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SyncError(f"sqlite3_rsync failed: {(result.stderr or result.stdout).strip()}")
    # sqlite3_rsync reads the store without cleaning up after itself; one more read-write
    # open lets SQLite remove the -wal and -shm when nothing else holds the store.
    tidy = open_store(master)
    try:
        tidy.execute("PRAGMA schema_version").fetchone()
    finally:
        tidy.close()
    sent = next((line for line in result.stdout.splitlines() if line.startswith("sent")), "")
    say(f"replica refreshed from {master} ({sent or 'no transfer summary'})")


def _restore(master: Path, replica: Path, say: Callable[[str], None]) -> None:
    holders = processes_holding(f"{master} (deleted)")
    if holders:
        # Their writes after the last sync live only in the deleted file they still hold.
        raise StoreError(
            f"{master} was deleted while {', '.join(holders)} still holds it open, with writes "
            "the replica lacks. Copy it from /proc/<pid>/fd/ before the process exits, or stop "
            "the process to fall back to the replica."
        )
    if not replica.exists():
        raise StoreError(f"neither {master} nor its replica {replica} exists")
    taken = datetime.fromtimestamp(replica.stat().st_mtime, timezone.utc)
    master.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    copy_store(replica, master)
    say(f"{master} was missing: restored from the replica of {taken:%Y-%m-%d %H:%M} UTC")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="the checkout (default: this one)")
    commands = parser.add_subparsers(dest="command", required=True)
    adopt_command = commands.add_parser("adopt", help="move the store to the SSD, once")
    adopt_command.add_argument(
        "--master", type=Path, default=None, help="default: /var/tmp/<user>/<checkout>/mlflow.db"
    )
    commands.add_parser("sync", help="refresh the replica, or restore the SSD store from it")
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()

    def say(message: str) -> None:
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC {message}", flush=True)

    try:
        if args.command == "adopt":
            adopt(root, (args.master or default_master(root)).absolute(), say)
        else:
            sync(root, say)
    except StoreError as error:
        say(f"REFUSED: {error}")
        return EXIT_REFUSED
    except SyncError as error:
        say(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
