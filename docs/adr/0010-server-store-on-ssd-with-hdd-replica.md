# Keep gorgona8's store on its SSD, copied to its hard disk every ten minutes

## Status

Accepted (2026-09-29). Decided in conversation the evening of ADR 0009's first push.
Implemented the same evening as `scripts/store_replica.py` with
`tests/unit/test_store_replica.py`, and as three changes to `scripts/sync_remote.py`
(decision 6) with new cases in `tests/unit/test_sync_remote.py`.

Touches no training code: every run still logs to `sqlite:///mlflow.db` in the checkout,
which on gorgona8 is now a link. The Windows checkout is unchanged.

## Context

gorgona8's checkout, `/scratch2/diogoneiss/TRIDENT`, sits on a 4 TB SATA hard disk (WD
Purple, a surveillance line). `/` and `/home` are on an NVMe SSD; `/home` is full, `/` had
153 GB free. No volume is a network mount.

Commit latency measured on 2026-09-29, 500 transactions shaped like MLflow's `log_metric`
(one insert and one upsert per commit):

| journal mode, `synchronous` | SSD (`/var/tmp`) | hard disk (`/scratch2`) |
|---|---|---|
| rollback (`DELETE`), `FULL`: the store until now | 0.97 ms | 225 ms |
| WAL, `FULL`: what MLflow gets in WAL | 0.39 ms | 77 ms |
| WAL, `NORMAL` | 0.04 ms | 1.35 ms |

MLflow never sets `synchronous`, so its connections run with SQLite's default, `FULL`, and
WAL alone leaves 77 ms per commit on the hard disk. The store held 3701 runs and 2.03 M
metric rows (about 550 per run, most logged one `log_metric` call at a time), so a 15 to
20 minute cell spent on the order of two minutes committing.

Facts verified while deciding:

- **`/var/tmp` is not cleaned on this server.** Its 30-day rule in
  `/usr/lib/tmpfiles.d/tmp.conf` is commented out; `/tmp` is emptied at boot and after 30
  days. `/var/tmp` has the sticky bit; `/data` is mode 777 without it, so any user can
  delete what is there.
- **In rollback mode a reader blocks the writer.** `sqlite3 -readonly` against the live
  store, while a cell trained, failed with `database is locked`. A copy holding a read
  transaction for seconds could push MLflow past its 5 s busy timeout and kill a run.
- **`sqlite3_rsync` copies only the changed pages and does not block a WAL writer.** With
  SQLite 3.53.4's `sqlite-tools` against a copy of the real store (713 MB): the first copy
  took 7.6 s; after 200 new tags, 1.8 s and 107 KB sent. Run four times while a writer
  committed every 2 ms for 15 s with a 5 s timeout: 6297 commits, no error, the slowest
  9.7 ms, and the replica passed `quick_check`. Ubuntu 24.04 ships SQLite 3.45.1, without
  it; the WAL and page-size requirements it once had were lifted in 3.50.0.
- **The backup API carries the journal mode.** A snapshot of a WAL store is a WAL store,
  so once the store is WAL, every hand-off of ADR 0009 keeps it so, in both directions.
- **A read-only connection cannot clean up after itself in WAL mode.** It leaves an empty
  `-wal` and a `-shm` when it closes, `sqlite3_rsync`'s and ADR 0009's snapshot included;
  a read-write connection closing last removes both.
- **SQLite follows a linked database to its target** and keeps `-journal`, `-wal` and
  `-shm` next to the target, not the link. `/proc/<pid>/fd` also names the target, which
  `sync_remote.py` already resolves.

Rejected on the way:

- *WAL on the hard disk alone.* 77 ms per commit under MLflow's `FULL`.
- *Pointing `MLFLOW_TRACKING_URI` at the SSD.* Every script defaults to
  `sqlite:///mlflow.db`, and `start_mlflow.sh` and `sync_remote.py` name
  `<root>/mlflow.db`; a link keeps every one of them as it is.
- *The replica at `<root>/mlflow.db`, the store elsewhere.* That path is what ADR 0009
  hands off: a push would be overwritten by the next copy, a pull would take a copy up to
  ten minutes old, and the digest check would refuse every sync.
- *A full copy with the backup API every ten minutes.* About 700 MB written to the hard
  disk each time, about 100 GB a day, competing with training for the disk.
- *`/data`* (no sticky bit), *`/dev/shm`* (RAM, lost at reboot).
- *A `systemd --user` timer* (lingering is off, so it stops at logout) or *a loop in tmux*
  (gone at reboot). Cron survives both and is open to every user.

## Decision

1. **The store lives on the SSD, in WAL mode:** `/var/tmp/diogoneiss/TRIDENT/mlflow.db`.
   `<root>/mlflow.db` is an absolute link to it. That file is the only one written.

2. **The replica is `<root>/sync/mlflow.replica.db`, on the hard disk.** `sync/` is ignored
   by git and never sent by ADR 0009 in either direction. Nothing writes the replica but
   the copy.

3. **`store_replica.py sync` runs from cron every ten minutes and by hand at any time.**
   When the SSD store exists, it must pass `quick_check` (a damaged store never overwrites
   the replica), then `sqlite3_rsync` copies it to the replica, then one read-write open
   lets SQLite remove the `-wal` and `-shm` the copy left. When the SSD store is gone, it
   is restored from the replica (backup API, WAL, `quick_check`, one rename into place),
   unless a process still holds the deleted file: its writes since the last copy exist
   only there, and the refusal names the process.

4. **`store_replica.py adopt` moves the store once.** It copies the store to the SSD with
   the backup API, switches the copy to WAL and checks it; the file on the hard disk
   becomes the first replica, and the link takes its place. It refuses while anything
   holds the store.

5. **One lock, `sync/store.lock` (`flock`), keeps the copy and the hand-off apart.**
   `store_replica.py` holds it while it reads or restores the store, and every server-side
   step of `sync_remote.py` holds it too, so a hand-off waits for a copy in progress, and a
   copy never starts in the middle of a hand-off.

6. **`sync_remote.py` changes, on the server side only** (on Windows the store is never a
   link and the lock does nothing):
   - `install_store` replaces the file behind a link, never the link. The file in place is
     copied to `sync/mlflow.db.prev`, the incoming store is copied next to the target and
     renamed over it. Replacing the link would put the store back on the hard disk and
     orphan the SSD file.
   - `refuse_if_busy` looks for `-journal` and `-wal` next to the link's target.
   - An empty `-wal` no longer refuses: it is a read-only reader's leftover and holds no
     write. A `-wal` holding pages still refuses, since those writes are invisible to the
     digest ADR 0009 compares.

## Consequences

- Losing the SSD store loses at most the ten minutes since the last copy. The replica is a
  mirror, not a backup: a run deleted in the UI leaves the replica at the next copy.
- The store shares `/` with the system. When `/` fills, the machine breaks for every user,
  not only this store.
- A hand-off from Windows can wait a few seconds for the lock while a copy runs.
- A store that arrives in rollback mode (a Windows store never converted) makes every
  `sync` warn until it is switched to WAL by hand; the hand-offs keep WAL once it is set.
- The replica shows the store as of the last copy: query through the link.
- `sync` waits for the lock rather than skipping a turn, so a copy that hangs holds every
  later one back: a `sync/replica.log` that stopped growing means a stuck `sqlite3_rsync`.
- `sqlite3_rsync` lives in `~/.local/bin`, taken from sqlite.org's `sqlite-tools` zip;
  `store_replica.py` finds it there even under cron's bare `PATH`.

## How to use

The server-side steps of `sync_remote.py` run the Windows checkout's copy of the script,
so on Windows, `git pull` before the first sync after this change. Until then `push` and
`code` refuse (the server holds a commit Windows lacks), but `pull` runs the old script,
without the lock.

```bash
python3 scripts/store_replica.py adopt   # once, while nothing holds the store
python3 scripts/store_replica.py sync    # copy now, or restore the SSD store from the replica
tail sync/replica.log                    # what cron's copies did (UTC)
```

Cron entry (`crontab -e`):

```
*/10 * * * * cd /scratch2/diogoneiss/TRIDENT && python3 scripts/store_replica.py sync >> sync/replica.log 2>&1
```

If `sync` refuses with "not a link", the store is back on the hard disk, most likely
replaced by a hand-off from a `sync_remote.py` older than decision 6. Check which file holds
the newest runs, SSD or hard disk, before moving either aside and running `adopt`.
