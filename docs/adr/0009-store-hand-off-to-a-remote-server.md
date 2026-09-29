# Hand the tracking store to a remote server whole, one writer at a time

## Status

Accepted (2026-09-29). Decided in conversation on 2026-09-29, after a shared SQLite over
SMB and an HTTP tracking server were ruled out (Context below). Implemented the same day
as `scripts/sync_remote.py` with `tests/unit/test_sync_remote.py`, and rehearsed end to
end against a directory in this machine's WSL (Outcome). Not yet run against the server.

Touches no training code: every run still logs to `sqlite:///mlflow.db` in the checkout
it is launched from, as [ADR 0002](0002-curated-cross-validation-mlflow-runs.md) and
[ADR 0006](0006-mirror-experiments-per-task.md) assume.

### Outcome (2026-09-29, rehearsal)

The rehearsal checkout held a copy of the real store (709 MB, 12 experiments, 3730 runs)
and one run's artifacts; the "server" was `/tmp/trident-rehearsal` in WSL (Ubuntu 22.04,
Python 3.10, rsync 3.2.7), reached with the same rsync and `python3 -` invocations a real
host uses, minus ssh.

- The dry-run push translated 11 experiment roots and 3730 run roots and passed every
  check in 33 s. The applied push took 1 min 27 s: 667 code files, the `.git` directory,
  the artifacts and the store. On the Linux side all 3730 roots began with
  `file:///tmp/trident-rehearsal/mlruns/`, `integrity_check` was `ok`, and
  `git log -1` read the checkout's commit.
- A tag written on the server side and a new artifact file came back with the pull
  (31 s). Compared table by table with the original copy, the returned store differed in
  exactly one row of one table of 53: that tag.
- Refusals, each through the real CLI: the Windows store held open (first by the
  training batch on the real store, then by a leftover MLflow server), the server's store
  held open by a Linux process (named by pid), and a store changed on both sides, which
  `status` reported as diverged and which both directions refused.
- Serving the returned store with `mlflow server` and reading 200 runs through it left its
  sha256 unchanged, so browsing a store does not break the rule of decision 2.
- A plain `cp` of the live store, taken while the batch was writing, failed
  `quick_check` (invalid page numbers in six trees). This is why the store is never
  copied by a file tool (decision 3).

## Context

Training moves to a remote Linux server with a much stronger GPU. The user wants the
files on this Windows machine as well, and accepts that the two machines will not train
against the store at the same time. Planned use: move nearly everything to the server,
then pull periodically.

Facts verified while deciding, against a copy of `mlflow.db` and MLflow 3.14.0:

- **Artifact roots are absolute paths.** All 11 non-default experiments carry
  `experiments.artifact_location = file:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlruns/<id>`
  (a literal space, not `%20`), and every run copies its experiment's root into
  `runs.artifact_uri`. The default experiment is `mlflow-artifacts:/0`. MLflow on Linux
  reads `file:///C:/Users/...` as the path `/C:/Users/...`, a directory at the root of the
  filesystem, so the first artifact a server run logs fails without root access.
- **Those two columns are the only ones that locate files.** A scan of every text column
  found the Windows user path in four more places, all provenance: `runs.user_id` and the
  `mlflow.user` tag (who ran), `mlflow.source.name` (199 runs, the script that ran), and
  the `optuna_storage` param (41 runs). Params are immutable in MLflow, and all four record
  where a run happened.
- **Artifacts need no rewriting.** 85 `tracking/diagnostic_manifest.json` artifacts embed
  Windows paths, only in `retained_artifact_paths.*.source_path`, which
  `src/training/artifacts.py` writes and nothing reads.
- **The commit tag needs `.git`.** 3513 of 3706 runs carry `mlflow.source.git.commit`, which
  MLflow reads from the repository the entry point sits in; pre-registrations cite
  commits. A code copy without `.git` would drop the tag on every server run.
- **SQLite has no merge.** `mlflow-export-import` recreates runs under new ids, which
  breaks `mlflow.parentRunId` nesting and the `source_run_id` links of ADR 0006.

Rejected on the way:

- *The store on an SMB share.* SQLite's locking is unreliable over network filesystems,
  and the result is a corrupt store.
- *An MLflow tracking server at home, reached from the server through an ssh tunnel.*
  Every multi-hour run would die when the home machine sleeps or the tunnel drops, and the
  `file:///C:/` roots still fail on Linux.
- *Rewriting the roots once to `mlflow-artifacts:/<id>`.* It makes the store portable,
  but then no process can reach an artifact without an MLflow server running on its
  machine, and the default `sqlite:///mlflow.db` used everywhere stops being enough.
- *`git push` to the server for code.* The user's choice is rsync; decision 5 keeps what
  git would have preserved.

## Decision

1. **One writer side between two syncs; the store moves whole.** A push makes the
   server's store a snapshot of this checkout's; a pull does the reverse. Rows are never
   merged. Several runs at once on the same side are fine. What the protocol forbids is
   writing on both sides between two syncs, and that covers more than training:
   `backfill_*.py --apply`, `mirror_runs.py --apply`, and renaming, tagging or deleting a
   run in the UI all write.

2. **A sync overwrites only a store unchanged since the last sync.** `sync/state.json`
   (on Windows; everything is driven from there) records the sha256 of each side's store
   as the last sync left it. Before replacing a destination, the sync hashes it again and
   refuses on any difference. A destination with a store no sync recorded is refused
   too (a first pull must not erase a store full of runs). A side with no store accepts
   one, since there is nothing to lose there. The digest of a source is
   taken *before* its snapshot, so a write slipping in between leaves a stale record that
   the next sync refuses, rather than a record that hides the write. When both sides
   changed, `status` reports the stores as diverged and neither direction runs; the user
   decides which side's new runs to keep.

3. **A store is moved only while nothing holds it open.** Before a snapshot or a
   replacement, the script checks for processes holding the file open: `/proc/*/fd` on
   Linux, an exclusive `CreateFileW` on Windows. It also refuses when a `-journal` or
   `-wal` file shows an interrupted write. The snapshot uses SQLite's backup API on a
   read-only connection, never a file copy. The replaced store is kept as
   `sync/mlflow.db.prev`.

4. **Two columns are translated, on a snapshot, on the Windows side.**
   `experiments.artifact_location` and `runs.artifact_uri` get their prefix swapped
   (`file:///C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlruns/` ↔
   `file:///<remote root>/mlruns/`), compared with `substr`, not `LIKE`, since `_` and
   `%` are ordinary characters in a path. The translation fails closed, rolling back the
   snapshot:
   - a root under neither prefix (other than `mlflow-artifacts:`) stops it before any
     change;
   - the source prefix still present in any text column of any table afterwards stops it,
     so a place a later MLflow starts storing roots in is caught, not missed;
   - `PRAGMA integrity_check` must return `ok`.

   Provenance is never rewritten. The source store itself is never modified.

5. **Code travels by rsync, as git sees the working tree.** Sent: tracked files plus
   untracked files that are not ignored (`git ls-files --cached --others
   --exclude-standard`), plus `datasets/processed_datasets/`. Never sent: `.venv`,
   `sync/`, `mlruns/`, `results/`, `metrics/`, and anything named `mlflow.db*`, matched on
   the top-level name only, so `stubs/sklearn/metrics/` still goes. `.git` is mirrored,
   so server runs record the commit. A push deletes on the server only the files the
   previous push sent and the checkout no longer has; the server's own files survive.
   `code` refuses while the server's store is open, because a batch still starting
   processes would run the new code for its remaining runs (`--while-busy` overrides).
   When tracked files differ from `HEAD`, it notes that runs will record `HEAD` but run
   the working tree.

6. **Outputs move additively in both directions.** `mlruns/`, `results/` and `metrics/`
   are copied without deletion; run directories are unique by id and result directories
   by timestamp. On the same path the sender wins, as the owner should.
   `metrics/<dataset>_metrics.csv` already holds only its dataset's latest run, so a pull
   replaces it with the server's latest.

7. **Nothing is installed on the server.** Transfers start from Windows, because the
   server cannot reach a machine behind a home router. rsync and ssh run through WSL
   (`wsl -e`), where the keys and `known_hosts` live. Each server-side step runs this same
   file through `ssh <host> python3 - remote <command>`, so the script is standard
   library only and runs on Python 3.9+. ssh runs with `BatchMode=yes`: key
   authentication only.

## Consequences

- Training on both sides between two syncs is detected, not prevented, and resolving it
  is a manual choice of which store to keep.
- Every sync moves the whole store (709 MB today; rsync compresses it in transit).
- Server runs must be launched from the server checkout's root. MLflow creates a new
  experiment's root under the working directory, and a root under neither prefix stops
  the next translation.
- `mlflow.user` on server runs is the Linux account, which marks the machine a run came
  from.
- Moving the server checkout means `init` again with the new root, and a store pushed
  under the old one keeps the old prefix until translated from it.
- Stopping the MLflow UI must stop its uvicorn workers. Killing only the parent process
  leaves them holding the store open, and the sync refuses until they exit.

## How to use

```powershell
uv run --python 3.11 python scripts/sync_remote.py init --host <ssh alias> --remote-root /abs/path/TRIDENT
uv run --python 3.11 python scripts/sync_remote.py status          # which direction may run
uv run --python 3.11 python scripts/sync_remote.py push            # dry run: checks, counts
uv run --python 3.11 python scripts/sync_remote.py push --apply    # code, outputs, store
uv run --python 3.11 python scripts/sync_remote.py code --apply    # code only
uv run --python 3.11 python scripts/sync_remote.py pull --apply    # outputs, store
```

The first push waits for nothing to hold `mlflow.db` open here (stop training batches and
the MLflow UI). A batch paused here must finish here before the push, or resume on the
server after it: resuming it here after the push writes both stores, and they diverge on
the first day. The server needs rsync, Python 3.9+, git for the commit tag, and
`uv sync --python 3.11` in its checkout before training. When the two stores have
diverged, keep one: rename the other side's `mlflow.db` aside, then sync from the side
that is kept (the renamed store keeps its runs for a later manual look).
