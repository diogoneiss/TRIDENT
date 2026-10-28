# ADR 0033: A second server as leader or replica, with promotion after 24 hours of silence

## Status

Accepted (2026-10-27, 20:16 GMT-3) by the user ("apoio, escreva a adr"), after a conversation
the same evening that settled each decision below. Numbered 0033 by the user, since 0032 was
taken by the TabPFN ADR. Decisions 1 and 9 were replaced the same evening, before any
implementation, when medusa1's `/scratch2` turned out to belong to root (below): the first
version linked each machine's path to the other's; this one gives every artifact root one
prefix both machines resolve, accepted by the user ("de acordo com vc").

Preparation done on 2026-10-27, after the experiments ended (the user stopped the MLflow UI):
- 23:15 GMT-3: backup `sync/backups/2026-10-27T2315-pre-0033-migration.db` (backup API,
  `quick_check` ok, 17998 runs, 13 experiments), brought up to the store's state at 23:27 by
  `sqlite3_rsync` once nothing held the store.
- 23:25 GMT-3: the link `/scratch/diogoneiss/TRIDENT -> /scratch2/diogoneiss/TRIDENT`.
- 23:30 GMT-3: the migration of decision 1 on gorgona8's store, under `sync/store.lock` and in
  one transaction: 13 experiment roots and 17998 run roots rewritten, none left outside the
  canonical prefix, `integrity_check` ok (114 s). MLflow then listed a run's artifacts through
  the new root.
- The user's git credentials copied to medusa1 (`~/.git-credentials`, mode 600); GitHub's API
  accepted them from there.

- 2026-10-28: `_MLFLOW_SERVER_ARTIFACT_ROOT` exported on both machines; medusa1 has its checkout
  (torch 2.11.0+cu128, commit `579e524`), `sqlite3_rsync`, and `MLFLOW_TRACKING_URI` pointing at
  its tunnel. Interim, until `leadership.py`: `scripts/leader_keepalive.sh` (gorgona8, cron)
  keeps the MLflow server up; `scripts/follower_sync.sh` (medusa1, cron) keeps the tunnel on
  port 47731 and sends `mlruns/`, `results/` and `metrics/` to gorgona8 (decisions 3 and 5). A
  run logged from medusa1 landed in gorgona8's store with the canonical root and its artifact
  arrived (00:38 GMT-3; the run and its experiment `_adr0033_smoke` were deleted after).

Not implemented: `leadership.py` (replica on medusa1, roles, epochs, switchover, fork,
reconciliation, root guard). The Windows `config.json` was edited to the canonical prefix by
hand instead of decision 9.

The peer is **medusa1** (hostname `medusa`, Ubuntu 24.04.4, two RTX 5090), checked over ssh
from gorgona8 on 2026-10-27 around 20:25 GMT-3:

- **ssh works both ways** with key authentication: alias `medusa1` from gorgona8, `gorgona8`
  from medusa1. Home directories are separate (each machine's own `/home`), so a file in
  `~` on one side is not on the other.
- **`/scratch` is SSD only:** one LVM volume of 6.2 TB spanning both NVMe disks (Samsung 990
  EVO Plus), 550 GB free (91% used). The store lives directly in the checkout, with no link
  and no hard-disk replica: ADR 0010's problem does not exist there. The volume is linear,
  not mirrored, so losing either disk loses `/scratch`. The backups of decision 7 on medusa1
  protect against mistakes, not against a disk failure; gorgona8's copy does that.
- **`/` on medusa1 is 96% full (43 GB free):** nothing of this ADR writes there.
- **`/scratch/diogoneiss` exists** and belongs to the user (it holds `uv-cache`, `uv-tools`);
  the checkout `/scratch/diogoneiss/TRIDENT` does not exist yet.
- **`/scratch2` on medusa1 is an empty directory owned by root, mode `755`.** The user cannot
  create `/scratch2/diogoneiss` there, and no administrator can be asked to. Hence decision 1.
- **No way around it without root.** Unprivileged user namespaces are blocked by AppArmor
  (`kernel.apparmor_restrict_unprivileged_userns = 1`): `unshare --user --mount` fails with
  "write failed /proc/self/uid_map: Operation not permitted", and `bwrap` with "setting up uid
  map: Permission denied", so `/scratch` cannot be bind-mounted on `/scratch2`. Docker is
  installed but the user is not in its group.
- **Missing on medusa1:** `sqlite3_rsync` (runs itself on both ends of a copy) and the
  `sqlite3` CLI. gorgona8's `~/.local/bin/sqlite3_rsync` (Ubuntu 24.04.5, same glibc
  family) can be copied over. Present: `rsync`, `git`, Python 3.12, `uv` in `~/.local/bin`.

Builds on [ADR 0009](0009-store-hand-off-to-a-remote-server.md) and ADR 0010, and amends 0009's
decision 4 (decision 9 below). Touches no training code: every run still logs to
`sqlite:///mlflow.db` in the checkout it is launched from, or to `MLFLOW_TRACKING_URI` when set.

## Context

A second Linux server, medusa1, joins gorgona8 for training. The user wants both to run experiments and
write files, with code moving through git, and asked for:

- runs on the peer written into gorgona8's store while both are up (the "option A" of the
  conversation: an MLflow tracking server reached over HTTP);
- a leader and a replica, the store written only on the leader, and a way to swap roles;
- a fork mode, so the replica can take over after a long silence of the leader and the two
  stores can be reconciled when they meet again, with logs of every step;
- backups at the first sign of a partition and before every reconciliation.

The user set the trigger for the fork: **24 hours without a signal from the leader, or an
explicit promotion**. Shorter gaps only wait.

Facts verified while deciding (2026-10-27, against gorgona8's store, 15.5 GB, MLflow 3.14):

- **The peer's checkout is at `/scratch/diogoneiss/TRIDENT`, gorgona8's at
  `/scratch2/diogoneiss/TRIDENT`.** Both machines have both `/scratch` and `/scratch2`, but medusa1's `/scratch2` belongs to root (Status).
  On gorgona8, `/scratch/diogoneiss` exists, is empty and belongs to the user.
- **The MLflow client writes artifacts straight to the run's `artifact_uri`.** A run on the
  peer that logs to gorgona8's server receives `file:///scratch2/diogoneiss/TRIDENT/mlruns/...`
  and writes that path on the peer's own disk, where it cannot exist. MLflow has no public
  setting to remap it.
- **A local client takes new experiments' roots from `_MLFLOW_SERVER_ARTIFACT_ROOT`.**
  Without it, MLflow resolves `./mlruns` against `getcwd()`, which returns the physical
  path (`/scratch2/...` on gorgona8 even when launched through a link). With
  `_MLFLOW_SERVER_ARTIFACT_ROOT=file:///scratch/diogoneiss/TRIDENT/mlruns`, a run against a
  throwaway `sqlite:///` store got `file:///scratch/diogoneiss/TRIDENT/mlruns/1` as its
  experiment root and `.../mlruns/1/<run>/artifacts` as its own (2026-10-27, MLflow 3.14,
  `_get_sqlalchemy_store` in `mlflow/tracking/_tracking_service/utils.py`). The name is
  private: a later MLflow may drop it.
- **Reports read `runs.artifact_uri` as a path.** `scripts/experiments/report_common.py`
  (`artifact_path`) and the other `*_report.py` open the path in the URI directly.
- **`experiments.experiment_id` has no `AUTOINCREMENT`.** SQLite gives a new experiment
  `max(experiment_id) + 1`; the highest today is 12. `experiments` is also
  `UNIQUE (workspace, name)`.
- **Almost every row hangs off a uuid, but datasets do not.** Keys of the tables a run writes:

  | Table | Primary key | Reaches a run through |
  |---|---|---|
  | `runs` | `run_uuid` | |
  | `params` | `(key, run_uuid)` (immutable in MLflow) | `run_uuid` |
  | `tags` | `(key, run_uuid)` | `run_uuid` |
  | `metrics` | `(key, timestamp, step, run_uuid, value, is_nan)` | `run_uuid` |
  | `latest_metrics` | `(key, run_uuid)` | `run_uuid` |
  | `datasets` | `(experiment_id, name, digest)`; `dataset_uuid` is not the key | |
  | `inputs` | `(source_type, source_id, destination_type, destination_id)` | `destination_id`; `source_id` is a `dataset_uuid` |
  | `input_tags` | `(input_uuid, name)` | `inputs.input_uuid` |
  | `logged_models*`, `trace_*` | string ids | 0 rows today |

  The same dataset logged on both sides during a split gets two `dataset_uuid`s for one key.
- **`mlflow server` serves artifacts by default** (`--serve-artifacts`, default true in
  3.14's `--help`). An experiment created through such a server gets
  `mlflow-artifacts:/<id>` as its root, the scheme ADR 0009 rejected and its translation
  refuses.
- **Some roots have no scheme.** In the store backed up on 2026-10-27 at 23:15 GMT-3 (17998
  runs), 11 experiments and 17980 runs carry `file:///scratch2/diogoneiss/TRIDENT/mlruns/`,
  but two experiments and 18 runs carry the bare path `/scratch2/diogoneiss/TRIDENT/mlruns/`:
  `Default` (0), recreated on 2026-10-26 at 18:09 GMT-3, and `TRIDENT/synthblock` (12),
  created by `scripts/experiments/occlusion_map.py` on 2026-10-27 at 02:13 GMT-3, with its 18
  runs. ADR 0009's translation finds these under neither prefix, so a Windows pull of this
  store refuses. The migration of decision 1 rewrites both forms to the canonical `file://`
  prefix. Rolled back in a transaction on the backup, it rewrote 13 experiments and 17998
  runs, left no root outside the canonical prefix and no `/scratch2/diogoneiss/TRIDENT/mlruns/`
  in any column.
- **A read-only open of a WAL store leaves an empty `-wal` and a `-shm`** when it is the last
  connection (ADR 0010, seen again on the replica on 2026-10-27 and cleaned by
  `store_replica.py sync`).

Rejected on the way:

- *A shared SQLite over a network filesystem*, *mlflow-export-import*: as in ADR 0009.
- *Postgres as the backend.* It takes concurrent writers from both machines, but means
  migrating 15 GB, running a service, and changing `sync_remote.py`, `store_replica.py` and
  about twenty scripts that open the store with `sqlite3`.
- *Translating artifact roots on every switchover* (`/scratch2/` ↔ `/scratch/`, as ADR 0009
  translates for Windows). It rewrites `runs`, and its pages would travel again on every
  replication. Decision 1 makes it unnecessary.
- *A link on each machine to the other's path*, leaving roots under either prefix. The first
  version of decision 1; medusa1's `/scratch2` belongs to root.
- *A private mount namespace or container on medusa1* binding `/scratch` on `/scratch2`:
  blocked (Status).
- *An MLflow artifact-repository plugin that takes over the `file` scheme* and maps one prefix
  to the other. MLflow's registry lets an entry point replace `file`, so it would work for
  MLflow's own reads and writes, but every report that opens `artifact_uri` as a path would
  need the same mapping. Kept as the fallback if `_MLFLOW_SERVER_ARTIFACT_ROOT` disappears.
- *A fork that may only add runs, enforced by SQLite triggers.* Designed first: it keeps
  reconciliation trivial. It does not fit the user's purpose, which is a replica that
  *takes over* after a day: a leader must be able to tag, backfill and delete.
- *Rewriting roots to `mlflow-artifacts:/`*: as in ADR 0009.

## Decision

1. **Every artifact root in the store is under one prefix,
   `file:///scratch/diogoneiss/TRIDENT/mlruns/`, which both machines resolve.** On medusa1 it
   is the checkout itself. On gorgona8, where `/scratch/diogoneiss` belongs to the user, one
   link makes it so: `/scratch/diogoneiss/TRIDENT -> /scratch2/diogoneiss/TRIDENT`. No
   translation between the servers, and no link on medusa1.
   - **Migration, once:** while nothing holds gorgona8's store, after a checked backup
     (decision 7), `experiments.artifact_location` and `runs.artifact_uri` have their prefix
     swapped to the canonical one from `file:///scratch2/diogoneiss/TRIDENT/mlruns/` and from
     the bare `/scratch2/diogoneiss/TRIDENT/mlruns/` (Context), with the
     checks of ADR 0009 decision 4: `substr`, not `LIKE`; a root under neither prefix stops
     it; no old prefix left in any text column; `integrity_check` returns `ok`. On gorgona8
     both prefixes resolve to the same files, so nothing reading the store breaks, before or
     after.
   - **New roots:** both machines export
     `_MLFLOW_SERVER_ARTIFACT_ROOT=file:///scratch/diogoneiss/TRIDENT/mlruns` for every
     process that writes the store, and the leader's `mlflow server` takes the same value as
     `--default-artifact-root` (decision 3). Runs are still launched from the physical
     checkout.
   - **Guard:** since the variable is private, every cron tick of the leader looks for roots
     outside the canonical prefix. It rewrites one under `/scratch2/diogoneiss/TRIDENT/` (safe
     on gorgona8, where both resolve) and logs `root_normalized`. Any other prefix is logged
     as `root_foreign` and left for the user.

2. **One leader at a time, recorded in `sync/role.json` on each side.** The file holds the
   role (`leader`, `follower`, `deposed`), an **epoch** (an integer, raised by every
   switchover and promotion), the peer's ssh alias and root, and the time of the last
   successful contact. The higher epoch is the legitimate leader. A side that sees a higher
   epoch on its peer never writes under its own lower epoch: it demotes itself (decision 8).
   - **Leader:** its `mlflow.db` is writable. It runs `mlflow server` over the store for the
     follower's HTTP clients (decision 3).
   - **Follower:** its `mlflow.db` is mode `444`, so a forgotten `sqlite:///mlflow.db` fails
     at once with "readonly database" instead of starting a second store. The replicator
     sets `u+w` around each copy and `u-w` after it, under `sync/store.lock`. Only
     promotion leaves the file writable. That a `444` WAL store can still be read (its
     `-shm` is created in the writable directory), and serve as the origin of
     `store_replica.py sync`, is checked during implementation.

3. **The follower trains through the leader's server.** The leader runs
   `mlflow server --backend-store-uri sqlite:///mlflow.db --no-serve-artifacts
   --default-artifact-root file:///scratch/diogoneiss/TRIDENT/mlruns` from its checkout. Both flags are
   required, so an experiment created over HTTP gets a `file://` root like every other
   (Context). The leader's cron tick starts the server when it is not running: ADR 0010 ruled
   out `systemd --user` and tmux, which do not survive logout or reboot. Runs started on the
   follower set
   `MLFLOW_TRACKING_URI` to the leader's `mlflow server`, through an ssh tunnel. Rows go to
   the leader's store; artifacts land on the follower's disk under the canonical prefix of decision 1
   and travel to the leader with the outputs (decision 5). Such a run dies if the leader or
   the tunnel goes away; the fork of decision 6 is for the runs launched after that.

4. **Replication: the leader's store reaches the follower with `sqlite3_rsync` every ten
   minutes.** `leadership.py replicate` runs from cron on both sides. By default the follower
   pulls (`sqlite3_rsync <leader>:<path> mlflow.db`): the lock, the `u+w`/`u-w` toggle and the
   last-contact time all live on the follower, so one local process does them. A leader push
   is the fallback when only the leader can reach the follower, and then needs a remote step
   on the follower, as `sync_remote.py` runs `python3 - remote`. `sqlite3_rsync` sends only
   changed pages and does not block a WAL writer (ADR 0010). On gorgona8 it reads the SSD
   store, not the hard-disk replica. Whoever copies, the follower's own cron tick checks the
   age of the last contact and takes the steps of decision 6, so a dead leader cannot stop
   the timer. **Last contact** is the time of the
   last copy that finished, written into the follower's `role.json`. The first contact after
   a split is also the first chance to read the peer's epoch.

5. **Outputs move additively in both directions**, as in ADR 0009 decision 6: `mlruns/`,
   `results/` and `metrics/` are copied without deletion, after every replication and
   before every reconciliation. Paths inside them are relative, so `/scratch` and
   `/scratch2` need no translation. Files a copy replaces are kept by
   `rsync --backup --backup-dir=sync/backups/<ts>-outputs/`.

6. **The fork: the follower promotes itself after 24 hours without contact, or on
   `leadership.py promote`.**
   - **At 30 minutes without contact** the follower logs `peer_unreachable` and takes the
     **fork base** backup (decision 7). Nothing reached its store since the last copy, so
     this file is the state both sides last shared. It is taken once per split. If contact
     returns before promotion, it is deleted and replication resumes.
   - **At 24 hours**, before promoting, the follower probes the leader itself:
     `ssh <leader> true`, then reads the leader's `role.json`. If the leader answers, the
     silence was a stalled replication, not a dead leader: it logs `replication_stalled`,
     does not promote, and leaves the user to fix the copy. This is the failure ADR 0010
     already names, a stuck `sqlite3_rsync`, which must never become a fork.
   - **`promote` by hand refuses while the leader answers**, and points to `switchover`.
     A planned role swap stays clean and needs no reconciliation.
   - **Promotion:**
     1. Raise the epoch by one and make the store writable.
     2. Insert a placeholder experiment, `lifecycle_stage = 'deleted'`, named
        `_epoch_<n>_reserved`, with id `epoch × 100000`. Every experiment the new leader
        creates then takes an id above it, while the old leader, if alive, keeps creating ids
        just above 12. The ids never collide, in the store or under `mlruns/<id>/`.
     3. Start `mlflow server` and log `promoted`, with the base, the epoch and the reason
        (`timeout` or `manual`).

     `mlflow gc` would delete the placeholder and free the range, so it is never run while a
     fork is open.

7. **Backups, each checked before the step it protects.** Each one is made with `VACUUM INTO`
   or the backup API, never a file copy, and goes to the hard disk (`sync/backups/` in the
   checkout), never the SSD. Each passes `quick_check`, has its runs counted, and is logged
   as `backup_created` with path, size, run count and check result. A backup that fails its
   check aborts the step. Every backup refuses to start with less than twice the store's
   size free.

   | Backup | When | Where | Deleted |
   |---|---|---|---|
   | fork base | first detection, decision 6 | the follower | after `reconcile_done`, or when contact returns before promotion |
   | pre-reconcile, new leader | before the merge writes it | the new leader | after N days, never while a fork is open |
   | pre-reconcile, old leader | before its store is fetched | the old leader | likewise |
   | incoming store | the old leader's store, fetched for the merge | `sync/incoming-<ts>.db` on the new leader | after `reconcile_done` |

   At 15.5 GB a store, a reconciliation keeps about four copies at once, about 62 GB.
   `status` reports what `sync/backups/` holds.

8. **Reconciliation: a three-way merge, run by hand with `leadership.py reconcile`.**
   - **Reconnection.** Every cron tick that reaches the peer compares epochs. The side with
     the lower epoch demotes itself: it writes `deposed` into its own `role.json` and stops
     its `mlflow server`, uvicorn workers included (ADR 0009 notes that killing only the
     parent leaves them holding the store). The merge then waits until nothing holds its
     store (`refuse_if_busy`). Its running trainings finish, and its store becomes mode
     `444` once they do. A `switchover` stops the old leader's server the same way. Nothing stops
     a new launch on it before that, because that would touch training code. A run launched
     there in the meantime is merged like any other.
   - **Dry run.** `reconcile` without `--apply` lists what would be inserted, applied and
     refused.
   - **The merge** compares base, new leader and old leader. It writes into the new
     leader's live store **one run per transaction**, so no transaction holds the write lock
     near MLflow's 5 s busy timeout.

   | On the old leader, since the base | Action |
   |---|---|
   | a new run | insert it with all its rows |
   | a run `RUNNING` at the base that went on there | insert every row the new leader lacks (metrics, tags, params, inputs, and child runs such as `best_fold`/`worst_fold`), recompute `latest_metrics`, take `status` and `end_time` |
   | a tag added or changed on a base run the new leader did not change | apply it |
   | the same tag changed on both sides to different values | conflict: the higher epoch wins, the lost value is logged |
   | a run deleted on one side only | apply it, logged |
   | a new experiment | insert it under its own id (decision 6 keeps ids apart) |
   | a new experiment whose name exists on the new leader | **refuse the whole merge**; rename it on the old leader and run again |
   | a dataset whose `(experiment_id, name, digest)` exists on the new leader | keep the existing row, and point the incoming `inputs.source_id` at its `dataset_uuid` |
   | rows in `logged_models*` or `trace_*` | refuse: none exist today, and no rule is written for them |

   - **Logging and undo.** Each inserted or changed row is logged (`reconcile_run_inserted`,
     `reconcile_tag_applied`, `reconcile_conflict`). `reconcile --undo <ts>` removes exactly
     those rows. The full backup of decision 7 is the last resort, because restoring it
     would also drop every run the new leader logged since.
   - **After `reconcile_done`**, the old leader becomes a follower, and replication from the
     new leader overwrites its store.

9. **Windows (amends ADR 0009).**
   - `sync_remote.py` reads the current leader from the server's `role.json` instead of the
     host fixed by `init`.
   - Decision 4's translation takes the canonical prefix of decision 1 as the server's,
     whichever machine leads. It is set apart from the remote root used for transfers, which
     stays each machine's physical checkout (`/scratch2/diogoneiss/TRIDENT` on gorgona8,
     `/scratch/diogoneiss/TRIDENT` on medusa1).
   - A pull of a store taken before the migration still carries `/scratch2/...` roots;
     until then the translation keeps today's prefix.

10. **Logs.** Every step appends a JSON line to `sync/leadership.log` (UTC): role changes,
    epochs, contact lost and regained, backups, promotion, every reconciliation row, and
    refusals. `leadership.py status` summarises them:
    - role, epoch and last contact;
    - how long a fork has been open;
    - runs waiting for a merge;
    - space used by backups.

## Consequences

- With two machines, a split cannot be told from a dead leader. A promotion while the old
  leader still trains makes two writers; the 24 hours make this rare, and the fork base and
  the three-way merge make it recoverable. A third witness (for example, the leader's last
  push to GitHub) could rule out the case and is left for later.
- A follower run that logs over HTTP dies with the tunnel; only runs launched after a
  promotion use the local store.
- The leader's `mlflow server` keeps its store open at all times, so ADR 0009's ordinary
  `pull` and `push` refuse for good. `pull --while-busy` is the way Windows reads the store,
  and a push from Windows is retired in practice.
- Experiment ids jump to `epoch × 100000` after the first promotion. Scripts that compare
  ids by size, or assume they are small, would break; none is known.
- A merge refuses on a duplicate experiment name and on traces or logged models. Both are
  decisions for the user, not for a script.
- Each reconciliation needs about 62 GB of free disk on the new leader's side: hard disk on
  gorgona8, the SSD `/scratch` on medusa1.
- New experiments get the canonical root only through a private MLflow variable. If an MLflow
  upgrade drops it, roots fall back to `getcwd()`: on gorgona8 the guard of decision 1
  rewrites them, but a follower run in such an experiment fails on medusa1 until then. An
  upgrade of MLflow repeats the check of Context before the variable is trusted again; the
  plugin of Context is the fallback.
- Once migrated, every root in gorgona8's store reads `/scratch/...` while the checkout sits
  on `/scratch2`: the link of decision 1 must never be removed.
- Until `role.json` exists, nothing changes: gorgona8 stays the only server, as in ADR 0009.

## How to use

Planned interface, not yet implemented:

```bash
ln -s /scratch2/diogoneiss/TRIDENT /scratch/diogoneiss/TRIDENT   # on gorgona8, once
export _MLFLOW_SERVER_ARTIFACT_ROOT=file:///scratch/diogoneiss/TRIDENT/mlruns   # both machines
python3 scripts/leadership.py migrate-roots      # once, gorgona8, nothing holding the store; --apply
python3 scripts/leadership.py init --peer <alias> --peer-root <path> --role leader|follower
python3 scripts/leadership.py status
python3 scripts/leadership.py replicate          # from cron, every ten minutes
python3 scripts/leadership.py switchover         # planned swap, both sides reachable
python3 scripts/leadership.py promote            # refused while the leader answers
python3 scripts/leadership.py reconcile          # dry run; --apply merges; --undo <ts> reverts
```
