@AGENTS.md

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## MLflow run analysis

Every training invocation is a top-level MLflow run. Filter `tags.run_role = parent` before comparing runs — `best_fold`/`worst_fold` children exist only for diagnosis and double-count if included. Every run also has a mirror run in `TRIDENT/mirror/<task>` ([ADR 0006](docs/adr/0006-mirror-experiments-per-task.md)); any query that spans experiments must add `tags.is_mirror = false`, or every source run is counted twice. Also compare within one `tags.lr_scheduler` value (`cosine_legacy` is the pre-ADR-0003 schedule; backfilled runs carry `lr_scheduler_backfilled = true`). Each machine's `mlflow.db` is a snapshot as of the last sync ([ADR 0009](docs/adr/0009-store-hand-off-to-a-remote-server.md)): query the side that holds the store, or the newest runs are missing. Full tagging/CI-metric layout: README.md § "MLflow Cross-Validation Comparisons", [ADR 0002](docs/adr/0002-curated-cross-validation-mlflow-runs.md).

## gorgona8: the store lives on the SSD

On gorgona8, `mlflow.db` in the checkout is a link to `/var/tmp/diogoneiss/TRIDENT/mlflow.db` on the SSD, the only copy ever written; `/scratch2` is a hard disk where each SQLite commit costs ~225 ms ([ADR 0010](docs/adr/0010-server-store-on-ssd-with-hdd-replica.md)). Cron copies it every 10 minutes to `sync/mlflow.replica.db` on the hard disk, logging to `sync/replica.log` (UTC).
- Keep logging and querying through `sqlite:///mlflow.db`, i.e. the link. The replica can be up to 10 minutes behind; never write it.
- Never `mv`, `cp`, `rm` or replace the link, and never copy the store with a file tool; SQLite's backup API, `sqlite3_rsync` or `scripts/store_replica.py` only. `-wal` and `-shm` next to the SSD store while something holds it open are normal.
- To copy right now, or to recreate the SSD store from the replica after it was lost: `python3 scripts/store_replica.py sync`.
- Windows runs its own copy of `sync_remote.py` on the server: it must `git pull` this change before its first sync, or a push would replace the link with a file on the hard disk.
- If `sync` refuses with "not a link", the store is back on the hard disk: read ADR 0010 before running `adopt`.
- The SSD store shares `/` with the system: check `df -h /` before writing anything large there.

## Experimentation log

[docs/EXPERIMENTATION_LOG.md](docs/EXPERIMENTATION_LOG.md) holds one entry per hypothesis-driven experiment (Optuna study, ablation, sweep, cross-machine check), oldest first, in the template at its top. Keep it current as part of the experiment itself:
- **Pre-registering:** append the entry with the same commit as the ticket: hypothesis, scenarios (arms, variants, seeds, budgets), measures and decision rule, status `in progress`.
- **Amending mid-run:** add the dated amendment to the entry.
- **Finishing or stopping:** fill in Results and Conclusion in that entry, in the same commit as the ticket's Outcome.

The ticket stays the contract; the entry summarises it, copies its numbers, and links to it.

## Times

The user reads GMT-3. gorgona8's system clock is UTC, but since 2026-09-30 23:20 GMT-3 the user's shells export `TZ=America/Sao_Paulo` (top of `~/.bashrc`). So `date`, Python's `datetime.now()` (run names) and the ablation kit's `progress.log` now come out in GMT-3. `progress.log` lines also carry their offset (`-0300`); its older lines without one are UTC. These stay UTC whatever `TZ` says:
- logs written before that time;
- any process started before it, or from an environment that did not read `~/.bashrc`;
- `sync/replica.log`, which `store_replica.py` stamps in UTC explicitly.

MLflow stores epoch timestamps, and its UI shows the browser's zone. Every time you report, in chat and in files you write (tickets, the experimentation log), is in GMT-3 and marked, e.g. `21:45 GMT-3`, including the date when the conversion crosses midnight; convert a UTC source before reporting it.

## Background & full reference

README.md — paper/method background (TabularEmbedder, pretraining/fine-tuning stages), complete CLI flag list, hyperparameter config schema, citation. Read it when asked about the paper, model architecture, or full CLI usage.
