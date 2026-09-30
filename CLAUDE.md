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

## Experimentation log

[docs/EXPERIMENTATION_LOG.md](docs/EXPERIMENTATION_LOG.md) holds one entry per hypothesis-driven experiment (Optuna study, ablation, sweep, cross-machine check), oldest first, in the template at its top. Keep it current as part of the experiment itself:
- **Pre-registering:** append the entry with the same commit as the ticket: hypothesis, scenarios (arms, variants, seeds, budgets), measures and decision rule, status `in progress`.
- **Amending mid-run:** add the dated amendment to the entry.
- **Finishing or stopping:** fill in Results and Conclusion in that entry, in the same commit as the ticket's Outcome.

The ticket stays the contract; the entry summarises it, copies its numbers, and links to it.

## Times

gorgona8's clock and its logs (`date`, `progress.log`, MLflow timestamps) run in UTC; the user reads GMT-3. Every time you report, in chat and in files you write (tickets, the experimentation log), is converted to GMT-3 and marked, e.g. `21:45 GMT-3`, including the date when the conversion crosses midnight.

## Background & full reference

README.md — paper/method background (TabularEmbedder, pretraining/fine-tuning stages), complete CLI flag list, hyperparameter config schema, citation. Read it when asked about the paper, model architecture, or full CLI usage.
