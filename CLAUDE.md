@AGENTS.md

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## MLflow run analysis

Every training invocation is a top-level MLflow run. Filter `tags.run_role = parent` before comparing runs — `best_fold`/`worst_fold` children exist only for diagnosis and double-count if included. Every run also has a mirror run in `TRIDENT/mirror/<task>` ([ADR 0006](docs/adr/0006-mirror-experiments-per-task.md)); any query that spans experiments must add `tags.is_mirror = false`, or every source run is counted twice. Also compare within one `tags.lr_scheduler` value (`cosine_legacy` is the pre-ADR-0003 schedule; backfilled runs carry `lr_scheduler_backfilled = true`). Full tagging/CI-metric layout: README.md § "MLflow Cross-Validation Comparisons", [ADR 0002](docs/adr/0002-curated-cross-validation-mlflow-runs.md).

## Background & full reference

README.md — paper/method background (TabularEmbedder, pretraining/fine-tuning stages), complete CLI flag list, hyperparameter config schema, citation. Read it when asked about the paper, model architecture, or full CLI usage.
