# Imputation decoder task: implementation tickets

The specification is [ADR 0004](../../adr/0004-imputation-decoder-task.md). The ordering
and the per-task test outlines are in the
[implementation plan](../../wayfinder/imputation-decoder/plan.md). The reasoning behind
every decision is on the [wayfinder map](../../wayfinder/imputation-decoder/map.md) and its
fourteen decision tickets. This directory holds the **execution** tickets, one per plan
task, so each can be picked up with `mattpocock-skills:tdd` (or `code-review` afterwards)
by path.

Branch: `feat/imputation-task`. Commit after each ticket; commit only that ticket's files.

## Tickets

| # | Ticket | Status | Blocked by |
|---|---|---|---|
| 01 | [Per-task fold-ranking contract](issues/01-ranking-contract.md) | done (`cbf991c`) | none |
| 02 | [Request, hyperparameters and command line](issues/02-request-and-cli.md) | done | 01 |
| 03 | [Imputation metrics as pure functions](issues/03-imputation-metrics.md) | done | none |
| 04 | [The decoder model](issues/04-decoder-model.md) | done | none |
| 05 | [Data support for evaluation](issues/05-evaluation-data-support.md) | done | none |
| 06 | [The decode stage](issues/06-decode-stage.md) | done | 01, 02, 03, 04, 05 |
| 07 | [Imputation artifacts](issues/07-imputation-artifacts.md) | done | 05, 06 |
| 08 | [Runner, MLflow identity, batch summary](issues/08-runner-and-mlflow-identity.md) | done | 02, 06, 07 |
| 09 | [Tag backfill](issues/09-tag-backfill.md) | awaiting-approval | 08 |
| 10 | [Optuna for the imputation task](issues/10-optuna-imputation.md) | ready-for-agent | 02, 08 |
| 11 | [Regression fixture and documentation](issues/11-fixture-and-docs.md) | ready-for-agent | 08, 10 |

The frontier after ticket 01 is 02, 03, 04 and 05, which are independent of each other.

## Global constraints (apply to every ticket)

- `uv run --python 3.10 ...` for everything.
- Classification stays bit-identical: every new argument and dataclass field defaults to
  the classification behaviour; `tests/integration/test_vehicle_regression.py` passes
  **with no edit** after every ticket that touches the training package.
- The decode stage never calls `model.apply(initialize_weights)`; that sweep erases the
  pretrained encoder.
- `[MASK]`, `[NULL]` and the literal `"nan"` never enter a categorical head's output
  space; targets are encoded from `preprocess_table(..., fine_tunning=True)`.
- Training never reads the `_00nan` sibling; it is read for test-fold scoring only, after
  an alignment assertion.
- Tests at pre-agreed public seams only; no mocking of internal collaborators; expected
  values are independent literals, never recomputed the way the code computes them.
- Preserve unrelated worktree changes (`metrics/`, `results/`, MLflow files, user edits).

## Status vocabulary

`ready-for-agent` (fully specified, take it), `in-progress` (claimed; name who), `done`
(with the commit hash), `needs-info` (blocked on a question; say what). Record progress
and findings under each ticket's `## Comments`.
