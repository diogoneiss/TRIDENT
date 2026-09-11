# Reduced Optuna search for imputation: implementation tickets

The specification is [ADR 0005](../../adr/0005-reduced-optuna-search-for-imputation.md).
The ordering and the per-task test outlines are in the
[implementation plan](../../wayfinder/imputation-optuna-reduced/plan.md). The reasoning
behind every decision is on the
[wayfinder map](../../wayfinder/imputation-optuna-reduced/map.md) and its tickets. This
directory holds the **execution** tickets, one per plan task, so each can be picked up
with `mattpocock-skills:tdd` (or `code-review` afterwards) by path.

Branch: `feat/imputation-task`. Commit after each ticket; commit only that ticket's files.

## Tickets

| # | Ticket | Status | Blocked by |
|---|---|---|---|
| 01 | [Search-space profiles](issues/01-search-space-profiles.md) | done (`14ceea6`) | none |
| 02 | [The search objective on the validation split](issues/02-validation-search-objective.md) | ready-for-agent | 01 |
| 03 | [Task-keyed lookup and explicit promotion](issues/03-lookup-and-promotion.md) | ready-for-agent | 02 |
| 04 | [The importance artifact](issues/04-importance-artifact.md) | ready-for-agent | 03 |
| 05 | [The launcher and the six studies](issues/05-launcher-and-studies.md) | ready-for-agent | 04 |
| 06 | [The comparison and the documentation](issues/06-comparison-and-docs.md) | ready-for-agent | 05 |

The four code tickets are serialised because every one edits `opt.py`; two sessions on
them at once would conflict in the same functions.

## Global constraints (apply to every ticket)

- `uv run --python 3.10 ...` for everything.
- Classification training behaviour is untouched: every new argument, dataclass field and
  flag defaults to today's behaviour; `tests/integration/test_vehicle_regression.py` and
  `tests/integration/test_credit_g_imputation_regression.py` pass **with no edit** after
  every ticket that touches the training package.
- `src/training/data.py` is protected: the column-mix extraction in ticket 01 is pure.
- Tests at pre-agreed public seams only; no mocking of internal collaborators; expected
  values are independent literals, never recomputed the way the code computes them.
- Every real run inside tickets 01 to 04 goes to a scratch tracking store and scratch
  output directories, and its `mlruns/` artifact folders are removed afterwards; only
  tickets 05 and 06 write to the shared `mlflow.db`, because they are the experiment.
- Preserve unrelated worktree changes (`metrics/`, `results/`, MLflow files, user edits).

## Status vocabulary

`ready-for-agent` (fully specified, take it), `in-progress` (claimed; name who), `done`
(with the commit hash), `needs-info` (blocked on a question; say what). Record progress
and findings under each ticket's `## Comments`.
