# 10. Record the design: ADR and implementation plan

Type: task
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 03, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14

## Question

Nothing left to decide; this is the destination handoff. Write:

- `docs/adr/0004-imputation-decoder-task.md` following the ADR format used by
  `docs/adr/0003-selectable-learning-rate-schedule.md` (Status, Context, Decision,
  How to compare, Considered options, Consequences), linking every resolved ticket.
- `docs/wayfinder/imputation-decoder/plan.md`, an implementation plan in
  the style of the existing plans there, with tasks, files touched, tests, and the
  backfill script.
- `CONTEXT.md` entries for the terms that settled along the route.

## Answer

Resolved 2026-09-10. The destination is reached; these three artifacts are the handoff.

| Artifact | Path | What it holds |
|---|---|---|
| Decision record | `docs/adr/0004-imputation-decoder-task.md` | Fourteen numbered decisions synthesised from tickets 03-14, each linking its ticket; the comparison recipe; considered options; consequences; what stays fog and what is out of scope |
| Implementation plan | `docs/wayfinder/imputation-decoder/plan.md` | Eleven TDD tasks in dependency order, with files, interfaces, failing-test outlines, and commit points |
| Glossary | `CONTEXT.md` | Six imputation terms (*self-masked cell*, *induced-missing cell*, *imputation ground truth*, *imputation task*, *decode stage*, *decoder*) and the generalised *fold-ranking metric* |

The plan's task order is deliberate and is the one thing a reader should not reorder: the
per-task ranking contract (Task 1) lands before the decode stage (Task 6), because the
cross-validation summariser raises when the ranking metric is absent and it runs after
every fold has trained. Implementing the stage first would let the first cross-validated
imputation run train to completion and then die at aggregation.

Research and prototype sources remain on their branches and are linked from the ADR:
`research/imputation-eval-protocol`, `research/masked-cell-decoder-heads`,
`prototype/imputation-preview`.

Nothing on this map was committed to the working branch; the ADR, plan, glossary and the
`docs/wayfinder/` tree sit in the working tree for the user to review and commit.

## Comments
