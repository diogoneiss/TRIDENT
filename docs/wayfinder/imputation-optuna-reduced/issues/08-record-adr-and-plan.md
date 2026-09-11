# 08. Record the design: ADR 0005 and the implementation plan

Type: task
Status: open
Assignee: unassigned
Blocked by: 07

## Question

Synthesise tickets 01 to 07 into the destination:

- `docs/adr/0005-reduced-optuna-search-for-imputation.md`, amending ADR 0004 decision 13,
  with a link back to every ticket, in the format of `docs/adr/0003` and `0004`. It must
  state: the two profiles and what each samples and holds, with ranges; the search
  objective and its split, per task; the study protocol (three datasets, two variants,
  40 trials, `cosine`, seeded TPE, single split, no pruner) with the measured costs; the
  task-keyed lookup and explicit promotion, closing backlog I2; the sparse `search_space`
  tag; the importance artifact; the comparison protocol; the repair from ticket 07 as done,
  including the `HEAD_DIM` rename and the trained-configuration user attribute, with
  `docs/BACKLOG.md` noting that `--retrain_best` was broken from `356bcca` to `161fc92`
  and that the decode stage logs its timing under `time/finetune_seconds`; and the fog and
  out-of-scope lists from the map as considered options and future work.
- `docs/wayfinder/imputation-optuna-reduced/plan.md`, ordering TDD tasks so nothing can
  crash mid-study, in the format of the previous effort's plan. Expected shape: (1) the
  profile flag, the `SEARCH_SPACE_PROFILES` tuple, the reduced space, the widened range,
  the `LAMBDA_NUM` conditional and the `search_space` tag; (2) the validation-scored
  objective and the `TaskSpec` objective field; (3) the task-keyed lookup and
  `--promote_best`; (4) the importance artifact; (5) launching the six studies and
  promoting; (6) the twelve comparison runs and the comparison table; docs (`README.md`
  flags and lookup order, `docs/BACKLOG.md` I2 closed, `AGENTS.md` if a flag needs naming).
  Execution tickets go under `docs/tickets/imputation-optuna-reduced/` (`spec.md` plus
  `issues/NN-<slug>.md`) so each can be handed to `mattpocock-skills:tdd` by path, as the
  previous effort did.
- `CONTEXT.md` already holds the four terms; check them against the ADR's wording.
- Close this map: mark the status line "destination reached" as the previous map does.

## Comments
