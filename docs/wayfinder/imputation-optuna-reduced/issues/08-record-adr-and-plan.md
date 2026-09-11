# 08. Record the design: ADR 0005 and the implementation plan

Type: task
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
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

## Answer

Resolved 2026-09-10. The destination is recorded:

- [ADR 0005](../../../adr/0005-reduced-optuna-search-for-imputation.md): seven decisions,
  each linking its ticket; amends ADR 0004 decision 13; records ticket 07's repair and the
  interval in which `--retrain_best` was broken.
- [`plan.md`](../plan.md): six tasks, the four code tasks serialised because every one
  edits `opt.py`, then the studies, then the comparison and the documentation.
- Execution tickets at [`docs/tickets/imputation-optuna-reduced/`](../../../tickets/imputation-optuna-reduced/spec.md),
  01 to 06, all `ready-for-agent`; 01 is the frontier.
- `docs/BACKLOG.md`: I2 marked decided (lands with execution ticket 03), a new H5 for the
  decode stage's timing key, B3's fix noted as the cause of the `--retrain_best` break,
  and the housekeeping note about `vehicle_00nan.json` corrected to what was verified.
- `CONTEXT.md`'s four terms read against the ADR: consistent, unchanged.

**Plumbing pinned in the ADR beyond what the grilling settled**, each a choice a later
TDD session would otherwise have had to make alone, listed so the user can veto any:

1. `opt.py`'s `__main__` builds its parser from `build_training_parser`, so the two entry
   points share every flag (its private parser lacked `--task` and three others).
2. The validation score reaches Optuna through a programmatic, defaulted
   `TrainingRequest` field that only `opt.py` sets, logged as its own `validation/`
   family and merged into the fold metrics only then, so no ordinary or CV run ever
   carries a `validation/` key and the `cv/test/` summariser never sees one.
3. A promoted file is complete: the task's full key set with resolved values,
   `LR_SCHEDULER` included, so it cannot move if a default changes.
4. A dense `config_source` param on every run (`defaults`, `override`, or the file's
   relative path), so tuned and default runs are told apart without a new tag.
5. The column mix for the `LAMBDA_NUM` conditional comes from a pure helper extracted from
   `prepare_dataset`, computed once per study.
6. The running best of a **classification** study also moves into the study directory,
   which is backlog I2's fix applied to both tasks rather than a flag that ignores one.
7. The launcher's `-Compare` switch moves the promoted file aside for the default run and
   restores it, since the lookup is by path.

## Comments
