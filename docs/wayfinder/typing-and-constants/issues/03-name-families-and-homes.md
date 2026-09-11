# 03. Five StrEnums, and where constants live

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, round 2)
Blocked by: 01 (the Python floor decides whether enums are safe)

## Question

Five families of names recur as bare string literals throughout the codebase: tasks,
learning-rate schedules, search-space profiles, tracking run roles, and column kinds. Four
of them already have a `tuple[str, ...]` declaration, and all five are still written as
bare literals at call sites.

How should they be represented, and where should the constants live?

## Answer

### What the diagnosis actually was

The obvious reading — "this codebase lacks constants" — is wrong, and getting it wrong
would have produced the wrong fix. The constants exist:

| Family | Existing declaration | Home |
|---|---|---|
| Tasks | `TASK_NAMES: tuple[str, ...]` | `src/training/types.py:24` |
| Schedules | `LR_SCHEDULER_NAMES: tuple[str, ...]` | `src/training/types.py:10` |
| Search-space profiles | `SEARCH_SPACE_PROFILES: tuple[str, ...]` | `src/training/types.py:31` |
| Run roles | `TRACKING_RUN_ROLES: tuple[str, ...]` | `src/training/types.py:203` |
| Column kinds | `NUMERICAL` / `CATEGORICAL` | `src/training/imputation_metrics.py:14` |

And alongside them, in `src/training/tracking.py:222`:

```python
tags.update({"dataset": record.result.dataset_name, "run_role": role})
...
tags[TASK_TAG] = self.task            # two lines later, correctly symbolic
```

`TRACKING_RUN_ROLES` was declared and then not used. **The convention was never missing;
it was unenforced, so it drifted.** About 36 bare literals of these families survive
across ten files.

That reframes the fix. Adding a sixth constant to a codebase that already ignores five
changes nothing. The fix has to remove the *possibility* of drift.

### Decision: `StrEnum`, with the tuples derived

```python
class Task(StrEnum):
    CLASSIFICATION = "classification"
    IMPUTATION = "imputation"

TASK_NAMES = tuple(Task)        # derived, not maintained
DEFAULT_TASK = Task.CLASSIFICATION
```

The load-bearing word is **derived**. Today `TASK_NAMES` and every `"imputation"` literal
are two independent sources that agree only by vigilance. With the enum as the single
source, the tuple cannot disagree with it — not because someone remembers to keep them in
step, but because there is nothing left to keep in step.

The same for `LrScheduler`, `SearchSpace`, `RunRole` and `ColumnKind`.

`StrEnum` over `Literal` aliases, which was the serious alternative. `Literal` has one real
advantage: it is a pure annotation, so it provably cannot change runtime behaviour and both
regression fixtures stay bit-identical by construction. It was rejected because it leaves
the tuple and the literal as two sources — the exact structure that produced the
half-migration — and because the enum's guarantee is now free (ticket 01 verified every
serialisation path on 3.11). The bit-identity `Literal` would have given for free is
instead bought by the fixtures, which every execution ticket must pass unedited.

`StrEnum` over a hybrid (enums for values crossing a persistence boundary, `Literal` for
internal ones): the boundary is not stable. `ColumnKind` looks internal until you notice it
is written into the imputation preview artifact at `decoding.py:298` and `:313`. Two
mechanisms with a moving line between them is worse than one mechanism.

### Decision: homes split keys from values

| What | Where | Why |
|---|---|---|
| Value enums — `Task`, `LrScheduler`, `SearchSpace`, `RunRole`, `ColumnKind` | `src/training/types.py` | It already owns the domain vocabulary and the `TaskSpec` contract. `ColumnKind` **moves here** from `imputation_metrics.py`. |
| MLflow tag and param **keys** | `src/mlflow_utils.py` | It already owns the tracking surface and four of these keys. Gains `RUN_ROLE_TAG` and `DATASET_TAG`, currently bare literals. |
| Metric keys | with `TaskSpec` | Already derived there — `loss_keys` composes them from `self.stage`. Leave it. |

One rule: **a value lives with the vocabulary, a key lives with the surface it addresses.**
Rejected: a single `src/constants.py`, which is easier to police with a grep but inverts the
dependency direction and strips `types.py` of vocabulary it already owns coherently.

### Domain-modeling note: no CONTEXT.md changes

`CONTEXT.md` is a glossary and is explicitly free of implementation detail. `Task`,
`LrScheduler` and the rest are not new domain terms — *task*, *search-space profile* and
*decode stage* are already in the glossary from previous efforts, and this effort only
changes how they are spelled in Python. **This effort is expected to add no CONTEXT.md
terms.** If a session finds itself wanting to add one, that is a signal it has wandered
into domain work and should stop.

## Implementation cautions for the execution tickets

Three places where the enum conversion can change behaviour, all verified as real concerns
during charting:

1. **argparse does not return enum members.** `src/training/config.py` already passes
   `choices=LR_SCHEDULER_NAMES` (line 211), `choices=TASK_NAMES` (221) and
   `choices=SEARCH_SPACE_PROFILES` (244). With derived tuples those become tuples of enum
   members, and `in` still matches a plain string — but **`args.task` will still be a
   plain `str`**, not `Task.IMPUTATION`, unless `type=Task` is also passed. Pass
   `type=Task` so the boundary converts once, at the edge, rather than leaving
   `str`-vs-enum ambiguity to flow inward. Check the `--help` output afterwards: it renders
   choices via `str()`, which is the bare value on 3.11.

2. **`ColumnKind` members flow into a pandas DataFrame.** `decoding.py:298` and `:313`
   build cell records with `"kind": CATEGORICAL`, and `imputation_metrics.py:39` filters
   with `cells["kind"] == NUMERICAL`. Equality survives, and so does `json.dumps`, but the
   column's dtype becomes `object` holding enum members rather than plain strings. This is
   the single highest-risk conversion in the effort because it feeds the
   `credit-g_20nan` imputation fixture. Convert it in its own commit and run
   `pytest -m integration` before moving on.

3. **`Hyperparameters.lr_scheduler` is validated by hand** at `types.py:128` against
   `LR_SCHEDULER_NAMES`, and set from JSON via `from_mapping`. Once the field is typed
   `LrScheduler`, `from_mapping` must coerce (`LrScheduler(values[...])`) and the
   hand-rolled check can become the enum's own `ValueError` — but the error *message* is
   user-facing, so preserve its wording or improve it deliberately, not accidentally.
