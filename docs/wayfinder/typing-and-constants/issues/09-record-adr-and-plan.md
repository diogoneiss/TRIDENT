# 09. Record the design: ADR 0006 and the implementation plan

Type: grilling
Status: open
Assignee: unclaimed
Blocked by: 05, 06, 07, 08

## Question

This is the destination. Every decision on this map has to land somewhere a future reader
finds it without reading the map, and the execution work has to be ordered precisely enough
to run without another decision.

## The work

1. **Write `docs/adr/0006-typing-and-naming-standard.md`**, synthesising the resolved
   tickets with a link back to each: the Python floor (01), the checker and its ramp and
   gate (02), the five StrEnums and where constants live (03), the task-branch union and
   exhaustiveness (04), the stub strategy (05), the hyperparameter schema (06), and the
   enforcement question (07).

   The ADR earns its place on all three of the usual tests, and the reasoning should be
   visible in it: the Python floor is **hard to reverse** (it touches every invocation in
   the repo); the `StrEnum`-over-`Literal` choice is **surprising without context** (a
   future reader will reasonably ask why a codebase with zero `Optional[` imports needed
   enums at all, and the answer is the 3.10 `str(member)` hazard plus the two-source drift
   that produced the half-migration); and the strictness ramp is **a real trade-off**
   against a plain `--strict` cliff.

2. **Write `plan.md` beside the map**, ordering the execution tickets. The serialisation
   constraints known at charting:
   - The **Python bump and the checker configuration are both already done**, inside the
     map, by ticket 08 (the bump is its step 0). So the plan starts from a green tree on
     3.11 and contains no version-bump task. Do not re-add one.
   - The **enum conversion** serialises on `src/training/types.py`, and its three risky
     conversions want separate commits: argparse `type=`, the `ColumnKind`-into-DataFrame
     path, and `Hyperparameters.lr_scheduler`'s coercion (ticket 03's cautions).
   - The **ramp tickets** — one per deleted `[[tool.mypy.overrides]]` block — are
     independent of each other and of the enum work. They parallelise.
   - The **literal replacement** (~36 sites across ten files) should follow the enum
     conversion, not precede it.

3. **Create the execution tickets** under `docs/tickets/typing-and-constants/`, following
   the layout of `docs/tickets/imputation-optuna-reduced/` (a `spec.md` plus
   `issues/NN-<slug>.md`).

4. **Update `docs/BACKLOG.md`** with anything this effort found but did not fix. Known
   candidates at charting: the duplicated defaults in `from_mapping` if ticket 06 finds any
   pair disagreeing; the `#TODO store those defaults elsewhere` at `types.py:86`; and
   whatever ticket 08's re-measurement turns up.

5. **Pin any plumbing choices the grilling did not cover**, and list them explicitly in
   this ticket's resolution so the user can veto them — the pattern ticket 08 of the
   previous map used.

## Acceptance

- Every resolved ticket on this map is reachable from the ADR.
- The plan is precise enough that an execution session needs no decision from it, only
  work.
- `AGENTS.md` is updated: the `uv run --python 3.11` invocation, and a line about the
  typing gate so the next agent knows the constraint exists before tripping over it.
- The map's header gains a "destination reached" note, as
  `docs/wayfinder/imputation-optuna-reduced/map.md` did.

## Notes

Call `mattpocock-skills:grilling` and `mattpocock-skills:domain-modeling`.

Expect **no CONTEXT.md changes** — ticket 03 settled that this effort adds no domain
vocabulary. If writing the ADR makes a genuinely new domain term surface, that is worth
noticing rather than suppressing, but the default expectation is none.

Run `graphify update .` at the end; the knowledge graph indexes `docs/wayfinder/`.
