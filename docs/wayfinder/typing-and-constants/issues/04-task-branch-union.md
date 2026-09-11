# 04. The task-branch union and exhaustiveness

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, round 2)
Blocked by: 01 (`assert_never` needs 3.11)

## Question

mypy's default-mode baseline reports, at `src/training/runner.py:93`:

```
error: Incompatible types in assignment
       (expression has type "FinetuningOutcome", variable has type "DecodingOutcome")
```

The code is:

```python
if task.name == "imputation":
    finetuning = train_and_evaluate_decoder(...)      # -> DecodingOutcome
else:
    finetuning = train_and_evaluate_classifier(...)   # -> FinetuningOutcome
```

This is not a runtime bug — the two-task pipeline works, and both integration fixtures
pass. It is an *undeclared* union: `finetuning` is inferred from whichever branch mypy sees
first, and the other branch is an error against an inference nobody intended.

Is fixing this in scope, and how far does the fix go?

## Answer

**In scope: declare the union and add an exhaustiveness check. Out of scope: unifying the
two outcome types behind a `Protocol` or ABC.**

### Why this one error earns its own ticket

Look at what guards the branch: `if task.name == "imputation"` — a bare string comparison,
one of the ~36 literals ticket 03 is replacing. And look at what the branch produces: two
structurally similar but distinct outcome types that the type system never relates.

**The typing question and the enum question are the same question here**, which is why this
is a ticket rather than a line in a cleanup list. It is the one place in the codebase where
the two halves of this effort meet, and it sits on the two-task invariant that `AGENTS.md`
protects most carefully.

### The fix

```python
finetuning: FinetuningOutcome | DecodingOutcome
if request.task is Task.IMPUTATION:
    finetuning = train_and_evaluate_decoder(...)
elif request.task is Task.CLASSIFICATION:
    finetuning = train_and_evaluate_classifier(...)
else:
    assert_never(request.task)
```

**The `is` comparison is contingent, and the execution ticket must check it.** Today the
guard is `task.name == "imputation"` where `task` is a `TaskSpec`, so what `request.task`
actually holds — a `TaskSpec`, a `Task` member, or a plain `str` straight from argparse —
has to be confirmed at the call site before `is` is sound. Identity only holds if the
boundary coerces, which is exactly [ticket 03's caution 1](03-name-families-and-homes.md)
(`argparse` returns a plain `str` unless `type=Task` is passed). Get the coercion in first,
or write `==`; a stray `str` reaching an `is` chain falls through to `assert_never` and
raises at runtime — loudly, but for the wrong reason, and only on the imputation path.

Two distinct gains, worth separating:

- **The declared union** removes the error and states the contract that already holds at
  runtime.
- **`assert_never`** (3.11, from ticket 01) makes the branch *exhaustive*. Today a third
  task added to `Task` would fall silently into the `else` and train a classifier — the
  failure would surface as a `KeyError` on a missing metric much later, if at all. That is
  exactly the shape of the defect ticket 07 of the previous map found and repaired, where
  a missing `task` on the trial namespace meant every Optuna trial silently trained a
  classification model. **With `assert_never`, adding a task without handling it here
  fails the checker.**

That second point is the ticket's real justification. A declared union is tidiness; an
exhaustiveness check is a seam that catches a silent-wrongness risk this repo has already
been bitten by once.

### Why the `Protocol` is out of scope

The tempting next step is a shared `StageOutcome` protocol so the two outcomes have a
common supertype and the union disappears. Declined, and recorded in the map's
**Out of scope**:

- It changes module structure, not types. This effort annotates and names what exists.
- It would require deciding what the two outcomes genuinely share versus what merely looks
  alike, which is a deep-module design question — `mattpocock-skills:codebase-design` work,
  and a separate effort.
- The union is honest. The two branches really do produce different things; a protocol that
  hides that is a worse description of the program than a union that states it.

### Related, and deliberately not bundled

The other two default-mode errors are *not* this ticket:

- `summary.py:234-235` — fold keys typed `int | str` flowing into `dict[int, str]`. A
  straightforward tightening; belongs to ticket 08's "make it green" work.
- `artifacts.py:318` — `StandardScaler` typed `object`. That is the third-party stub
  question; ticket 05.
