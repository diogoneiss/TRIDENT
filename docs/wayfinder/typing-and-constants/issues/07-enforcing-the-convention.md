# 07. Enforcing the constants convention beyond what the checker catches

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, round 3)
Blocked by: 03 (what the convention *is*), 02 (what the gate mechanism *is*)

## Question

Ticket 03 established the diagnosis: the constants convention already existed and decayed
because nothing enforced it. `TRACKING_RUN_ROLES` has been declared in `types.py` while
`tracking.py` wrote `"run_role"` as a bare literal two lines above a correct `TASK_TAG`
reference. Replacing the ~36 remaining literals fixes the codebase as it stands today —
it does not stop the thirty-seventh.

The enum decision does a lot of the work automatically. Once
`Hyperparameters.lr_scheduler: LrScheduler`, assigning `"cosine"` is a type error the
checker reports, with no extra machinery. **But the enum only protects values that flow
through a typed field**, and it leaves two gaps:

1. **MLflow tag and param keys are plain `str` constants**, not enum members.
   `mlflow.set_tag("run_role", ...)` is perfectly well-typed — `set_tag` takes a `str` —
   so the checker will never object. `RUN_ROLE_TAG` and the bare `"run_role"` are
   indistinguishable to mypy.
2. **String literals in dict construction and comparison** that never touch an annotated
   parameter. `tags.update({"dataset": ...})` builds a `dict[str, str]`; every key is a
   valid `str`.

What, if anything, enforces the convention in those two gaps?

## What to decide

1. **Is any enforcement warranted at all beyond the checker?** The honest case for "no":
   the enum covers the five name families, which is where the silent-wrongness risk lives
   (a wrong *value* silently trains the wrong thing); a wrong tag *key* produces a missing
   tag, which shows up as an empty column the first time anyone runs the comparison
   protocol. Those are not equally dangerous, and a seam has to catch a silent-wrongness
   risk to earn its place. Argue this down before building anything.
2. **If yes, what mechanism?** A grep-style unit test asserting that a fixed list of known
   key strings appears only at its definition site is the cheap option, and it fits the
   pytest gate ticket 02 already chose. Consider also whether `Literal` types on the
   tag-writing helpers' parameters would let the checker do it properly instead — that is
   strictly better than a grep if the writes funnel through few enough helpers.
   `execution_tags` in `src/training/tracking.py` is the obvious funnel.
3. **False positives.** A grep rule has to tolerate the definition site, docstrings, test
   fixtures that deliberately spell the raw tag, and `docs/`. If the allow-list grows
   faster than the rule catches anything, the rule is a liability.
4. **Does the same question apply to metric keys?** `TRAINING_SECONDS_KEY`,
   `TIMING_METRIC_KEYS` and `TaskSpec.loss_keys` are the same shape of problem — string
   keys addressing a persisted surface — and ticket 03 left them where they are.

## A prior worth stating

The map's whole diagnosis is that documentation without enforcement decays. It would be
consistent to conclude that a documented convention plus a checker that covers the
dangerous half is *enough*, and that policing the safe half costs more than it returns. It
would be inconsistent to conclude "no enforcement needed" on the grounds that people will
remember — that is precisely the assumption that failed.

Resolve it either way, but say which argument you are making.

## Answer

**Narrow the tag helpers' own parameter types and let the checker do it properly. No grep
rule.**

```python
def execution_tags(*, run_role: RunRole, ...) -> dict[TagKey, str]:
```

A bare `"parent"` passed as `run_role` becomes a checker error, with no allow-list, no
false positives and nothing to maintain.

The argument being made is **not** "people will remember" — that is the assumption this
map's whole diagnosis says already failed here. It is that a grep rule is the *weaker*
enforcement of the two, and choosing it would be settling. A grep rule tests spelling at
text level: it needs an allow-list for definition sites, docstrings, test fixtures and
`docs/`, it cannot distinguish a legitimate mention from a violation, and if it never fires
it rots into noise that the next person deletes. Narrowed parameter types test the thing
that actually matters — that a value reaching a tag write came from the declared
vocabulary — and they are checked by the gate ticket 02 already chose.

The grep rule was the fallback for tag *keys*, which the enum cannot cover because
`mlflow.set_tag` takes a plain `str`. Narrowing the helpers covers them too, by moving the
key out of the call site and into the helper's signature and return type.

### The measurement this answer depends on

**This decision is contingent on how much actually funnels through helpers, and that was
not measured during charting.** The execution ticket's first step is to count, in `src/`,
`opt.py` and `scripts/`:

- tag and param writes that go through `execution_tags` or another helper, versus
- writes that call `mlflow.set_tag` / `set_tags` / `log_param` / `log_params` directly.

If the funnel is narrow — most writes through few helpers — narrowing the signatures covers
the surface and this answer stands as written. **If a long tail writes tags directly, say
so and reopen rather than quietly adding a grep rule to cover the remainder**; the right
response to a wide surface is probably to funnel the direct writes through a helper first,
which is a structural change and therefore its own decision.

Known funnels to start from: `execution_tags` and the fold-tag builder around
`tracking.py:222`, the `OptunaTrialTracker` tag block near `tracking.py:281`, and the study
parent's tags in `opt.py`.

### Metric keys: same treatment where it is free, no new machinery

`TRAINING_SECONDS_KEY`, `TIMING_METRIC_KEYS` and `TaskSpec.loss_keys` are the same shape of
problem. `loss_keys` is already derived from `self.stage` and needs nothing. For the rest,
narrow a parameter type where a helper already exists; do not introduce a helper solely to
have somewhere to narrow. Metric keys are lower-stakes than tag keys — a wrong metric key
shows up as a missing metric on the run, which is loud — so this is opportunistic, not
required.

## Notes

Call `mattpocock-skills:grilling`. HITL.

Read before starting: `src/training/tracking.py` (`execution_tags`, the fold-tag builder
around line 222, the Optuna trial tracker), `src/mlflow_utils.py` (the existing `*_TAG`
constants), and ticket 02's gate decision.
