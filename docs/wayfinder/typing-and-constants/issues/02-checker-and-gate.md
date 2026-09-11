# 02. The checker, the strictness ramp and the gate

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, round 1)
Blocked by: none

## Question

The repo has no mypy, pyright or ruff configuration, no `.github/`, and no pre-commit
hook. Every annotation currently in the codebase — and `src/training/` is extensively
annotated — is therefore a comment: nothing verifies it. Adding more annotations without a
checker produces more comments.

Which checker, at what strictness, over which files, and what actually enforces it in a
repo with no CI?

## Answer

### The baseline this rests on

Measured in the charting session, before any decision:

```
mypy --ignore-missing-imports --explicit-package-bases --namespace-packages
       src main.py opt.py train.py scripts

  default strictness:    7 errors in  3 files
  --strict:            136 errors in 17 files (27 source files checked)
```

Strict errors by file:

| File | Errors | | File | Errors |
|---|---|---|---|---|
| `opt.py` | 43 | | `src/training/finetuning.py` | 4 |
| `src/transformer.py` | 22 | | `src/training/artifacts.py` | 4 |
| `src/models.py` | 14 | | `src/training/tracking.py` | 3 |
| `src/training/decoding.py` | 8 | | `src/training/summary.py` | 3 |
| `src/mlflow_utils.py` | 8 | | `main.py` | 3 |
| `src/embedder.py` | 8 | | `train.py` | 2 |
| `src/utils.py` | 6 | | `src/training/runner.py` | 2 |
| `src/training/pretraining.py` | 4 | | `src/training/data.py`, `cli.py` | 1 each |

**79 of 136 errors — 58% — are in three files.** That distribution is what makes a
per-module ramp viable; a codebase with 136 errors spread evenly would need a different
strategy.

### Checker: mypy

Over pyright/basedpyright, for one practical reason and one honest admission.

- **Practical**: pyright needs a Node toolchain. The user is on Windows with a
  uv-managed Python environment and no Node dependency anywhere in the project. mypy
  installs as a Python dev dependency alongside pytest, which is the toolchain that
  already exists.
- **Honest admission**: pyright is faster and narrows `Literal` types better. If ticket 05
  concludes that sklearn needs inline inference rather than stubs, or if mypy's run time
  makes the pytest gate annoying, revisiting this is cheap — the gate mechanism below is
  checker-agnostic.

### Strictness: `strict = true` with per-module overrides

Not a global `--strict` (a 136-error cliff nobody pays down), and not default mode either.
**Default mode is the trap here**: it reports only 7 errors because it does not check the
bodies of unannotated functions at all. `opt.py` is 11/12 unannotated and
`src/transformer.py` is 9/9, so default mode is blind to precisely the two worst files.
Seven green errors would measure almost nothing.

```toml
[tool.mypy]
python_version = "3.11"
strict = true
explicit_package_bases = true
namespace_packages = true

[[tool.mypy.overrides]]
module = ["opt", "src.transformer", "src.models"]
# relaxations sufficient to make these three green today
```

**The ramp is one execution ticket per deleted override.** Delete `src.models`, fix its
14 errors, fixtures green, commit. That gives the plan a natural unit of work with an
unambiguous done condition, and it means the gate never regresses: a file that is strict
today cannot silently stop being strict.

The exact relaxation keys per override are left to ticket 08, which will measure them
against the real configuration rather than the `--ignore-missing-imports` approximation
used for this baseline.

### Scope: all of `src/` plus the entry points

In: `src/training/*`, the legacy `src/*.py` (`models`, `transformer`, `embedder`, `utils`,
`mlflow_utils`), `main.py`, `opt.py`, `train.py`, `scripts/`.

The reason for including legacy `src/*.py` rather than stopping at the well-typed core:
**untyped edges return `Any` into the typed core and defeat the checker at exactly the
seam where it matters**. `src/training/decoding.py` calls into `src/models.py`; if
`TridentDecoder` is untyped, every value flowing back is `Any` and the eight strict errors
reported in `decoding.py` are an undercount.

Out of the annotation gate: `tests/` (4,253 lines, roughly the size of `src/`). Checked —
it will be visible to the checker so that a test calling a source function with the wrong
type is still caught — but not annotation-gated. Typing test bodies pays least and would
double every execution ticket's surface. Recorded in the map's **Out of scope**, with a
softer "revisit after the source ramp" note in **Not yet specified**.

### The gate: a unit test that shells out to mypy

Ranked against the alternatives:

1. **A pytest test** — `tests/unit/test_typing_gate.py` runs mypy in a subprocess and
   asserts a clean exit. **Chosen.**
2. A pre-commit hook — earlier, but needs `pre-commit install` per clone, is skippable
   with `--no-verify`, and never fires in an agent session that runs tests without
   committing.
3. A documented command in `AGENTS.md` — pure convention, no enforcement.
4. Real CI — strongest, but see **Out of scope**.

Option 1 wins because of what this map found about *why* the existing constants convention
decayed. `src/training/types.py:203` has declared `TRACKING_RUN_ROLES` for some time, and
`src/training/tracking.py:222` still writes `"run_role"` as a bare literal two lines above
a line that correctly uses `TASK_TAG`. The convention was documented and unenforced, and it
drifted. Option 3 would repeat that exactly.

`AGENTS.md` already instructs every agent and every human to run
`pytest -m "not integration"`. **A gate placed inside that command is a gate placed where
people already look.** It needs no new tool, no install step, and no habit change.

Implementation notes for ticket 08, so the gate does not become a nuisance:

- Run mypy via `subprocess` against the configured package set, assert `returncode == 0`,
  and put the captured stdout in the assertion message so a failure names the file and
  line rather than just failing.
- The test must not be marked `integration` — it has to run in the default suite.
- If run time turns out to be objectionable, mypy's incremental cache makes repeat runs
  fast; do not reach for `-m slow` markers before measuring.
