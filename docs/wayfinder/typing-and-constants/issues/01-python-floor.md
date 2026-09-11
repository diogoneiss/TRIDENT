# 01. The Python floor moves to 3.11

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, rounds 1 and 2)
Blocked by: none

## Question

The user asked for "latest Python features on annotations". `pyproject.toml` says
`requires-python = ">=3.10"`, `.python-version` pins `3.10`, and `AGENTS.md` hardcodes
`uv run --python 3.10` as the invocation for every command in the repo. So "latest" is
bounded by whatever floor this effort adopts, and the floor has to be settled before
anything else: it decides whether enums are cheap or hazardous, which in turn decides how
the five name families are represented (ticket 03).

Does the floor move, and to what?

## Answer

**The floor moves to 3.11.** The user chose this over the recommended research-first
option, with the unverified wheel risk explicitly listed in front of them; the resolver
check below was then run to retire that risk rather than leave it as an assumption.

### Why the floor mattered more than it looks

The deciding fact is a serialisation difference, not a feature list. On 3.10 there is no
`enum.StrEnum`, so an enum has to be written `class LrScheduler(str, Enum)`. Verified on
both interpreters in the charting session:

| Operation | 3.10 `(str, Enum)` | 3.11 `StrEnum` |
|---|---|---|
| `str(member)` | `S.A` | `cosine_legacy` |
| `format(member)` / f-string / `"%s" %` | `cosine_legacy` | `cosine_legacy` |
| `json.dumps({"lr": member})` | `{"lr": "cosine_legacy"}` | `{"lr": "cosine_legacy"}` |
| `member == "cosine_legacy"` | `True` | `True` |
| `{member: 1}.get("cosine_legacy")` | `1` | `1` |
| `Sched("cosine_legacy") is Sched.COSINE_LEGACY` | `True` | `True` |

**MLflow stringifies tag and param values.** So on 3.10, converting
`Hyperparameters.lr_scheduler` to an enum would have written
`LrScheduler.COSINE_LEGACY` into the tag surface that the cross-run comparison protocol in
`README.md` and `CLAUDE.md` queries — while `json.dumps` stayed correct, both regression
fixtures stayed green, and every unit test passed. That is a silent-wrongness failure that
the repo's existing seams would not have caught.

At 3.11 the hazard is simply gone: every path yields the bare value. This is the whole
reason the floor question had to precede ticket 03 rather than run beside it.

### What 3.11 buys, concretely

- `enum.StrEnum` — ticket 03's five name families, safely.
- `typing.Self` — `Hyperparameters.from_mapping` currently returns the stringified
  `"Hyperparameters"` forward reference at `types.py:134`.
- `typing.assert_never` — ticket 04's exhaustiveness check on the task branch.
- `typing.LiteralString`, `typing.Never`, and variadic generics, none of which this effort
  has an immediate use for.

### Why not further

3.12's PEP 695 (`type X = ...`, the new generic parameter syntax) and `@override` were
weighed and declined. They buy little here, tooling support for PEP 695 aliases is still
uneven, and each additional version multiplies the cu121 wheel re-verification that has to
be done by hand on the user's Windows box. Recorded in the map's **Out of scope**.

### What was verified, and what was not

**Verified in-session:**

- `uv lock --python 3.11 --dry-run` → `Resolved 136 packages in 2ms`,
  `No lockfile changes detected`. No files were modified (`git status` on `uv.lock` and
  `pyproject.toml` clean afterwards).
- `uv.lock` already carries resolution markers
  `python_full_version == '3.11.*' and sys_platform == 'win32'` for `torch 2.5.1+cu121`
  from the `download.pytorch.org/whl/cu121` registry.
- `pyarrow 23.0.1` ships `pyarrow-23.0.1-cp311-cp311-win_amd64.whl`, so the `pyarrow<24`
  pin (held because PyArrow 24 caused intermittent Windows access violations during pytest
  collection) survives the bump.
- `uv run --python 3.11 --no-project python` resolved a real 3.11.15 interpreter on this
  machine, so the toolchain is available.

**Not verified — this is the bump's acceptance criterion, not an assumption:**

The 2.5 GB CUDA wheel was not downloaded or imported. Resolution is not installation.

### Acceptance for the bump

**The bump is performed by [ticket 08](08-configure-the-checker.md) as its step 0**, not by
a separate execution ticket in the plan. It has to be: ticket 08 writes
`python_version = "3.11"`, uses `assert_never`, and is verified with
`uv run --python 3.11` — so a bump scheduled after it would leave it unable to run.

1. A 3.11 environment installs and `import torch` succeeds with
   `torch.cuda.is_available()` returning `True` on the RTX 3050 Laptop GPU.
2. `uv run --python 3.11 pytest -m "not integration"` is green.
3. `uv run --python 3.11 pytest -m integration` passes **both** fixtures unedited —
   `vehicle_00nan` for classification and `credit-g_20nan` for imputation. A version bump
   must not move a single number; if it does, that is a finding, not a fixture update.
4. The pin is updated everywhere it appears, which is at minimum: `.python-version`,
   `requires-python` in `pyproject.toml`, and every `uv run --python 3.10` occurrence in
   `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/` and the PowerShell scripts
   (`experiment.ps1`, `start_mlflow.ps1`). Grep for `3.10` rather than trusting this list.

If step 1 or 3 fails, the bump is reverted and this ticket reopens — the floor decision is
contingent on the wheel, and no part of the rest of the map is worth a broken CUDA build.
