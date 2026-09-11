# 08. Configure the checker and make it green

Type: task
Status: open
Assignee: unclaimed
Blocked by: 01 (the floor decision), 02 (the gate design), 05 (the stub strategy is part of
the configuration block). **Also gated on a worktree condition the tracker cannot express:
do not start until the previous effort's execution ticket 03 is committed** — at charting
it had `src/training/runner.py`, `types.py`, `tracking.py`, `config.py` and `opt.py`
modified and uncommitted, and step 5 below edits `runner.py:93`.

## Question

This is the map's one **execution** carve-out, and the only ticket that changes code. It
exists because every later strictness decision is opinion until a checker actually runs:
the ramp ticket 02 designed is only credible if its starting point is measured against a
real configuration rather than the `--ignore-missing-imports` approximation the baseline
used.

Nothing is decided here. The work is to make tickets 01, 02 and 05 real and prove the
result.

## The work

0. **Perform the Python floor bump to 3.11**, per ticket 01's decision and its acceptance
   list. This is step zero rather than a separate execution ticket because **every step
   below requires it**: step 3 writes `python_version = "3.11"`, step 5 uses `assert_never`,
   and the acceptance runs `uv run --python 3.11`. A version bump scheduled after this
   ticket would leave this ticket unable to run at all.

   Update the pin everywhere it appears — at minimum `.python-version`, `requires-python`
   in `pyproject.toml`, and every `uv run --python 3.10` in `AGENTS.md`, `CLAUDE.md`,
   `README.md`, `docs/` and the PowerShell scripts. **Grep for `3.10` rather than trusting
   that list.** Verify `import torch` with `torch.cuda.is_available()` true before going
   further; if the CUDA wheel does not install, stop and reopen ticket 01 — nothing below
   is worth a broken CUDA build.

   Keep the bump in its own commit, separate from everything below.

1. **Add the package marker.** `src/` has no `__init__.py`, which is why mypy reports
   `Source file found twice under different module names: "utils" and "src.utils"` and
   needs `--explicit-package-bases`. Imports are already consistent
   (`from src.X import ...` everywhere), so only the marker is missing. **Check the
   interaction with `pytest.ini`'s `pythonpath = .` before and after** — this is a
   one-line change that touches import resolution for the whole test suite, which is
   exactly why it is here and not a drive-by.
2. **Add mypy as a dev dependency** in `pyproject.toml`'s `[dependency-groups] dev`,
   pinned.
3. **Write `[tool.mypy]`** per ticket 02 and ticket 05. **Take the configuration block
   verbatim from [`research/05-stub-strategy.md`](../research/05-stub-strategy.md) §8.1** —
   it is already type-checked and carries three details that are easy to get wrong:
   `mypy_path = "$MYPY_CONFIG_FILE_DIR/stubs"` (a bare Windows path splits on its `:`), the
   deliberate **absence** of `ignore_missing_imports` for `sklearn.*` and `scipy.*`, and
   the `tqdm.*` override. Add `scipy-stubs` to the dev dependency group alongside mypy.

   If `ignore_missing_imports` for sklearn or scipy, or `follow_untyped_imports` anywhere,
   ends up in the final configuration, ticket 05's answer has been reversed — re-argue it,
   do not quietly override it.

3b. **Wire the stub-drift check correctly.** If a `stubtest` gate is added for the local
   sklearn stubs, it **must** receive `--mypy-config-file pyproject.toml`. Without it,
   stubtest finds no stubs, exits 0, and says nothing — verified by planting a typo: the
   configured run exits 1, the unconfigured run exits 0. **Prove the check can fail before
   trusting it**; a drift check that passes vacuously is worse than none, because it reads
   as coverage.
4. **Write the `[[tool.mypy.overrides]]` blocks** relaxing `opt`, `src.transformer` and
   `src.models` to whatever minimum makes them pass. **Record the exact relaxation keys in
   the resolution** — they are the ramp's inventory, and ticket 09's plan turns each into
   an execution ticket.
5. **Fix the default-mode errors** not covered by an override. **There are 22, not 7** —
   ticket 05 re-measured on mypy 2.3.1 and the map's original figure was wrong; 9 of the
   extra ones become visible only once the sklearn stubs land, and all 9 are real, all in
   `src/training/data.py` (a `splitter` variable reassigned across four splitter types,
   `ndarray` versus `Sequence[int]` dataclass fields, `.values` returning
   `ndarray | ExtensionArray`). The originally-known four:
   - `runner.py:93` — **the declared union only**, per ticket 04. Not `assert_never`:
     today the branch tests `task.name == "imputation"` where `task.name` is `str`, so
     after both comparisons mypy still sees `str`, not `Never`, and `assert_never(...)`
     would itself be a type error. Exhaustiveness narrowing needs the branch on an enum,
     so the check lands with ticket 03's enum conversion during plan execution.
   - `summary.py:234-235` — fold keys typed `int | str` reaching a `dict[int, str]`.
   - `artifacts.py:318` — `StandardScaler` as `object`; resolved by ticket 05's choice.
     If ticket 05 chose `ignore_missing_imports`, note honestly in the resolution that this
     error is *silenced*, not fixed.
   - `artifacts.py:412` — `asdict` receiving `DataclassInstance | type[DataclassInstance]`.
6. **Write the gate test** — `tests/unit/test_typing_gate.py`, subprocess mypy, assert a
   clean exit, put captured stdout in the assertion message. Not marked `integration`.
7. **Re-measure `--strict` against the real configuration** and record the new per-file
   counts. The baseline's 136/17 was taken with `--ignore-missing-imports`; the true
   numbers will differ, and the map's **Not yet specified** patch about the ramp order
   graduates on these numbers.

## Two corrections to the numbers above, made before execution

**The override list is an output of this ticket, not the number 3 from ticket 02.** The
gate runs `strict = true`, so every unrelaxed file reports its *strict* errors — from the
charting baseline that is roughly 136 − 79 ≈ **57**, not the 22 default-mode errors in step
5 (which are largely a subset). Reconcile it this way: write the configuration at step 3,
**run it, and let the measured tree decide the override list**. Where the strict errors are
missing `-> None` and `no-any-return`, annotate them — that is not mass annotation and it
keeps the file strict. Where a file needs design decisions, relax it and record the
override. Ticket 02's three files are the expectation, not the budget.

**Expect coupling between relaxed and strict modules.** `main.py:97` reports
`[no-untyped-call]` *because* `opt.py` is untyped, so relaxing `opt.py` leaves `main.py`
red. Same shape at `decoding.py` → `models.py`. Two honest fixes: annotate the handful of
functions in the relaxed module that strict callers actually touch (cleanest — it types the
seam, which is where it matters), or set `disallow_untyped_calls = false` on the *callers*.
Prefer the first. Note that `strict` is not itself a per-module option; an override must
list the individual flags it turns off.

**Annotations follow runtime, never the reverse.** `FoldSplit.*_indices` holds ndarrays
today, so widen the annotation; do not wrap in `list()` to satisfy a `Sequence[int]`
declaration. The two `np.asarray` calls at `data.py:191` and `:209` are the only runtime
touch this ticket should make, and the fixtures prove them.

## Acceptance

- `uv run --python 3.11 pytest -m "not integration"` green, including the new gate test.
- `uv run --python 3.11 pytest -m integration` passes **both** fixtures **unedited** —
  `vehicle_00nan` and `credit-g_20nan`. This ticket must not move a single number.
- A deliberate regression is caught: introduce a type error, confirm the gate test fails
  and names the file and line, revert. **A gate nobody has seen fail is not a gate.**
- The resolution records the post-configuration strict counts per file and the exact
  override keys.

## Notes

Call `mattpocock-skills:tdd` for the gate test — write it failing first, which for a gate
test means writing it before the configuration exists.

The fixtures are the "run the model, not only tests" proof for this ticket. For a change
whose entire claim is behaviour-neutrality, an unedited fixture pass *is* the real run;
there is no separate training run that would tell you more. If either fixture moves, stop:
that is a finding about the change, not a fixture to update.

Keep step 1 in its own commit. If the `__init__.py` disturbs pytest's import resolution,
you want that isolated and revertable, not tangled with the mypy configuration.
