# 05. Third-party stubs for sklearn and scipy

Type: research
Status: resolved
Assignee: Claude (AFK research subagent); resolved 2026-09-11 (fired at charting)
Blocked by: none

## Question

Checked in the charting session, by looking for a `py.typed` marker in the venv:

| Typed | Untyped |
|---|---|
| `torch`, `optuna`, `mlflow`, `numpy`, `matplotlib` | `sklearn`, `scipy`, `pandas` |

`pandas-stubs~=2.3.3` is already a dependency, so pandas is covered. **`sklearn` is the
only gap that bites**, and the mypy baseline shows exactly where:

```
src/training/artifacts.py:318: error: "object" has no attribute "scale_"
src/training/artifacts.py:318: error: "object" has no attribute "mean_"
```

That is the one place the codebase hand-inverts a `StandardScaler` — undoing a single
column's scaling because `inverse_transform` wants a whole row — inside the imputation
preview artifact writer. The scaler is typed `object`, so the two attributes that make the
arithmetic correct are entirely unchecked.

**The surface to cover is small and fully enumerated** (verified by grepping every
`from sklearn` / `from scipy` import in `src/`, the entry points and `scripts/`):

```
sklearn.metrics         accuracy_score, confusion_matrix, f1_score,
                        precision_score, recall_score
sklearn.model_selection KFold, ShuffleSplit, StratifiedKFold,
                        StratifiedShuffleSplit, train_test_split
sklearn.preprocessing   LabelEncoder, StandardScaler
sklearn.utils.class_weight  compute_class_weight
scipy.stats             t
```

Twelve sklearn symbols and one scipy symbol. What should ticket 08's mypy configuration do
about them?

## What to find out

1. **Does a maintained stub package exist?** Check PyPI and `typeshed`'s `stubs/` directory
   for scikit-learn and scipy. Establish maintenance status and version alignment with
   `scikit-learn>=1.0.0` and `scipy>=1.10`, not just existence — an abandoned stub package
   pinned to an old sklearn is worse than none.
2. **Does modern sklearn ship any inline types?** scikit-learn has been adding annotations
   without shipping `py.typed`. If the annotations are there but the marker is not,
   `follow_untyped_imports` (mypy 1.14+) may give useful inference where
   `ignore_missing_imports` gives `Any`. Confirm against the installed version, not the
   changelog.
3. **What would a minimal local stub cost?** A `stubs/sklearn/` tree declaring only the
   thirteen symbols above. Estimate the line count and, more importantly, the maintenance
   story: who notices when it drifts from the real sklearn, and does the pytest gate catch
   a stub that has gone stale?
4. **What does each option do to `artifacts.py:318` specifically?** This is the concrete
   test. `ignore_missing_imports` makes `StandardScaler` `Any`, which *silences* the error
   without checking anything — `.scale_` and `.mean_` would be as unchecked as they are
   today, just quieter. A stub that declares `scale_: np.ndarray` and `mean_: np.ndarray`
   actually checks the arithmetic. Say plainly which options are real coverage and which
   are suppression.

## The three candidate answers

- **A. Per-module `ignore_missing_imports`** for `sklearn.*` and `scipy.*`. One config
  block, zero maintenance, and no coverage — every sklearn value becomes `Any` and leaks
  inward.
- **B. A maintained stub package** as a dev dependency, if one exists and is current.
  Real coverage, someone else's maintenance burden.
- **C. Minimal local stubs** in `stubs/`, covering only the thirteen symbols. Real coverage
  on exactly the surface used; this repo maintains them.

Option A is the honest fallback, not the default. Prefer B if a credible package exists;
fall back to C if the surface stays as small as it is today; take A only for scipy's single
`t` symbol if the cost of B or C is disproportionate to one import.

## Resolution

**Findings: [`research/05-stub-strategy.md`](../research/05-stub-strategy.md)** — the full
evidence, the exact configuration block, the 177 lines of stub source, and the post-fix
`artifacts.py` code. Ticket 08 should work from that document, not from this summary.

**Different answer per library, and option A for neither.**

- **scipy → B, `scipy-stubs` as a dev dependency.** An official package exists under the
  `scipy` GitHub org, actively maintained, versioned `{scipy_version}.{stubs_version}`.
  Verified against scratch copies of `pyproject.toml` and `uv.lock`: it resolves **without
  moving a single existing pin**, on the current 3.10 floor as well as 3.11 — so this
  decision never depended on ticket 01. Coverage check: `t.ppf(0.975, df=n-1)` reveals
  `float`; `t.ppff` is a type error.
- **sklearn → C, 177 lines of local stubs in `stubs/`** at the repository root, wired by
  `mypy_path = "$MYPY_CONFIG_FILE_DIR/stubs"`. **Option B does not exist**: typeshed never
  carried sklearn, upstream closed
  [scikit-learn#16705](https://github.com/scikit-learn/scikit-learn/issues/16705) as
  `not_planned` in 2024 ("we're not going to introduce typing on-mass"), 1.9.1 still ships
  no `py.typed`, and the only PyPI candidate is an abandoned one-day republish of
  Microsoft's Pylance tree pinned `<1.7.0` — below the 1.7.2 already installed here.
- **`follow_untyped_imports`** was evaluated as an unenumerated fourth option and
  **rejected on measured grounds**: globally it *breaks* `scipy.stats.t`, and under
  `strict = true` it fires `[no-untyped-call]` at every sklearn call site — noisier than
  ignoring the import outright.

**`artifacts.py:318` ends up genuinely checked, not silenced.** Under the stubs, `scale_`
and `mean_` resolve to `ndarray[…, dtype[float64]] | None`; mypy rejects indexing them
until narrowed, and `scaler.meen_` is caught with a spelling suggestion. Under option A
they would be `Any` — as unchecked as today, only quieter. The research document carries
the exact post-fix code, type-checked clean under strict, and notes that
`ArtifactWriter.write_imputation_preview`'s `scaler: object = None` at line 231 must change
with it. It argues for an `assert` over a widened guard, because widening would silently
alter artifact contents.

### Three things ticket 08 must not rediscover the hard way

1. **This map's default-mode baseline of 7 errors is wrong.** Measured on mypy 2.3.1 it is
   **13 substantive** (+8 sklearn/scipy, +3 tqdm). The charting session did not record its
   mypy version and mypy 2.0 changed several defaults, so the discrepancy's cause is
   unverified. **With the stubs in place it becomes 22 substantive and zero sklearn/scipy
   import errors** — the 9 newly visible errors are all real, all in `src/training/data.py`
   (a `splitter` variable reassigned across four splitter types, `ndarray` versus
   `Sequence[int]` dataclass fields, `.values` returning `ndarray | ExtensionArray`).
   Ticket 08 fixes 22, not 7.
2. **A mis-wired `stubtest` passes vacuously.** With no config and no `MYPYPATH` it finds
   no stubs, exits 0, and says nothing. The gate **must** pass
   `--mypy-config-file pyproject.toml`. Proven by planting a typo: the configured run exits
   1, the unconfigured run exits 0. A stub-drift check that cannot fail is worse than none,
   because it looks like coverage.
3. **`tqdm` is an untyped gap this map missed** (3 errors). Out of scope here; the config
   block parks it behind `ignore_missing_imports` so ticket 08 can reach green, with
   `types-tqdm` as a one-line follow-up for the backlog.

There is deliberately **no** `ignore_missing_imports` for `sklearn.*` or `scipy.*`, and
**no** `follow_untyped_imports` anywhere. If either appears in ticket 08's configuration,
this ticket's answer has been reversed and should be re-argued, not quietly overridden.

## Notes

Call `mattpocock-skills:research` and capture findings against primary sources — PyPI,
the typeshed repository, the scikit-learn release notes, the mypy documentation for
`follow_untyped_imports`. This ticket is AFK; it needs no human in the loop.

This ticket **blocks ticket 08**: the checker cannot be configured and made green until the
stub strategy is decided, because the configuration block is part of the answer.
