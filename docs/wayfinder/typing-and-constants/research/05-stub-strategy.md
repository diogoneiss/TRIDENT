# Research: third-party stub strategy for sklearn and scipy

Resolves: [ticket 05](../issues/05-third-party-stubs.md) · Blocks: ticket 08
Researched: 2026-09-11 · Skill: `mattpocock-skills:research`

Everything below was measured, not inferred. Every mypy result in this document was
produced by `uvx mypy@latest` (**mypy 2.3.1**, the current release) with
`--python-executable` pointed at this repo's `.venv`, so mypy resolved the *installed*
`scikit-learn 1.7.2` and `scipy 1.15.3`. Nothing was installed into `.venv`; nothing
outside this file was written. Where a claim could not be verified it says so.

---

## 1. Answer

| Library | Option | One line |
|---|---|---|
| **scipy** | **B — `scipy-stubs` dev dependency** | An official stub package exists under the scipy GitHub org, is actively maintained, versions itself against scipy releases, and resolves into this repo's lock *without moving a single existing pin*. |
| **sklearn** | **C — minimal local stubs in `stubs/`** | B does not exist: typeshed never had sklearn, upstream closed typing as `not_planned` in 2024, and the only PyPI package is a one-shot 2025 republish of Microsoft's Pylance tree that is itself pinned `<1.7.0`. 177 lines of local `.pyi` cover all twelve symbols and pass `stubtest` against both sklearn 1.7.2 and 1.9.0. |

Option **A** (`ignore_missing_imports`) is taken for **neither**. Under A, `artifacts.py:318`
is silenced, not checked.

**`artifacts.py:318` under the recommendation is genuinely checked** — `scale_` and `mean_`
resolve to `ndarray[…, dtype[float64]] | None`, mypy rejects indexing them without a
narrowing guard, and a misspelled attribute is reported with a suggestion. Section 7 has
the exact post-fix code, type-checked clean under `strict = true`.

---

## 2. Does a maintained stub package exist?

### 2.1 typeshed — no, for either library

- `https://api.github.com/repos/python/typeshed/contents/stubs` lists 204 entries. There is
  **no** `scikit-learn`, **no** `sklearn`, **no** `scipy`. The only `sc*` entry is `scp`.
- They were never there and never removed: the commits API for each of those paths returns
  `[]`.
- typeshed's acceptance policy is in
  [CONTRIBUTING.md § Third-party library stubs](https://github.com/python/typeshed/blob/main/CONTRIBUTING.md#third-party-library-stubs):
  a package must be on PyPI, support a Python version typeshed supports, and "does not ship
  with its own stubs or type annotations". Both libraries are *eligible*; they are absent
  because nobody contributed them, not because they were rejected.

### 2.2 scipy — `scipy-stubs` is official and current

| Fact | Source |
|---|---|
| Latest `scipy-stubs` **1.18.1.0**, uploaded **2026-08-22**; summary is literally "The official type stubs for SciPy"; homepage `scipy.org` | <https://pypi.org/pypi/scipy-stubs/json> |
| Repository is **`github.com/scipy/scipy-stubs`** — inside the scipy org (it was transferred from `jorenham/scipy-stubs`, which now redirects). `pushed_at` **2026-09-09**, not archived | <https://api.github.com/repos/scipy/scipy-stubs> |
| scipy's own 1.15.0 release notes: "A separate accompanying type stubs package, `scipy-stubs`, will be made available with the 1.15.0 release." 1.16.0 notes announce `v1.16.0.0`; 1.17.0 makes classes generic "for compatibility with `scipy-stubs`"; 1.18.0 adds `scipy-stubs` to the dev test env | `doc/source/release/1.{15,16,17,18}.0-notes.rst` in <https://github.com/scipy/scipy> |
| Versioning is `{scipy_version}.{stubs_version}`, so alignment is structural | <https://github.com/scipy/scipy-stubs/blob/master/README.md> |
| `scipy` is **not** a hard dependency (only the `[scipy]` extra pins it); the hard dep is `optype[numpy]` | <https://pypi.org/pypi/scipy-stubs/json> |

**scipy has not vendored them.** No scipy release through 1.18.1 ships a top-level
`scipy/py.typed`; the 14 `.pyi` files inside the wheel are module stubs for compiled
extensions and, absent the PEP 561 marker, are ignored by type checkers. Tracking issue
[scipy#17158](https://github.com/scipy/scipy/issues/17158) is still open with the
scipy-stubs lead at "-0.01… I don't see why anyone interested in static typing wouldn't
just use scipy-stubs" ([comment, 2025-12-18](https://github.com/scipy/scipy/issues/17158#issuecomment-3670810426)).
Verified locally too: `find .venv/Lib/site-packages/scipy -name py.typed` returns nothing.

mypy itself knows this package. With no configuration, mypy 2.3.1 emits:

```
src/training/summary.py:10: error: Library stubs not installed for "scipy.stats"  [import-untyped]
src/training/summary.py:10: note: Hint: "python3 -m pip install scipy-stubs"
```

It emits **no such hint for sklearn** — mypy's stub-distribution table has an entry for one
and not the other.

### 2.3 sklearn — nothing credible exists

| Candidate | Verdict | Source |
|---|---|---|
| `sklearn-stubs`, `types-scikit-learn`, `types-sklearn` | **HTTP 404 — do not exist** | PyPI JSON API |
| `scikit-learn-stubs` | Exists (v0.0.3, **2025-08-28**), declares no `Requires-Dist`. All three releases landed inside ~3 hours on one day and the repo has not been touched since. Its README: *"This repository simply publishes the stubs from Microsoft's Python Type Stubs repository to PyPI."* **Abandoned republish.** | <https://pypi.org/pypi/scikit-learn-stubs/json>, <https://github.com/hoel-bagard/scikit-learn-stubs> |
| `microsoft/python-type-stubs` | Still contains `stubs/sklearn`, repo `pushed_at` 2026-06-04, **not on PyPI** ("The stubs are not currently published on PyPI"). Its `pyproject.toml` pins **`scikit-learn <1.7.0`** with an explicit `# TODO: Update stubs for sklearn` — three minor releases behind current 1.9.1, and behind the 1.7.2 already in this venv. Pylance-oriented "work in progress". | <https://github.com/microsoft/python-type-stubs> |

**Upstream will not fix this.** [scikit-learn#16705 "Support typing"](https://github.com/scikit-learn/scikit-learn/issues/16705)
was **closed as `not_planned` on 2024-10-18**, with the maintainer's closing comment:

> "It seems the consensus has been that we're not going to introduce typing on-mass in
> scikit-learn… We also don't maintain the stubs, so the maintainers of that package need
> to take care of the changes in the code."

The v1.7 / v1.8 / v1.9 changelogs contain zero matches for `py.typed`, `type stub`, `type
annotation`, `typing` or `PEP 561`. The current release, **scikit-learn 1.9.1 (2026-09-10)**,
does not ship `py.typed`; the only markers in its wheel belong to the vendored
`sklearn/externals/array_api_compat` and `array_api_extra` subpackages — which is exactly
what `find` reports inside this repo's venv for 1.7.2 as well.

**This is the whole case for option C on sklearn.** There is no maintained package to
depend on, there is no prospect of one, and the abandoned candidate is pinned below the
version already installed here.

---

## 3. Does sklearn ship inline annotations? (and would `follow_untyped_imports` help?)

### 3.1 No inline annotations on the surface that matters

`scikit-learn 1.7.2` as installed, `sklearn/preprocessing/_data.py`:

```python
class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
    def fit(self, X, y=None, sample_weight=None):
    def transform(self, X, copy=None):
    def inverse_transform(self, X, copy=None):
```

Not one annotation. `scale_`, `mean_` and `var_` are never declared in the class body; they
are only assigned inside `partial_fit`, where they take `None`, `0.0` and ndarray values on
different branches. The only description of their types is prose in the docstring
("`scale_` : ndarray of shape (n_features,) or None"). The ticket's hypothesis —
"annotations are there but the marker is not" — **is false for this library**.

### 3.2 `follow_untyped_imports` — real but wrong for a `strict = true` repo

The option is genuine and the ticket's version claim checks out. Verbatim from
[the mypy config reference](https://mypy.readthedocs.io/en/stable/config_file.html#confval-follow_untyped_imports)
(type boolean, default `False`):

> Makes mypy analyze imports from installed packages even if missing a py.typed marker or
> stubs. … If this option is used in a per-module section, the module name should match the
> name of the *imported* module, not the module containing the import statement.
> **Warning** — Note that analyzing all unannotated modules might result in issues when
> analyzing code not designed to be type checked and may significantly increase how long
> mypy takes to run.

First shipped in **mypy 1.14.0** (2024-12-20): the flag and the docs entry are absent at tag
`v1.13.0` and present at `v1.14.0`; PR [python/mypy#17712](https://github.com/python/mypy/pull/17712)
merged 2024-12-04. It is in `PER_MODULE_OPTIONS`, so `[[tool.mypy.overrides]]` works.

It does *not* make the module `Any` — it analyses the source. Measured on the probe:

| | `ignore_missing_imports` | `follow_untyped_imports` |
|---|---|---|
| `reveal_type(scaler)` | `Any` | `sklearn.preprocessing._data.StandardScaler` |
| `reveal_type(scaler.scale_)` | `Any` | `Any` |
| `scaler.scalee_` (typo) | **not caught** | `"StandardScaler" has no attribute "scalee_"; maybe "scale_"?` |
| `f1_score`, `train_test_split`, `compute_class_weight` | `Any` | `Any` (all are `validate_params`-decorated) |

So it buys attribute-*existence* checking and nothing about the attribute's *type* — which
is precisely the half that `artifacts.py:318` needs.

Two hard blockers:

1. **It breaks `scipy.stats.t`.** Applied globally it produces a false positive:
   `error: Module "scipy.stats" has no attribute "t"  [attr-defined]` — `t` arrives through a
   star-import chain mypy cannot follow. Scoping it to `sklearn.*` avoids this, but then
   scipy still needs its own answer.
2. **It is unusable under `strict = true`.** Ticket 02 chose `strict = true` globally. With
   `[mypy-sklearn.*] follow_untyped_imports = true` and `strict = True`, every sklearn call
   site becomes an error, because unannotated defs are untyped functions:

```
calls.py:7: error: Call to untyped function "StandardScaler" in typed context  [no-untyped-call]
calls.py:8: error: Call to untyped function "fit" in typed context  [no-untyped-call]
calls.py:9: error: Call to untyped function "StratifiedKFold" in typed context  [no-untyped-call]
calls.py:10: error: Call to untyped function "split" in typed context  [no-untyped-call]
```

Under strict, following untyped imports is **noisier than ignoring them** and would need
either a blanket `# type: ignore[no-untyped-call]` at every sklearn call site or
`disallow_untyped_calls = false` for the whole repo. Rejected.

---

## 4. What minimal local stubs cost

**177 lines of sklearn `.pyi`**, in four files plus two empty package markers, covering
exactly the twelve symbols the repo imports:

```
stubs/sklearn/__init__.pyi                 0   (empty package marker)
stubs/sklearn/metrics/__init__.pyi        89   accuracy_score, confusion_matrix,
                                               f1_score, precision_score, recall_score
stubs/sklearn/model_selection/__init__.pyi 55  KFold, StratifiedKFold, ShuffleSplit,
                                               StratifiedShuffleSplit, train_test_split
stubs/sklearn/preprocessing/__init__.pyi   21  LabelEncoder, StandardScaler
stubs/sklearn/utils/__init__.pyi            0   (empty package marker)
stubs/sklearn/utils/class_weight.pyi       12   compute_class_weight
```

The metrics file is 89 lines only because `f1_score` / `precision_score` / `recall_score`
each need **two `@overload`s** keyed on `average` (see §6.2). A single-signature version is
45 lines but costs six call-site errors in `finetuning.py`.

### 4.1 How mypy finds them, and what "minimal" actually means

Measured, and matching [the mypy stubs docs](https://mypy.readthedocs.io/en/stable/stubs.html):

- **Use the plain package name, not a `-stubs` suffix.** `stubs/sklearn-stubs/` on
  `mypy_path` is silently ignored: "PEP 561 stub-only packages must be installed, and may
  not be pointed at through the `MYPYPATH`". `stubs/sklearn/` works.
- **No `py.typed` marker is needed, and adding one changes nothing.** A `py.typed`
  containing `partial\n` inside a `mypy_path` tree produced byte-identical output — the
  `partial` machinery in `mypy/modulefinder.py` only runs over site-packages.
- **Unstubbed *submodules* fall through to the runtime package automatically.** With only
  `preprocessing` stubbed, `sklearn.metrics` still reported plain `[import-untyped]` rather
  than an attribute error. So the tree is additive module by module.
- **A stubbed module is closed.** Every name imported *from* a module you stub must be
  declared in it, or mypy reports `Module "sklearn.metrics" has no attribute "f1_score"`.
  That is the drift alarm for *new* imports (§5.2).
- **`mypy_path` is resolved against the working directory, not the config file**, and may
  only appear in the global section. Use `$MYPY_CONFIG_FILE_DIR/stubs`, which also avoids
  the Windows drive-letter `:` splitting problem. Verified: the same config run from a
  different cwd resolved the stubs correctly.

### 4.2 Writing them found four real bugs in the first draft

The first hand-written draft was checked by `stubtest` and was wrong in four ways —
`StratifiedKFold.split` and `StratifiedShuffleSplit.split` require `y` at runtime,
`compute_class_weight` has a `sample_weight` parameter, the splitters carry `ABCMeta`, and
`BaseCrossValidator.get_n_splits` is abstract. That is the maintenance story in miniature:
hand-written stubs drift immediately, and the only thing that makes them honest is the gate
in §5.

---

## 5. The drift story: `stubtest` as a pytest gate

[`stubtest`](https://mypy.readthedocs.io/en/stable/stubtest.html) ships with mypy and
"will import your code and introspect your code objects at runtime… then analyse the stub
files, and compare the two, pointing out things that differ". It is the answer to "who
notices when the stub drifts".

### 5.1 Stub-vs-runtime drift — caught, and clean today

```
stubtest --ignore-missing-stub --concise \
    sklearn.preprocessing sklearn.metrics sklearn.model_selection sklearn.utils.class_weight
```

- `--ignore-missing-stub` ("Ignore errors for stub missing things that are present at
  runtime") is exactly the flag for a deliberately partial stub. Without it, every sklearn
  symbol this repo does not use is an error; with it, the run is quiet.
- It still catches real drift through that flag — a renamed or invented parameter is
  reported regardless.
- **The final stubs in §9 pass with zero findings and exit status 0 against both
  `scikit-learn==1.7.2` (what is installed) and `scikit-learn==1.9.0` (what a 3.11 floor
  resolves to — see §8.3).** That cross-version clean run is the evidence that this surface
  is stable enough to own.

Four operational notes for whoever writes the gate test, all measured:

- stubtest needs mypy **and** the real sklearn importable in the same interpreter. Ticket 02
  already puts mypy in the dev group, so `uv run --python 3.11 pytest` satisfies both.
- **`--mypy-config-file` does work, and it does expand `$MYPY_CONFIG_FILE_DIR`.** Run from an
  unrelated working directory with `MYPYPATH` unset, stubtest read `mypy_path =
  "$MYPY_CONFIG_FILE_DIR/stubs"` out of the config and found the stubs. So the gate needs no
  environment variable.
- **⚠ A mis-wired stubtest passes vacuously.** With no config file and no `MYPYPATH`,
  stubtest found no stubs at all and **exited 0 in silence** — `--ignore-missing-stub`
  swallows the "no stub" case. A gate that forgets to pass the config would therefore be
  green while checking nothing. Proof: with a deliberate `with_mean` → `with_meen` typo
  planted in the stub, the configured run reported
  `StandardScaler.__init__ is inconsistent, runtime does not have parameter "with_meen"` and
  exited 1, while the unconfigured run still exited 0.

  The guard against that is **the mypy gate itself**: if `stubs/` is not on `mypy_path`,
  `from sklearn.metrics import f1_score` is `[import-untyped]` and fails under `strict`. The
  two checks are complementary — mypy proves the stubs are *found*, stubtest proves they are
  *true* — and neither alone is sufficient.
- **On `scikit-learn >= 1.9`, stubtest cannot import `sklearn.model_selection` cold**: it
  walks `__all__` and trips `ImportError: HalvingGridSearchCV is experimental…`. Importing
  `sklearn.experimental.enable_halving_search_cv` first fixes it. Invoking stubtest
  in-process works and is the simplest form for a pytest gate:

  ```python
  import sys

  import sklearn.experimental.enable_halving_search_cv  # noqa: F401  (sklearn >= 1.9)
  from mypy.stubtest import main as stubtest_main

  sys.argv = [
      "stubtest",
      "--mypy-config-file", "pyproject.toml",   # required: see the vacuous-pass note above
      "--ignore-missing-stub",
      "--concise",
      "sklearn.preprocessing", "sklearn.metrics",
      "sklearn.model_selection", "sklearn.utils.class_weight",
  ]
  assert stubtest_main() == 0
  ```

  One caveat carried over from the throwaway runs: stubtest applies `python_version` from
  the config to the *whole* build, including numpy's own stubs. In an environment whose
  numpy is newer than the declared `python_version`, that surfaces as
  `numpy/__init__.pyi: error: Type statement is only supported in Python 3.12 and greater`.
  It did not occur against this repo's numpy 2.2.6 at `python_version = "3.11"`, only in a
  scratch env carrying numpy 2.4.

### 5.2 New-import drift — caught by the checker itself

The second drift direction is someone importing a *thirteenth* symbol. Because a stubbed
module is closed (§4.1), `from sklearn.metrics import roc_auc_score` fails the type check
with `Module "sklearn.metrics" has no attribute "roc_auc_score"`, and a brand-new submodule
fails with `[import-untyped]` under `strict`. Either way the mypy gate — not a human —
raises it, and the fix is to add the declaration. **Nothing silently degrades to `Any`.**

---

## 6. The concrete test: `artifacts.py:318`

The code, `src/training/artifacts.py:297-320`:

```python
def _to_original_units(
    ledger: pd.DataFrame, column: str, scaler: object, numerical_columns: Sequence[str]
) -> list:
    ...
        if cell["kind"] != "numerical" or scaler is None or cell["column"] not in order:
            values.append(cell[column])
            continue
        # inverse_transform wants a whole row, so undo this one column by hand.
        index = order.index(cell["column"])
        values.append(
            float(cell[column]) * float(scaler.scale_[index]) + float(scaler.mean_[index])
        )
```

`scaler: object` is what produces today's two baseline errors. Any option requires ticket 08
to replace that annotation; the question is what the annotation then *means*.

### 6.1 What each option does — measured, not argued

| | A: `ignore_missing_imports` | F: `follow_untyped_imports` | **C: local stubs** |
|---|---|---|---|
| `StandardScaler` resolves to | `Any` | the real class | the real class |
| `reveal_type(scaler.scale_)` | `Any` | `Any` | `ndarray[…, dtype[float64]] \| None` |
| `float(scaler.scale_[index])` | unchecked | unchecked | **checked** — `Value of type "ndarray[…] \| None" is not indexable  [index]` until narrowed |
| `scaler.scalee_` (typo) | **not caught** | caught | caught, with `maybe "scale_"?` |
| cost under `strict = true` | none | `[no-untyped-call]` at every sklearn call site | 177 lines + a stubtest gate |
| verdict for line 318 | **silenced** | **half-silenced** (attribute exists; arithmetic unchecked) | **genuinely checked** |

Only **C** gives real coverage of the arithmetic. A is suppression: after the annotation is
changed to `StandardScaler | None`, the value is `Any`, the errors disappear, and `.scale_`
and `.mean_` are exactly as unchecked as they are today — just quieter, which is worse,
because the baseline no longer records the gap. F is a half-measure that catches a
misspelling but tells you nothing about the numbers.

### 6.2 A bonus the stub surfaced

`StandardScaler.scale_` and `.mean_` are `ndarray | None` in sklearn's own documented
contract — both are `None` when `with_std=False` / `with_mean=False`. `artifacts.py:312`
guards `scaler is None` but **not** `scaler.scale_ is None`. That path is unreachable in
TRIDENT (`src/training/data.py:68` is the only construction site and it is a bare
`StandardScaler()`), but the checker cannot know that, and only option C surfaces it at all.

The same fidelity question appeared in `sklearn.metrics`: a single-signature
`f1_score(...) -> float | NDArray[np.float64]` is honest but produces six new `[dict-item]`
errors at `finetuning.py:29-39`, where the results go into a `dict[str, float | int | str]`.
Two `@overload`s keyed on `average` (`None → NDArray`, a literal → `float`) remove all six
while staying true to sklearn's behaviour. **The recommended stubs use the overloads**; that
is the whole reason `metrics/__init__.pyi` is 89 lines rather than 45.

---

## 7. What `artifacts.py:318` looks like under the recommendation

Type-checked clean under `strict = true` with the §9 stubs (only the deliberate typo probe
at the end errors, as intended):

```python
from sklearn.preprocessing import StandardScaler

def _to_original_units(
    ledger: pd.DataFrame,
    column: str,
    scaler: StandardScaler | None,
    numerical_columns: Sequence[str],
) -> list[object]:
    ...
        index = order.index(cell["column"])
        scale, mean = scaler.scale_, scaler.mean_
        # StandardScaler() is built with with_mean/with_std defaults, so both are arrays.
        assert scale is not None and mean is not None
        values.append(float(cell[column]) * float(scale[index]) + float(mean[index]))
```

**The caller changes with it.** `ArtifactWriter.write_imputation_preview` declares
`scaler: object = None` at `src/training/artifacts.py:231` and forwards it at lines 251-255.
That parameter becomes `StandardScaler | None = None` too, or the `object` simply moves up
one frame and the two errors reappear at the call site.

Three things are now true that are not true today:

1. `scale[index]` and `mean[index]` are indexing a `float64` array, checked.
2. `scaler.meen_` is a type error with a spelling suggestion, not a runtime `AttributeError`
   found by a user.
3. The `None` case is *stated* rather than assumed. The `assert` is deliberately chosen over
   widening the guard at line 312: widening would send a `None`-scaled column down the
   "already readable" path and silently change the artifact's contents, which
   `AGENTS.md`'s behaviour-neutrality rule forbids. The `assert` keeps the happy path
   byte-identical and turns an unreachable state into a loud one.

   The alternative — declaring `scale_: NDArray[np.float64]` non-optional in the stub, which
   removes the need for any change at 318 — was rejected: it is a lie about sklearn's
   contract that `stubtest` cannot catch, and it would hide the case rather than record it.

---

## 8. The configuration for ticket 08

### 8.1 `pyproject.toml`

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "mypy>=2.0",
    "scipy-stubs",       # official scipy stubs; covers the single `scipy.stats.t` import
]

[tool.mypy]
python_version = "3.11"
strict = true
# src/ has no __init__.py, so mypy needs these to resolve `src.X` consistently.
explicit_package_bases = true
namespace_packages = true
# Hand-written stubs for scikit-learn only. Config-file-relative: mypy_path is otherwise
# resolved against the working directory, and a bare Windows path would split on its `:`.
mypy_path = "$MYPY_CONFIG_FILE_DIR/stubs"

# Ticket 02's strictness ramp; one override deleted per execution ticket.
[[tool.mypy.overrides]]
module = ["opt", "src.transformer", "src.models"]
disallow_untyped_defs = false
# ... the rest of ticket 02's relaxations

# tqdm ships no stubs and is out of this ticket's scope; see §10.
[[tool.mypy.overrides]]
module = ["tqdm.*"]
ignore_missing_imports = true
```

There is deliberately **no** `ignore_missing_imports` for `sklearn.*` or `scipy.*`, and **no**
`follow_untyped_imports` anywhere. If either appears in ticket 08's config, this ticket's
answer has been reversed.

### 8.2 Where the stubs live

`stubs/` at the repository root — a sibling of `src/`, never importable at runtime, six
files, contents in §9. `mypy_path` is the only wiring.

### 8.3 Resolver impact — verified against copies of the real `pyproject.toml` + `uv.lock`

Run in a scratch copy; the repo's lock was not touched.

- **On today's `requires-python = ">=3.10"`**, adding `mypy` and `scipy-stubs` to the dev
  group is **purely additive — not one existing pin moves.** uv adds `scipy-stubs`
  1.15.3.0 / 1.17.1.5 / 1.18.1.0 across the Python branches plus `optype` and
  `numpy-typing-compat`. On the 3.10 branch it picks **1.15.3.0**, which matches the
  installed **scipy 1.15.3** exactly. **This ticket's answer therefore does not depend on
  ticket 01 landing first.**
- `scipy-stubs` is unpinned on purpose: its `{scipy_version}.{stubs_version}` scheme plus
  `requires-python` floors make uv track the scipy the lock already resolves. The 3.12
  requirement on `scipy-stubs 1.18.x` is not an extra constraint — scipy 1.18 itself
  requires 3.12.

**Clarification for ticket 01 (not a correction — the map's resolver fact stands).** Today's
3.10 `.venv` runs **scikit-learn 1.7.2 and scipy 1.15.3**, but the 3.11 environment ticket 01
must stand up will run **scikit-learn 1.9.0 and scipy 1.17.1**. The lock already carries
those as the 3.11-branch resolutions, which is exactly why the map's `uv lock --python 3.11
--dry-run` reported no lockfile changes. Editing `requires-python` to `>=3.11` only *removes
the 3.10 branch*:

```
Update numpy         v2.2.6, v2.4.6, v2.5.0    -> v2.4.6, v2.5.0
Update scikit-learn  v1.7.2, v1.9.0            -> v1.9.0
Update scipy         v1.15.3, v1.17.1, v1.18.0 -> v1.17.1, v1.18.0
```

So the lines that disappear are the 3.10-only pins, not a change to what 3.11 gets. The
consequence for *this* ticket is that the stubs must be true of **two** sklearn versions
across the floor bump, which is why §5.1 verified them against 1.9.0 as well as 1.7.2.

### 8.4 Honest cost to ticket 08

Default-mode mypy over `src main.py opt.py train.py scripts`, mypy 2.3.1, same tree:

| | without stubs | with the §9 stubs |
|---|---|---|
| total errors | 24 | 25 |
| `[import-untyped]` from sklearn/scipy | 8 | **0** |
| `[import-untyped]` from tqdm (§10) | 3 | 3 |
| **substantive errors** | **13** | **22** |

Adding real types does not reduce the count; it **changes what the count means.** The nine
newly visible errors are all genuine and all in `src/training/data.py`:

- `205`, `212`, `217` — `splitter` is reassigned across `KFold`, `StratifiedKFold`,
  `StratifiedShuffleSplit` and `ShuffleSplit` in one function. Fix: one annotation,
  `splitter: BaseCrossValidator`.
- `80`, `224`-`226` — `LabelEncoder.classes_` and `split()` return `ndarray`, while
  `PreparedDataset.label_classes` and `FoldSplit.*_indices` are declared `Sequence[...]`.
  (`166`-`168` are the same mismatch from `np.array(...)` and are already in the baseline.)
  Fix: widen those dataclass fields — a typing decision ticket 08 or 06 owns.
- `191`, `209` — `frame[label_column].values` is `ndarray | ExtensionArray` under
  pandas-stubs, and `ExtensionArray` is not `ArrayLike`. Fix: `np.asarray(...)` at the call
  site, a no-op at runtime on the ndarray this actually is.

These are the nine fixes option C buys with the coverage. Option A buys none of them and
leaves the count at 16 — a smaller number describing less checking.

**The map's default-mode baseline needs restating, and not only because of this ticket.**
The map records "7 errors at default strictness". Measured here on mypy 2.3.1 with
pandas-stubs installed and `explicit_package_bases`/`namespace_packages` set, the
default-mode baseline is **13 substantive errors** (plus 8 sklearn/scipy and 3 tqdm import
errors). The charting session did not record which mypy version it used, and mypy 2.0
changed defaults that plausibly account for the difference — `local_partial_types` and
`strict_bytes` both became on-by-default. **The cause is unverified**; what is verified is
the number. So ticket 08's budget is:

- **13 substantive errors today**, not 7;
- **22 after the stubs land** — the same 13 plus the 9 above;
- **0 sklearn/scipy import errors** in both post-stub columns.

Ticket 08 should re-measure against its own pinned mypy before committing to a figure.

---

## 9. The stubs, verbatim

These exact files produced every "with stubs" result above and pass `stubtest
--ignore-missing-stub` against scikit-learn 1.7.2 and 1.9.0 with zero findings. They assume
`python_version = "3.11"` (`typing.Self` is 3.11+); on a 3.10 floor, swap to
`typing_extensions.Self`.

<details>
<summary><code>stubs/sklearn/preprocessing/__init__.pyi</code> (21 lines)</summary>

```python
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LabelEncoder:
    classes_: NDArray[Any]
    def fit(self, y: ArrayLike) -> Self: ...
    def fit_transform(self, y: ArrayLike) -> NDArray[np.int64]: ...
    def transform(self, y: ArrayLike) -> NDArray[np.int64]: ...
    def inverse_transform(self, y: ArrayLike) -> NDArray[Any]: ...

class StandardScaler:
    scale_: NDArray[np.float64] | None
    mean_: NDArray[np.float64] | None
    var_: NDArray[np.float64] | None
    n_features_in_: int
    def __init__(self, *, copy: bool = ..., with_mean: bool = ..., with_std: bool = ...) -> None: ...
    def fit(self, X: ArrayLike, y: Any = ..., sample_weight: ArrayLike | None = ...) -> Self: ...
    def transform(self, X: ArrayLike, copy: bool | None = ...) -> NDArray[np.float64]: ...
    def fit_transform(self, X: ArrayLike, y: Any = ..., **fit_params: Any) -> NDArray[np.float64]: ...
    def inverse_transform(self, X: ArrayLike, copy: bool | None = ...) -> NDArray[np.float64]: ...
```
</details>

<details>
<summary><code>stubs/sklearn/model_selection/__init__.pyi</code> (55 lines)</summary>

```python
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Indices = NDArray[np.int64]
_Folds = Iterator[tuple[_Indices, _Indices]]

class BaseCrossValidator(metaclass=ABCMeta):
    def split(
        self, X: ArrayLike, y: ArrayLike | None = ..., groups: ArrayLike | None = ...
    ) -> _Folds: ...
    @abstractmethod
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class KFold(BaseCrossValidator):
    def __init__(
        self, n_splits: int = ..., *, shuffle: bool = ..., random_state: int | None = ...
    ) -> None: ...
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class StratifiedKFold(KFold):
    # sklearn narrows `y` to required here; mirroring that is an LSP break it also has.
    def split(  # type: ignore[override]
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = ...
    ) -> _Folds: ...

class BaseShuffleSplit(BaseCrossValidator):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: float | int | None = ...,
        train_size: float | int | None = ...,
        random_state: int | None = ...,
    ) -> None: ...
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class ShuffleSplit(BaseShuffleSplit): ...

class StratifiedShuffleSplit(BaseShuffleSplit):
    def split(  # type: ignore[override]
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = ...
    ) -> _Folds: ...

def train_test_split(
    *arrays: Any,
    test_size: float | int | None = ...,
    train_size: float | int | None = ...,
    random_state: int | None = ...,
    shuffle: bool = ...,
    stratify: ArrayLike | None = ...,
) -> list[Any]: ...
```
</details>

<details>
<summary><code>stubs/sklearn/metrics/__init__.pyi</code> (89 lines, complete)</summary>

```python
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Average = Literal["micro", "macro", "samples", "weighted", "binary"]
_ZeroDivision = Literal["warn"] | float

def accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    normalize: bool = ...,
    sample_weight: ArrayLike | None = ...,
) -> float: ...
def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    sample_weight: ArrayLike | None = ...,
    normalize: Literal["true", "pred", "all"] | None = ...,
) -> NDArray[Any]: ...
@overload
def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
@overload
def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
@overload
def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
```
</details>

<details>
<summary><code>stubs/sklearn/utils/class_weight.pyi</code> (12 lines) — plus empty <code>stubs/sklearn/__init__.pyi</code> and <code>stubs/sklearn/utils/__init__.pyi</code></summary>

```python
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

def compute_class_weight(
    class_weight: Mapping[int, float] | str | None,
    *,
    classes: NDArray[np.int64],
    y: ArrayLike,
    sample_weight: ArrayLike | None = ...,
) -> NDArray[np.float64]: ...
```
</details>

**No scipy stub is written** — `scipy-stubs` covers it. The fallback below is recorded only
so it never has to be rediscovered: if `scipy-stubs` were ever unavailable, this 11-line
file is sufficient for the one call in play, and it passes `stubtest --ignore-missing-stub
--concise scipy.stats` against scipy 1.15.3 with zero findings.

<details>
<summary><code>stubs/scipy/stats/__init__.pyi</code> — fallback only, NOT part of the recommendation</summary>

```python
from typing import Any

from numpy.typing import ArrayLike

class _TDistribution:
    def ppf(self, q: ArrayLike, df: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...) -> Any: ...
    def cdf(self, x: ArrayLike, df: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...) -> Any: ...
    def sf(self, x: ArrayLike, df: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...) -> Any: ...
    def interval(
        self, confidence: float, df: ArrayLike, loc: ArrayLike = ..., scale: ArrayLike = ...
    ) -> tuple[Any, Any]: ...

t: _TDistribution
```

Note it returns `Any` from `ppf`, where the real `scipy-stubs` returns `float`. That is the
gap between a fiction written for one call site and a maintained package — another reason B
wins here.
</details>

### The scipy side, verified

`src/training/summary.py:214` is the entire scipy surface:
`t.ppf(0.975, df=len(value_array) - 1)`. With `scipy-stubs` installed in a throwaway 3.11
venv (`scipy 1.17.1` + `scipy-stubs 1.17.1.5`):

- `reveal_type(t)` → `scipy.stats._continuous_distns.t_gen`
- `reveal_type(t.ppf(0.975, df=n - 1))` → **`float`** (an overload resolves the scalar case)
- `t.ppff(...)` → `error: "t_gen" has no attribute "ppff"  [attr-defined]`

Real coverage, zero maintenance for this repo. (One limitation: a wrong keyword such as
`degrees_of_freedom=` was *not* rejected, because the overload ends in `**kwds`.)

---

## 10. Out of scope, but found

**The map's "sklearn is the only gap that bites" is not quite right — `tqdm` is untyped too.**
The repo-wide run reports it three times:

```
src/training/pretraining.py:6: error: Library stubs not installed for "tqdm"  [import-untyped]
src/training/finetuning.py:10: error: Library stubs not installed for "tqdm"  [import-untyped]
src/training/decoding.py:13: error: Library stubs not installed for "tqdm"  [import-untyped]
```

`types-tqdm` exists in typeshed and mypy hints it. It was not in this ticket's enumerated
surface, so it is not decided here — the §8.1 block parks it behind
`ignore_missing_imports` so ticket 08 can go green, and the real choice (add `types-tqdm` to
the dev group) is a one-line follow-up. `tqdm` wraps iterables and produces no values the
codebase does arithmetic on, so the stakes are nothing like sklearn's.

---

## 11. What could not be verified

- **conda-forge** packaging state for `scipy-stubs` or any sklearn stub — not checked.
- **`scikit-learn-stubs` 0.0.3's actual sklearn target.** It declares no `Requires-Dist`.
  The `<1.7.0` figure is Microsoft's *current* pin read forward, not read out of the
  August 2025 snapshot. It does not change the verdict: the package is an unmaintained
  republish either way.
- **Microsoft's repo carries no explicit deprecation notice.** Its staleness for sklearn is
  inferred from the `<1.7.0` pin plus the `# TODO: Update stubs for sklearn` comment plus
  the last push date, not from a statement by its maintainers.
- **The stubs were not run against `scikit-learn 1.9.1`**, only 1.7.2 and 1.9.0. 1.9.1 was
  released the day before this research and is not what either floor resolves to.
- **`uv lock` was only dry-run**, never applied, and no install was performed into `.venv`.
  The resolver output in §8.3 is what uv reports it *would* do.
- **`scipy-stubs` was exercised on `scipy 1.17.1`, not on the `scipy 1.15.3` in this venv.**
  Installing it here would resolve `scipy-stubs 1.15.3.0`, which is the matched pair; the
  one call in play (`t.ppf`) has been stable across every scipy version in range, but that
  specific combination was not run.

---

## 12. Sources

**typeshed** · <https://api.github.com/repos/python/typeshed/contents/stubs> ·
<https://github.com/python/typeshed/blob/main/CONTRIBUTING.md#third-party-library-stubs> ·
<https://github.com/python/typeshed/blob/main/CONTRIBUTING.md#third-party-library-removal-policy>

**scipy** · <https://pypi.org/pypi/scipy-stubs/json> · <https://api.github.com/repos/scipy/scipy-stubs> ·
<https://github.com/scipy/scipy-stubs/blob/master/README.md> ·
<https://github.com/scipy/scipy/issues/17158> ·
<https://github.com/scipy/scipy/issues/17158#issuecomment-3670810426> ·
`doc/source/release/1.{15,16,17,18}.0-notes.rst` in <https://github.com/scipy/scipy> ·
<https://pypi.org/pypi/scipy/json>

**scikit-learn** · <https://github.com/scikit-learn/scikit-learn/issues/16705> ·
<https://github.com/scikit-learn/scikit-learn/issues/16705#issuecomment-2421887710> ·
<https://pypi.org/pypi/scikit-learn/json> · <https://pypi.org/pypi/scikit-learn-stubs/json> ·
<https://github.com/hoel-bagard/scikit-learn-stubs> ·
<https://github.com/microsoft/python-type-stubs> ·
<https://github.com/microsoft/python-type-stubs/blob/main/pyproject.toml>

**mypy** · <https://mypy.readthedocs.io/en/stable/config_file.html> ·
<https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports> ·
<https://mypy.readthedocs.io/en/stable/stubs.html> ·
<https://mypy.readthedocs.io/en/stable/installed_packages.html> ·
<https://mypy.readthedocs.io/en/stable/stubtest.html> ·
<https://mypy.readthedocs.io/en/stable/error_code_list.html#code-import-untyped> ·
<https://github.com/python/mypy/pull/17712> ·
<https://github.com/python/mypy/blob/master/CHANGELOG.md> · <https://pypi.org/pypi/mypy/json>

**typing spec** · <https://peps.python.org/pep-0561/> ·
<https://typing.readthedocs.io/en/latest/spec/distributing.html#partial-stub-packages>

**Local** · `.venv/Lib/site-packages/sklearn/preprocessing/_data.py` (1.7.2) ·
`.venv/Lib/site-packages/{sklearn,scipy}/` `py.typed` absence ·
`src/training/artifacts.py:297-320` · `src/training/data.py:68,151-226` ·
`src/training/finetuning.py:20-42` · `src/training/summary.py:207-218`
