# Map: Typing and naming standard for TRIDENT

Label: wayfinder:map
Charted: 2026-09-11
Tracker: local markdown. `AGENTS.md` keeps tickets, plans and decisions under `docs/`,
so this effort lives at `docs/wayfinder/typing-and-constants/`. Tickets are
`issues/NN-<slug>.md`; each carries `Type:`, `Status:` (`open` / `claimed` / `resolved`)
and `Blocked by:` lines. The frontier is every open, unblocked, unclaimed ticket, lowest
number first.

Work it with `/wayfinder docs/wayfinder/typing-and-constants/map.md` (optionally naming a
ticket).

Charting note: the charting session ran two grilling rounds, which settled tickets 01 to
04 on the spot; they are recorded as resolved so each decision and its reasoning has one
home. The user asked the session to continue until an answer was needed rather than stop
at one ticket per turn, so later sessions should expect the same cadence unless told
otherwise.

## Destination

A decided **typing and naming standard for TRIDENT**, recorded as ADR 0006 under
`docs/adr/` plus an implementation plan beside this map at `plan.md`, precise enough that
the annotation and literal-replacement work can be executed from the plan without another
decision. It covers: the Python floor and what the bump must prove; the type checker, its
per-module strictness ramp and what enforces it in a repo with no CI; how the five name
families are represented and where constants live; the task-branch union and its
exhaustiveness check; the third-party stub strategy; the hyperparameter JSON schema and
the three naming layers; and the enforcement of the constants convention beyond what the
checker can catch.

Planning only, with one exception: **the Python floor is bumped and the checker is
configured and made green inside the map** (ticket 08), by configuration and the seven
default-mode fixes, not by mass annotation. Every later strictness decision is opinion
until a checker actually runs, and the ramp in ticket 02 is only credible if its starting
point is real. The bump rides along in the same ticket because the checker configuration
depends on it — `python_version = "3.11"`, `assert_never`, and a `uv run --python 3.11`
acceptance cannot be verified on a 3.10 tree. The annotation work itself and the literal
replacement are execution, launched from the plan.

## Notes

- **Domain**: TRIDENT. Read `AGENTS.md` (the protected invariants), `CONTEXT.md`
  (glossary; this effort is implementation, not domain vocabulary, so it is expected to
  add **no** CONTEXT.md terms — see ticket 03's answer), `README.md` (CLI flags, the
  hyperparameter config schema, the MLflow comparison protocol) and `docs/BACKLOG.md`
  (no existing typing or constants entry; this effort is new ground). Code: the typed core
  `src/training/` (especially `types.py`, `tracking.py`, `config.py`, `runner.py`), the
  legacy `src/*.py` (`models.py`, `transformer.py`, `embedder.py`, `utils.py`,
  `mlflow_utils.py`) and `opt.py`. For codebase questions run `graphify query "<question>"`
  first.
- **Skills each session should call**: `mattpocock-skills:grilling` and
  `mattpocock-skills:domain-modeling` for any grilling ticket;
  `mattpocock-skills:research` for ticket 05; `mattpocock-skills:tdd` for ticket 08's
  code, followed by the real run its acceptance names.
  `mattpocock-skills:codebase-design` is the right vocabulary if a ticket starts arguing
  about module structure — but see **Out of scope**: restructuring is not this effort.
- **Standing preferences** (from `AGENTS.md`, `MEMORY.md` and prior decisions):
  - **This effort is behaviour-neutral by construction.** Classification must stay
    bit-identical and imputation must stay bit-identical; the `vehicle_00nan` and
    `credit-g_20nan` integration fixtures pass **unedited** as the acceptance criterion on
    every execution ticket. That is also the "run the model, not only tests" proof this
    repo requires: for a refactor with no intended behaviour change, the fixtures are the
    run.
  - `train.main(args, return_metrics=False)` and the Optuna-compatible return shape stay
    stable. Type them; do not reshape them.
  - Behaviour choices ship as a flag with the old behaviour as default plus an MLflow tag
    (the ADR 0003 pattern). Nothing in this effort should need that pattern — if a ticket
    finds itself reaching for it, the ticket has strayed out of scope.
  - Tickets, plans and decisions live under `docs/`; no external issues.
- **Facts established while charting** (verified in-session against the code, the venv and
  the resolver, so sessions need not re-derive them):
  - **mypy baseline on the tree at charting**: 7 errors at default strictness, **136
    errors in 17 files at `--strict`**. The strict errors concentrate: `opt.py` 43,
    `src/transformer.py` 22, `src/models.py` 14 — **79 of 136, 58%, in three files**. The
    remaining fourteen files average four each. This is what makes the per-module ramp in
    ticket 02 viable.

    **Correction — the default-mode count of 7 is wrong.** Ticket 05's research re-measured
    on mypy 2.3.1 and found **13 substantive errors** (+8 sklearn/scipy, +3 tqdm). The
    charting session did not record its mypy version and mypy 2.0 changed several defaults,
    so the cause is unverified. **With ticket 05's stubs in place the figure is 22
    substantive and zero sklearn/scipy import errors**: the 9 newly visible errors are all
    real, all in `src/training/data.py`. Ticket 08 fixes 22, not 7. The `--strict` figures
    below were not re-measured and carry the same uncertainty.

    **Caveat on these numbers.** They were measured with
    `mypy --ignore-missing-imports --explicit-package-bases --namespace-packages` against a
    worktree that was mid-change on execution ticket 03 of the *previous* effort
    (`src/training/config.py`, `runner.py`, `tracking.py`, `types.py` and `opt.py` all
    modified, uncommitted). Treat the distribution as sound — 58% in three files is not
    going to be overturned by one ticket's diff — but treat the exact counts as indicative.
    Ticket 08 re-measures against the real configuration on a settled tree, and its numbers
    are the ones the ramp is built on.
  - The seven default-mode errors are three distinct kinds, one per later ticket:
    `runner.py:93` an undeclared task-branch union (ticket 04); `summary.py:234-235` fold
    keys typed `int | str` flowing into a `dict[int, str]`; `artifacts.py:318` a
    `StandardScaler` typed `object` so `.scale_` / `.mean_` go unchecked (ticket 05).
  - **mypy cannot resolve the package as passed**: `src/` has no `__init__.py`, so mypy
    reports `Source file found twice under different module names: "utils" and "src.utils"`
    and needs `--explicit-package-bases --namespace-packages`. Imports themselves are
    consistent (`from src.X import ...` everywhere); only the package marker is missing.
  - **`StrEnum` on 3.11 is safe on every serialisation path this repo uses.** Verified on
    3.11.15: `str()`, `format()`, f-string and `%s` all yield the bare value,
    `json.dumps` yields the value, `member == "value"` is `True`, a member works as a
    plain-string dict key, and `Sched("value") is Sched.MEMBER`. **On 3.10 `(str, Enum)`
    is not**: `str(member)` yields `"S.A"`, and MLflow stringifies tag and param values,
    so a 3.10 enum would have silently written `LrScheduler.COSINE_LEGACY` into the tag
    surface while JSON fixtures stayed green. This single difference is why the Python
    floor had to be settled before the name-family question (tickets 01, 03).
  - **The 3.11 dependency set resolves clean.** `uv lock --python 3.11 --dry-run` reports
    "Resolved 136 packages", no lockfile changes, no files touched. `uv.lock` already
    carries `python_full_version == '3.11.*' and sys_platform == 'win32'` resolution
    markers for `torch 2.5.1+cu121`, and `pyarrow 23.0.1` ships `cp311-cp311-win_amd64`.
    **Not verified: installation.** The 2.5 GB CUDA wheel was not downloaded or imported,
    which is why ticket 01 makes that the bump's acceptance criterion rather than an
    assumption.
  - **Stub coverage in the venv** (`py.typed` marker present): `torch`, `optuna`,
    `mlflow`, `numpy`, `matplotlib` **typed**; `sklearn`, `scipy`, `pandas` **untyped**.
    `pandas-stubs~=2.3.3` is already a dependency, so pandas is covered. Ticket 05 resolved
    the rest and also found **`tqdm`, which this list missed** — it ships no stubs and
    accounts for 3 errors. Parked behind `ignore_missing_imports` so ticket 08 can reach
    green, with `types-tqdm` as a backlog follow-up.
  - **The constants convention already exists and is half-applied.** `src/training/types.py`
    declares `LR_SCHEDULER_NAMES`, `TASK_NAMES`, `SEARCH_SPACE_PROFILES` and
    `TRACKING_RUN_ROLES` as `tuple[str, ...]`; `src/mlflow_utils.py` declares
    `LR_SCHEDULER_TAG`, `TASK_TAG`, `IS_OPTUNA_TAG`, `SEARCH_SPACE_TAG`;
    `src/training/imputation_metrics.py` declares `NUMERICAL` / `CATEGORICAL`. And
    `src/training/tracking.py:222` still writes `{"dataset": ..., "run_role": role}` with
    bare literals, two lines above `tags[TASK_TAG]`. The problem is not a missing
    convention; it is that **nothing enforces the one that exists** (tickets 03, 07).
  - **Size of the literal surface**: about 36 bare literals of the five name families
    across ten source files, excluding their definition sites — `tracking.py` 7,
    `runner.py` 5, `config.py` 5, `opt.py` 5, `schedulers.py` 4, `artifacts.py` 3,
    `imputation_metrics.py` 2, `mlflow_utils.py` 2, `embedder.py` 2, `main.py` 1.
  - **No legacy typing anywhere.** Zero occurrences of `Optional[`, `Dict[`, `List[`,
    `Tuple[` or `Union[` in `src/`, the entry points or `scripts/`. The modernisation this
    effort needs is not `Optional` to `|`; it is annotations where there are none, and
    enforcement where there is none.
  - **Unannotated function returns**, worst first: `src/training/tracking.py` 11/35,
    `opt.py` 11/12, `src/transformer.py` 9/9, `src/training/summary.py` 6/14,
    `src/training/artifacts.py` 6/14, `src/models.py` 5/11, `src/utils.py` 4/4.
  - **The duplicated defaults in `from_mapping` all agree.** Verified by walking the AST
    and comparing each innermost `.get()` default against `dataclasses.fields()`: 22 fields
    are read, every field with a default is read (none missed), 21 restate the default as a
    literal and **all 21 agree**. The 22nd, `lr_scheduler` at `types.py:160`, passes
    `DEFAULT_LR_SCHEDULER` instead of restating it — the pattern ticket 06 should
    generalise. So collapsing the duplication is behaviour-neutral *today*; it is a design
    decision, not a risk assessment.
  - **No enforcement infrastructure exists**: no mypy, pyright or ruff configuration, no
    `.github/`, no pre-commit. `pytest.ini` sets `pythonpath = .` and declares the
    `integration` marker. Whatever gates the checker has to be something this repo already
    runs (ticket 02).

## Decisions so far

<!-- one line per resolved ticket: gist, then the link for detail -->

- [01. The Python floor moves to 3.11](issues/01-python-floor.md): the floor becomes 3.11
  so `StrEnum`, `Self` and `assert_never` are available, chosen over a research-first
  approach after the resolver verified the 3.11 dependency set clean. The bump's
  acceptance is an installed 3.11 environment that imports `torch` with
  `cuda.is_available()` true and passes both integration fixtures unedited; the pin lives
  in `.python-version`, `pyproject.toml`, `AGENTS.md` and every `uv run --python 3.10`.
- [02. The checker, the strictness ramp and the gate](issues/02-checker-and-gate.md):
  mypy, gated by a unit test inside `pytest -m "not integration"` because that is the
  ritual `AGENTS.md` already imposes; `strict = true` globally with
  `[[tool.mypy.overrides]]` relaxing the modules that cannot meet it yet, so the ramp is
  one execution ticket per deleted override. Scope is all of `src/` plus `main.py`,
  `opt.py`, `train.py` and `scripts/`; `tests/` is checked but not annotation-gated.
  **The ramp turned out to be eleven blocks, not the three this ticket expected** — ticket
  08 measured it against the real configuration, which is why the override list was left as
  that ticket's output rather than fixed here.
- [03. Five StrEnums, and where constants live](issues/03-name-families-and-homes.md):
  `Task`, `LrScheduler`, `SearchSpace`, `RunRole` and `ColumnKind` become `StrEnum`s and
  the existing name tuples are **derived** from them (`TASK_NAMES = tuple(Task)`) rather
  than maintained alongside, which removes the second source that let the convention
  drift. Value enums consolidate in `src/training/types.py`; MLflow tag and param **keys**
  stay in `src/mlflow_utils.py`, gaining `RUN_ROLE_TAG` and `DATASET_TAG`. Metric keys
  stay with `TaskSpec`. No CONTEXT.md terms.
- [04. The task-branch union and exhaustiveness](issues/04-task-branch-union.md): declare
  `finetuning: FinetuningOutcome | DecodingOutcome` at `runner.py:93` and add
  `assert_never` on the task branch so a third task fails the checker instead of falling
  silently into the classification arm. A shared `Protocol` or ABC unifying the two
  outcomes is **out of scope** — that changes module structure, not types.
- [05. Third-party stubs for sklearn and scipy](issues/05-third-party-stubs.md):
  different answer per library, `ignore_missing_imports` for neither. **scipy** takes the
  official `scipy-stubs` dev dependency, which resolves without moving a single existing
  pin. **sklearn** takes 177 lines of local stubs in `stubs/` — no maintained package
  exists, upstream closed the request as `not_planned`. `follow_untyped_imports` was
  measured and rejected: it breaks `scipy.stats.t` and is noisier than ignoring the import.
  `artifacts.py:318` ends up **genuinely checked, not silenced**. Full evidence, the
  configuration block and the stub source: [`research/05-stub-strategy.md`](research/05-stub-strategy.md).
  Also corrected this map's default-mode baseline (7 → 13, or 22 with the stubs) and found
  that a mis-wired `stubtest` **passes vacuously** unless given
  `--mypy-config-file pyproject.toml`.
- [06. The hyperparameter JSON schema](issues/06-hyperparameter-schema.md): the JSON key
  moves into each field's `dataclasses.field(metadata=...)` and `from_mapping` becomes a
  `fields()` loop — ticket 03's single-source principle applied to the second naming layer,
  chosen over a `TypedDict` or a key-to-field table because only this option removes the
  duplication rather than typing it. The 22 duplicated defaults collapse, which the
  charting measurement showed is behaviour-neutral today. Unknown keys stay ignored; both
  naming layers stay spelled exactly as they are.
- [07. Enforcing the convention beyond the checker](issues/07-enforcing-the-convention.md):
  narrow the tag helpers' own parameter types (`run_role: RunRole`) so the checker rejects
  a bare `"parent"`, rather than adding a grep rule — a grep rule needs an allow-list,
  cannot tell a mention from a violation, and rots into noise if it never fires.
  **Contingent on a measurement the execution ticket must take first**: how many tag writes
  funnel through helpers versus calling `mlflow.set_tag` directly. A wide surface reopens
  this rather than bolting a grep rule onto the remainder.
- [08. Configure the checker and make it green](issues/08-configure-the-checker.md): the
  execution carve-out, done in `d99d5d7` (floor to 3.11, torch cu121 verified on the GPU),
  `30cef72` (`src/__init__.py`) and `3fa848d` (mypy, stubs, ramp, gate). **17 of 28 files
  pass `strict = true`**; eleven override blocks are the ramp inventory, each relaxing only
  the flags its module owes. The checker found four defects on its first run, the sharpest
  being `runner.py` reading `DecodingOutcome.scored_cells` behind a string compare nothing
  could verify — the same shape as the Optuna defect that silently trained classifiers.
  Both fixtures pass unedited; the gate was proven to fail before being trusted.

## Not yet specified

- **The per-module ramp order for the three relaxed files.** Ticket 02 fixes the
  mechanism (delete an override, fix that module, one execution ticket each) but not the
  order or the per-module cost. `opt.py` at 43 errors is the biggest and the one where
  defects have historically hidden; `src/transformer.py` at 22 is pure torch tensor
  shapes and may want shape-aware annotations or may want none. Unsharp until
  ticket 08 lands and the strict counts are re-measured against the real configuration
  rather than an `--ignore-missing-imports` approximation.
- **Whether `Hyperparameters` should keep its `#TODO store those defaults elsewhere`.**
  The dataclass carries that signpost at `src/training/types.py:86`, above 20-odd default
  values that are simultaneously the dataclass defaults, the MLflow logged params and the
  fallback when no promoted configuration exists. Moving them is a config-architecture
  question that typing only brushes against; it may turn out to be one ticket, several, or
  a separate effort once ticket 06 has typed the JSON schema.
- **Typing the test suite.** Ticket 02 settled that `tests/` is checked but not
  annotation-gated. Whether the 4,253 lines should eventually be annotated — and whether
  the fixture builders in particular would catch anything — is a question for after the
  source ramp finishes, not before.
- **Runtime validation at the untyped edges.** A checker proves nothing about a JSON file
  or a CLI string arriving from outside. Ticket 06 will type the hyperparameter schema,
  but whether `from_mapping` should also *validate* at runtime (and what it should do with
  an unknown key, which it currently ignores) is downstream of that ticket's shape.

## Out of scope

- **Changing any persisted value.** MLflow tag and param values, the hyperparameter JSON
  keys (`EPOCHS_PRE` and friends), metric key strings, run-role values, and CLI flag
  spellings are a contract across 89 parent runs and the comparison protocol documented in
  `README.md` and `CLAUDE.md`. **This effort changes symbols, never values.** A rename
  routes through the flag-plus-tag-plus-backfill pattern as its own effort. This boundary
  is stated rather than assumed because a symbol refactor is precisely where a value
  rename gets smuggled in unnoticed.
- **Restructuring modules.** No new `Protocol` or ABC to unify `FinetuningOutcome` and
  `DecodingOutcome` (ticket 04), no splitting of `tracking.py` or `opt.py`, no moving the
  legacy `src/*.py` under `src/training/`. Deep-module work is a `codebase-design` effort;
  this one annotates and names what is already there.
- **Dataclass `slots=True` and `kw_only=True`.** Reads as free modernisation and is not:
  `kw_only=True` breaks every positional construction, including
  `TaskSpec("classification", "f1_macro", "maximize", "finetune", "f1_macro")` in
  `types.py` and the fixture builders throughout `tests/`, and `slots=True` interacts
  badly with the `@property` accessors `TaskSpec` already has. No type-safety gain for a
  call-site-breaking change.
- **Standing up CI.** A `.github/workflows/` would be the strongest gate, but it opens its
  own questions — runner OS, whether the GPU-bound integration fixtures run there, MLflow
  store access — that belong to a CI effort, not this one. Ticket 02 chose the gate this
  repo can actually enforce today.
- **Python 3.12 or newer.** PEP 695 (`type X = ...`, the new generic syntax) and
  `@override` were weighed and declined with the floor decision: 3.11 buys the features
  this effort actually needs, and each further version multiplies the cu121 wheel
  re-verification that ticket 01 must do by hand on Windows.
- **Annotating `tests/`.** 4,253 lines, roughly the size of `src/`, where strict typing
  pays least and doubles every execution ticket's surface. Checked, not gated; see also
  the softer question in **Not yet specified**.
