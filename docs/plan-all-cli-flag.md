# Add `--all` CLI Flag for Batch Dataset Training

Run all available datasets in a single CLI invocation, with companion flags to control the NaN level and limit the number of datasets.

## Example Usage

```bash
# Run all 9 datasets with 0% missing data
uv run --python 3.10 python main.py --all --cv_folds 3

# Run first 3 datasets (alphabetical) with 40% missing data
uv run --python 3.10 python main.py --all --limit 3 --nan_level 40 --cv_folds 3

# Single dataset (unchanged behavior)
uv run --python 3.10 python main.py --dataset_name vehicle_00nan --cv_folds 3
```

## Design Decisions

- **Mutual exclusivity**: `--all` is mutually exclusive with both `--dataset_name` and `--use_optuna`. The parser will enforce this at argument-parsing time.
- **Dataset discovery**: Scans `datasets/processed_datasets/` at runtime, treating each subdirectory (except `splits/`) as a base dataset name.
- **Ordering**: Datasets are sorted alphabetically for deterministic `--limit` behavior.
- **Representative variant**: Each base dataset is combined with a NaN suffix (default `00nan`) to form the full dataset name (e.g. `vehicle` → `vehicle_00nan`).
- **Execution**: Sequential, one dataset at a time. `run_from_namespace()` and `run_training()` remain single-dataset.
- **Error handling**: A failed dataset is logged and skipped; the loop continues. A summary table is printed at the end.
- **`--nan_level`**: Only valid with `--all`. Accepts an integer from `{0, 20, 40, 60, 80}` (default `0`). The code appends `nan` internally.

## Proposed Changes

### CLI Parser

#### [MODIFY] `src/training/config.py`

Add three new arguments to `build_training_parser()`:

| Flag | Type | Default | Constraint |
|------|------|---------|------------|
| `--all` | `store_true` | `False` | Mutually exclusive with `--dataset_name` |
| `--limit` | `int` | `None` | Only valid with `--all` |
| `--nan_level` | `int` | `0` | Only valid with `--all`; choices `{0, 20, 40, 60, 80}` |

Use an `argparse` mutually exclusive group for `--all` and `--dataset_name`. This also means `--dataset_name` is no longer `required=True` — instead, the parser's validation will ensure exactly one of the two is provided.

Add a post-parse validation function (`validate_parsed_args`) that checks:
- At least one of `--all` or `--dataset_name` is provided.
- `--limit` and `--nan_level` are rejected when `--all` is not set.
- `--use_optuna` is rejected when `--all` is set.

---

### Dataset Discovery

#### [NEW] `src/training/discovery.py`

New module with a single function:

```python
def discover_datasets(nan_level: int = 0, limit: int | None = None) -> list[str]:
    """Return sorted dataset names from datasets/processed_datasets/.

    Scans subdirectories (excluding 'splits'), combines each base name
    with the nan_level suffix, and validates the resulting CSV exists.
    """
```

Returns e.g. `["biodeg_00nan", "credit-g_00nan", ..., "vehicle_00nan"]`.

---

### Batch Orchestration

#### [MODIFY] `main.py`

Add a `run_all(args)` function that:

1. Calls `discover_datasets(args.nan_level, args.limit)` to get the dataset list.
2. Loops over each dataset name, setting `args.dataset_name` on a copy of the namespace.
3. Calls `run_from_namespace(copied_args)` inside a `try/except`.
4. Tracks per-dataset status, key metrics (f1_macro), and wall-clock runtime.
5. Prints a summary table at the end.

The existing `main()` dispatches to `run_all()` when `args.all` is `True`.

---

### CLI Module

#### [MODIFY] `src/training/cli.py`

Update `parse_args()` to call the new `validate_parsed_args()` after parsing.

---

## What Is NOT Changing

- `run_from_namespace()` — remains single-dataset.
- `run_training()` — remains single-dataset.
- `train.py` — the `train.main(args, return_metrics)` API/contract is untouched.
- `opt.py` — its standalone parser and logic are untouched.
- All existing tests and integration baselines.

## Verification Plan

### Automated Tests
```bash
# Existing tests must still pass
uv run --python 3.10 pytest -m "not integration"
```

### Manual Verification
- `uv run --python 3.10 python main.py --all --limit 2 --cv_folds 3 --disable_mlflow` — should run biodeg_00nan and credit-g_00nan, print a summary table.
- `uv run --python 3.10 python main.py --all --nan_level 40 --limit 1 --disable_mlflow` — should run biodeg_40nan only.
- `uv run --python 3.10 python main.py --all --dataset_name foo` — should fail with a mutual exclusivity error.
- `uv run --python 3.10 python main.py --all --use_optuna` — should fail with an error.
- `uv run --python 3.10 python main.py --dataset_name vehicle_00nan` — should work unchanged.
- `uv run --python 3.10 python main.py` — should fail asking for `--all` or `--dataset_name`.
