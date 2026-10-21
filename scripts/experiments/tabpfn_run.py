"""E55's offline pass: TabPFN v2 on one variant's folds (imputation-protocol/10).

    uv run --python 3.11 python -B scripts/experiments/tabpfn_run.py <out_dir> <variant> <seed> [<seed> ...]

For each seed, the variant's five folds are rebuilt exactly as a run built them
(``recompute_folds``: EVAL_MASK_RATE 0.2, no extra rates). TabPFN v2 (``src/training/tabpfn_baseline.py``,
its v2 checkpoints, column-wise) is fitted on each training fold, on the GPU, and scored on the
fold's two populations by ``score_baselines``, the same cells and denominators as every baseline.
Needs the ``tabpfn`` extra: ``uv run --extra tabpfn``.

One CSV per (variant, seed), ``<out_dir>/<variant>_s<seed>.csv``, one row per fold, keys as in a
run's ``raw_fold_metrics.csv`` (``impute/<population>/baseline/tabpfn/<metric>``) plus the fit's
seconds. A finished file is skipped, so a relaunch resumes. Nothing is written to the store.
Run from the checkout root.
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.backfill_baseline_imputers import FoldScorer, recompute_folds  # noqa: E402
from src.training.imputation_baselines import score_baselines  # noqa: E402
from src.training.tabpfn_baseline import fit_tabpfn_imputers  # noqa: E402

FOLDS = 5
RATE = 0.2


def _scorer(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> FoldScorer:
    started = time.time()
    imputers = fit_tabpfn_imputers(train, numerical, categorical)
    seconds = time.time() - started

    def score(test: pd.DataFrame, cells: pd.DataFrame, naive: Mapping[str, float | str]) -> Mapping[str, float]:
        metrics = {key: float(value) for key, value in score_baselines(imputers, test, cells, naive).metrics.items()}
        return metrics | {"baseline/tabpfn/fit_seconds": seconds}

    return score


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.exit(__doc__)
    out_dir, variant, seeds = Path(argv[0]), argv[1], [int(seed) for seed in argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    for seed in seeds:
        path = out_dir / f"{variant}_s{seed}.csv"
        if path.exists():
            print(f"SKIP {path.name}: done", flush=True)
            continue
        started = time.time()
        folds = recompute_folds(variant, seed, FOLDS, RATE, [], fold_scorer=_scorer)
        columns = sorted({key for metrics in folds.values() for key in metrics})
        partial = path.with_suffix(".partial")
        with partial.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", "seed", "fold", *columns])
            for fold, metrics in sorted(folds.items()):
                writer.writerow([variant, seed, fold, *(metrics.get(column, "") for column in columns)])
        partial.rename(path)
        print(f"DONE {variant} s{seed}: {len(folds)} folds, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
