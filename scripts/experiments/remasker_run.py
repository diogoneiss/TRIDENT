"""E157's offline pass: ReMasker on one variant's folds (imputation-protocol/55).

    uv run --python 3.11 python -B scripts/experiments/remasker_run.py <out_dir> <variant> <seed> [<seed> ...]

For each seed, the variant's five folds are rebuilt exactly as a run built them
(``recompute_folds``: EVAL_MASK_RATE 0.2, no extra rates). ReMasker (``src/training/remasker.py``, the
paper's settings) is fitted on each training fold, on the GPU when there is one, and scored on the
fold's two populations by ``score_baselines``, the same cells and denominators as every baseline.

One CSV per (variant, seed), ``<out_dir>/<variant>_s<seed>.csv``, one row per fold, keys as in a
run's ``raw_fold_metrics.csv`` (``impute/<population>/baseline/remasker/<metric>``) plus the fit's
seconds. A finished file is skipped, so a relaunch resumes. Nothing is written to the store.
Run from the checkout root.

``REMASKER_REMASK=paper`` in the environment selects the paper's re-mask count (E158); the default is the reference
code's (E157). Each row records it as ``baseline/remasker/remask_paper`` (1 or 0).

A table with categorical columns (E161's MIMIC tables, read from ``$TRIDENT_DATASETS_ROOT``) is fitted on its
numerical columns alone, its categorical cells filled with the training mode (``baseline/remasker/categorical_mode``
1); the comparison that matters there is on the numerical cells.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.backfill_baseline_imputers import FoldScorer, recompute_folds  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, MeanModeImputer, score_baselines  # noqa: E402
from src.training.remasker import fit_remasker_imputers  # noqa: E402

FOLDS = 5
RATE = 0.2
REMASK = os.environ.get("REMASKER_REMASK", "reference")


class _WithModes:
    """ReMasker on the numerical columns; a categorical column (the MIMIC tables, E161) gets the training mode,
    since ReMasker has no categorical path. Without this the hidden categorical cells would keep their true value."""

    name = "remasker"

    def __init__(self, remasker: BaselineImputer, modes: MeanModeImputer, categorical: Sequence[str]) -> None:
        self._remasker, self._modes, self._categorical = remasker, modes, list(categorical)

    def fit(self, frame: pd.DataFrame) -> None:
        self._remasker.fit(frame)
        self._modes.fit(frame)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        out = self._remasker.impute(frame, hidden)
        out[self._categorical] = self._modes.impute(frame, hidden)[self._categorical]
        return out


def _scorer(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> FoldScorer:
    started = time.time()
    imputers = fit_remasker_imputers(train, numerical, (), remask=REMASK)
    if categorical:
        modes = MeanModeImputer(numerical, categorical)
        modes.fit(train)
        imputers = [_WithModes(imputers[0], modes, categorical)]
    seconds = time.time() - started

    def score(test: pd.DataFrame, cells: pd.DataFrame, naive: Mapping[str, float | str]) -> Mapping[str, float]:
        metrics = {key: float(value) for key, value in score_baselines(imputers, test, cells, naive).metrics.items()}
        return metrics | {"baseline/remasker/fit_seconds": seconds, "baseline/remasker/remask_paper": float(REMASK == "paper"),
                          "baseline/remasker/categorical_mode": float(bool(categorical))}

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
