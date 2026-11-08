"""E161's diagnostic, run after the read (mimic/09): why ReMasker trails mean fill on the MIMIC tables. One fold.

    uv run --python 3.11 python -B scripts/experiments/remasker_mimic_diagnostic.py <variant> <seed> <out.txt>

Fold 1 only. Per numerical column, rmse over the training fold's standard deviation, for ReMasker (paper rule, 600 and
60 epochs) and mean fill, on the fold's masked test cells and on 20% of the training fold's observed cells hidden after
the fit (the training error tells under- from over-fitting). Explanatory, not a test. Nothing in the store.
Run from the checkout root with ``TRIDENT_DATASETS_ROOT`` set.
"""

from __future__ import annotations

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
from src.training.imputation_baselines import BaselineImputer, MeanModeImputer  # noqa: E402
from src.training.remasker import fit_remasker_imputers  # noqa: E402


def _per_column(fills: Mapping[str, pd.DataFrame], frame: pd.DataFrame, hidden: NDArray[np.bool_],
                numerical: Sequence[str], sd: pd.Series) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for c in numerical:
        mask = hidden[:, frame.columns.get_loc(c)] & frame[c].notna().to_numpy()
        truth = frame[c].to_numpy(float)[mask]
        rows[c] = {"n": float(mask.sum())} | {
            k: float(np.sqrt(np.mean((f[c].to_numpy(float)[mask] - truth) ** 2)) / sd[c]) for k, f in fills.items()}
    return pd.DataFrame(rows).T


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    variant, seed, out = argv[0], int(argv[1]), Path(argv[2])
    warnings.filterwarnings("ignore")
    lines: list[str] = []

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)
        out.write_text("\n".join(lines) + "\n")

    def scorer(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> FoldScorer:
        sd = train[list(numerical)].std()
        models: dict[str, BaselineImputer] = {}
        for epochs in (600, 60):
            started = time.time()
            models[f"rm{epochs}"] = fit_remasker_imputers(train, numerical, (), epochs=epochs, remask="paper")[0]
            log(f"fit {epochs} epochs: {time.time() - started:.0f} s")
        mean = MeanModeImputer(numerical, categorical)
        mean.fit(train)
        models["mean"] = mean
        hide_train = (np.random.default_rng(0).random(train.shape) < 0.2) & train.notna().to_numpy()
        fitted = _per_column({k: m.impute(train, hide_train) for k, m in models.items()}, train, hide_train, numerical, sd)
        log("## training cells hidden after fit (rmse / training sd)\n" + fitted.round(3).to_string())

        def score(test: pd.DataFrame, cells: pd.DataFrame, naive: Mapping[str, float | str]) -> Mapping[str, float]:
            hidden = np.zeros(test.shape, dtype=bool)
            hidden[cells["row"].to_numpy(int), test.columns.get_indexer(pd.Index(cells["column"]))] = True
            table = _per_column({k: m.impute(test, hidden) for k, m in models.items()}, test, hidden, numerical, sd)
            log("## test masked cells (rmse / training sd)\n" + table.round(3).to_string())
            w = table["n"] / table["n"].sum()
            log("cell-weighted mean: " + ", ".join(f"{k} {(table[k] * w).sum():.3f}" for k in models))
            return {}

        return score

    recompute_folds(variant, seed, 5, 0.2, [], done={k: {} for k in (2, 3, 4, 5)}, fold_scorer=scorer)
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
