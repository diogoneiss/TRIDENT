"""E41's offline pass: every baseline, the linear arms included, on one variant's folds (imputation-protocol/01).

    uv run --python 3.11 python -B scripts/experiments/linear_baselines_run.py <out_dir> <variant> <seed> [<seed> ...]

For each seed, the variant's five folds are rebuilt exactly as a run built them
(``recompute_folds``: EVAL_MASK_RATE 0.2, no extra rates), and each fold's two populations are
scored by:

- every live family (mean_mode, knn5, knn10, hgb, lgbm, missForest) through the baseline cache,
  where the runs this study pairs with left their entries on gorgona8;
- the linear arms of ``src/training/linear_baselines.py`` (condexp, softimpute, ice), fitted here;
- on the small tables (credit-g, vehicle, biodeg, kc2), ``missforest_seq``, missForest with the
  sequential fill.

One CSV per (variant, seed), ``<out_dir>/<variant>_s<seed>.csv``, one row per fold with every
metric under the keys a run's ``raw_fold_metrics.csv`` uses, plus each arm's diagnostics
(SoftImpute's rank and kept energy, ICE's rounds). A finished file is skipped, so a relaunch
resumes. Nothing is written to the store. The missForest and LightGBM threads are capped at 4.
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
from src.training import lightgbm_baseline, missforest  # noqa: E402
from src.training.baseline_cache import BaselineScorer  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, score_baselines  # noqa: E402
from src.training.linear_baselines import (  # noqa: E402
    IceImputer,
    SequentialRoundsImputer,
    SoftImputeImputer,
    fit_linear_imputers,
)
from src.training.types import DEFAULT_BASELINE_CACHE_DIR  # noqa: E402

THREADS = 4
FOLDS = 5
RATE = 0.2
SMALL_TABLES = ("credit-g", "vehicle", "biodeg", "kc2")


def _factory(variant: str):  # type: ignore[no-untyped-def]
    def scorer(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> FoldScorer:
        cached = BaselineScorer(train, numerical, categorical, DEFAULT_BASELINE_CACHE_DIR)
        arms: list[BaselineImputer] = fit_linear_imputers(train, numerical, categorical)
        if variant.split("_")[0] in SMALL_TABLES:
            sequential = SequentialRoundsImputer(numerical, categorical)
            sequential.fit(train)
            arms.append(sequential)
        diagnostics: dict[str, float] = {}
        for arm in arms:
            if isinstance(arm, SoftImputeImputer):
                diagnostics |= {"baseline/softimpute/rank": float(arm.rank), "baseline/softimpute/energy": arm.energy}
            if isinstance(arm, (IceImputer, SequentialRoundsImputer)):
                diagnostics[f"baseline/{arm.name}/rounds"] = float(arm.rounds)

        def score(test: pd.DataFrame, cells: pd.DataFrame, naive: Mapping[str, float | str]) -> Mapping[str, float]:
            metrics = {key: float(value) for key, value in cached.score(test, cells, naive).metrics.items()}
            metrics |= {key: float(value) for key, value in score_baselines(arms, test, cells, naive).metrics.items()}
            return metrics | diagnostics

        return score

    return scorer


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.exit(__doc__)
    out_dir, variant, seeds = Path(argv[0]), argv[1], [int(seed) for seed in argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    warnings.filterwarnings("ignore", category=UserWarning)
    for seed in seeds:
        path = out_dir / f"{variant}_s{seed}.csv"
        if path.exists():
            print(f"SKIP {path.name}: done", flush=True)
            continue
        started = time.time()
        folds = recompute_folds(variant, seed, FOLDS, RATE, [], fold_scorer=_factory(variant))
        columns = sorted({key for metrics in folds.values() for key in metrics})
        partial = path.with_suffix(".partial")
        with partial.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", "seed", "fold", *columns])
            for fold, metrics in sorted(folds.items()):
                writer.writerow([variant, seed, fold, *(metrics.get(column, "") for column in columns)])
        partial.rename(path)
        print(f"DONE {variant} s{seed}: {len(folds)} folds, {len(columns)} columns, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
