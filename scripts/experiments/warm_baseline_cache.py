"""Fill the baseline cache for cells a study will run, before its GPU pass (plan 2026-10-17, §5).

    uv run --python 3.11 python scripts/experiments/warm_baseline_cache.py <variant> <seed> [<seed> ...]

The baselines never depend on the model (ADR 0007 decision 10), so the cells' every family can be
scored on the CPU ahead of time. A later live run of the same (variant, seed, folds, rate) finds
them in `results/baseline_cache/`, under the very keys it would write: ADR 0016 checked that the
backfill's recomputed cells hit the live runs' entries. Five folds and EVAL_MASK_RATE 0.2 with no
extra rates, as a default run scores. Nothing is written to the store.

The missForest and LightGBM thread counts are capped at 4, so the pass can run beside the GPU
queue (their fills are the same at any thread count, ADR 0017 decision 6). Run from the checkout
root, with ``OMP_NUM_THREADS=4``.
"""

from __future__ import annotations

import sys
import time
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
from src.training.types import DEFAULT_BASELINE_CACHE_DIR  # noqa: E402

THREADS = 4
FOLDS = 5
RATE = 0.2


def _scorer(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> FoldScorer:
    scorer = BaselineScorer(train, numerical, categorical, DEFAULT_BASELINE_CACHE_DIR)

    def score(test: pd.DataFrame, cells: pd.DataFrame, naive: Mapping[str, float | str]) -> Mapping[str, float]:
        return {key: float(value) for key, value in scorer.score(test, cells, naive).metrics.items()}

    return score


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.exit(__doc__)
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    variant, seeds = argv[0], [int(seed) for seed in argv[1:]]
    for seed in seeds:
        started = time.time()
        folds = recompute_folds(variant, seed, FOLDS, RATE, [], fold_scorer=_scorer)
        mean_mode = [value for fold in folds.values() for key, value in fold.items() if key.endswith("/baseline/mean_mode/impute_score")]
        assert all(abs(value - 1.0) < 1e-9 for value in mean_mode), f"{variant} s{seed}: mean/mode is not 1"
        print(f"WARM {variant} s{seed}: {len(folds)} folds, {len(mean_mode)} populations, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
