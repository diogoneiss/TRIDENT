"""A synthetic table whose column dependencies are known, for E78's truth test (T38).

    uv run --python 3.11 python datasets/generate_synthblock.py

Writes ``processed_datasets/synthblock/synthblock_00nan.csv``, ``synthblock_40nan.csv``,
``synthblock_60nan.csv`` (E79) and ``categorical_columns/synthblock.txt``. 3,000 rows; three blocks of four columns, each block driven
by its own standard-normal latent ``z``, and three columns of pure noise:

- block a: ``a1 = z + e``, ``a2 = z**2 - 1 + e`` (Pearson ~0 with a1), ``a3 = sin(2z) + e``,
  ``a4 = |z| + e``;
- block b: ``b1 = z + e``, ``b2 = exp(z / 2) + e``, ``b3`` the tercile of ``z + e`` (categorical),
  ``b4 = -z + 2e``;
- block c: ``c1 = z + e``, ``c2 = tanh(2z) + e``, ``c3`` the sign of ``z + e`` (categorical),
  ``c4 = z**3 / 3 + e``;
- noise: ``n1``, ``n2`` standard normal, ``n3`` uniform over four categories.

``e`` is fresh N(0, 0.3^2) noise per column. Blocks are independent of each other and of the noise,
so a column's informative sources are exactly the other three of its block. ``class`` is a fair coin,
independent of everything, because the pipeline needs a label to stratify folds on. The 40nan and
60nan variants are ``generate_splits.inject_nans`` with seed ``42 + pp``, the shipped MCAR mechanism. The data
directory is gitignored, so this script is the record; a rerun rewrites identical files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_splits import RANDOM_SEED, TARGET_COLUMN, inject_nans  # noqa: E402

BASE = "synthblock"
ROWS = 3000
SEED = 7800
NOISE = 0.3
CATEGORICAL = ("b3", "c3", "n3")
BLOCKS = {"a": ("a1", "a2", "a3", "a4"), "b": ("b1", "b2", "b3", "b4"), "c": ("c1", "c2", "c3", "c4")}


def complete_table() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    z = {block: rng.standard_normal(ROWS) for block in BLOCKS}

    def e() -> np.ndarray:
        return NOISE * rng.standard_normal(ROWS)

    za, zb, zc = z["a"], z["b"], z["c"]
    columns = {
        "a1": za + e(), "a2": za**2 - 1 + e(), "a3": np.sin(2 * za) + e(), "a4": np.abs(za) + e(),
        "b1": zb + e(), "b2": np.exp(zb / 2) + e(),
        "b3": np.array(["low", "mid", "high"])[np.digitize(zb + e(), [-0.43, 0.43])],
        "b4": -zb + 2 * e(),
        "c1": zc + e(), "c2": np.tanh(2 * zc) + e(), "c3": np.where(zc + e() > 0, "pos", "neg"), "c4": zc**3 / 3 + e(),
        "n1": rng.standard_normal(ROWS), "n2": rng.standard_normal(ROWS),
        "n3": rng.choice(np.array(["w", "x", "y", "z"]), ROWS),
        TARGET_COLUMN: rng.choice(np.array(["yes", "no"]), ROWS),
    }
    frame = pd.DataFrame(columns)
    numerical = [c for c in frame.columns if c not in CATEGORICAL and c != TARGET_COLUMN]
    frame[numerical] = frame[numerical].round(6)
    return frame


def main() -> None:
    out = HERE / "processed_datasets" / BASE
    out.mkdir(parents=True, exist_ok=True)
    complete = complete_table()
    complete.to_csv(out / f"{BASE}_00nan.csv", index=False)
    for pp in (40, 60):
        gapped = inject_nans(complete, pp / 100, TARGET_COLUMN, seed=RANDOM_SEED + pp)
        assert gapped is not None
        gapped.to_csv(out / f"{BASE}_{pp}nan.csv", index=False)
    (HERE / "categorical_columns" / f"{BASE}.txt").write_text(",".join(CATEGORICAL))
    shares = gapped.drop(columns=[TARGET_COLUMN]).isna().mean()
    print(f"{BASE}: {complete.shape}, missing share per feature {shares.min():.3f}-{shares.max():.3f}")
    print(complete.drop(columns=[*CATEGORICAL, TARGET_COLUMN]).corr(method="pearson").round(2).to_string())


if __name__ == "__main__":
    main()
