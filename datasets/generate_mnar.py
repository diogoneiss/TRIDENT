"""Value-dependent (MNAR) variants of the complete tables, at the MCAR siblings' counts (E45).

    uv run --python 3.11 python datasets/generate_mnar.py [--bases biodeg kc2 vehicle] [--pct 0.6]

For each base table, reads ``processed_datasets/<base>/<base>_00nan.csv`` and writes
``processed_datasets/<base>/<base>_probit_<pp>nan.csv``. Every feature column loses exactly
``round(n * pct)`` cells, the count ``generate_splits.py`` takes from its MCAR sibling, drawn
without replacement with weights Phi(SLOPE * z - OFFSET), z being the column standardised on the
complete table: large values are hidden more often. Weighted sampling at a fixed count is not
exactly a probit mechanism (the inclusion probabilities are not Phi), and the ticket says so. The
label column is never touched, rows keep their order, and every observed cell equals the complete
table's, so the variant is row-aligned with its ``_00nan`` sibling and the training code finds the
induced cells' truth as it does for MCAR variants.

Only numerical tables for now (the smoke family): a categorical column has no order to tilt by, and
the script refuses a base with one. ``generate_splits.py`` is not touched; it rewrites its outputs
in place. The data directory is gitignored; this script is the record.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent / "processed_datasets"
CATEGORICAL = Path(__file__).resolve().parent / "categorical_columns"
TARGET = "class"
SLOPE = 0.5
OFFSET = 0.0
SEED = 1060


def tilted(frame: pd.DataFrame, pct: float, seed: int = SEED, slope: float = SLOPE, offset: float = OFFSET) -> pd.DataFrame:
    """The frame with round(n * pct) cells of every feature column hidden, larger values likelier."""
    stream = np.random.default_rng(seed)
    out = frame.copy()
    count = int(round(len(frame) * pct))
    for column in [c for c in frame.columns if c != TARGET]:
        values = frame[column].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"column '{column}' already has gaps; the complete table must be complete")
        spread = values.std()
        z = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
        weights = norm.cdf(slope * z - offset)
        hidden = stream.choice(len(values), size=count, replace=False, p=weights / weights.sum())
        out.loc[out.index[hidden], column] = np.nan
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", default=["biodeg", "kc2", "vehicle"])
    parser.add_argument("--pct", type=float, default=0.6)
    options = parser.parse_args(argv)
    for base in options.bases:
        if (CATEGORICAL / f"{base}.txt").exists() and (CATEGORICAL / f"{base}.txt").read_text().strip():
            sys.exit(f"{base} has categorical columns; the smoke generator handles numerical tables only")
        complete = pd.read_csv(ROOT / base / f"{base}_00nan.csv")
        variant = tilted(complete, options.pct)
        path = ROOT / base / f"{base}_probit_{round(options.pct * 100):02d}nan.csv"
        variant.to_csv(path, index=False)
        print(f"wrote {path} ({variant.drop(columns=[TARGET]).isna().to_numpy().mean():.3f} of feature cells missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
