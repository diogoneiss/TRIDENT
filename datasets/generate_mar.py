"""Missing-at-random (MAR) variants of the complete tables, with a matched MCAR control (T30, E63).

    uv run --python 3.11 python datasets/generate_mar.py [--bases biodeg kc2 vehicle kr-vs-kp] [--pct 0.6]

For each base table, reads ``processed_datasets/<base>/<base>_00nan.csv`` and writes two variants
that differ only in the mechanism:

- ``<base>_mar_<pp>nan.csv``: a cell's chance of being hidden depends on *other*, observed columns;
- ``<base>_marctl_<pp>nan.csv``: the matched MCAR control, the same columns and counts, drawn
  uniformly.

How:

- **Drivers:** ``round(0.3 * features)`` feature columns, at least one, are chosen at random and
  never lose a cell, in both variants. A numerical driver enters standardised on the complete table;
  a categorical one as the indicator of its mode, standardised, so all-categorical tables can enter.
- **Every other feature column** loses exactly ``round(n * pct)`` cells, the count
  ``generate_splits.py`` takes for its MCAR sibling, drawn without replacement.
  - **MAR:** with weights ``sigmoid(SLOPE * s)``, where ``s`` is the column's own random
    combination of the drivers (normal weights), standardised. Weighted sampling at a fixed count
    is not exactly a logistic mechanism, and the ticket says so.
  - **Control:** with equal weights.
- So a ``_60nan`` variant hides 60% of each non-driver column, about 42% of all feature cells, and
  its control hides the same cells' count in the same columns.

**The checks,** printed and enforced:

- the drivers are fully observed and every other column has exactly its count, in both variants;
- the manipulation: in the MAR variant, the mean over non-driver columns of the standardised
  difference in ``s`` between hidden and observed rows is at least ``MIN_SHIFT``; in the control
  its absolute value is below ``MAX_CONTROL_SHIFT``.

The label column is never touched, rows keep their order, and every observed cell equals the
complete table's, so the training code finds the induced cells' truth in ``<base>_00nan``. An
existing file is rewritten only if identical. The data directory is gitignored; this script is
the record.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "processed_datasets"
CATEGORICAL = Path(__file__).resolve().parent / "categorical_columns"
TARGET = "class"
SEED = 3060
DRIVER_SHARE = 0.3
SLOPE = 1.5
MIN_SHIFT = 0.3
MAX_CONTROL_SHIFT = 0.1


def _categorical(base: str) -> set[str]:
    """The declared categorical columns, as ``src/training/data.py`` reads them: comma-separated."""
    path = CATEGORICAL / f"{base}.txt"
    content = path.read_text().strip() if path.exists() else ""
    return {c.strip() for c in content.replace("\n", ",").split(",") if c.strip()}


def _standardised(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


def make(complete: pd.DataFrame, categorical: set[str], pct: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[float], list[float]]:
    """The MAR variant, its control, the drivers, and each hidden column's shift in both."""
    rng = np.random.default_rng(SEED)
    features = [c for c in complete.columns if c != TARGET]
    if complete[features].isna().any().any():
        raise ValueError("the complete table has gaps")
    n_drivers = max(1, round(DRIVER_SHARE * len(features)))
    drivers = sorted(rng.choice(features, size=n_drivers, replace=False).tolist(), key=features.index)
    z = np.column_stack([
        _standardised((complete[d].astype(str) == complete[d].astype(str).mode().iloc[0]).to_numpy(dtype=float))
        if d in categorical else _standardised(complete[d].to_numpy(dtype=float))
        for d in drivers
    ])
    n = len(complete)
    count = int(round(n * pct))
    mar, control = complete.copy(), complete.copy()
    shifts_mar: list[float] = []
    shifts_control: list[float] = []
    for column in features:
        if column in drivers:
            continue
        score = _standardised(z @ rng.normal(size=len(drivers)))
        weights = 1.0 / (1.0 + np.exp(-SLOPE * score))
        hidden = rng.choice(n, size=count, replace=False, p=weights / weights.sum())
        uniform = rng.choice(n, size=count, replace=False)
        mar.loc[mar.index[hidden], column] = np.nan
        control.loc[control.index[uniform], column] = np.nan
        for rows, out in ((hidden, shifts_mar), (uniform, shifts_control)):
            mask = np.zeros(n, dtype=bool)
            mask[rows] = True
            out.append(float(score[mask].mean() - score[~mask].mean()))
    return mar, control, drivers, shifts_mar, shifts_control


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.shape == b.shape and a.isna().equals(b.isna()) and bool(
        a.astype(str).where(a.notna(), "").equals(b.astype(str).where(b.notna(), ""))
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", default=["biodeg", "kc2", "vehicle", "kr-vs-kp"])
    parser.add_argument("--pct", type=float, default=0.6)
    options = parser.parse_args(argv)
    pp = round(options.pct * 100)
    for base in options.bases:
        complete = pd.read_csv(ROOT / base / f"{base}_00nan.csv")
        mar, control, drivers, shift_mar, shift_control = make(complete, _categorical(base), options.pct)
        count = int(round(len(complete) * options.pct))
        for name, frame in (("mar", mar), ("marctl", control)):
            gaps = frame.drop(columns=[TARGET]).isna().sum()
            assert (gaps[drivers] == 0).all(), f"{base} {name}: a driver lost cells"
            assert (gaps.drop(index=drivers) == count).all(), f"{base} {name}: a column misses its count"
            observed = frame.notna().to_numpy()
            assert (frame.to_numpy()[observed] == complete.to_numpy()[observed]).all(), f"{base} {name}: an observed cell moved"
        mean_mar, mean_control = float(np.mean(shift_mar)), float(np.mean(shift_control))
        if mean_mar < MIN_SHIFT or abs(mean_control) >= MAX_CONTROL_SHIFT:
            sys.exit(f"{base}: the manipulation check failed (MAR shift {mean_mar:.3f}, control {mean_control:.3f})")
        for name, frame in (("mar", mar), ("marctl", control)):
            path = ROOT / base / f"{base}_{name}_{pp:02d}nan.csv"
            if path.exists() and not _same(pd.read_csv(path), pd.read_csv(io_roundtrip(frame))):
                sys.exit(f"{path} exists and differs; refusing to overwrite")
            frame.to_csv(path, index=False)
        share = mar.drop(columns=[TARGET]).isna().to_numpy().mean()
        print(f"{base}: {len(drivers)} drivers of {complete.shape[1] - 1} ({', '.join(drivers)}); {count} gaps in each other column; "
              f"{share:.3f} of feature cells missing; shift in s, hidden minus observed: MAR {mean_mar:+.3f} "
              f"(min {min(shift_mar):+.3f}), control {mean_control:+.3f}")
    return 0


def io_roundtrip(frame: pd.DataFrame) -> "object":
    import io

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
