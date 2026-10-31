"""The MAR family: E63's mechanism on more tables and rates, with a slope dial (E64).

    uv run --python 3.11 python datasets/generate_mar_family.py

The mechanism is ``generate_mar.py``'s, with a new seed, so drivers and draws are independent of
E63's smoke, and with separate random streams:

- the drivers and each column's driver combination come from one stream;
- the MAR draws from a second, one per column;
- the uniform control from a third.

So the dial's slopes share one control and one set of driver scores, and differ only in how
strongly the scores tilt the draw. For each base and rate it writes ``<base>_marf_<pp>nan`` (slope
1.5) and its control ``<base>_marfctl_<pp>nan``. On the dial bases, at the dial rate,
``<base>_marf<slope>_<pp>nan`` is added for each dial slope: ``0p75`` for 0.75, ``3`` for 3.0.

**The checks** are ``generate_mar.py``'s:

- drivers fully observed, every other column at exactly round(n * pct) gaps, observed cells
  unchanged;
- at slope 1.5, a mean shift in the driver score of at least ``MIN_SHIFT`` between hidden and
  observed rows;
- in the control, a shift below ``MAX_CONTROL_SHIFT`` in absolute value.

The dial's shifts are printed, not enforced. An existing file is rewritten only if identical. The
data directory is gitignored; this script is the record.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_mar import DRIVER_SHARE, MAX_CONTROL_SHIFT, MIN_SHIFT, TARGET, _categorical, _standardised  # noqa: E402

ROOT = HERE / "processed_datasets"
SEED = 4060
SLOPE = 1.5
BASES = ("biodeg", "kc2", "vehicle", "kr-vs-kp", "spambase", "credit-g")
RATES = (0.2, 0.4, 0.6)  # 0.4 added for E105; each rate has its own streams, so 0.2 and 0.6 are unchanged
DIAL_BASES, DIAL_RATE, DIAL_SLOPES = ("biodeg",), 0.6, (0.75, 3.0)


def _slope_tag(slope: float) -> str:
    return "" if slope == SLOPE else f"{slope:g}".replace(".", "p")


def family(complete: pd.DataFrame, categorical: set[str], pct: float, slopes: tuple[float, ...]) -> tuple[dict[float, pd.DataFrame], pd.DataFrame, list[str], dict[float, list[float]], list[float]]:
    structure = np.random.default_rng([SEED, 0])
    features = [c for c in complete.columns if c != TARGET]
    if complete[features].isna().any().any():
        raise ValueError("the complete table has gaps")
    n_drivers = max(1, round(DRIVER_SHARE * len(features)))
    drivers = sorted(structure.choice(features, size=n_drivers, replace=False).tolist(), key=features.index)
    z = np.column_stack([
        _standardised((complete[d].astype(str) == complete[d].astype(str).mode().iloc[0]).to_numpy(dtype=float))
        if d in categorical else _standardised(complete[d].to_numpy(dtype=float))
        for d in drivers
    ])
    n = len(complete)
    count = int(round(n * pct))
    control_stream = np.random.default_rng([SEED, 2, round(pct * 100)])
    mar = {s: complete.copy() for s in slopes}
    control = complete.copy()
    shifts: dict[float, list[float]] = {s: [] for s in slopes}
    control_shifts: list[float] = []
    for index, column in enumerate(features):
        if column in drivers:
            continue
        score = _standardised(z @ structure.normal(size=len(drivers)))
        uniform = control_stream.choice(n, size=count, replace=False)
        control.loc[control.index[uniform], column] = np.nan
        control_shifts.append(_shift(score, uniform))
        for slope in slopes:
            weights = 1.0 / (1.0 + np.exp(-slope * score))
            draw = np.random.default_rng([SEED, 1, round(pct * 100), index])
            hidden = draw.choice(n, size=count, replace=False, p=weights / weights.sum())
            mar[slope].loc[mar[slope].index[hidden], column] = np.nan
            shifts[slope].append(_shift(score, hidden))
    return mar, control, drivers, shifts, control_shifts


def _shift(score: np.ndarray, rows: np.ndarray) -> float:
    mask = np.zeros(len(score), dtype=bool)
    mask[rows] = True
    return float(score[mask].mean() - score[~mask].mean())


def _roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.shape == b.shape and a.isna().equals(b.isna()) and bool(
        a.astype(str).where(a.notna(), "").equals(b.astype(str).where(b.notna(), ""))
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    if path.exists() and not _same(pd.read_csv(path), _roundtrip(frame)):
        sys.exit(f"{path} exists and differs; refusing to overwrite")
    frame.to_csv(path, index=False)


def main() -> int:
    for base in BASES:
        complete = pd.read_csv(ROOT / base / f"{base}_00nan.csv")
        for pct in RATES:
            pp = round(pct * 100)
            slopes = (SLOPE, *DIAL_SLOPES) if base in DIAL_BASES and pct == DIAL_RATE else (SLOPE,)
            mar, control, drivers, shifts, control_shifts = family(complete, _categorical(base), pct, slopes)
            count = int(round(len(complete) * pct))
            for frame in (*mar.values(), control):
                gaps = frame.drop(columns=[TARGET]).isna().sum()
                assert (gaps[drivers] == 0).all() and (gaps.drop(index=drivers) == count).all(), f"{base} {pp}: counts"
                observed = frame.notna().to_numpy()
                assert (frame.to_numpy()[observed] == complete.to_numpy()[observed]).all(), f"{base} {pp}: an observed cell moved"
            main_shift, ctl_shift = float(np.mean(shifts[SLOPE])), float(np.mean(control_shifts))
            if main_shift < MIN_SHIFT or abs(ctl_shift) >= MAX_CONTROL_SHIFT:
                sys.exit(f"{base} {pp}: the manipulation check failed (MAR {main_shift:.3f}, control {ctl_shift:.3f})")
            for slope, frame in mar.items():
                _write(frame, ROOT / base / f"{base}_marf{_slope_tag(slope)}_{pp:02d}nan.csv")
            _write(control, ROOT / base / f"{base}_marfctl_{pp:02d}nan.csv")
            dial = "; ".join(f"slope {s:g} {float(np.mean(v)):+.3f}" for s, v in shifts.items())
            share = control.drop(columns=[TARGET]).isna().to_numpy().mean()
            print(f"{base} {pp}nan: {len(drivers)} drivers of {complete.shape[1] - 1}; {count} gaps per other column, "
                  f"{share:.3f} of feature cells; shift MAR {dial}; control {ctl_shift:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
