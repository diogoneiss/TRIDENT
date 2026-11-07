"""C15's seedable missingness generators (docs/plans/2026-11-03-c15-mnar-prediction.md, Stage 0).

    uv run --python 3.11 python datasets/mnar_family.py outcome --bases biodeg kc2 spambase --pct 0.6 --odds 3 --seed 2060
    uv run --python 3.11 python datasets/mnar_family.py noise --bases biodeg kc2 spambase --pct 0.6 --odds 3 --seed 2060
    uv run --python 3.11 python datasets/mnar_family.py probit --bases biodeg --pct 0.6 --slope 0.5 --seed 3070

Every feature column loses exactly ``round(n * pct)`` cells, the MCAR sibling's count, drawn without
replacement with per-cell weights; each column draws from its own stream, ``SeedSequence([seed, j])``
for the column at position ``j``, so adding or dropping a column leaves the others' gaps unchanged.
The seed is in every output name, and a file that exists with other contents is never overwritten.
``generate_mnar.py`` (one stream, seed 1060) is left as it is, so E45's and E46's files still
reproduce.

- ``outcome`` (C15's F3): weights ``odds ** y``, y the label encoded as ``LabelEncoder`` does (classes
  sorted), so class 1 loses cells ``odds`` times as readily. Binary tables only. Writes
  ``<base>_outcome<odds>_s<seed>_<pp>nan``.
- ``noise`` (F5, the positive control): the MCAR file ``<base>_<pp>nan`` plus one N(0, 1) column,
  ``c15_noise``, whose ``round(n * pct)`` gaps follow ``odds ** y``; the other columns are the MCAR
  file's. Writes ``<base>_noise<odds>_s<seed>_<pp>nan``. Its gap is the column's only signal.
- ``sign`` (C15's F4): self-masking with column j's slope ``+0.5 * sign_j`` (``--direction plus``) or
  ``-0.5 * sign_j`` (``minus``), sign_j the sign of j's coefficient in an L2 logistic regression on the
  standardised complete table, so the hidden tail is the one that raises (plus) or lowers (minus) P(y = 1).
  Binary tables only. Writes ``<base>_sign<direction>_s<seed>_<pp>nan``.
- ``mcar`` (fresh MCAR siblings for confirmation): uniform weights, ``<base>_mcar_s<seed>_<pp>nan``.
- ``probit`` (fresh self-masking draws for confirmation): weights Phi(slope * z), z the column
  standardised on the complete table, as ``generate_mnar.py`` but seeded per column. Writes
  ``<base>_probit<slope>_s<seed>_<pp>nan``.

The label is never touched and rows keep their order. The data directory is gitignored; this script
is the record.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent / "processed_datasets"
TARGET = "class"
NOISE = "c15_noise"
Weights = Callable[[str, NDArray[np.float64]], NDArray[np.float64]]


def dial(value: float) -> str:
    """1.5 -> 1p5, 3 -> 3p0: a number spelt into a variant name."""
    text = f"{value:g}".replace(".", "p")
    return text if "p" in text else text + "p0"


def labels(frame: pd.DataFrame) -> NDArray[np.float64]:
    return np.asarray(LabelEncoder().fit_transform(frame[TARGET]), dtype=float)


def hide(frame: pd.DataFrame, columns: list[str], pct: float, seed: int, weights: Weights) -> pd.DataFrame:
    """``frame`` with round(n * pct) cells of each of ``columns`` hidden, drawn with ``weights``."""
    out = frame.copy()
    count = int(round(len(frame) * pct))
    for position, column in enumerate(frame.columns):
        if column not in columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"column '{column}' already has gaps")
        w = weights(column, values)
        stream = np.random.default_rng(np.random.SeedSequence([seed, position]))
        hidden = stream.choice(len(values), size=count, replace=False, p=w / w.sum())
        out.loc[out.index[hidden], column] = np.nan
    return out


def outcome(complete: pd.DataFrame, pct: float, odds: float, seed: int) -> pd.DataFrame:
    y = labels(complete)
    if len(np.unique(y)) != 2:
        raise ValueError("the outcome-dependent family needs a binary label")
    features = [c for c in complete.columns if c != TARGET]
    return hide(complete, features, pct, seed, lambda _c, _v: odds ** y)


def noise(mcar: pd.DataFrame, pct: float, odds: float, seed: int) -> pd.DataFrame:
    if NOISE in mcar.columns:
        raise ValueError(f"'{NOISE}' already exists")
    y = labels(mcar)
    if len(np.unique(y)) != 2:
        raise ValueError("the positive control needs a binary label")
    out = mcar.copy()
    out.insert(len(out.columns) - (1 if out.columns[-1] == TARGET else 0), NOISE,
               np.random.default_rng(np.random.SeedSequence([seed, 10**6])).standard_normal(len(out)))
    return hide(out, [NOISE], pct, seed, lambda _c, _v: odds ** y)


def probit(complete: pd.DataFrame, pct: float, slope: float, seed: int) -> pd.DataFrame:
    def weights(_column: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        spread = values.std()
        z = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
        return np.asarray(norm.cdf(slope * z), dtype=float)
    return hide(complete, [c for c in complete.columns if c != TARGET], pct, seed, weights)


def signs(complete: pd.DataFrame) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = labels(complete)
    if len(np.unique(y)) != 2:
        raise ValueError("the sign family needs a binary label")
    features = [c for c in complete.columns if c != TARGET]
    coef = LogisticRegression(max_iter=5000).fit(StandardScaler().fit_transform(complete[features]), y).coef_[0]
    return {c: float(np.sign(v)) or 1.0 for c, v in zip(features, coef)}


def sign(complete: pd.DataFrame, pct: float, direction: str, seed: int) -> pd.DataFrame:
    s = signs(complete)
    flip = 1.0 if direction == "plus" else -1.0

    def weights(column: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        spread = values.std()
        z = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
        return np.asarray(norm.cdf(flip * 0.5 * s[column] * z), dtype=float)
    return hide(complete, [c for c in complete.columns if c != TARGET], pct, seed, weights)


def mcar(complete: pd.DataFrame, pct: float, seed: int) -> pd.DataFrame:
    return hide(complete, [c for c in complete.columns if c != TARGET], pct, seed, lambda _c, v: np.ones(len(v)))


def write(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame``; an existing file with the same contents is kept, one with others is refused."""
    text = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() == text:
            print(f"kept {path} (unchanged)")
            return
        sys.exit(f"refusing to overwrite {path}: it exists with other contents")
    path.write_text(text)
    print(f"wrote {path}")


def report(frame: pd.DataFrame, reference: pd.DataFrame | None, name: str) -> None:
    """The integrity checks: per-column counts, the hidden-minus-observed shift, per-class rates."""
    features = [c for c in frame.columns if c != TARGET]
    gaps = frame[features].isna()
    y = labels(frame)
    rates = {int(k): float(gaps[y == k].to_numpy().mean()) for k in np.unique(y)}
    line = f"{name}: {gaps.to_numpy().mean():.3f} of feature cells missing; by class {rates}"
    if reference is not None:
        shifts = []
        for column in features:
            if column not in reference.columns or not gaps[column].any():
                continue
            truth = reference[column].to_numpy(dtype=float)
            spread = truth.std()
            if spread > 0:
                mask = gaps[column].to_numpy()
                shifts.append((truth[mask].mean() - truth[~mask].mean()) / spread)
        line += f"; hidden − observed shift {np.mean(shifts):+.3f}" if shifts else ""
    print(line)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["outcome", "noise", "probit", "mcar", "sign"])
    parser.add_argument("--direction", choices=["plus", "minus"], default="plus")
    parser.add_argument("--bases", nargs="+", required=True)
    parser.add_argument("--pct", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--odds", type=float, default=3.0)
    parser.add_argument("--slope", type=float, default=0.5)
    parser.add_argument("--root", type=Path, default=ROOT, help="the tables' root (default processed_datasets)")
    o = parser.parse_args(argv)
    level = f"{round(o.pct * 100):02d}nan"
    root = o.root
    for base in o.bases:
        complete = pd.read_csv(root / base / f"{base}_00nan.csv")
        if o.family == "outcome":
            frame, name = outcome(complete, o.pct, o.odds, o.seed), f"{base}_outcome{dial(o.odds)}_s{o.seed}_{level}"
        elif o.family == "noise":
            frame = noise(pd.read_csv(root / base / f"{base}_{level}.csv"), o.pct, o.odds, o.seed)
            name = f"{base}_noise{dial(o.odds)}_s{o.seed}_{level}"
        elif o.family == "sign":
            frame, name = sign(complete, o.pct, o.direction, o.seed), f"{base}_sign{o.direction}_s{o.seed}_{level}"
        elif o.family == "mcar":
            frame, name = mcar(complete, o.pct, o.seed), f"{base}_mcar_s{o.seed}_{level}"
        else:
            frame, name = probit(complete, o.pct, o.slope, o.seed), f"{base}_probit{dial(o.slope)}_s{o.seed}_{level}"
        write(frame, root / base / f"{name}.csv")
        report(frame, complete, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
