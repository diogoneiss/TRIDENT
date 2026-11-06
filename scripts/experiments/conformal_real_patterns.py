"""E153 (docs/tickets/mimic/07-conformal-real-patterns.md): ADR 0030 decision 8's binned interval on E149's ledgers,
real gap patterns transplanted onto complete MIMIC-IV rows. Offline, nothing written.

    uv run --python 3.11 python -B scripts/experiments/conformal_real_patterns.py

E151's arms C and MC (``conformal_row_gaps.fold_arms``, unchanged) on the ``induced`` numerical cells; Q1 only, since
E149 wrote no ``spread``. E151's synthetic self-check runs first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import conformal_row_gaps as e151  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-11-06-mimic-e149")
TABLES = ("mimic-cardio-donor", "mimic-sofa-donor")
KINDS = ("tmatch", "trand", "mcar")
SEEDS = (42, 7, 13)


def cells(variant: str, seed: int, fold: int) -> pd.DataFrame:
    """A fold's induced numerical cells with error, b (the row's induced cells) and a unit spread E151's arms ignore."""
    found = sorted((STATE / "results" / f"{variant}_s{seed}_D" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    induced = frame[frame["population"] == "induced"]
    out = induced[induced["kind"] == "numerical"][["row", "column", "actual", "imputed"]].copy()
    out["error"] = (out["imputed"].astype(float) - out["actual"].astype(float)).abs()
    out["b"] = out["row"].map(induced.groupby("row").size()).astype(float)
    return out.assign(spread=1.0).reset_index(drop=True)


def read(variant: str) -> tuple[str, str]:
    per_seed = []
    for seed in SEEDS:
        folds = {k: cells(variant, seed, k) for k in range(1, 6)}
        per_seed.append([e151.fold_measures(e151.fold_arms(folds[k], pd.concat([f for j, f in folds.items() if j != k], ignore_index=True)))
                         for k in range(1, 6)])
    arrays = {key: [np.array([f[key] for f in folds]) for folds in per_seed] for key in per_seed[0][0]}
    q1 = common.verdict([mc - c for mc, c in zip(arrays["u_MC"], arrays["u_C"])], sesoi=e151.SESOI)
    width = common.verdict([(mc - c) / c for mc, c in zip(arrays["width_MC"], arrays["width_C"])], sesoi=e151.SESOI)
    mean = {key: float(np.concatenate(v).mean()) for key, v in arrays.items()}
    coverage = ", ".join(f"{arm} {mean[f'cover_{arm}']:.3f} (low bin {mean[f'low_{arm}']:.3f}, top {mean[f'top_{arm}']:.3f})" for arm in ("C", "MC"))
    line = (f"- `{variant}`: Q1 u(MC) − u(C) {q1.mean:+.4f} [{q1.low:+.4f}, {q1.high:+.4f}] {q1.call}, u(C) {mean['u_C']:.3f}, "
            f"u(MC) {mean['u_MC']:.3f}; width (MC − C) / C {width.mean:+.3f} [{width.low:+.3f}, {width.high:+.3f}] {width.call}; "
            f"coverage {coverage}; fallback {mean['fallback']:.2f}")
    return q1.call, line


def reading(calls: dict[str, str]) -> str:
    better = sum(c == "better" for c in calls.values())
    worse = sum(c == "worse" for c in calls.values())
    return "supported" if better == 2 else "mixed" if better == 1 and worse == 0 else "refuted"


def main() -> None:
    e151.check()
    assert reading({"a": "better", "b": "better"}) == "supported" and reading({"a": "better", "b": "none"}) == "mixed"
    assert reading({"a": "better", "b": "worse"}) == "refuted" and reading({"a": "none", "b": "equivalent"}) == "refuted"
    calls: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}
    for kind in KINDS:
        print(f"\n## {kind}{' (primary)' if kind == 'tmatch' else ' (descriptive)'}\n")
        for table in TABLES:
            call, line = read(f"{table}_{kind}_20nan")
            calls[kind][table] = call
            print(line)
        print(f"- Q1: {common.tally(calls[kind].values())}")
    print(f"\nPrimary (E153, Q1 on tmatch): {calls['tmatch']} -> {reading(calls['tmatch'])} ({common.RULE})")


if __name__ == "__main__":
    if sys.argv[1:]:
        sys.exit(__doc__)
    main()
