"""E154 (docs/tickets/mimic/08-conformal-natural-gaps.md): ADR 0030 decision 8's binned interval on E148's ledgers,
observed cells hidden on MIMIC-IV's natural tables. Offline, nothing written.

    uv run --python 3.11 python -B scripts/experiments/conformal_natural_gaps.py

E151's arms C and MC (``conformal_row_gaps.fold_arms``, unchanged) on the ``masked`` numerical cells, b the row's
natural gaps plus its masked cells; Q1 only, read by E153's two-table rule. E151's synthetic self-check runs first.
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

from scripts.experiments import conformal_real_patterns as e153  # noqa: E402
from scripts.experiments import conformal_row_gaps as e151  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-11-06-mimic-e148")
DATA = Path("/scratch/diogoneiss/trident-data")
TABLES = ("mimic-cardio_natural", "mimic-sofa_natural")
SEEDS = (42, 7, 13)


def cells(variant: str, seed: int, fold: int, natural: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """A fold's masked numerical cells with error and b, the row's natural gaps plus its masked cells. Asserts the row
    mapping: each cell's original-unit truth is the table's value at the mapped row."""
    found = sorted((STATE / "results" / f"{variant}_s{seed}_M" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    split = pd.read_csv(found[0].parent / f"imputation_fold_{fold}_partition.csv")
    test = split.loc[split["split"] == "test", "row"].to_numpy()  # the ledger's row is a position among these
    masked = frame[frame["population"] == "masked"]
    out = masked[masked["kind"] == "numerical"][["row", "column", "actual", "imputed", "actual_original_units"]].copy()
    rows = test[out["row"].to_numpy()]
    truth = table.to_numpy()[rows, table.columns.get_indexer(pd.Index(out["column"]))].astype(float)
    assert np.allclose(truth, out["actual_original_units"].astype(float), rtol=1e-4, atol=1e-6), f"{variant} s{seed} fold {fold}: row mapping"
    out["error"] = (out["imputed"].astype(float) - out["actual"].astype(float)).abs()
    out["b"] = (natural[rows] + out["row"].map(masked.groupby("row").size()).to_numpy()).astype(float)
    return out.assign(spread=1.0).reset_index(drop=True)


def worst_bin(arms: pd.DataFrame, arm: str) -> int:
    """The bin (0 the fewest gaps) with the largest under-coverage of 0.90."""
    return int((e151.TARGET - (arms["error"] <= arms[arm]).groupby(arms["bin"]).mean()).idxmax())


def read(variant: str) -> tuple[str, str]:
    table = pd.read_csv(DATA / variant.rsplit("_", 1)[0] / f"{variant}.csv")
    natural = table.drop(columns=["class"]).isna().sum(axis=1).to_numpy()
    per_seed, worst = [], []
    for seed in SEEDS:
        folds = {k: cells(variant, seed, k, natural, table) for k in range(1, 6)}
        measures = []
        for k in range(1, 6):
            arms = e151.fold_arms(folds[k], pd.concat([f for j, f in folds.items() if j != k], ignore_index=True))
            measures.append(e151.fold_measures(arms))
            worst.append(worst_bin(arms, "C"))
        per_seed.append(measures)
    arrays = {key: [np.array([f[key] for f in folds]) for folds in per_seed] for key in per_seed[0][0]}
    q1 = common.verdict([mc - c for mc, c in zip(arrays["u_MC"], arrays["u_C"])], sesoi=e151.SESOI)
    width = common.verdict([(mc - c) / c for mc, c in zip(arrays["width_MC"], arrays["width_C"])], sesoi=e151.SESOI)
    mean = {key: float(np.concatenate(v).mean()) for key, v in arrays.items()}
    coverage = ", ".join(f"{arm} {mean[f'cover_{arm}']:.3f} (low bin {mean[f'low_{arm}']:.3f}, top {mean[f'top_{arm}']:.3f})" for arm in ("C", "MC"))
    line = (f"- `{variant}`: Q1 u(MC) − u(C) {q1.mean:+.4f} [{q1.low:+.4f}, {q1.high:+.4f}] {q1.call}, u(C) {mean['u_C']:.3f}, "
            f"u(MC) {mean['u_MC']:.3f}; C's worst bin per fold (0 fewest gaps) {dict(sorted(pd.Series(worst).value_counts().items()))}; "
            f"width (MC − C) / C {width.mean:+.3f} [{width.low:+.3f}, {width.high:+.3f}] {width.call}; coverage {coverage}; "
            f"fallback {mean['fallback']:.2f}")
    return q1.call, line


def main() -> None:
    e151.check()
    calls: dict[str, str] = {}
    for variant in TABLES:
        calls[variant], line = read(variant)
        print(line)
    print(f"- Q1: {common.tally(calls.values())}")
    print(f"\nPrimary (E154, Q1 on the two natural tables): {calls} -> {e153.reading(calls)} ({common.RULE})")


if __name__ == "__main__":
    if sys.argv[1:]:
        sys.exit(__doc__)
    main()
