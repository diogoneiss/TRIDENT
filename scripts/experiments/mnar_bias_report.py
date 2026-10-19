"""E45's and E46's measure 5: the signed bias per column under MNAR against MCAR (descriptive).

    uv run --python 3.11 python scripts/experiments/mnar_bias_report.py [<panel_dir>]

Reads ``metric_panel.py``'s CSVs of arm G's ledgers (``--cell_suffix _G``) with every baseline
refitted, in ``<panel_dir>/e45`` and ``<panel_dir>/e46``. Per pair of variants (MNAR, MCAR) and
method: the mean over the 15 fold pairs of the signed bias (fill minus truth, averaged over the
numerical columns, in scaled units; negative means under-predicting), under each mechanism, and
its change from MCAR to MNAR, paired fold by fold (the two variants share their folds), with the
verdict rule. The model's fills are its ledger's, the run's own checkpoint (K), not H. Under MNAR
the hidden values are larger, so every imputer is expected to fall below them; the reading is which
one falls least.
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

from scripts.experiments import report_common as common  # noqa: E402

DEFAULT_PANEL = Path("/scratch2/diogoneiss/trident-2026-10-17-metric-panel")
# The verdict rule names the sign of a difference; here a negative Δ means the fills fall further
# below the truth under MNAR.
SHIFT = {"better": "falls under MNAR", "worse": "rises under MNAR", "none": "no verdict", "equivalent": "equivalent"}
METHODS = ("model", "mean_mode", "knn5", "knn10", "hgb", "lgbm", "missforest", "condexp", "softimpute", "ice")
PAIRS = [
    *((f"{t}_probit_60nan", "e45", f"{t}_60nan", "e45") for t in ("biodeg", "kc2", "vehicle")),
    ("spambase_probit_60nan", "e46", "spambase_60nan", "e46"),
    *((f"{t}_probit_20nan", "e46", f"{t}_20nan", "e46") for t in ("biodeg", "kc2", "vehicle")),
    *((f"biodeg_probit{s}_60nan", "e46", "biodeg_60nan", "e45") for s in ("0p25", "1p0", "2p0")),
]


def _load(panel: Path, tag: str, variant: str) -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in sorted((panel / tag).glob(f"{variant}_s*.csv"))]
    if len(frames) != 3:
        return None
    return pd.concat(frames, ignore_index=True)


def _per_seed(frame: pd.DataFrame, method: str, metric: str) -> list[np.ndarray]:
    rows = frame[frame["method"] == method]
    return [g.sort_values("fold")[metric].to_numpy(dtype=float) for _, g in rows.groupby("seed", sort=True)]


def main(argv: list[str]) -> int:
    panel = Path(argv[0]) if argv else DEFAULT_PANEL
    print("Signed bias (fill − truth, scaled units, mean over numerical columns), mean over 15 folds;")
    print("Δ = MNAR − MCAR, fold-paired, with the verdict rule (" + common.RULE + ").")
    for mnar, mnar_tag, mcar, mcar_tag in PAIRS:
        a, b = _load(panel, mnar_tag, mnar), _load(panel, mcar_tag, mcar)
        if a is None or b is None:
            print(f"\n### {mnar} against {mcar}: incomplete")
            continue
        print(f"\n### {mnar} against {mcar}")
        print()
        print("| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |")
        print("|---|---|---|---|---|---|")
        shifts = {}
        for method in METHODS:
            x, y = _per_seed(a, method, "signed_bias"), _per_seed(b, method, "signed_bias")
            v = common.verdict([p - q for p, q in zip(x, y)])
            shifts[method] = v.mean
            under = float(np.concatenate(_per_seed(a, method, "share_under")).mean())
            print(f"| {method} | {np.concatenate(x).mean():+.4f} | {np.concatenate(y).mean():+.4f} | "
                  f"{v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] | {SHIFT[v.call]} | {under:.0%} |")
        order = sorted(shifts, key=lambda m: abs(shifts[m]))
        rank = order.index("model") + 1
        print(f"\nBy |Δ|, the model ranks {rank} of {len(order)} (1 moves least; least: {order[0]}, most: {order[-1]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
