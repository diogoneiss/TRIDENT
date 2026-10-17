"""Reading of the secondary metric panel (T24; D16): where the per-kind readings agree with impute_score.

    uv run --python 3.11 python scripts/experiments/metric_panel_report.py <panel_dir> [<variant> ...]

Reads ``metric_panel.py``'s CSVs. Per (variant, seed) the opponent is the weak tier's best by mean
``impute_score`` (ADR 0007 decision 8, ADR 0020); per variant and metric, the model's fold values
minus the opponent's, over the 15 fold pairs, with the project's verdict rule. Lower is better for
every ratio, for the absolute bias, the Wasserstein distance and the category TV; for dispersion
the reading is the distance from 1. A claim stands where the per-kind readings agree with
``impute_score``; every disagreement is listed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

PANEL = ("impute_score", "rmse_ratio", "mae_ratio", "colnorm_ratio", "err_ratio", "balacc_error_ratio",
         "abs_bias", "dispersion", "wasserstein", "tv", "impute_score_other", "impute_score_with_context")


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit(__doc__)
    folder = Path(argv[0])
    frames = [pd.read_csv(path) for path in sorted(folder.glob("*.csv"))]
    if not frames:
        sys.exit(f"no panel CSV in {folder}")
    panel = pd.concat(frames, ignore_index=True)
    variants = argv[1:] or sorted(panel["variant"].unique())
    disagreements = []
    for variant in variants:
        rows = panel[panel["variant"] == variant]
        per_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in PANEL}
        names: list[str] = []
        for seed, cell in rows.groupby("seed"):
            weak = cell[cell["method"].isin(common.WEAK)]
            bar = str(weak.groupby("method")["impute_score"].mean().idxmin())
            names.append(bar)
            model = cell[cell["method"] == "model"].sort_values("fold")
            opponent = cell[cell["method"] == bar].sort_values("fold")
            for metric in PANEL:
                if metric not in cell.columns or model[metric].isna().all():
                    continue
                a, b = model[metric].to_numpy(dtype=float), opponent[metric].to_numpy(dtype=float)
                if metric == "dispersion":
                    a, b = np.abs(a - 1), np.abs(b - 1)
                per_metric[metric].append(a - b)
        calls = {}
        for metric, diffs in per_metric.items():
            if len(diffs) == rows["seed"].nunique() and diffs and not any(np.isnan(d).any() for d in diffs):
                calls[metric] = common.verdict(diffs)
        if "impute_score" not in calls:
            continue
        headline = calls["impute_score"].call
        line = ", ".join(f"{m} {v.mean:+.3f} {v.call}" for m, v in calls.items())
        print(f"- `{variant}` (opponent {'/'.join(sorted(set(names)))}): {line}")
        for metric, v in calls.items():
            if metric != "impute_score" and {headline, v.call} == {"better", "worse"}:
                disagreements.append(f"{variant}: impute_score {headline}, {metric} {v.call}")
    print()
    print("Disagreements between impute_score and a panel metric (opposite verdicts, " + common.RULE + "):")
    for item in disagreements or ["none"]:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
