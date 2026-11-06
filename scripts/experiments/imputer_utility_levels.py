"""E147 (ticket imputation-protocol/53; run as E149), post hoc and descriptive only: the primary by missingness level,
the per-variant mean correlations, and the imputer each score picks.

    uv run --python 3.11 python -B scripts/experiments/imputer_utility_levels.py <csv_imputers dir>

Reads ``imputer_utility.py``'s CSVs; no fit, nothing written. The block it heads "per variant (MCAR)" lists every
mechanism's variants; the label is kept so that the output matches the one in the ticket.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main(out: Path) -> int:
    d = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*.csv"))], ignore_index=True)
    d = d[~d["method"].isin(["mean_mode", "truth"])]
    d["f1"] = d[["f1_complete_logistic", "f1_complete_mlp"]].mean(axis=1)
    d["mech"] = d["variant"].map(lambda v: "MAR" if "_marf_" in v else "MNAR" if "_probit_" in v else "MCAR")
    d["level"] = d["variant"].str.extract(r"_(\d+)nan$")[0]
    rows: list[dict[str, object]] = []
    for (v, s, f), g in d.groupby(["variant", "seed", "fold"]):
        assert len(g) == 12, (v, s, f, len(g))
        rows.append({"variant": v, "seed": s, "fold": f, "mech": g["mech"].iloc[0], "level": g["level"].iloc[0],
                     "rho_i": stats.spearmanr(-g["impute_score"], g["f1"]).statistic, "rho_d": stats.spearmanr(-g["dist_score"], g["f1"]).statistic,
                     "pick_i": g.loc[g["impute_score"].idxmin(), "method"], "pick_d": g.loc[g["dist_score"].idxmin(), "method"],
                     "best": g.loc[g["f1"].idxmax(), "method"]})
    o = pd.DataFrame(rows)
    o["d"] = o["rho_d"] - o["rho_i"]
    for (m, level), g in o.groupby(["mech", "level"]):
        p = g.pivot_table(index=["seed", "fold"], columns="variant", values="d").mean(axis=1).to_numpy()
        h = stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p))
        print(f"{m} {level}nan: {g['variant'].nunique()} variants, rho_impute {g['rho_i'].mean():+.3f}, rho_dist {g['rho_d'].mean():+.3f}, "
              f"diff {p.mean():+.4f} [{p.mean()-h:+.4f}, {p.mean()+h:+.4f}]")
    print("\nper variant (MCAR): diff of rho, mean over 15 folds")
    print(o.groupby(["mech", "variant"])[["rho_i", "rho_d", "d"]].mean().round(3).to_string())
    for m, g in o.groupby("mech"):
        print(f"\n{str(m)}: picked by impute_score {g['pick_i'].value_counts().to_dict()}\n   picked by dist_score {g['pick_d'].value_counts().to_dict()}\n"
              f"   classifier's best {g['best'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raise SystemExit(main(Path(sys.argv[1])))
