"""T29's descriptive report: why the boosters win on electricity (read-only).

    uv run --python 3.11 python scripts/experiments/electricity_report.py

From the store and the data, nothing written:

- the table's profile: the rows where ``vicprice``, ``vicdemand`` and ``transfer`` hold one
  constant value each (the source's fill before Victoria's data begins), and where they end in
  ``date``;
- per variant, today's default K against the bar and the strong tier, on 15 fold pairs (20nan:
  E51, seeds 1001/2002/3003; 40, 60 and 80nan: E44, seeds 707/808/909, the headline K). E44's
  fold tables carry missForest but not E41's linear arms, so its strong tier here is missForest
  alone; the tuned bar (D4) is in E44's own report;
- per column, each method's error ratio to the mean/mode fill (z-space RMSE for numbers, error for
  ``day``), the mean over the 15 folds of ``metrics/per_column_imputation.csv``, with the column's
  share of the induced cells. The last two columns are the model's ratio minus the better
  booster's, in RMSE and in MAE (numbers only), so a heavy tail cannot carry a column alone.
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

TABLE = Path("datasets/processed_datasets/electricity/electricity_00nan.csv")
REGIME_COLUMNS = ("vicprice", "vicdemand", "transfer")
SOURCES = {
    "electricity_20nan": ("default-refresh-2026-10-20", [1001, 2002, 3003]),
    "electricity_40nan": ("confirmation-2026-10-17", [707, 808, 909]),
    "electricity_60nan": ("confirmation-2026-10-17", [707, 808, 909]),
    "electricity_80nan": ("confirmation-2026-10-17", [707, 808, 909]),
}
METHODS = ("model", "hgb", "lgbm", "missforest", "condexp", "softimpute", "ice", "knn10")
SESOI = 0.01


def regime(frame: pd.DataFrame) -> pd.Series:
    """Rows where the three Victoria columns all hold their single most frequent value."""
    chosen = pd.Series(True, index=frame.index)
    for column in REGIME_COLUMNS:
        chosen &= frame[column] == frame[column].mode().iloc[0]
    return chosen


def profile() -> None:
    frame = pd.read_csv(TABLE)
    inside = regime(frame)
    first = int(frame.index[~inside].min())
    print("## The table")
    print(f"- {len(frame)} rows, {frame.shape[1] - 1} features; the constant regime: {int(inside.sum())} rows "
          f"({100 * inside.mean():.1f}%), rows 0–{first - 1}, date {frame.date[inside].min():.4f}–{frame.date[inside].max():.4f}; "
          f"none after row {first - 1}")
    for column in REGIME_COLUMNS:
        value = frame[column].mode().iloc[0]
        outside = frame.loc[~inside, column]
        print(f"  - {column}: constant {value} on the regime; elsewhere sd {outside.std():.4f}, and {100 * (outside == value).mean():.2f}% equal to it")
    print()


def _cells(variant: str) -> "common_cells":
    experiment, seeds = SOURCES[variant]
    found = common.parents(experiment)
    out = []
    for seed in seeds:
        runs = found.get((variant, seed), [])
        if len(runs) != 1:
            return []
        out.append((runs[0], common.fold_table(runs[0])))
    return out


common_cells = list[tuple[common.Run, pd.DataFrame]]


def _verdict(found: common_cells, which: str) -> tuple[str, common.Verdict]:
    per_seed, scale, names = [], [], []
    for run, frame in found:
        name, theirs = common.opponent(frame, which)
        per_seed.append(common.model(frame, run, "K") - theirs)
        scale.append(float(theirs.mean()))
        names.append(name)
    return "/".join(sorted(set(names))), common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def _per_column(found: common_cells) -> pd.DataFrame:
    rows = []
    for run, _ in found:
        table = pd.read_csv(common.artifact_path(run.uri) / "metrics" / "per_column_imputation.csv")
        table = table[table.population == "induced"]
        wide = table.pivot_table(index=["fold", "column"], columns="metric", values="value").reset_index()
        rows.append(wide.assign(seed=run.seed))
    return pd.concat(rows, ignore_index=True)


def columns_table(variant: str, frame: pd.DataFrame) -> None:
    numbers = frame["n_num_cells"].fillna(0) > 0
    out = []
    for column, group in frame.groupby("column"):
        numeric = bool(numbers[group.index].any())
        metric = "rmse_num_z" if numeric else "acc_cat"
        naive = group[f"baseline/mean_mode/{metric}"]
        row: dict[str, object] = {"column": column, "cells": float((group["n_num_cells"] + group["n_cat_cells"]).sum())}
        for method in METHODS:
            key = metric if method == "model" else f"baseline/{method}/{metric}"
            if key not in group.columns:
                row[method] = float("nan")
                continue
            ratio = group[key] / naive if numeric else (1 - group[key]) / (1 - naive)
            row[method] = float(ratio.mean())
        if numeric:
            mae_naive = group["baseline/mean_mode/mae_num_z"]
            mae = {m: float((group[m_key] / mae_naive).mean()) for m, m_key in
                   (("model", "mae_num_z"), ("hgb", "baseline/hgb/mae_num_z"), ("lgbm", "baseline/lgbm/mae_num_z"))}
            row["MAE model − best booster"] = mae["model"] - min(mae["hgb"], mae["lgbm"])
        else:
            row["MAE model − best booster"] = float("nan")
        out.append(row)
    table = pd.DataFrame(out)
    table["share"] = table["cells"] / table["cells"].sum()
    table["model − best booster"] = table["model"] - table[["hgb", "lgbm"]].min(axis=1)
    print(f"Per column ({variant}; ratio to the mean/mode fill, mean over 15 folds; lower is better):\n")
    shown = table[["column", "share", *METHODS, "model − best booster", "MAE model − best booster"]]
    print("| " + " | ".join(shown.columns) + " |")
    print("|" + "---|" * len(shown.columns))
    for _, values in shown.iterrows():
        print("| " + " | ".join(f"`{v}`" if isinstance(v, str) else f"{v:.3f}" for v in values) + " |")
    print()


def main() -> int:
    profile()
    for variant in SOURCES:
        found = _cells(variant)
        if not found:
            print(f"## {variant}: runs missing")
            continue
        k = float(np.concatenate([common.model(f, r, "K") for r, f in found]).mean())
        print(f"## {variant} ({SOURCES[variant][0]}, seeds {'/'.join(str(s) for s in SOURCES[variant][1])})")
        for which in ("bar", "strong"):
            name, v = _verdict(found, which)
            print(f"- K {k:.4f} − {which} ({name}): {v.mean:+.4f} ({100 * v.mean / (k - v.mean):+.1f}% of it) "
                  f"[{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print()
        columns_table(variant, _per_column(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
