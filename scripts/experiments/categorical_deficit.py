"""E61: where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read, offline
(imputation-protocol/16). Nothing is written to the store.

    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/categorical_deficit.py <variant> <seed> <out_dir>
    uv run --python 3.11 python -B scripts/experiments/categorical_deficit.py --tally <out_dir>

On E40's runs (seeds 404/505/606), per fold and per column of the ledger's induced cells:

- **The model's errors** come from the run's fold ledger: the headline K's fills.
- **TabPFN's errors** come from TabPFN v2 refitted on the same training fold, as E60 did, and asked
  for exactly the ledger's cells.
- **The mode's (or mean's) errors** come from the training fold, as ``impute_score``'s naive fill.

A categorical column's error is its count of wrong cells, a numerical one's its sum of squared
errors. The model's fold ``impute_score`` must equal the run's K, and TabPFN's must stay within
``TABPFN_TOLERANCE`` of E58's CSV, or the cell stops. ``--tally`` applies the rule fixed in the
ticket. Threads are capped at 4.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEFAULT_TABPFN  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-16-calibration")
EXPERIMENT = "calibration-2026-10-16"
VARIANTS = [f"{t}_{level}nan" for t in ("kr-vs-kp", "credit-g") for level in (40, 60, 80)]
SEEDS = (404, 505, 606)
FOLDS = 5
# E58's fills came from one pass over the masked and induced views together, which moves a fold's
# score by up to 3.6e-4 (E55's note); this pass asks for the induced cells alone.
TABPFN_TOLERANCE = 2e-3
# The rule's thresholds, fixed in the ticket: a column where the model is stuck at the marginal.
STUCK_MODEL_GAIN, STUCK_TABPFN_GAIN, STUCK_SHARE = 0.05, 0.20, 0.50


def _ledger(variant: str, seed: int, fold: int) -> pd.DataFrame:
    found = sorted((STATE / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    return frame[frame["population"] == "induced"].reset_index(drop=True)


def _score(cells: pd.DataFrame, naive: dict[str, float | str]) -> float:
    return float(score_cells(cells, naive).metrics.get("impute_score", float("nan")))


def _errors(cells: pd.DataFrame, filled: Sequence[object]) -> np.ndarray:
    numeric = (cells["kind"] == NUMERICAL).to_numpy()
    out = np.empty(len(cells))
    actual = cells["actual"].to_numpy()
    guess = np.asarray(filled, dtype=object)
    out[numeric] = (guess[numeric].astype(float) - actual[numeric].astype(float)) ** 2
    out[~numeric] = (guess[~numeric].astype(str) != actual[~numeric].astype(str)).astype(float)
    return out


def _entropy(shares: np.ndarray) -> float:
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum())


def run(variant: str, seed: int, out: Path) -> None:
    from src.training.tabpfn_baseline import fit_tabpfn_imputers

    warnings.filterwarnings("ignore")
    found = common.parents(EXPERIMENT)[(variant, seed)]
    if len(found) != 1:
        raise RuntimeError(f"{variant} s{seed}: {len(found)} runs")
    logged = common.model(common.fold_table(found[0]), found[0], "K")
    e58 = pd.read_csv(DEFAULT_TABPFN / f"{variant}_s{seed}.csv").sort_values("fold")["impute/induced/baseline/tabpfn/impute_score"].to_numpy()
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    rows: list[dict[str, object]] = []
    started = time.time()
    drift = 0.0
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        cells = _ledger(variant, seed, ordinal)
        numeric = (cells["kind"] == NUMERICAL).to_numpy()
        cells.loc[numeric, "actual"] = cells.loc[numeric, "actual"].astype(float)
        cells.loc[numeric, "imputed"] = cells.loc[numeric, "imputed"].astype(float)
        positions = cells["row"].to_numpy(dtype=int)
        columns = test.columns.get_indexer(pd.Index(cells["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, columns] = True
        imputer = fit_tabpfn_imputers(train, numerical, categorical)[0]
        filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, columns]
        tabpfn = [float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])]
        model_score, tabpfn_score = _score(cells, naive), _score(cells.assign(imputed=tabpfn), naive)
        assert abs(model_score - float(logged[ordinal - 1])) < 1e-9, f"{variant} s{seed} fold {ordinal}: {model_score} against {logged[ordinal - 1]}"
        assert abs(tabpfn_score - float(e58[ordinal - 1])) < TABPFN_TOLERANCE, f"{variant} s{seed} fold {ordinal}: TabPFN {tabpfn_score} against E58's {e58[ordinal - 1]}"
        drift = max(drift, abs(tabpfn_score - float(e58[ordinal - 1])))
        mode = [naive[c] for c in cells["column"]]
        frame = cells.assign(model=_errors(cells, list(cells["imputed"])), tabpfn=_errors(cells, tabpfn), mode=_errors(cells, mode))
        for column, group in frame.groupby("column", sort=True):
            kind = str(group["kind"].iloc[0])
            row: dict[str, object] = {"variant": variant, "seed": seed, "fold": ordinal, "column": column, "kind": kind,
                                      "cells": len(group), "model": group["model"].sum(), "tabpfn": group["tabpfn"].sum(),
                                      "mode": group["mode"].sum(), "model_score": model_score, "tabpfn_score": tabpfn_score}
            if kind == CATEGORICAL:
                shares = train[column].dropna().astype(str).value_counts(normalize=True).to_numpy()
                row |= {"cardinality": len(shares), "mode_share": float(shares.max()), "entropy": _entropy(shares)}
            rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{variant}_s{seed}.csv", index=False)
    print(f"DONE {variant} s{seed}: model equals the run's K on 5 folds, TabPFN within {drift:.1e} of E58, {time.time() - started:.0f} s", flush=True)


def _parts(rows: pd.DataFrame, method: str) -> tuple[float, float]:
    """A fold's ``impute_score`` split into its categorical and numerical terms."""
    total = rows["cells"].sum()
    cat, num = rows[rows.kind == CATEGORICAL], rows[rows.kind == NUMERICAL]
    parts = []
    for part, kind_rows in ((cat, CATEGORICAL), (num, NUMERICAL)):
        if part.empty:
            parts.append(0.0)
            continue
        n = part["cells"].sum()
        mine, naive = part[method].sum() / n, part["mode"].sum() / n
        if kind_rows == NUMERICAL:
            mine, naive = np.sqrt(mine), np.sqrt(naive)
        parts.append(float(n / total * (mine / naive if naive else 1.0 + mine)))
    return parts[0], parts[1]


def tally(out: Path) -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted(out.glob("*.csv"))], ignore_index=True)
    counted, holding = 0, []
    for variant in VARIANTS:
        rows = table[table.variant == variant]
        if rows["seed"].nunique() < len(SEEDS):
            print(f"- `{variant}`: incomplete")
            continue
        folds = [g for _, g in rows.groupby(["seed", "fold"])]
        for g in folds:
            assert abs(sum(_parts(g, "model")) - g["model_score"].iloc[0]) < 1e-9, "the split does not add up to the score"
        gap_cat = float(np.mean([_parts(g, "model")[0] - _parts(g, "tabpfn")[0] for g in folds]))
        gap_num = float(np.mean([_parts(g, "model")[1] - _parts(g, "tabpfn")[1] for g in folds]))
        gap = gap_cat + gap_num
        print(f"## `{variant}`: K − TabPFN {gap:+.4f} = categorical {gap_cat:+.4f} + numerical {gap_num:+.4f}")
        cat = rows[rows.kind == CATEGORICAL].groupby("column").agg(
            cells=("cells", "sum"), model=("model", "sum"), tabpfn=("tabpfn", "sum"), mode=("mode", "sum"),
            cardinality=("cardinality", "max"), mode_share=("mode_share", "mean"), entropy=("entropy", "mean"))
        if cat.empty or gap <= 0 or gap_cat < 0.5 * gap:
            print("  - not counted: K is not behind TabPFN here, or the categorical term carries under half of the gap\n")
            continue
        counted += 1
        cat["excess"] = cat["model"] - cat["tabpfn"]
        cat["model_gain"] = 1 - cat["model"] / cat["mode"].where(cat["mode"] > 0)
        cat["tabpfn_gain"] = 1 - cat["tabpfn"] / cat["mode"].where(cat["mode"] > 0)
        net = float(cat["excess"].sum())
        stuck = cat[(cat["model_gain"] <= STUCK_MODEL_GAIN) & (cat["tabpfn_gain"] >= STUCK_TABPFN_GAIN)]
        share = float(stuck["excess"].sum() / net) if net > 0 else float("nan")
        top = max(1, int(np.ceil(0.2 * len(cat))))
        concentration = float(cat["excess"].nlargest(top).sum() / net) if net > 0 else float("nan")
        proportional = float(cat["tabpfn"].nlargest(top).sum() / cat["tabpfn"].sum())
        print(f"  - net excess errors {net:.0f} over {len(cat)} columns; the top {top} hold {concentration:.0%} "
              f"(TabPFN's own errors, top {top}: {proportional:.0%})")
        print(f"  - stuck at the marginal (model gain ≤ {STUCK_MODEL_GAIN:.0%}, TabPFN gain ≥ {STUCK_TABPFN_GAIN:.0%}): "
              f"{len(stuck)} columns, {share:.0%} of the net excess")
        shown = cat.sort_values("excess", ascending=False).head(8)
        print("  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |")
        print("  |---|---|---|---|---|---|---|---|---|")
        for name, c in shown.iterrows():
            print(f"  | {name} | {c.cells:.0f} | {c.model:.0f} | {c.tabpfn:.0f} | {c['mode']:.0f} | {c.model_gain:+.0%} | {c.tabpfn_gain:+.0%} | {c.cardinality:.0f} | {c.mode_share:.2f} |")
        corr = float(np.corrcoef(cat["model_gain"].fillna(0), cat["tabpfn_gain"].fillna(0))[0, 1])
        print(f"  - correlation of K's and TabPFN's per-column gains: {corr:+.2f}\n")
        if share >= STUCK_SHARE:
            holding.append(variant)
    if counted < 3:
        print(f"Primary: only {counted} variants counted; the rule needs 3")
        return
    reading = "a target" if len(holding) >= 3 else "no target" if len(holding) <= 1 else "mixed"
    print(f"Primary: stuck columns hold ≥ {STUCK_SHARE:.0%} of the net excess on {len(holding)} of {counted} counted variants "
          f"({holding or 'none'}) -> {reading}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"] and len(argv) == 2:
        tally(Path(argv[1]))
    elif len(argv) == 3:
        run(argv[0], int(argv[1]), Path(argv[2]))
    else:
        sys.exit(__doc__)
