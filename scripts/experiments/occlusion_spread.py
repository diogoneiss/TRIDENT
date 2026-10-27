"""E80: does a numerical cell's re-masking spread grow when its top sources are missing in its row?
(imputation-interpretability/03, T38). Offline, on E78's and E79's cells; no training.

    uv run --python 3.11 python -B scripts/experiments/occlusion_spread.py [--check]

For each fold and numerical target j: j's top 3 sources come from that fold's validation map
(``occlusion_fold_<k>.csv``, split validation), exactly as E78's B ranks them. A test gap cell (r, j)
counts how many of them are missing in row r; a row's gaps are read from the fold ledger's ``induced``
rows, which hold every gap of the test split. The cell's spread is its ``induced_remask`` row's (K).
One fold's value is the mean over targets of Spearman(spread, top-3 missing count) minus the mean of
the same over 20 draws of 3 random sources (private stream ``[seed, fold, 80]``): the comparator
separates "its informative sources are missing" from "more of the row is missing". ``--check``
prints the alignment checks only.
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

from scripts.experiments import occlusion_map as om  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

DRAWS = 20


def _ledger(variant: str, seed: int, fold: int) -> pd.DataFrame:
    found = sorted((om.STATE / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    return pd.read_csv(found[0], dtype={"actual": object, "imputed": object})


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < om.MIN_CELLS or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def fold_values(ledger: pd.DataFrame, validation: pd.DataFrame, seed: int, fold: int, truth: bool = False) -> dict[str, float]:
    gaps = ledger[ledger["population"] == "induced"].groupby("row")["column"].agg(set).to_dict()
    cells = ledger[(ledger["population"] == "induced_remask") & (ledger["kind"] == "numerical")]
    generator = np.random.default_rng([seed, fold, 80])
    out: dict[str, list[float]] = {"specific": [], "top": [], "all": [], "error": []}
    for target, group in cells.groupby("column"):
        name = str(target)
        ranked = validation[(validation["target"] == name) & validation["effect"].notna()].sort_values("effect", ascending=False)
        others = sorted(set(validation.loc[validation["target"] == name, "source"]))
        if truth:
            block = om._block_of(name)
            top = [c for c in om.BLOCKS[block] if c != name] if block else []
        else:
            top = list(ranked["source"].head(om.TOP))
        if len(top) < om.TOP:
            continue
        row_gaps = [gaps.get(int(r), set()) for r in group["row"]]
        spread = group["spread"].to_numpy(dtype=float)
        top_missing = np.array([len(g & set(top)) for g in row_gaps])
        rho_top = _spearman(spread, top_missing)
        drawn = [set(generator.choice(others, size=om.TOP, replace=False)) for _ in range(DRAWS)]
        rho_random = _mean([_spearman(spread, np.array([len(g & d) for g in row_gaps])) for d in drawn])
        out["specific"].append(rho_top - rho_random)
        out["top"].append(rho_top)
        out["all"].append(_spearman(spread, np.array([len(g) - 1 for g in row_gaps])))
        error = (group["imputed"].astype(float) - group["actual"].astype(float)).abs().to_numpy()
        out["error"].append(_spearman(error, top_missing))
    return {key: _mean(values) for key, values in out.items()}


def check() -> None:
    """The ledger's rows and the occlusion files' rows are the same test positions."""
    for study in ("e78", "e79"):
        if study == "e79":
            om._use_sixty()
        variant = next(v for v in om.REAL if not v.startswith("kr-vs-kp"))
        seed = om.SEEDS[variant][0]
        ledger = _ledger(variant, seed, 1)
        induced = ledger[ledger["population"] == "induced"]
        per_row = induced.groupby("row")["column"].nunique()
        columns = om._files(variant, seed, "occlusion")[0]["source"].nunique()
        remask_rows = set(ledger.loc[ledger["population"] == "induced_remask", "row"])
        print(f"{study} {variant} s{seed} fold 1: {len(induced)} gap cells over {len(per_row)} rows, "
              f"gaps a row {per_row.min()}-{per_row.max()} of {columns} columns; remask rows inside the gap rows: "
              f"{remask_rows <= set(per_row.index)}; duplicates: {int(induced.duplicated(['row', 'column']).sum())}")


def tally() -> None:
    marks: dict[str, str] = {}
    for study in ("e78", "e79"):
        if study == "e79":
            om._use_sixty()
        for variant, seeds in om.SEEDS.items():
            if variant.startswith("kr-vs-kp"):
                continue
            per_fold = {seed: [fold_values(_ledger(variant, seed, f), om._files(variant, seed, "occlusion")[f - 1].query("split == 'validation'"), seed, f)
                               for f in range(1, 6)] for seed in seeds}
            v = common.verdict([np.array([d["specific"] for d in per_fold[seed]]) for seed in seeds])
            mark = "specific" if v.low > 0 and all(s > 0 for s in v.seeds) else "against" if v.high < 0 and all(s < 0 for s in v.seeds) else "no signal"
            described = {key: float(np.nanmean([d[key] for seed in seeds for d in per_fold[seed]])) for key in ("top", "all", "error")}
            line = (f"- `{variant}`: top-3 − random-3 {v.mean:+.3f} [{v.low:+.3f}, {v.high:+.3f}] seeds {', '.join(f'{s:+.3f}' for s in v.seeds)} -> {mark}; "
                    f"descriptive: ρ(spread, top-3 missing) {described['top']:+.3f}, ρ(spread, row's gaps) {described['all']:+.3f}, ρ(error, top-3 missing) {described['error']:+.3f}")
            if variant.startswith("synthblock"):
                true = [fold_values(_ledger(variant, seed, f), om._files(variant, seed, "occlusion")[f - 1].query("split == 'validation'"), seed, f, truth=True)["specific"]
                        for seed in seeds for f in range(1, 6)]
                line += f"; with the true block as the top 3: {np.nanmean(true):+.3f} (descriptive)"
            else:
                marks[variant] = mark
            print(line)
    specific = [v for v, m in marks.items() if m == "specific"]
    against = [v for v, m in marks.items() if m == "against"]
    reading = "supported" if len(specific) >= 12 and not against else "refuted" if len(specific) <= 6 else "mixed"
    print(f"\nPrimary: specific on {len(specific)} of {len(marks)}, against on {against or 'none'} -> {reading} ({common.RULE})")


if __name__ == "__main__":
    check() if "--check" in sys.argv[1:] else tally()
