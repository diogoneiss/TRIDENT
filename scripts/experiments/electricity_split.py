"""E54, T29's test: is the boosters' lead on electricity carried by ``date`` as a key?
(imputation-protocol/09). Offline: nothing is written to the store.

    uv run --python 3.11 python -B scripts/experiments/electricity_split.py <variant> <seed> <out_dir>
    uv run --python 3.11 python -B scripts/experiments/electricity_split.py --tally <out_dir>

Per (variant, seed), on each of the five folds:

- **The model's fills** are its induced fills, from the run's fold ledger. They are the headline
  K on these runs: E51 at 20nan, E44 at 40, 60 and 80nan.
- **The boosters' fills** (``hgb`` and ``lgbm``) are refitted on the same training fold and asked
  for exactly the ledger's cells.
- **Integrity:** on all the cells, the model's and both boosters' ``impute_score`` must equal the
  run's fold table to 1e-9.

``impute_score`` (against the fold's mean/mode fill) is then written on these subsets of the
induced cells:

- ``other7_date_observed`` and ``other7_date_missing``: the cells of the seven columns other than
  ``date``, split by whether their row's ``date`` is observed (the primary pair). A ``date`` cell's
  row never has its date, so those cells are left out of both;
- the same split for the ``day`` and ``nswdemand`` cells alone;
- ``regime`` and ``rest``: rows 0–17,423 (the three Victoria columns constant) and the others;
- ``all``.

``--tally`` reads the CSVs and applies the rule fixed in the ticket. Each run's bar is the booster
that the full-cell rule (``report_common.opponent``) picked for it. Threads are capped at 4.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from src.training import lightgbm_baseline  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_baselines import GradientBoostingImputer  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.lightgbm_baseline import LightGbmImputer  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

FOLDS = 5
REGIME_ROWS = 17424
SOURCES = {
    "electricity_20nan": ("default-refresh-2026-10-20", Path("/scratch2/diogoneiss/trident-2026-10-20-default-refresh"), (1001, 2002, 3003)),
    "electricity_40nan": ("confirmation-2026-10-17", Path("/scratch2/diogoneiss/trident-2026-10-17-confirmation"), (707, 808, 909)),
    "electricity_60nan": ("confirmation-2026-10-17", Path("/scratch2/diogoneiss/trident-2026-10-17-confirmation"), (707, 808, 909)),
    "electricity_80nan": ("confirmation-2026-10-17", Path("/scratch2/diogoneiss/trident-2026-10-17-confirmation"), (707, 808, 909)),
}
WORSE_RATES = ("electricity_20nan", "electricity_60nan")
SESOI = 0.01


def _ledger(state: Path, variant: str, seed: int, fold: int) -> pd.DataFrame:
    found = sorted((state / "results" / f"{variant}_s{seed}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    return frame[frame["population"] == "induced"].reset_index(drop=True)


def _score(cells: pd.DataFrame, naive: dict[str, float | str]) -> float:
    return float(score_cells(cells, naive).metrics.get("impute_score", float("nan"))) if len(cells) else float("nan")


def split(variant: str, seed: int, out: Path) -> None:
    experiment, state, _ = SOURCES[variant]
    runs = common.parents(experiment).get((variant, seed), [])
    assert len(runs) == 1, f"{variant} s{seed}: {len(runs)} runs"
    logged = common.fold_table(runs[0])
    lightgbm_baseline.THREADS = 4
    warnings.filterwarnings("ignore")
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    rows: list[dict[str, object]] = []
    started = time.time()
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, FOLDS, seed), start=1):
        train_index, test_index = np.asarray(fold.train_indices), np.asarray(fold.test_indices)
        train = features.iloc[train_index].reset_index(drop=True)
        test = features.iloc[test_index].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        cells = _ledger(state, variant, seed, ordinal)
        numeric = cells["kind"] == NUMERICAL
        cells.loc[numeric, "actual"] = cells.loc[numeric, "actual"].astype(float)
        cells.loc[numeric, "imputed"] = cells.loc[numeric, "imputed"].astype(float)
        positions = cells["row"].to_numpy(dtype=int)
        column_index = test.columns.get_indexer(pd.Index(cells["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, column_index] = True
        date_observed = test["date"].notna().to_numpy()[positions]
        in_regime = test_index[positions] < REGIME_ROWS
        other = (cells["column"] != "date").to_numpy()
        subsets = {
            "all": np.ones(len(cells), dtype=bool),
            "other7_date_observed": other & date_observed,
            "other7_date_missing": other & ~date_observed,
            "regime": in_regime,
            "rest": ~in_regime,
        }
        for column in ("day", "nswdemand"):
            mine = (cells["column"] == column).to_numpy()
            subsets[f"{column}_date_observed"] = mine & date_observed
            subsets[f"{column}_date_missing"] = mine & ~date_observed
        fills = {"model": cells}
        for imputer in (GradientBoostingImputer(numerical, categorical), LightGbmImputer(numerical, categorical)):
            imputer.fit(train)
            filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, column_index]
            fills[imputer.name] = cells.assign(imputed=[float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])])
        for method, frame in fills.items():
            key = "impute/induced/impute_score" if method == "model" else f"impute/induced/baseline/{method}/impute_score"
            if method == "model":
                key = common.readout_key(logged.columns, runs[0].decode_checkpoint, "K", "induced", "impute_score")
            full, expected = _score(frame, naive), float(logged[key].to_numpy(dtype=float)[ordinal - 1])
            assert abs(full - expected) < 1e-9, f"{variant} s{seed} fold {ordinal} {method}: {full} against {expected}"
            for name, chosen in subsets.items():
                rows.append({"variant": variant, "seed": seed, "fold": ordinal, "method": method, "subset": name,
                             "n_cells": int(chosen.sum()), "impute_score": _score(frame[chosen], naive)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{variant}_s{seed}.csv", index=False)
    print(f"DONE {variant} s{seed}: integrity passed on 5 folds, {time.time() - started:.0f} s", flush=True)


def _bar(variant: str, seed: int) -> str:
    experiment, _, _ = SOURCES[variant]
    run = common.parents(experiment)[(variant, seed)][0]
    return common.opponent(common.fold_table(run), "bar")[0]


def _contrast(table: pd.DataFrame, variant: str, subset: str) -> common.Verdict | None:
    per_seed, scale = [], []
    for seed in SOURCES[variant][2]:
        rows = table[(table.variant == variant) & (table.seed == seed) & (table.subset == subset)]
        if rows.empty:
            return None
        bar = _bar(variant, seed)
        model = rows[rows.method == "model"].sort_values("fold")["impute_score"].to_numpy(dtype=float)
        theirs = rows[rows.method == bar].sort_values("fold")["impute_score"].to_numpy(dtype=float)
        per_seed.append(model - theirs)
        scale.append(float(np.mean(theirs)))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def tally(out: Path) -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted(out.glob("electricity_*_s*.csv"))], ignore_index=True)
    calls: dict[tuple[str, str], str] = {}
    for variant in SOURCES:
        print(f"=== {variant} (bars: {', '.join(_bar(variant, s) for s in SOURCES[variant][2])})")
        for subset in ("all", "other7_date_observed", "other7_date_missing", "day_date_observed", "day_date_missing",
                       "nswdemand_date_observed", "nswdemand_date_missing", "regime", "rest"):
            v = _contrast(table, variant, subset)
            if v is None:
                print(f"  {subset}: incomplete")
                continue
            cells = int(table[(table.variant == variant) & (table.subset == subset) & (table.method == "model")]["n_cells"].mean())
            print(f"  K − bar, {subset} (≈{cells} cells a fold): {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            calls[variant, subset] = v.call
    needed = [(v, s) for v in WORSE_RATES for s in ("other7_date_observed", "other7_date_missing")]
    if any(k not in calls for k in needed):
        print("\nPrimary: incomplete")
        return
    observed_worse = all(calls[v, "other7_date_observed"] == "worse" for v in WORSE_RATES)
    missing_not_worse = all(calls[v, "other7_date_missing"] != "worse" for v in WORSE_RATES)
    missing_worse = all(calls[v, "other7_date_missing"] == "worse" for v in WORSE_RATES)
    reading = "supported" if observed_worse and missing_not_worse else "refuted" if missing_worse else "mixed"
    print(f"\nPrimary (20 and 60nan, the rates where K − bar is worse): date observed "
          f"{[calls[v, 'other7_date_observed'] for v in WORSE_RATES]}, date missing "
          f"{[calls[v, 'other7_date_missing'] for v in WORSE_RATES]} -> {reading} ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"] and len(argv) == 2:
        tally(Path(argv[1]))
    elif len(argv) == 3:
        split(argv[0], int(argv[1]), Path(argv[2]))
    else:
        sys.exit(__doc__)
