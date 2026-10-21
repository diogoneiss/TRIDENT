"""The secondary metric panel, offline (T24; plan 2026-10-17 §2d, D16).

    uv run --python 3.11 python -B scripts/experiments/metric_panel.py <state_dir> <out_dir> <variant> <seed> [<seed> ...] [--baselines] [--cell_suffix _G] [--fitters tabpfn]

For each (variant, seed) of a queue study, the model's induced fills are read from the run's five
fold ledgers (``<state_dir>/results/<variant>_s<seed>/<variant>/<timestamp>/imputation/
imputation_fold_<n>_cells.csv``: the headline's fills, K on every run since ADR 0015). With
``--baselines``, every baseline family (the weak tier, missForest, E41's arms) is refitted on the
same training fold, as ``recompute_folds`` rebuilds it, and asked for exactly the ledger's cells.
``--fitters tabpfn`` refits TabPFN v2 alone instead (E56; run under ``uv run --extra tabpfn``).

Per fold and method it writes, beside ``impute_score``:

- the per-kind ratios to the mean/mode fill: RMSE, MAE, categorical error, balanced-accuracy error;
- a column-normalised numeric ratio (the mean over columns of each column's RMSE ratio);
- the mean absolute signed bias over numeric columns, in scaled units, and the dispersion ratio
  (sd of the fills over sd of the truths, averaged over columns);
- the mean 1-D Wasserstein distance over numeric columns, and the mean category-frequency total
  variation over categorical ones;
- ``impute_score`` on the cells of rows whose complete feature vector also appears in the training
  fold (twins), on the other cells, and on the cells of rows that are not context-free (rows with
  every feature missing); and those rows' share of the cells.

One CSV per (variant, seed) in ``<out_dir>``; nothing is written to the store; threads capped at 4.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training import lightgbm_baseline, missforest  # noqa: E402
from src.training.data import build_folds, load_complete_sibling, prepare_dataset  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, fit_baseline_imputers  # noqa: E402
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.lightgbm_baseline import fit_lightgbm_imputers  # noqa: E402
from src.training.linear_baselines import fit_linear_imputers  # noqa: E402
from src.training.missforest import fit_missforest_imputers  # noqa: E402
from src.training.tabpfn_baseline import fit_tabpfn_imputers  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

THREADS = 4
FOLDS = 5
FITTERS: Sequence[Callable[[pd.DataFrame, Sequence[str], Sequence[str]], list[BaselineImputer]]] = (
    fit_baseline_imputers, fit_lightgbm_imputers, fit_missforest_imputers, fit_linear_imputers,
)
# E56: TabPFN v2 alone, the optional extra's baseline (``src/training/tabpfn_baseline.py``).
NAMED_FITTERS = {"tabpfn": (fit_tabpfn_imputers,)}


def _ratio(value: float, naive: float) -> float:
    return value / naive if naive > 0 else float("nan")


def panel(cells: pd.DataFrame, naive: Mapping[str, float | str]) -> dict[str, float]:
    """The panel on scored cells carrying ``actual`` and ``imputed``."""
    out: dict[str, float] = {"impute_score": float(score_cells(cells, naive).metrics.get("impute_score", float("nan")))}
    numbers = cells[cells["kind"] == NUMERICAL]
    if len(numbers):
        actual = numbers["actual"].to_numpy(dtype=float)
        imputed = numbers["imputed"].to_numpy(dtype=float)
        fill = numbers["column"].map(naive).to_numpy(dtype=float)
        out["rmse_ratio"] = _ratio(float(np.sqrt(np.mean((imputed - actual) ** 2))), float(np.sqrt(np.mean((fill - actual) ** 2))))
        out["mae_ratio"] = _ratio(float(np.mean(np.abs(imputed - actual))), float(np.mean(np.abs(fill - actual))))
        ratios, biases, signed, spreads, distances = [], [], [], [], []
        for column, group in numbers.groupby("column"):
            a, i = group["actual"].to_numpy(dtype=float), group["imputed"].to_numpy(dtype=float)
            f = float(naive[str(column)])
            ratios.append(_ratio(float(np.sqrt(np.mean((i - a) ** 2))), float(np.sqrt(np.mean((f - a) ** 2)))))
            biases.append(abs(float(np.mean(i - a))))
            signed.append(float(np.mean(i - a)))
            if len(a) > 1 and a.std() > 0:
                spreads.append(float(i.std() / a.std()))
            distances.append(float(wasserstein_distance(i, a)))
        out["colnorm_ratio"] = float(np.nanmean(ratios))
        out["abs_bias"] = float(np.mean(biases))
        # E45/E46's measure 5: the signed bias per column, averaged, and the share of columns
        # whose fills fall below the truth on average (MNAR hides large values).
        out["signed_bias"] = float(np.mean(signed))
        out["share_under"] = float(np.mean(np.asarray(signed) < 0))
        out["dispersion"] = float(np.mean(spreads)) if spreads else float("nan")
        out["wasserstein"] = float(np.mean(distances))
    categories = cells[cells["kind"] == CATEGORICAL]
    if len(categories):
        actual_labels = categories["actual"].astype(str).to_numpy()
        imputed_labels = categories["imputed"].astype(str).to_numpy()
        modes = categories["column"].map(naive).astype(str).to_numpy()
        out["err_ratio"] = _ratio(float(np.mean(imputed_labels != actual_labels)), float(np.mean(modes != actual_labels)))
        balanced, balanced_naive, weights, variations = [], [], [], []
        for column, group in categories.groupby("column"):
            a = group["actual"].astype(str).to_numpy()
            i = group["imputed"].astype(str).to_numpy()
            mode = str(naive[str(column)])
            classes = np.unique(a)
            balanced.append(float(np.mean([np.mean(i[a == c] == c) for c in classes])))
            balanced_naive.append(float(np.mean([np.mean(np.full((a == c).sum(), mode) == c) for c in classes])))
            weights.append(len(a))
            labels = np.union1d(a, i)
            variations.append(0.5 * float(sum(abs(np.mean(i == c) - np.mean(a == c)) for c in labels)))
        out["balacc_error_ratio"] = _ratio(1 - float(np.average(balanced, weights=weights)), 1 - float(np.average(balanced_naive, weights=weights)))
        out["tv"] = float(np.mean(variations))
    for subset in ("twin", "other", "with_context"):
        chosen = cells[cells[subset]] if subset in cells.columns else cells.iloc[0:0]
        out[f"impute_score_{subset}"] = float(score_cells(chosen, naive).metrics.get("impute_score", float("nan"))) if len(chosen) else float("nan")
    out["share_twin"] = float(cells["twin"].mean())
    out["share_context_free"] = float(1 - cells["with_context"].mean())
    return out


def _ledger(state: Path, variant: str, seed: int, fold: int, suffix: str = "") -> pd.DataFrame:
    found = sorted((state / "results" / f"{variant}_s{seed}{suffix}" / variant).glob(f"*/imputation/imputation_fold_{fold}_cells.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"{variant} s{seed} fold {fold}: {len(found)} ledgers")
    frame = pd.read_csv(found[0], dtype={"actual": object, "imputed": object})
    return frame[frame["population"] == "induced"].reset_index(drop=True)


def main(argv: list[str]) -> int:
    baselines = "--baselines" in argv
    argv = [a for a in argv if a != "--baselines"]
    # A study whose cell directories carry the arm (E45/E46: ``<variant>_s<seed>_G``).
    suffix = ""
    fitters = FITTERS
    if "--fitters" in argv:
        at = argv.index("--fitters")
        fitters, argv = NAMED_FITTERS[argv[at + 1]], argv[:at] + argv[at + 2:]
    if "--cell_suffix" in argv:
        at = argv.index("--cell_suffix")
        suffix, argv = argv[at + 1], argv[:at] + argv[at + 2:]
    if len(argv) < 4:
        sys.exit(__doc__)
    state, out_dir, variant, seeds = Path(argv[0]), Path(argv[1]), argv[2], [int(s) for s in argv[3:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    warnings.filterwarnings("ignore")
    spec = DatasetSpec.from_name(variant, None)
    dataset = prepare_dataset(spec)
    sibling = load_complete_sibling(spec)
    if sibling is None:
        sys.exit(f"{variant} has no complete sibling")
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    complete_rows = sibling[list(features.columns)].astype(str).agg("|".join, axis=1).to_numpy()
    for seed in seeds:
        path = out_dir / f"{variant}_s{seed}.csv"
        if path.exists():
            print(f"SKIP {path.name}", flush=True)
            continue
        started = time.time()
        rows: list[dict[str, float | int | str]] = []
        for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, FOLDS, seed), start=1):
            train_index, test_index = np.asarray(fold.train_indices), np.asarray(fold.test_indices)
            train = features.iloc[train_index].reset_index(drop=True)
            test = features.iloc[test_index].reset_index(drop=True)
            naive = mean_mode_baselines(train, numerical, categorical)
            cells = _ledger(state, variant, seed, ordinal, suffix)
            numeric = cells["kind"] == NUMERICAL
            cells.loc[numeric, "actual"] = cells.loc[numeric, "actual"].astype(float)
            cells.loc[numeric, "imputed"] = cells.loc[numeric, "imputed"].astype(float)
            seen = set(complete_rows[train_index])
            twin_rows = np.array([complete_rows[i] in seen for i in test_index])
            context_free = test.isna().all(axis=1).to_numpy()
            positions = cells["row"].to_numpy(dtype=int)
            cells["twin"] = twin_rows[positions]
            cells["other"] = ~cells["twin"]
            cells["with_context"] = ~context_free[positions]
            rows.append({"variant": variant, "seed": seed, "fold": ordinal, "method": "model", **panel(cells, naive)})
            if not baselines:
                continue
            hidden = np.zeros(test.shape, dtype=bool)
            hidden[positions, test.columns.get_indexer(pd.Index(cells["column"]))] = True
            for fitter in fitters:
                for imputer in fitter(train, numerical, categorical):
                    filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, test.columns.get_indexer(pd.Index(cells["column"]))]
                    guesses = [float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])]
                    rows.append({"variant": variant, "seed": seed, "fold": ordinal, "method": imputer.name, **panel(cells.assign(imputed=guesses), naive)})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"DONE {variant} s{seed}: {len(rows)} rows, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
