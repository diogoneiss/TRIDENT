"""E182: TRIDENT against its baselines on a distributional score that is not marginal, the sliced Wasserstein
distance between the filled rows and the true rows (docs/tickets/imputation-protocol/71-sliced-wasserstein.md).

    SLICED_STATE=<dir> python -B scripts/experiments/sliced_run.py run <variant> <seed> <group>   # one CSV under <dir>/csv
    python -B scripts/experiments/sliced_run.py --jobs <medusa|gorgona8>                         # that machine's jobs, small tables first
    SLICED_STATE=<dir> python -B scripts/experiments/sliced_run.py --tally

No training and no MLflow: TRIDENT's fills are E181's ``Kc`` fold ledgers (``SLICED_LEDGERS``, the directory holding
their ``results/``), and each baseline is refitted on the fold's training rows and asked for exactly the ledger's
cells, as ``imputer_utility.py`` does. A group is a set of imputers: ``reference`` (TRIDENT, the mean/mode fill, the
class-conditional fill, the truth, and the control: the truths of each column shuffled among its scored cells, so
every marginal is exact and the rows are wrong), ``adr0007`` (knn5, knn10, hgb), ``lgbm``, ``missforest`` and
``linear`` (condexp, softimpute, ice). ``--tally`` refuses a partial read. Run from the checkout root.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.decode_metric_report import _scores  # noqa: E402
from scripts.experiments.hybrid_fresh_seeds import MCAR  # noqa: E402
from src.training import lightgbm_baseline, missforest  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, fit_baseline_imputers  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, _ratio, class_mean_mode_baselines, mean_mode_baselines  # noqa: E402
from src.training.lightgbm_baseline import fit_lightgbm_imputers  # noqa: E402
from src.training.linear_baselines import fit_linear_imputers  # noqa: E402
from src.training.missforest import fit_missforest_imputers  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

STATE = Path(os.environ.get("SLICED_STATE", "/scratch/diogoneiss/trident-2026-11-18-sliced"))
LEDGERS = Path(os.environ.get("SLICED_LEDGERS", "/scratch/diogoneiss/trident-2026-11-17-classcond"))
THREADS = int(os.environ.get("IMPUTER_THREADS", "4"))
VARIANTS: tuple[str, ...] = MCAR
SEEDS: tuple[int, ...] = (101, 202, 303)
GROUPS = ("reference", "linear", "adr0007", "lgbm", "missforest")
TABLES = ("kc2", "vehicle", "biodeg", "credit-g", "kr-vs-kp", "spambase", "pendigits", "letter", "electricity")  # by size
GORGONA8 = ("vehicle", "kr-vs-kp", "pendigits")  # whole tables to one machine: hgb's last digits differ between the two
BAR = ("knn5", "knn10", "hgb", "lgbm")
STRONG = ("missforest", "condexp", "softimpute", "ice")
SCORES = ("sw_score", "dist_score", "impute_score")
P = "impute/induced"
DIRECTIONS = 2048  # 128 against 256 moved a fold of kc2_20nan by up to 0.03; 2048 against 4096 is checked in the ticket
SESOI = 0.01  # of the opponent's mean, as E181
CONTROL = 0.10  # the shuffled truths' sw_score a variant must reach for the score to be read on it
GATE = 0.005  # a refit whose impute_score is further than this from what E181's run logged is not read


def directions(seed: int, fold: int, width: int, count: int = DIRECTIONS) -> NDArray[np.float64]:
    """``count`` unit vectors, the same for every imputer of the fold."""
    drawn = np.random.default_rng([seed, fold]).normal(size=(count, width))
    return np.asarray(drawn / np.linalg.norm(drawn, axis=1, keepdims=True), dtype=np.float64)


def sliced(filled: NDArray[np.float64], truth: NDArray[np.float64], along: NDArray[np.float64]) -> float:
    """The sliced 1-Wasserstein distance between two sets of as many rows: over the directions, the mean gap
    between the sorted projections of the one and of the other."""
    return float(np.mean(np.abs(np.sort(filled @ along.T, axis=0) - np.sort(truth @ along.T, axis=0))))


def _fitter(group: str) -> Callable[[pd.DataFrame, Sequence[str], Sequence[str]], list[BaselineImputer]]:
    return {"adr0007": fit_baseline_imputers, "lgbm": fit_lightgbm_imputers, "missforest": fit_missforest_imputers,
            "linear": fit_linear_imputers}[group]


def _run_dir(variant: str, seed: int) -> Path:
    found = sorted((LEDGERS / "results" / f"{variant}_s{seed}_Kc" / variant).glob("*/metrics/cv_summary.json"))
    if not found:
        sys.exit(f"no finished Kc run of {variant} seed {seed} under {LEDGERS}")
    return found[-1].parent.parent


def _ledger(run: Path, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(run / "imputation" / f"imputation_fold_{fold}_cells.csv", dtype={"actual": object, "imputed": object})
    frame = frame[frame["population"] == "induced"].reset_index(drop=True)
    numeric = frame["kind"] == NUMERICAL
    for name in ("actual", "imputed"):
        frame.loc[numeric, name] = frame.loc[numeric, name].astype(float)
    return frame


def run(variant: str, seed: int, group: str) -> None:
    path = STATE / "csv" / f"{variant}_s{seed}__{group}.csv"
    if path.exists():
        print("SKIP", path.name)
        return
    warnings.filterwarnings("ignore")
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    classes = dataset.frame[dataset.label_column].astype(str).to_numpy()
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    is_number = np.array([column in numerical for column in features.columns])
    levels = {j: sorted(set(features.iloc[:, j].dropna().astype(str))) for j in np.flatnonzero(~is_number)}
    width = int(is_number.sum()) + sum(len(v) for v in levels.values())

    def encode(block: NDArray[np.object_]) -> NDArray[np.float64]:
        """Rows as the score sees them: numbers in the scaled space, each category one-hot over the table's
        categories and divided by sqrt(2), so that a wrong category is as far as one scaled unit."""
        parts = [block[:, is_number].astype(float)]
        for j, names in levels.items():
            codes = pd.Categorical(block[:, j].astype(str), categories=names).codes
            assert (codes >= 0).all(), f"{variant}: a category the table does not have, column {features.columns[j]}"
            parts.append(np.eye(len(names))[codes] / np.sqrt(2.0))
        return np.asarray(np.hstack(parts), dtype=np.float64)

    run_dir = _run_dir(variant, seed)
    logged = pd.read_csv(run_dir / "metrics" / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
    out: list[dict[str, object]] = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, 5, seed), start=1):
        train_index, test_index = np.asarray(fold.train_indices), np.asarray(fold.test_indices)
        train = features.iloc[train_index].reset_index(drop=True)
        test = features.iloc[test_index].reset_index(drop=True)
        naive = mean_mode_baselines(train, numerical, categorical)
        cells = _ledger(run_dir, ordinal)
        local = cells["row"].to_numpy(dtype=int)
        position = np.asarray(features.columns.get_indexer(pd.Index(cells["column"])))
        assert test.isna().to_numpy().sum() == len(cells) and test.isna().to_numpy()[local, position].all(), "the ledger does not cover the fold's gaps"
        kinds = cells["kind"].to_numpy()
        fills: dict[str, NDArray[np.object_]] = {}
        seconds = 0.0
        if group == "reference":
            by_class = class_mean_mode_baselines(train, classes[train_index], numerical, categorical)
            labels = classes[test_index][local]
            shuffled = cells["actual"].to_numpy(dtype=object).copy()
            order = np.random.default_rng([seed, ordinal, 1])
            for column in np.unique(position):
                where = np.flatnonzero(position == column)
                shuffled[where] = shuffled[order.permutation(where)]
            fills = {"trident": cells["imputed"].to_numpy(dtype=object), "mean_mode": cells["column"].map(naive).to_numpy(dtype=object),
                     "class_fill": np.array([by_class.get(str(label), naive)[column] for label, column in zip(labels, cells["column"])], dtype=object),
                     "shuffled": shuffled, "truth": cells["actual"].to_numpy(dtype=object)}
        else:
            hidden = np.zeros(test.shape, dtype=bool)
            hidden[local, position] = True
            started = time.monotonic()
            for imputer in _fitter(group)(train, numerical, categorical):
                guesses = imputer.impute(test, hidden).to_numpy(dtype=object)[local, position]
                fills[imputer.name] = np.array([float(g) if kind == NUMERICAL else str(g) for g, kind in zip(guesses, kinds)], dtype=object)
            seconds = time.monotonic() - started
            fills["mean_mode"] = cells["column"].map(naive).to_numpy(dtype=object)
        # the rows with a scored cell, as they are filled and as they truly are
        rows = np.unique(local)
        at = np.searchsorted(rows, local)
        base = test.iloc[rows].to_numpy(dtype=object)

        def rows_of(values: NDArray[np.object_]) -> NDArray[np.float64]:
            block = base.copy()
            block[at, position] = values
            return encode(block)

        truth = rows_of(cells["actual"].to_numpy(dtype=object))
        along = directions(seed, ordinal, width)
        assert truth.shape[1] == width
        distance = {name: sliced(rows_of(values), truth, along) for name, values in fills.items()}
        for name, values in fills.items():
            if name == "mean_mode" and group != "reference":
                continue
            scores = _scores(cells.assign(imputed=values), naive)
            key = f"{P}/impute_score" if name == "trident" else f"{P}/baseline/{name}/impute_score"
            gap = abs(scores["impute_score"] - float(logged[key].iloc[ordinal - 1])) if key in logged.columns else float("nan")
            out.append({"variant": variant, "seed": seed, "fold": ordinal, "group": group, "method": name, "machine": socket.gethostname(),
                        "impute_score": scores["impute_score"], "dist_score": scores["dist_score"], "sw": distance[name],
                        "sw_score": _ratio(distance[name], distance["mean_mode"]), "gate": gap, "rows": len(rows), "seconds": seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print("DONE", path.name, flush=True)


def jobs(machine: str) -> None:
    mine = [t for t in TABLES if (t in GORGONA8) == (machine == "gorgona8")]
    for table in mine:
        for group in GROUPS:
            for variant in VARIANTS:
                if common.table_of(variant) == table:
                    for seed in SEEDS:
                        print(variant, seed, group)


def tally() -> None:
    wanted = [STATE / "csv" / f"{v}_s{s}__{g}.csv" for v in VARIANTS for s in SEEDS for g in GROUPS]
    missing = [p.name for p in wanted if not p.exists()]
    if missing:
        print(f"incomplete: {len(missing)} of {len(wanted)} files missing, e.g. {missing[:5]}")
        return
    frame = pd.concat([pd.read_csv(p) for p in wanted], ignore_index=True)
    gaps = frame.groupby("method")["gate"].max().dropna()
    print("Gate, the largest gap between a refit's impute_score and the one E181's run logged: " + "; ".join(f"{m} {g:.2e}" for m, g in gaps.items()))
    assert frame.loc[frame["method"] == "truth", "sw"].max() == 0.0 and np.allclose(frame.loc[frame["method"] == "mean_mode", "sw_score"], 1.0)
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        part = frame[frame["variant"] == variant]

        def folds_of(method: str, score: str) -> list[NDArray[np.float64]]:
            rows = part[part["method"] == method].sort_values(["seed", "fold"])
            return [rows.loc[rows["seed"] == s, score].to_numpy(dtype=float) for s in SEEDS]

        def level(method: str, score: str) -> float:
            return float(np.concatenate(folds_of(method, score)).mean())

        failed = sorted(m for m in BAR + STRONG if float(part.loc[part["method"] == m, "gate"].max()) > GATE)
        call = calls.setdefault(variant, {"control": "seen" if level("shuffled", "sw_score") >= CONTROL else "blind"})
        print(f"- `{variant}`: TRIDENT sw_score {level('trident', 'sw_score'):.4f} (dist_score {level('trident', 'dist_score'):.4f}, impute_score {level('trident', 'impute_score'):.4f}); "
              f"control {level('shuffled', 'sw_score'):.4f} ({call['control']}; its dist_score {level('shuffled', 'dist_score'):.4f}); class fill {level('class_fill', 'sw_score'):.4f}"
              + (f"; refits outside the gate, not read: {failed}" if failed else ""))
        for label, names in (("bar", BAR), ("strong", STRONG)):
            for score in SCORES:
                have = [n for n in names if n not in failed]
                if not have:
                    continue
                name = min(have, key=lambda n: level(n, score))
                theirs = folds_of(name, score)
                v = common.verdict([o - b for o, b in zip(folds_of("trident", score), theirs)], sesoi=SESOI * level(name, score))
                call[f"{label}/{score}"] = v.call
                print(f"  - TRIDENT − {label} ({name}) on {score} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
    seen = [v for v in VARIANTS if calls[v]["control"] == "seen"]
    print(f"\nControl: the score sees the rows on {len(seen)} of {len(VARIANTS)} variants; blind on {[v for v in VARIANTS if v not in seen]}")
    worse = [v for v in seen if calls[v].get("strong/sw_score") == "worse"]
    reading = ("inconclusive (the control passes on fewer than 13 variants)" if len(seen) < 13 else "supported" if 4 * len(worse) <= len(seen)
               else "refuted" if 2 * len(worse) >= len(seen) else "mixed")
    print(f"Primary, TRIDENT against the best strong baseline on sw_score, the {len(seen)} variants the control passes: "
          f"{common.tally(calls[v]['strong/sw_score'] for v in seen if 'strong/sw_score' in calls[v])}; worse on {worse} -> {reading}")
    for label in ("bar", "strong"):
        for score in SCORES:
            print(f"Secondary, TRIDENT against the {label} on {score}, all variants: {common.tally(calls[v][f'{label}/{score}'] for v in VARIANTS if f'{label}/{score}' in calls[v])}")
        for other in ("dist_score", "impute_score"):
            moved = [v for v in VARIANTS if calls[v].get(f"{label}/sw_score") != calls[v].get(f"{label}/{other}")]
            print(f"Secondary, {label}: calls on sw_score that differ from {other}'s: {moved}")


def main(argv: list[str]) -> int:
    if argv[:1] == ["run"] and len(argv) == 4 and argv[3] in GROUPS:
        run(argv[1], int(argv[2]), argv[3])
    elif argv[:1] == ["--jobs"] and len(argv) == 2 and argv[1] in ("medusa", "gorgona8"):
        jobs(argv[1])
    elif argv == ["--tally"]:
        tally()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
