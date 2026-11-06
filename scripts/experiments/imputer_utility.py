"""E147 (ticket imputation-protocol/53; pre-registered and run as E149), a pilot: which score orders imputers as a downstream classifier does,
``impute_score`` or ``dist_score``, across TRIDENT and the baselines and across missingness mechanisms.

    DOWNSTREAM_OUT=<dir> python -B <this file> run <state_dir> <variant> <seed> <group>     # one CSV under <dir>/csv_imputers
    DOWNSTREAM_OUT=<dir> python -B <this file> tally

``state_dir`` holds the variant's TRIDENT cells in E145's layout (arms A and B, five fold ledgers each). A group
is a set of imputers: ``trident`` (arm A, the checkpoint by impute_score; arm B, by dist_score; the mean/mode
fill; the truth), ``adr0007`` (knn5, knn10, hgb), ``lgbm``, ``missforest``, ``linear`` (condexp, softimpute, ice)
and ``tabpfn`` (needs the extra and a GPU). Each baseline is fitted on the fold's
training rows and asked for exactly the ledger's cells, in the scaled space, as ``metric_panel.py`` does. Per
(fold, imputer): ``impute_score``, ``dist_score`` and its parts, the dispersion, and the macro-F1 of logistic
regression and a small MLP, fitted on the truth-filled rows outside the test fold (``complete``) or on the
imputer's own filled rows (``own``), scored on the test fold as the imputer filled it.

Run from the checkout root.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report, decode_metric_report as e145, downstream_utility as e146  # noqa: E402
from src.training import lightgbm_baseline, missforest  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, fit_baseline_imputers  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines  # noqa: E402
from src.training.lightgbm_baseline import fit_lightgbm_imputers  # noqa: E402
from src.training.linear_baselines import fit_linear_imputers  # noqa: E402
from src.training.missforest import fit_missforest_imputers  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

OUT = e146.OUT.parent / "csv_imputers"  # DOWNSTREAM_OUT, required by the import above
THREADS = int(os.environ.get("IMPUTER_THREADS", "4"))
GROUPS = ("trident", "adr0007", "lgbm", "missforest", "linear", "tabpfn")
LEARNERS = ("logistic", "mlp")
SCORES = ("impute_score", "dist_score", "wd_num_z", "tv_cat", "dispersion")
STRONG = ("missforest", "condexp", "softimpute", "ice", "tabpfn")  # ADR 0021's strong tier and ADR 0028's TabPFN


def _fitter(group: str) -> Callable[[pd.DataFrame, Sequence[str], Sequence[str]], list[BaselineImputer]]:
    if group == "tabpfn":
        from src.training.tabpfn_baseline import fit_tabpfn_imputers

        return fit_tabpfn_imputers
    return {"adr0007": fit_baseline_imputers, "lgbm": fit_lightgbm_imputers, "missforest": fit_missforest_imputers,
            "linear": fit_linear_imputers}[group]


def run(state: Path, variant: str, seed: int, group: str) -> None:
    path = OUT / f"{variant}_s{seed}__{group}.csv"
    if path.exists():
        print("SKIP", path.name)
        return
    warnings.filterwarnings("ignore")
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    labels = dataset.frame[dataset.label_column].reset_index(drop=True)
    assert labels.notna().all(), "a label is missing"
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = list(build_folds(dataset.frame, dataset.label_column, 5, seed))
    blank = features.to_numpy(dtype=object, copy=True)
    blocks: dict[str, np.ndarray] = {"truth": blank.copy()}
    scored: dict[tuple[int, str], dict[str, float]] = {}
    for ordinal, fold in enumerate(folds, start=1):
        test_index = np.asarray(fold.test_indices)
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[test_index].reset_index(drop=True)
        naive = mean_mode_baselines(train, numerical, categorical)
        a = e145._ledger(state, "A", variant, seed, ordinal)
        assert a is not None, f"fold {ordinal}: arm A's ledger is missing"
        local = a["row"].to_numpy(dtype=int)
        position = features.columns.get_indexer(pd.Index(a["column"]))
        assert test.isna().to_numpy().sum() == len(a) and test.isna().to_numpy()[local, position].all(), "the ledger does not cover the fold's gaps"
        rows = test_index[local]
        blocks["truth"][rows, position] = a["actual"].to_numpy(dtype=object)
        fills: dict[str, np.ndarray] = {}
        if group == "trident":
            b = e145._ledger(state, "B", variant, seed, ordinal)
            assert b is not None and a[["row", "column"]].equals(b[["row", "column"]]), f"fold {ordinal}: the arms' ledgers differ"
            fills = {"trident_A": a["imputed"].to_numpy(dtype=object), "trident_B": b["imputed"].to_numpy(dtype=object),
                     "naive": a["column"].map(naive).to_numpy(dtype=object)}
        else:
            hidden = np.zeros(test.shape, dtype=bool)
            hidden[local, position] = True
            for imputer in _fitter(group)(train, numerical, categorical):
                guesses = imputer.impute(test, hidden).to_numpy(dtype=object)[local, position]
                fills[imputer.name] = np.array([float(g) if kind == NUMERICAL else str(g) for g, kind in zip(guesses, a["kind"])], dtype=object)
        for name, values in fills.items():
            scored[(ordinal, name)] = e145._scores(a.assign(imputed=values), naive)
            blocks.setdefault(name, blank.copy())[rows, position] = values
    tables: dict[str, pd.DataFrame] = {}
    for name, block in blocks.items():
        table = pd.DataFrame(block, columns=features.columns)
        assert not table.isna().any().any(), f"{name}: a gap is left unfilled"
        table[numerical] = table[numerical].astype(float)
        table[categorical] = table[categorical].astype(str)
        tables[name] = table
    methods = [name for name in tables if name != "truth" or group == "trident"]
    out: list[dict[str, object]] = []
    everything = np.arange(len(features))
    for ordinal, fold in enumerate(folds, start=1):
        test_index = np.asarray(fold.test_indices)
        outside = np.setdiff1d(everything, test_index)
        f1: dict[str, dict[str, float]] = {name: {} for name in methods}
        for learner in LEARNERS:
            complete = e146._learner(learner, seed, categorical, numerical).fit(tables["truth"].iloc[outside], labels.iloc[outside])
            for name in methods:
                f1[name][f"f1_complete_{learner}"] = float(f1_score(labels.iloc[test_index], complete.predict(tables[name].iloc[test_index]), average="macro"))
                if name == "truth":
                    f1[name][f"f1_own_{learner}"] = f1[name][f"f1_complete_{learner}"]
                    continue
                own = e146._learner(learner, seed, categorical, numerical).fit(tables[name].iloc[outside], labels.iloc[outside])
                f1[name][f"f1_own_{learner}"] = float(f1_score(labels.iloc[test_index], own.predict(tables[name].iloc[test_index]), average="macro"))
        for name in methods:
            scores = scored.get((ordinal, name), {"impute_score": 0.0, "dist_score": 0.0})  # the truth scores zero by definition
            out.append({"variant": variant, "seed": seed, "fold": ordinal, "group": group, "method": name,
                        **{score: scores.get(score, float("nan")) for score in SCORES}, **f1[name]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print("DONE", path.name, flush=True)


def _mechanism(variant: str) -> str:
    return "MAR" if "_marf_" in variant else "MNAR" if "_probit_" in variant else "MCAR"


def _line(name: str, d: np.ndarray, words: tuple[str, str]) -> str:
    m, lo, hi = c15_report.interval(d, False)
    _, nlo, nhi = c15_report.interval(d)
    return f"{name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]) -> {words[0] if lo > 0 else words[1] if hi < 0 else 'no difference'}"


def _pooled(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A per-(variant, seed, fold) quantity averaged over the variants, one value a (seed, fold) replicate."""
    return frame.pivot_table(index=["seed", "fold"], columns="variant", values=column).mean(axis=1).to_numpy()


def tally() -> int:
    files = sorted(OUT.glob("*.csv"))
    if not files:
        print("no files")
        return 1
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data = data[data["method"] != "mean_mode"]  # the weak tier's mean/mode is the trident group's ``naive`` again
    data["mechanism"] = data["variant"].map(_mechanism)
    data["f1"] = data[[f"f1_complete_{name}" for name in LEARNERS]].mean(axis=1)
    data["f1_own"] = data[[f"f1_own_{name}" for name in LEARNERS]].mean(axis=1)
    key = ["variant", "seed", "fold"]
    print("## Imputers, scores and downstream use (E147): per fold, every imputer's impute_score, dist_score and the macro-F1 of a "
          "classifier fitted on complete rows and scored on the test fold as that imputer filled it (mean of logistic regression and an MLP)")
    primaries: dict[str, np.ndarray] = {}
    for mechanism in ("MCAR", "MCAR at 60nan", "MAR", "MNAR"):
        part = data[data["mechanism"] == mechanism.split()[0]]
        if mechanism == "MCAR at 60nan":  # beside: the six tables at the level MAR and MNAR are read at
            part = part[part["variant"].str.endswith("_60nan")]
        if part.empty:
            print(f"\n### {mechanism}: no file yet")
            continue
        variants = sorted(part["variant"].unique())
        cells = part.groupby(key).ngroups
        counts = part.groupby("method").size()
        methods = sorted(str(m) for m, n in counts.items() if n == cells and m != "truth")
        partial = sorted(str(m) for m, n in counts.items() if n != cells)
        complete_seeds = part.groupby("variant")["seed"].nunique().eq(3).all() and (part.groupby(["variant", "seed"])["fold"].nunique() == 5).all()
        print(f"\n### {mechanism}: {len(variants)} variants, {cells} folds, {'complete' if complete_seeds else 'INCOMPLETE'}; "
              f"imputers on every fold: {', '.join(methods)}" + (f"; left out, not on every fold: {', '.join(partial)}" if partial else ""))
        part = part[part["method"].isin([*methods, "truth"])]
        levels = part.groupby("method")[["impute_score", "dist_score", "dispersion", "f1", "f1_own"]].mean()
        for method in levels.sort_values("f1", ascending=False, kind="stable").index:
            row = levels.loc[method]
            print(f"- {method}: impute_score {row['impute_score']:.4f}, dist_score {row['dist_score']:.4f}, dispersion {row['dispersion']:.3f}, "
                  f"F1 {row['f1']:.4f}, F1 own {row['f1_own']:.4f}")

        ranked = part[part["method"] != "truth"]
        rows: list[dict[str, object]] = []
        for (variant, seed, fold), group in ranked.groupby(key):
            best = group.loc[group["f1"].idxmax()]
            by_impute, by_dist = group.loc[group["impute_score"].idxmin()], group.loc[group["dist_score"].idxmin()]
            rows.append({"variant": variant, "seed": seed, "fold": fold,
                         "rho_impute": float(stats.spearmanr(-group["impute_score"], group["f1"]).statistic),
                         "rho_dist": float(stats.spearmanr(-group["dist_score"], group["f1"]).statistic),
                         "regret_impute": float(best["f1"] - by_impute["f1"]), "regret_dist": float(best["f1"] - by_dist["f1"]),
                         "top_impute": by_impute["method"] == best["method"], "top_dist": by_dist["method"] == best["method"]})
        order = pd.DataFrame(rows)
        order["d_rho"] = order["rho_dist"] - order["rho_impute"]
        order["d_regret"] = order["regret_impute"] - order["regret_dist"]
        print(f"Mean Spearman across the {len(methods)} imputers of a fold, score against F1: impute_score {order['rho_impute'].mean():+.3f}, dist_score {order['rho_dist'].mean():+.3f}")
        print(_line("Primary, rho(dist_score) − rho(impute_score)", _pooled(order, "d_rho"), ("dist_score orders them more like the classifier", "impute_score does")))
        if mechanism != "MCAR at 60nan" and complete_seeds:
            primaries[mechanism] = _pooled(order, "d_rho")
        print(_line("F1 lost by taking each score's best imputer, impute_score's loss − dist_score's", _pooled(order, "d_regret"), ("dist_score picks better", "impute_score picks better")))
        print(f"Mean F1 lost: by impute_score {order['regret_impute'].mean():.4f}, by dist_score {order['regret_dist'].mean():.4f}; "
              f"folds where the score's best imputer is the classifier's best: impute_score {int(order['top_impute'].sum())}/{len(order)}, dist_score {int(order['top_dist'].sum())}/{len(order)}")

        wide = {score: part.pivot_table(index=key, columns="method", values=score).reset_index() for score in ("impute_score", "dist_score", "f1")}
        for score, better in (("dist_score", "lower"), ("impute_score", "lower"), ("f1", "higher")):
            sign = -1.0 if better == "lower" else 1.0
            frame = wide[score].assign(d=sign * (wide[score]["trident_B"] - wide[score]["trident_A"]))
            print(_line(f"TRIDENT B against A on {score} (+ favours B)", _pooled(frame, "d"), ("B", "A")))
        strong = [m for m in STRONG if m in methods]
        if not strong:
            print("No strong baseline on every fold yet.")
            continue
        print(f"Strong baselines in: {', '.join(strong)}")
        for arm in ("trident_A", "trident_B"):
            for score in ("dist_score", "impute_score", "f1"):
                sign = -1.0 if score != "f1" else 1.0
                tallies = {"better": 0, "no difference": 0, "worse": 0}
                frames = []
                for variant in variants:
                    one = wide[score][wide[score]["variant"] == variant]
                    rival = min(strong, key=lambda m: sign * -one[m].mean())  # the strong baseline that is best on this score, on this variant
                    d = sign * (one[arm] - one[rival])
                    m, lo, hi = c15_report.interval(d.to_numpy(), False)
                    tallies["better" if lo > 0 else "worse" if hi < 0 else "no difference"] += 1
                    frames.append(one[key].assign(d=d.to_numpy()))
                pooled = _pooled(pd.concat(frames), "d")
                print(_line(f"{arm} against the best strong baseline of each variant on {score} (+ favours TRIDENT)", pooled, ("TRIDENT", "the baseline"))
                      + f"; variants better/no difference/worse: {tallies['better']}/{tallies['no difference']}/{tallies['worse']}")
    if primaries:
        p = {name: float(stats.ttest_1samp(d, 0.0).pvalue) for name, d in primaries.items()}
        by_p = sorted(p, key=lambda name: p[name])
        holm = {name: min(1.0, max(p[other] * (len(p) - j) for j, other in enumerate(by_p[: i + 1]))) for i, name in enumerate(by_p)}
        print("\n### The primary over the mechanisms read complete, Holm over them")
        for name in primaries:
            m = float(primaries[name].mean())
            verdict = "no difference" if holm[name] >= 0.05 else "dist_score" if m > 0 else "impute_score"
            print(f"- {name}: rho(dist_score) − rho(impute_score) {m:+.4f}, p {p[name]:.4f}, Holm {holm[name]:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["run"] and len(argv) == 5 and argv[4] in GROUPS:
        run(Path(argv[1]), argv[2], int(argv[3]), argv[4])
    elif argv == ["tally"]:
        raise SystemExit(tally())
    else:
        sys.exit(__doc__)
