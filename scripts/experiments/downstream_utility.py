"""E146 (ticket imputation-protocol/52), a pilot: are K's fills more useful to a downstream classifier when the
decode checkpoint is chosen by ``impute_score`` (arm A) or by ``dist_score`` (arm B)?

    DOWNSTREAM_OUT=<dir> uv run --python 3.11 python -B <this file> run <variant> <seed>     # one CSV under <dir>/csv
    DOWNSTREAM_OUT=<dir> uv run --python 3.11 python -B <this file> tally

It trains nothing of TRIDENT's. For each of E145's (variant, seed) pairs it rebuilds the five folds, reads both
arms' fold ledgers and assembles four tables in the scaled space the ledgers use: every gap filled with the truth
(the ledger's ``actual``), with arm A's fill, with arm B's, and with the fold's mean/mode. A fixed classifier is
then fitted per fold:

- ``complete`` (primary): on the truth-filled rows outside the test fold, scored on the test fold under each of
  the four fills;
- ``own`` (beside): on a fill's own table outside the test fold, scored on that table's test fold.

Run from the checkout root; the ledgers are E145's (``decode_metric_run.STATE``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[attr-defined]

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import c15_report, decode_metric_report as e145, decode_metric_run as cells  # noqa: E402
from src.training.data import build_folds, prepare_dataset  # noqa: E402
from src.training.imputation_metrics import mean_mode_baselines  # noqa: E402
from src.training.types import DatasetSpec  # noqa: E402

if not os.environ.get("DOWNSTREAM_OUT"):
    sys.exit("set DOWNSTREAM_OUT: the directory that holds the study's CSVs, outside the checkout")
OUT = Path(os.environ["DOWNSTREAM_OUT"]) / "csv"
FILLS = ("truth", "A", "B", "naive")
LEARNERS = ("logistic", "mlp")


def _learner(name: str, seed: int, categorical: list[str], numerical: list[str]) -> Any:
    encode = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", "passthrough", numerical)])
    model = (LogisticRegression(max_iter=5000) if name == "logistic"
             else MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed))
    return make_pipeline(encode, model)


def run(variant: str, seed: int) -> None:
    path = OUT / f"{variant}_s{seed}.csv"
    if path.exists():
        print("SKIP", path.name)
        return
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    labels = dataset.frame[dataset.label_column].reset_index(drop=True)
    assert labels.notna().all(), "a label is missing"
    features = dataset.frame.drop(columns=[dataset.label_column]).reset_index(drop=True)
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = list(build_folds(dataset.frame, dataset.label_column, 5, seed))
    tables = {fill: features.astype(object) for fill in FILLS}
    metric: list[dict[str, float]] = []
    for ordinal, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_indices)
        naive = mean_mode_baselines(features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True), numerical, categorical)
        a, b = e145._ledger(cells.STATE, "A", variant, seed, ordinal), e145._ledger(cells.STATE, "B", variant, seed, ordinal)
        assert a is not None and b is not None, f"fold {ordinal}: a ledger is missing"
        assert a[["row", "column"]].equals(b[["row", "column"]]) and a["actual"].astype(str).equals(b["actual"].astype(str))
        rows = test[a["row"].to_numpy(dtype=int)]
        position = features.columns.get_indexer(pd.Index(a["column"]))
        fills = {"truth": a["actual"], "A": a["imputed"], "B": b["imputed"], "naive": a["column"].map(naive)}
        for fill, values in fills.items():
            block = tables[fill].to_numpy(dtype=object, copy=True)
            block[rows, position] = values.to_numpy(dtype=object)
            tables[fill] = pd.DataFrame(block, columns=features.columns)
        sa, sb = e145._scores(a, naive), e145._scores(b, naive)
        metric.append({f"{name}_{arm}": s[name] for arm, s in (("A", sa), ("B", sb)) for name in ("impute_score", "dist_score")})
    for fill in FILLS:
        assert not tables[fill].isna().any().any(), f"{fill}: a gap is left unfilled"
        tables[fill][numerical] = tables[fill][numerical].astype(float)
        tables[fill][categorical] = tables[fill][categorical].astype(str)
    out: list[dict[str, object]] = []
    everything = np.arange(len(features))
    for ordinal, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_indices)
        train = np.setdiff1d(everything, test)
        for name in LEARNERS:
            scored: dict[tuple[str, str], float] = {}
            model = _learner(name, seed, categorical, numerical).fit(tables["truth"].iloc[train], labels.iloc[train])
            for fill in FILLS:
                scored[("complete", fill)] = f1_score(labels.iloc[test], model.predict(tables[fill].iloc[test]), average="macro")
            for fill in ("A", "B", "naive"):
                own = _learner(name, seed, categorical, numerical).fit(tables[fill].iloc[train], labels.iloc[train])
                scored[("own", fill)] = f1_score(labels.iloc[test], own.predict(tables[fill].iloc[test]), average="macro")
            out.extend({"variant": variant, "seed": seed, "fold": ordinal, "learner": name, "protocol": protocol, "fill": fill,
                        "macro_f1": float(value), **metric[ordinal - 1]} for (protocol, fill), value in scored.items())
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print("DONE", path.name, flush=True)


def _line(name: str, d: np.ndarray) -> str:
    m, lo, hi = c15_report.interval(d, False)
    _, nlo, nhi = c15_report.interval(d)
    return f"{name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (NB [{nlo:+.4f}, {nhi:+.4f}]) -> {'B' if lo > 0 else 'A' if hi < 0 else 'no difference'}"


def tally() -> int:
    files = sorted(OUT.glob("*.csv"))
    expected = len(cells.VARIANTS) * len(cells.SEEDS)
    if len(files) != expected:
        print(f"incomplete: {len(files)} of {expected} files")
        return 1
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["variant", "seed", "fold"]
    print("## Downstream utility (E146): macro-F1 of a fixed classifier on E145's test folds under each fill of their gaps; "
          "B − A, 15 replicates a variant, pooled over the 12 variants per replicate; plain 95% t interval")

    def wide(protocol: str, learners: tuple[str, ...]) -> pd.DataFrame:
        part = data[(data["protocol"] == protocol) & data["learner"].isin(learners)]
        return part.groupby([*key, "fill"])["macro_f1"].mean().unstack("fill").reset_index()

    def pooled(frame: pd.DataFrame, column: pd.Series) -> np.ndarray:
        frame = frame.assign(d=column.to_numpy())
        return frame.pivot_table(index=["seed", "fold"], columns="variant", values="d").mean(axis=1).to_numpy()

    primary = wide("complete", LEARNERS)
    print(_line("Primary, classifier fitted on the complete rows, mean of the two learners, F1(B) − F1(A)", pooled(primary, primary["B"] - primary["A"])))
    for name in LEARNERS:
        one = wide("complete", (name,))
        print(_line(f"  {name}", pooled(one, one["B"] - one["A"])))
    own = wide("own", LEARNERS)
    print(_line("Beside, classifier fitted on each fill's own table, mean of the two learners, F1(B) − F1(A)", pooled(own, own["B"] - own["A"])))
    means = primary[list(FILLS)].mean()
    span = means["truth"] - means["naive"]
    print(f"Levels (complete, mean of learners): truth {means['truth']:.4f}, A {means['A']:.4f}, B {means['B']:.4f}, naive {means['naive']:.4f}; "
          f"share of the naive-to-truth gap closed: A {(means['A'] - means['naive']) / span:+.2f}, B {(means['B'] - means['naive']) / span:+.2f}")
    own_means = own[["A", "B", "naive"]].mean()
    print(f"Levels (own): A {own_means['A']:.4f}, B {own_means['B']:.4f}, naive {own_means['naive']:.4f}")

    scores = data.drop_duplicates(key)[[*key, "impute_score_A", "impute_score_B", "dist_score_A", "dist_score_B"]]
    joined = primary.merge(scores, on=key)
    joined["d_f1"] = joined["B"] - joined["A"]
    joined["d_impute"] = joined["impute_score_B"] - joined["impute_score_A"]
    joined["d_dist"] = joined["dist_score_B"] - joined["dist_score_A"]
    print("\n### Per variant (complete, mean of learners): which score pointed at the more useful fills")
    right = {"impute_score": 0, "dist_score": 0, "neither": 0}
    for variant in cells.VARIANTS:
        part = joined[joined["variant"] == variant]
        m, lo, hi = c15_report.interval(part["d_f1"].to_numpy(), False)
        verdict = "B" if lo > 0 else "A" if hi < 0 else "no difference"
        di, dd = part["d_impute"].mean(), part["d_dist"].mean()
        said_impute, said_dist = ("B" if di < 0 else "A"), ("B" if dd < 0 else "A")
        agree = [name for name, said in (("impute_score", said_impute), ("dist_score", said_dist)) if said == verdict]
        for name in agree or (["neither"] if verdict != "no difference" else []):
            right[name] += 1
        print(f"- {variant}: F1(B) − F1(A) {m:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}; impute_score said {said_impute} ({di:+.4f}), "
              f"dist_score said {said_dist} ({dd:+.4f}); truth {part['truth'].mean():.3f} A {part['A'].mean():.3f} B {part['B'].mean():.3f} naive {part['naive'].mean():.3f}")
    print(f"Variants with a verdict where each score was on the right side: {right}")

    print("\n### Replicate-level Spearman of F1(B) − F1(A) with each score's own B − A, sign turned so that + means agreement")
    rng = np.random.default_rng(0)
    for name, column in (("impute_score", "d_impute"), ("dist_score", "d_dist")):
        rho = float(stats.spearmanr(joined["d_f1"], -joined[column]).statistic)
        draws: list[float] = []
        for _ in range(2000):
            sample = pd.concat([joined[joined["variant"] == v] for v in rng.choice(cells.VARIANTS, size=len(cells.VARIANTS))])
            draws.append(float(stats.spearmanr(sample["d_f1"], -sample[column]).statistic))
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        print(f"- {name}: rho {rho:+.3f} [{lo:+.3f}, {hi:+.3f}] (bootstrap over variants)")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["run"] and len(argv) == 3:
        run(argv[1], int(argv[2]))
    elif argv == ["tally"]:
        raise SystemExit(tally())
    else:
        sys.exit(__doc__)
