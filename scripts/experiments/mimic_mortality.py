"""E150: fills and gap patterns judged by a mortality classifier (docs/tickets/mimic/05-downstream-mortality.md).

    QUEUE_STATE=<E148's state directory> TRIDENT_DATASETS_ROOT=<root> \\
        uv run --python 3.11 python -B scripts/experiments/mimic_mortality.py

No training. For each table, seed and fold of E148 it reads the fold's export, the decoder's fills of the natural gaps
and the fold's partition, and fits one fixed logistic regression on the fold's training rows under four inputs:

  N   the mean/mode fill          NI  N and one indicator per column with gaps
  T   the decoder's fills         TI  T and the indicators

The mean/mode fill and the scaler come from the fold's training rows and the score is on its test rows; the
validation rows, on which the decoder chose its epoch, are left out. Writes one row per (table, seed, fold, input) to
``<state>/mortality.csv`` and prints the levels and the paired contrasts: ``T − N`` and ``NI − N`` on both tables as
one Holm family of four, ``TI − NI`` beside them. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # type: ignore[attr-defined]

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments.report_common import holm  # noqa: E402

LABEL = "class"
METRICS = ("macro_f1", "auroc", "auprc")
FAMILY = (("T", "N"), ("NI", "N"))  # the primary, one Holm family over the tables
BESIDE = (("TI", "NI"),)


def inputs(features: pd.DataFrame, fills: pd.DataFrame, train: np.ndarray) -> dict[str, pd.DataFrame]:
    """The four tables a fold's classifier reads: numerical columns standardised, categories one-hot."""
    numerical = [name for name in features.columns if pd.api.types.is_numeric_dtype(features[name])]
    categorical = [name for name in features.columns if name not in numerical]
    naive = features.copy()
    naive[numerical] = naive[numerical].fillna(features.iloc[train][numerical].mean())
    for name in categorical:
        naive[name] = naive[name].fillna(features.iloc[train][name].mode().iloc[0])

    assert len(fills) == int(features.isna().sum().sum()) and not fills.duplicated(["row", "column"]).any()
    decoded = features.copy()
    for column, cells in fills.groupby("column"):
        rows = decoded.index[cells["row"].to_numpy()]
        assert decoded.loc[rows, str(column)].isna().all(), f"a fill of {column} lands on a measured cell"
        values = cells["imputed_original_units"].astype(float) if column in numerical else cells["imputed"].astype(str)
        decoded.loc[rows, str(column)] = values.to_numpy()
    assert not decoded.isna().any().any(), "a gap has no fill"

    def encoded(frame: pd.DataFrame) -> pd.DataFrame:
        fitted = frame.iloc[train][numerical]
        scaled = (frame[numerical] - fitted.mean()) / fitted.std().replace(0.0, 1.0)
        return pd.concat([scaled, pd.get_dummies(frame[categorical], dtype=float)], axis=1)

    gapped = [name for name in features.columns if features[name].isna().any()]
    flags = features[gapped].isna().astype(float).add_suffix("_missing")
    return {
        "N": encoded(naive), "NI": pd.concat([encoded(naive), flags], axis=1),
        "T": encoded(decoded), "TI": pd.concat([encoded(decoded), flags], axis=1),
    }


def score(table: pd.DataFrame, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    """The ticket's classifier, fitted on the training rows and scored on the test rows."""
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(table.iloc[train], labels[train])
    probability = model.predict_proba(table.iloc[test])[:, 1]
    return {
        "macro_f1": float(f1_score(labels[test], model.predict(table.iloc[test]), average="macro")),
        "auroc": float(roc_auc_score(labels[test], probability)),
        "auprc": float(average_precision_score(labels[test], probability)),
    }


def run() -> pd.DataFrame:
    import os

    from scripts.experiments import mimic_run  # needs QUEUE_STATE and TRIDENT_DATASETS_ROOT

    rows = []
    for variant in mimic_run.VARIANTS:
        base = variant.split("_", 1)[0]
        table = pd.read_csv(Path(os.environ["TRIDENT_DATASETS_ROOT"]) / base / f"{variant}.csv")
        features, labels = table.drop(columns=[LABEL]), table[LABEL].to_numpy()
        for seed in mimic_run.SEEDS:
            partitions = sorted((mimic_run.cell_dir("M", variant, seed) / variant).glob("*/imputation/*_partition.csv"))
            if not mimic_run.done("M", variant, seed) or len(partitions) != 5:
                sys.exit(f"{variant} seed {seed} is not done: the ticket reads only with every cell done")
            for fold, path in enumerate(partitions, start=1):
                split = pd.read_csv(path).set_index("row")["split"].reindex(range(len(table)))
                assert split.notna().all(), f"{path} does not cover the table"
                fills = pd.read_csv(path.with_name(path.name.replace("_partition", "_gap_fills")))
                train, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
                for name, frame in inputs(features, fills, train).items():
                    rows.append({"table": base, "seed": seed, "fold": fold, "input": name,
                                 **score(frame, labels, train, test)})
                print("scored", base, seed, fold, flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(mimic_run.STATE / "mortality.csv", index=False)
    return scores


def tally(scores: pd.DataFrame) -> None:
    print(scores.groupby(["table", "input"])[list(METRICS)].mean().round(4).to_string())
    wide = scores.pivot(index=["table", "seed", "fold"], columns="input", values="macro_f1")
    contrasts = {
        (table, a, b): (wide.loc[table, a] - wide.loc[table, b]).to_numpy()
        for table in wide.index.get_level_values("table").unique() for a, b in FAMILY + BESIDE
    }
    adjusted = holm({f"{table} {a}-{b}": float(stats.ttest_1samp(d, 0.0).pvalue)
                     for (table, a, b), d in contrasts.items() if (a, b) in FAMILY})
    for (table, a, b), d in contrasts.items():
        low, high = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        line = f"{table} {a} - {b} (macro-F1, n={len(d)}): {d.mean():+.4f} [{low:+.4f}, {high:+.4f}]"
        if (a, b) in FAMILY:
            p = adjusted[f"{table} {a}-{b}"]
            line += f", Holm p {p:.4f} -> {'holds' if p < 0.05 and d.mean() > 0 else 'does not hold'}"
        print(line)


if __name__ == "__main__":
    tally(run())
