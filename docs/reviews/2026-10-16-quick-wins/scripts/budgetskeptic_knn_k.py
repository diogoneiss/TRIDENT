"""Read-only: induced impute_score of KNN baselines at k in {5,10,20,50} on each variant's
own folds (same construction as scripts/backfill_baseline_imputers.recompute_folds),
induced population only. Writes nothing."""
import sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from src.embedder import as_category_strings
from src.training.data import build_folds, in_variant_dtypes, load_complete_sibling, prepare_dataset
from src.training.imputation_baselines import KnnImputer, MeanModeImputer, score_baselines
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines
from src.training.types import DatasetSpec
import pandas as pd

KS = [int(k) for k in os.environ.get("KS", "5,10,20,50").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
VARIANTS = sys.argv[1:]

def cells_table(positions, columns, numbers, categories):
    rows = []
    for row, position in zip(*np.nonzero(positions)):
        column = columns[position]
        if column in categories:
            actual = str(categories[column][row]); kind = CATEGORICAL
        else:
            actual = float(np.float32(numbers[column][row])); kind = NUMERICAL
        rows.append({"row": int(row), "column": column, "kind": kind, "population": "induced",
                     "actual": actual, "imputed": actual, "confidence": float("nan"), "actual_original": actual})
    return pd.DataFrame(rows, columns=["row", "column", "kind", "population", "actual", "imputed", "confidence", "actual_original"])

def values(frame, ds):
    return ({c: frame[c].to_numpy(dtype=float) for c in ds.numerical_columns},
            {c: as_category_strings(frame[c]) for c in ds.categorical_columns})

import contextlib, io
out = {}
for name in VARIANTS:
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        spec = DatasetSpec.from_name(name, None)
        ds = prepare_dataset(spec)
        sib = load_complete_sibling(spec)
    feats = ds.frame.drop(columns=[ds.label_column])
    num, cat = list(ds.numerical_columns), list(ds.categorical_columns)
    per = {}
    for seed in SEEDS:
        for ordinal, fold in enumerate(build_folds(ds.frame, ds.label_column, 5, seed), start=1):
            train = feats.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
            test = feats.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
            naive = mean_mode_baselines(train, num, cat)
            imps = [MeanModeImputer(num, cat)] + [KnnImputer(num, cat, n_neighbors=k) for k in KS]
            for i in imps:
                i.fit(train)
            columns = list(test.columns)
            truth = in_variant_dtypes(sib.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)[columns].copy(), test, cat)
            if ds.scaler is not None and num:
                truth[num] = ds.scaler.transform(truth[num])
            gaps = (test.isna() & truth.notna()).to_numpy()
            tn, tc = values(truth, ds)
            cells = cells_table(gaps, columns, tn, tc)
            m = score_baselines(imps, test, cells, naive).metrics
            for i in imps:
                per.setdefault(i.name, []).append(m[f"baseline/{i.name}/impute_score"])
    out[name] = {k: float(np.mean(v)) for k, v in per.items()}
    print(name, f"{time.time()-t0:.0f}s", " ".join(f"{k}={v:.4f}" for k, v in out[name].items()), flush=True)
