"""Read-only: split stored HEAD-era cell ledgers (induced population) by exact-twin flag.
Usage: variant parent_run_id[,..] ; knn on/off. Prints per fold: n cells, twin share, model score all/twin/notwin,
logged model fold score (check), replicated knn5/knn10 all/twin/notwin and logged knn fold score (check)."""
import sqlite3, sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
T = Path(r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT")
con = sqlite3.connect(f"file:{T/'mlflow.db'}?mode=ro", uri=True)
def folds(y, k, seed):
    out = []
    for trv, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=max(0.01, min(0.5, 0.1/(1-1/k))), random_state=seed).split(np.zeros(len(trv)), y[trv]))
        out.append((trv[a], trv[b], te))
    return out
def ratio(p, a, n):
    return np.sqrt(np.mean((p-a)**2)) / np.sqrt(np.mean((n-a)**2))
variant, parents, do_knn = sys.argv[1], sys.argv[2].split(","), sys.argv[3] == "knn"
base = variant.rsplit("_", 1)[0]
var = pd.read_csv(T/"datasets/processed_datasets"/base/f"{variant}.csv")
full = pd.read_csv(T/"datasets/processed_datasets"/base/f"{base}_00nan.csv")
feats = [c for c in var.columns if c != "class"]
y = LabelEncoder().fit_transform(var["class"])
X = var[feats].to_numpy(float); F = full[feats].to_numpy(float)
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd[sd == 0] = 1
Z = (X-mu)/sd; FZ = (F-mu)/sd
key = pd.util.hash_pandas_object(full[feats], index=False).to_numpy()
col = {c: j for j, c in enumerate(feats)}
out = []
for parent in parents:
    seed = int(con.execute("select value from params where run_uuid=? and key in ('seed','SEED')", (parent,)).fetchone()[0])
    exp = con.execute("select value from tags where run_uuid=? and key='experiment'", (parent,)).fetchone()
    arm = con.execute("select value from tags where run_uuid=? and key='arm'", (parent,)).fetchone()
    puri = con.execute("select artifact_uri from runs where run_uuid=?", (parent,)).fetchone()[0].replace("file:///", "")
    raw = pd.read_csv(Path(puri)/"metrics/raw_fold_metrics.csv")
    fl = folds(y, 5, seed)
    kids = con.execute("select r.artifact_uri from runs r join tags t on t.run_uuid=r.run_uuid where t.key='mlflow.parentRunId' and t.value=?", (parent,)).fetchall()
    for (uri,) in kids:
        p = Path(uri.replace("file:///", ""))/"imputation"
        for f in (p.glob("imputation_fold_*_cells.csv") if p.exists() else []):
            k = int(f.stem.split("_")[2]); tr, va, te = fl[k-1]
            d = pd.read_csv(f); d = d[d.population == "induced"]
            r = d.row.to_numpy(); c = d.column.map(col).to_numpy()
            idx = te[r]
            assert np.allclose(d.actual.to_numpy(float), FZ[idx, c], atol=1e-4), "row mapping/scale mismatch"
            assert np.isnan(X[idx, c]).all()
            tw = np.isin(key[idx], key[tr])
            naive = np.nanmean(Z[tr], 0)[c]; a = d.actual.to_numpy(float); m = d.imputed.to_numpy(float)
            logged = float(raw.loc[raw.fold == k, "impute/induced/impute_score"].iloc[0])
            rec = dict(seed=seed, fold=k, arm=(arm or exp)[0], n=len(d), twin_share=tw.mean(),
                       model=ratio(m, a, naive), logged=logged, model_tw=ratio(m[tw], a[tw], naive[tw]), model_nt=ratio(m[~tw], a[~tw], naive[~tw]))
            if do_knn:
                for kk in (5, 10):
                    P = KNNImputer(n_neighbors=kk, keep_empty_features=True).fit(Z[tr]).transform(Z[te])[r, c]
                    rec[f"knn{kk}"] = ratio(P, a, naive); rec[f"knn{kk}_logged"] = float(raw.loc[raw.fold == k, f"impute/induced/baseline/knn{kk}/impute_score"].iloc[0])
                    rec[f"knn{kk}_tw"] = ratio(P[tw], a[tw], naive[tw]); rec[f"knn{kk}_nt"] = ratio(P[~tw], a[~tw], naive[~tw])
            # store per-cell for pairing
            rec["_cells"] = pd.DataFrame(dict(row=r, c=c, a=a, m=m, naive=naive, tw=tw))
            out.append(rec)
df = pd.DataFrame([{k: v for k, v in r.items() if k != "_cells"} for r in out]).sort_values(["seed", "fold", "arm"])
pd.set_option("display.width", 250); print(variant); print(df.round(4).to_string(index=False))
# pairing across arms on same seed/fold
arms = sorted(df.arm.unique())
if len(arms) == 2:
    A, B = arms
    rows = []
    for r1 in out:
        for r2 in out:
            if r1["arm"] == A and r2["arm"] == B and r1["seed"] == r2["seed"] and r1["fold"] == r2["fold"]:
                c1, c2 = r1["_cells"], r2["_cells"]
                assert (c1.row.values == c2.row.values).all() and (c1.c.values == c2.c.values).all()
                tw = c1.tw.values
                f = lambda cc, s: ratio(cc.m.values[s], cc.a.values[s], cc.naive.values[s])
                allm = np.ones(len(tw), bool)
                rows.append(dict(seed=r1["seed"], fold=r1["fold"], diff_all=f(c1, allm)-f(c2, allm), diff_tw=f(c1, tw)-f(c2, tw), diff_nt=f(c1, ~tw)-f(c2, ~tw),
                                 rel_all=f(c1, allm)/f(c2, allm)-1, rel_tw=f(c1, tw)/f(c2, tw)-1, rel_nt=f(c1, ~tw)/f(c2, ~tw)-1))
    pr = pd.DataFrame(rows)
    print(f"paired {A} - {B} on shared folds:"); print(pr.round(4).to_string(index=False)); print("mean", pr.mean(numeric_only=True).round(4).to_dict())
