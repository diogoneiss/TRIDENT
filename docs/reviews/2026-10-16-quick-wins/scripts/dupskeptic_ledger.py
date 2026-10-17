"""Read-only: model vs knn5/knn10 on twin / no-twin induced cells, from current-config MLflow ledgers.
Run from the TRIDENT root (relative dataset paths). argv: variant, comma-separated experiment tags."""
import io, sys, os, glob, contextlib, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(os.getcwd())
sys.path.insert(0, str(REPO))
from src.training.types import DatasetSpec
from src.training.data import prepare_dataset, build_folds, load_complete_sibling
from src.training.imputation_baselines import KnnImputer
from src.training.imputation_metrics import mean_mode_baselines

variant, exps = sys.argv[1], sys.argv[2].split(",")
con = sqlite3.connect("file:" + str(REPO / "mlflow.db").replace("\\", "/") + "?mode=ro", uri=True)
def tags(r): return dict(con.execute("select key,value from tags where run_uuid=?", (r,)).fetchall())

ledgers = []
for p in glob.glob("mlruns/*/*/artifacts/imputation/imputation_fold_*_cells.csv"):
    rid = p.replace("\\", "/").split("/")[2]
    t = tags(rid)
    if t.get("dataset") != variant: continue
    pt = tags(t["mlflow.parentRunId"])
    if pt.get("experiment") not in exps: continue
    ledgers.append((pt.get("experiment"), pt.get("arm"), int(pt["seed"]), int(t["fold"]), int(t["total_folds"]), p, pt.get("pretrain_gap_token")))
ledgers.sort()

spec = DatasetSpec.from_name(variant, None)
with contextlib.redirect_stdout(io.StringIO()):
    ds = prepare_dataset(spec)
sib = load_complete_sibling(spec)
feats = [c for c in ds.frame.columns if c != ds.label_column]
num = list(ds.numerical_columns)
X = ds.frame[feats]
T = sib[feats].copy(); T[num] = ds.scaler.transform(sib[num])
key = pd.util.hash_pandas_object(sib[feats], index=False).to_numpy()
col_idx = {c: j for j, c in enumerate(feats)}
folds_cache = {}

def r(e, n): return np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(n ** 2))
tot = {}
for exp, arm, seed, fold, k, path, pgt in ledgers:
    if (seed, k) not in folds_cache:
        folds_cache[(seed, k)] = build_folds(ds.frame, ds.label_column, k, seed)
    f = folds_cache[(seed, k)][fold - 1]
    tr, te = np.asarray(f.train_indices), np.asarray(f.test_indices)
    train = X.iloc[tr].reset_index(drop=True); test = X.iloc[te].reset_index(drop=True)
    d = pd.read_csv(path); d = d[d.population == "induced"].copy()
    rows = d.row.to_numpy(int); cols = d.column.map(col_idx).to_numpy(int)
    truth = T.to_numpy(float)[te[rows], cols]
    rowcheck = np.mean(np.isclose(truth, d.actual.to_numpy(float), atol=1e-4))
    tw = set(key[tr]); twin = np.array([kk in tw for kk in key[te]])[rows]
    naive = np.array([mean_mode_baselines(train, num, [])[c] for c in feats], float)[cols]
    preds = {"model": d.imputed.to_numpy(float)}
    for kn in (5, 10):
        imp = KnnImputer(num, [], kn); imp.fit(train)
        P = imp.impute(test, np.zeros(test.shape, bool)).to_numpy(float)
        preds[f"knn{kn}"] = P[rows, cols]
    a = d.actual.to_numpy(float)
    line = f"{exp}/{arm} pgt={pgt} seed {seed} fold {fold}/{k} rowcheck={rowcheck:.3f} twin_cells={twin.mean():.3f} |"
    for name, P in preds.items():
        line += f" {name} all {r(P-a, naive-a):.3f} twin {r((P-a)[twin], (naive-a)[twin]) if twin.any() else float('nan'):.3f} notwin {r((P-a)[~twin], (naive-a)[~twin]):.3f};"
        for sub, m in (("all", np.ones_like(twin)), ("twin", twin), ("notwin", ~twin)):
            tot.setdefault((exp, arm, name, sub), []).append(r((P-a)[m], (naive-a)[m]) if m.any() else np.nan)
    print(line); sys.stdout.flush()
print("--- mean over listed ledgers (best/worst folds only) ---")
for (exp, arm) in sorted({(e, a) for e, a, *_ in tot}):
    s = f"{exp}/{arm}:"
    for name in ("model", "knn5", "knn10"):
        s += " " + name + " " + "/".join(f"{np.nanmean(tot[(exp, arm, name, sub)]):.3f}" for sub in ("all", "twin", "notwin"))
    print(s + "  (all/twin/notwin)")
