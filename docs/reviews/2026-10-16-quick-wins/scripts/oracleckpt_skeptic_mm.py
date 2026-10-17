import sqlite3, re
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np, pandas as pd
from scipy.stats import t
db = sqlite3.connect("file:C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlflow.db?mode=ro", uri=True, timeout=30)
def runs(exp):
    rows = db.execute("""
    select d.value, s.value, r.artifact_uri from runs r
    join tags x on x.run_uuid=r.run_uuid and x.key='experiment' and x.value=?
    join tags d on d.run_uuid=r.run_uuid and d.key='dataset'
    join tags s on s.run_uuid=r.run_uuid and s.key='seed'
    join tags p on p.run_uuid=r.run_uuid and p.key='run_role' and p.value='parent'
    where r.status='FINISHED' and r.lifecycle_stage='active'
    and r.run_uuid not in (select run_uuid from tags where key='is_mirror' and value='true')""",(exp,)).fetchall()
    out={}
    for v,s,uri in rows:
        p=unquote(urlparse(uri).path); p=Path(p[1:] if re.match(r"^/[A-Za-z]:",p) else p)
        out[(v,int(s))]=pd.read_csv(p/"metrics"/"raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True)
    return out
P="impute/induced/"; M="impute/masked/"
def verdict(per_seed):
    x=np.concatenate(per_seed); m=x.mean(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); s=[d.mean() for d in per_seed]
    if m+h<0 and all(v<0 for v in s): return "W"
    if m-h>0 and all(v>0 for v in s): return "L"
    return "-"
B4=("mean_mode","knn5","knn10","hgb")
for exp,seeds in (("selection-2026-10-03",(101,202,303)),("selection-replication-2026-10-04",(42,7,13))):
    R=runs(exp)
    print("=====",exp,len(R))
    print("variant | H mean, K mean | H-mm K-mm | bar(test-picked) ids | H-bar K-bar | fixed-bar(mm/knn/hgb chosen by OTHER seed set) | K epoch, H epoch")
    for v in sorted({v for v,_ in R}):
        cs=[R.get((v,s)) for s in seeds]
        if any(c is None for c in cs): continue
        H=[c[f"{P}impute_score"].to_numpy() for c in cs]; K=[c[f"{P}induced_checkpoint/impute_score"].to_numpy() for c in cs]
        mm=[c[f"{P}baseline/mean_mode/impute_score"].to_numpy() for c in cs]
        def bar(c):
            n=min(B4,key=lambda b:c[f"{P}baseline/{b}/impute_score"].mean()); return n,c[f"{P}baseline/{n}/impute_score"].to_numpy()
        bars=[bar(c) for c in cs]
        ek=np.mean([c["decode/induced_checkpoint_epoch"].mean() for c in cs]) if "decode/induced_checkpoint_epoch" in cs[0].columns else np.nan
        eh=np.mean([c["decode/loss_checkpoint_epoch"].mean() for c in cs]) if "decode/loss_checkpoint_epoch" in cs[0].columns else np.nan
        print(f"{v:18s} {np.mean(np.concatenate(H)):.4f} {np.mean(np.concatenate(K)):.4f} | {verdict([h-b for h,b in zip(H,mm)])} {verdict([k-b for k,b in zip(K,mm)])} | {sorted({n for n,_ in bars})} | {verdict([h-b for h,(_,b) in zip(H,bars)])} {verdict([k-b for k,(_,b) in zip(K,bars)])} | ep K {ek:.1f} H {eh:.1f}")
