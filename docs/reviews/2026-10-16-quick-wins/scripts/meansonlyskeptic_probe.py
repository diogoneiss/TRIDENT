import sqlite3, re
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np, pandas as pd
from scipy.stats import t
db = sqlite3.connect("file:C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlflow.db?mode=ro", uri=True, timeout=30)
def runs(exp):
    rows = db.execute("""select d.value, s.value, r.artifact_uri, r.run_uuid from runs r
    join tags x on x.run_uuid=r.run_uuid and x.key='experiment' and x.value=?
    join tags d on d.run_uuid=r.run_uuid and d.key='dataset'
    join tags s on s.run_uuid=r.run_uuid and s.key='seed'
    join tags p on p.run_uuid=r.run_uuid and p.key='run_role' and p.value='parent'
    where r.status='FINISHED' and r.lifecycle_stage='active'
    and r.run_uuid not in (select run_uuid from tags where key='is_mirror' and value='true')""",(exp,)).fetchall()
    out={}
    for v,s,uri,rid in rows:
        p=unquote(urlparse(uri).path); p=Path(p[1:] if re.match(r"^/[A-Za-z]:",p) else p)
        out.setdefault((v,int(s)),[]).append((pd.read_csv(p/"metrics"/"raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True), rid))
    return out
E34=runs("selection-2026-10-03"); E35=runs("selection-replication-2026-10-04"); E38=runs("architecture-large-2026-10-06")
dups=[k for d in (E34,E35,E38) for k,v in d.items() if len(v)>1]
print("duplicate parents:", dups)
P="impute/induced/"; K=P+"induced_checkpoint/impute_score"
B4=("mean_mode","knn5","knn10","hgb")
def ci(x):
    m=x.mean(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return m,m-h,m+h
def verdict(per_seed):
    m,lo,hi=ci(np.concatenate(per_seed)); s=[d.mean() for d in per_seed]
    if hi<0 and all(x<0 for x in s): return "better"
    if lo>0 and all(x>0 for x in s): return "worse"
    return "nodiff"
def bar(c,names):
    n=min(names,key=lambda b:c[f"{P}baseline/{b}/impute_score"].mean()); return n,c[f"{P}baseline/{n}/impute_score"].to_numpy()
V=sorted({v for v,_ in E35})
print("E35 csv has lgbm:", f"{P}baseline/lgbm/impute_score" in E35[(V[0],42)][0][0].columns)
print("E38 variants x seeds:", len(E38), "E38 csv has lgbm on all:", all(f"{P}baseline/lgbm/impute_score" in v[0][0].columns for v in E38.values()))
from collections import Counter
res={}
for label,store,seeds,names_fn in (("E35 bar4",E35,(42,7,13),lambda c:B4),("E34 bar4",E34,(101,202,303),lambda c:B4)):
    cnt=Counter(); below=0; n=0; detail={}
    for v in V:
        cells=[store.get((v,s)) for s in seeds]
        if any(c is None for c in cells): continue
        cells=[c[0][0] for c in cells]; n+=1
        bars=[bar(c,names_fn(c)) for c in cells]
        d=[c[K].to_numpy()-b for c,(_,b) in zip(cells,bars)]
        vv=verdict(d); cnt[vv]+=1
        mm=np.mean([c[K].mean() for c in cells]); bm=np.mean(np.concatenate([b for _,b in bars]))
        below+= mm<bm
        detail[v]=(vv, round(mm,4), round(bm,4), "/".join(sorted({x for x,_ in bars})))
    print(label, "n",n,"means-below",below, dict(cnt))
    res[label]=detail
# E34 seeds with lgbm and missforest from E38
cnt5=Counter(); cntmf=Counter(); below5=0; belowmf=0; n=0; d5={}
for v in V:
    a=[E34.get((v,s)) for s in (101,202,303)]; b=[E38.get((v,s)) for s in (101,202,303)]
    if any(x is None for x in a+b): continue
    a=[x[0][0] for x in a]; b=[x[0][0] for x in b]; n+=1
    for x,y in zip(a,b):
        for nm in B4: assert np.allclose(x[f"{P}baseline/{nm}/impute_score"],y[f"{P}baseline/{nm}/impute_score"],atol=1e-6)
    cells=[]
    for x,y in zip(a,b):
        c=x.copy(); c[f"{P}baseline/lgbm/impute_score"]=y[f"{P}baseline/lgbm/impute_score"] if f"{P}baseline/lgbm/impute_score" in y.columns else np.nan; c[f"{P}baseline/missforest/impute_score"]=y[f"{P}baseline/missforest/impute_score"]; cells.append(c)
    has=all(c[f"{P}baseline/lgbm/impute_score"].notna().all() for c in cells)
    if not has: print("no lgbm:",v)
    bars=[bar(c,B4+("lgbm",) if has else B4) for c in cells]
    vv=verdict([c[K].to_numpy()-bb for c,(_,bb) in zip(cells,bars)]); cnt5[vv]+=1
    mm=np.mean([c[K].mean() for c in cells]); bm=np.mean(np.concatenate([bb for _,bb in bars])); below5+=mm<bm
    mf=[c[f"{P}baseline/missforest/impute_score"].to_numpy() for c in cells]
    vm=verdict([c[K].to_numpy()-m for c,m in zip(cells,mf)]); cntmf[vm]+=1; belowmf+= mm<np.mean(np.concatenate(mf))
    d5[v]=(vv,round(mm,4),round(bm,4),vm)
print("E34+E38 bar5 n",n,"means-below",below5,dict(cnt5),"| missForest means-below",belowmf,dict(cntmf))
for v in V:
    print(v, "E35:",res["E35 bar4"].get(v), "E34:",res["E34 bar4"].get(v), "bar5/mf:", d5.get(v))
