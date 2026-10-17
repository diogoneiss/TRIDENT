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
E34=runs("selection-2026-10-03"); E35=runs("selection-replication-2026-10-04"); E38=runs("architecture-large-2026-10-06")
VARIANTS=sorted({v for v,_ in E34})
P="impute/induced/"
B4=("mean_mode","knn5","knn10","hgb"); B5=B4+("lgbm",)
def ci(x):
    x=np.asarray(x); m=x.mean(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return m,m-h,m+h
def verdict(per_seed):
    m,lo,hi=ci(np.concatenate(per_seed)); s=[d.mean() for d in per_seed]
    if hi<0 and all(x<0 for x in s): return "M",m
    if lo>0 and all(x>0 for x in s): return "B",m
    return "-",m
def bar(cell, names):
    name=min(names,key=lambda b: cell[f"{P}baseline/{b}/impute_score"].mean())
    return name, cell[f"{P}baseline/{name}/impute_score"].to_numpy()
rows=[]
for v in VARIANTS:
    r={"variant":v}
    # seeds 101/202/303: E34 for model, E38 for lgbm/missforest
    cells=[]
    ok=True
    for s in (101,202,303):
        a=E34.get((v,s)); b=E38.get((v,s))
        if a is None or b is None: ok=False; break
        c=a.copy()
        for name in B4:
            assert np.allclose(a[f"{P}baseline/{name}/impute_score"],b[f"{P}baseline/{name}/impute_score"],atol=1e-6),(v,s,name)
        for name in ("lgbm","missforest"):
            k=f"{P}baseline/{name}/impute_score"
            c[k]=b[k] if k in b.columns else np.nan
        c["ALL"]=b[f"{P}induced_checkpoint/impute_score"] if f"{P}induced_checkpoint/impute_score" in b.columns else b[f"{P}impute_score"]
        cells.append(c)
    if not ok: continue
    has_lgbm=all(c[f"{P}baseline/lgbm/impute_score"].notna().all() for c in cells)
    names=B5 if has_lgbm else B4
    bars=[bar(c,names) for c in cells]
    K=[c[f"{P}induced_checkpoint/impute_score"].to_numpy() for c in cells]
    H=[c[f"{P}impute_score"].to_numpy() for c in cells]
    MF=[c[f"{P}baseline/missforest/impute_score"].to_numpy() for c in cells]
    A=[c["ALL"].to_numpy() for c in cells]
    r["bar"]="/".join(sorted({n for n,_ in bars}))+("" if has_lgbm else "(no lgbm)")
    r["barmean"]=np.mean(np.concatenate([b for _,b in bars]))
    r["Kmean"]=np.mean(np.concatenate(K)); r["Hmean"]=np.mean(np.concatenate(H))
    r["K_vs_bar"],_=verdict([k-b for k,(_,b) in zip(K,bars)])
    r["H_vs_bar"],_=verdict([h-b for h,(_,b) in zip(H,bars)])
    r["K_vs_mf"],r["K-mf"]=verdict([k-m for k,m in zip(K,MF)])
    r["mfmean"]=np.mean(np.concatenate(MF))
    if v in ("pendigits_20nan","letter_20nan"):
        r["ALL_vs_bar"],_=verdict([a-b for a,(_,b) in zip(A,bars)]); r["ALL_vs_mf"],r["ALL-mf"]=verdict([a-m for a,m in zip(A,MF)]); r["ALLmean"]=np.mean(np.concatenate(A))
    # E35 seeds, bar of 4
    c35=[E35.get((v,s)) for s in (42,7,13)]
    if all(c is not None for c in c35):
        b35=[bar(c,B4) for c in c35]
        r["K35_vs_bar4"],_=verdict([c[f"{P}induced_checkpoint/impute_score"].to_numpy()-b for c,(_,b) in zip(c35,b35)])
        r["H35_vs_bar4"],_=verdict([c[f"{P}impute_score"].to_numpy()-b for c,(_,b) in zip(c35,b35)])
        r["K35mean"]=np.mean([c[f"{P}induced_checkpoint/impute_score"].mean() for c in c35]); r["H35mean"]=np.mean([c[f"{P}impute_score"].mean() for c in c35])
        r["bar4_35"]=np.mean(np.concatenate([b for _,b in b35]))
    rows.append(r)
df=pd.DataFrame(rows)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(df.round(4).to_string(index=False))
for col in ("K_vs_bar","H_vs_bar","K_vs_mf","K35_vs_bar4","H35_vs_bar4"):
    print(col, df[col].value_counts().to_dict())
print("means-only K<bar:", int((df.Kmean<df.barmean).sum()), " H<bar:", int((df.Hmean<df.barmean).sum()))
print("means-only E35 K<bar4:", int((df.K35mean<df.bar4_35).sum()), " H<bar4:", int((df.H35mean<df.bar4_35).sum()))
print("E38 cells without lgbm:", sorted(k for k,c in E38.items() if f"{P}baseline/lgbm/impute_score" not in c.columns))
print("--- selection optimism of the per-seed min over the bar (seeds 101/202/303; identity fixed from E35 seeds 42/7/13)")
opt=[]
for v in VARIANTS:
    c35=[E35.get((v,s)) for s in (42,7,13)]; c34=[E34.get((v,s)) for s in (101,202,303)]; c38=[E38.get((v,s)) for s in (101,202,303)]
    if any(c is None for c in c35+c34+c38): continue
    names=B4
    fixed=min(names,key=lambda b: np.mean([c[f"{P}baseline/{b}/impute_score"].mean() for c in c35]))
    per_seed=np.mean([bar(c,names)[1].mean() for c in c34])
    fixed_val=np.mean([c[f"{P}baseline/{fixed}/impute_score"].mean() for c in c34])
    oracle_pooled=min(np.mean([c[f"{P}baseline/{b}/impute_score"].mean() for c in c34]) for b in names)
    opt.append((v,fixed,round(per_seed,4),round(oracle_pooled,4),round(fixed_val,4),round(fixed_val-per_seed,4)))
for o in opt: print(o)
print("max optimism", max(o[-1] for o in opt))
print("--- K and H vs best of (bar of 5 + missForest), seeds 101/202/303; pendigits/letter also with ADR 0018's ALL")
tally={"K":{}, "H":{}}
for v in VARIANTS:
    cells=[]
    for s in (101,202,303):
        a=E34[(v,s)].copy(); b=E38[(v,s)]
        for name in ("lgbm","missforest"):
            k=f"{P}baseline/{name}/impute_score"; a[k]=b[k] if k in b.columns else np.nan
        a["ALL"]=b[f"{P}induced_checkpoint/impute_score"] if f"{P}induced_checkpoint/impute_score" in b.columns else b[f"{P}impute_score"]
        cells.append(a)
    names=[n for n in B5+("missforest",) if all(c[f"{P}baseline/{n}/impute_score"].notna().all() for c in cells)]
    bars=[bar(c,names) for c in cells]
    out=[v,"/".join(sorted({n for n,_ in bars}))]
    for arm,key in (("K",f"{P}induced_checkpoint/impute_score"),("H",f"{P}impute_score")):
        vv,m=verdict([c[key].to_numpy()-b for c,(_,b) in zip(cells,bars)]); out+= [arm,vv,round(m,4)]
        tally[arm][vv]=tally[arm].get(vv,0)+1
    if v in ("pendigits_20nan","letter_20nan"):
        vv,m=verdict([c["ALL"].to_numpy()-b for c,(_,b) in zip(cells,bars)]); out+=["ALL",vv,round(m,4)]
    print(out)
print(tally)
