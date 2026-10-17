import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from metric_load import runs, fold_frame
from scipy.stats import t
P="impute/induced/"
def parts(f, pre, num, cat):
    nn, nc = f[P+"n_num_cells"].to_numpy(float), f[P+"n_cat_cells"].to_numpy(float)
    tot = nn+nc; s = np.zeros(len(f))
    if (nn>0).all():
        k = "rmse_num_z" if num=="rmse" else "mae_num_z"
        s += nn/tot * f[pre+k].to_numpy()/f[P+"baseline/mean_mode/"+k].to_numpy()
    if (nc>0).all():
        if cat=="acc": s += nc/tot * (1-f[pre+"acc_cat"].to_numpy())/(1-f[P+"baseline/mean_mode/acc_cat"].to_numpy())
        else: s += nc/tot * (1-f[pre+"macro_f1_cat"].to_numpy())/(1-f[P+"baseline/mean_mode/macro_f1_cat"].to_numpy())
    return s
def call(per_seed):
    d=np.concatenate(per_seed); m=d.mean(); h=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    sg=[x.mean() for x in per_seed]
    if m+h<0 and all(x<0 for x in sg): return "M", m, h
    if m-h>0 and all(x>0 for x in sg): return "B", m, h
    return "-", m, h
modes=[("rmse","acc"),("mae","acc"),("rmse","f1"),("mae","f1")]
def run(exp, seeds, bar, opp, label, variants=None):
    R = runs(exp)
    V = sorted({v for v,_ in R}) if variants is None else variants
    tally={m:{"M":0,"B":0,"-":0} for m in modes}; topp={m:{"M":0,"B":0,"-":0} for m in modes}
    for v in V:
        try:
            frs=[fold_frame(R[(v,s)][0]) for s in seeds]
        except KeyError: continue
        if any(f is None for f in frs): continue
        mpre = P+"induced_checkpoint/" if P+"induced_checkpoint/acc_cat" in frs[0].columns or P+"induced_checkpoint/rmse_num_z" in frs[0].columns else P
        out=[f"{v:18s}"]
        for mode in modes:
            ps=[]; po=[]
            for f in frs:
                m=parts(f,mpre,*mode)
                sc={b:parts(f,P+f"baseline/{b}/",*mode) for b in bar if any(c.startswith(P+f"baseline/{b}/") for c in f.columns)}
                best=min(sc,key=lambda b: sc[b].mean()); ps.append(m-sc[best])
                if any(c.startswith(P+f"baseline/{opp}/") for c in f.columns): po.append(m-parts(f,P+f"baseline/{opp}/",*mode))
            c,dm,h=call(ps); tally[mode][c]+=1
            if len(po)==len(frs):
                co,do,ho=call(po); topp[mode][co]+=1; out.append(f"{mode[0]}/{mode[1]} bar:{c}{dm:+.3f} {opp}:{co}{do:+.3f}±{ho:.3f}")
            else: out.append(f"{mode[0]}/{mode[1]} bar:{c}{dm:+.3f}")
        print("  ".join(out))
    print(label, "bar", tally); print(label, opp, topp)
which = sys.argv[1]
if which=="e35":
    run("selection-replication-2026-10-04",[42,7,13],["mean_mode","knn5","knn10","hgb"],"hgb","E35 4bar/hgb")
elif which=="e35mf":
    run("selection-replication-2026-10-04",[42,7,13],["mean_mode","knn5","knn10","hgb","lgbm"],"missforest","E35 5bar/missforest")
elif which=="e34mf":
    run("selection-2026-10-03",[101,202,303],["mean_mode","knn5","knn10","hgb","lgbm"],"missforest","E34 5bar/missforest")
