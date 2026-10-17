import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from metric_load import runs, fold_frame
from scipy.stats import t
SEEDS=[101,202,303]
E34 = runs("selection-2026-10-03"); E38 = runs("architecture-large-2026-10-06")
VARIANTS = sorted({v for v,_ in E34})
P="impute/induced/"
BAR=["mean_mode","knn5","knn10","hgb"]
def parts(f, pre, num, cat):
    nn, nc = f[P+"n_num_cells"].to_numpy(float), f[P+"n_cat_cells"].to_numpy(float)
    tot = nn+nc; s = np.zeros(len(f))
    if (nn>0).all():
        if num=="rmse": s += nn/tot * f[pre+"rmse_num_z"].to_numpy()/f[P+"baseline/mean_mode/rmse_num_z"].to_numpy()
        else: s += nn/tot * f[pre+"mae_num_z"].to_numpy()/f[P+"baseline/mean_mode/mae_num_z"].to_numpy()
    if (nc>0).all():
        if cat=="acc": s += nc/tot * (1-f[pre+"acc_cat"].to_numpy())/(1-f[P+"baseline/mean_mode/acc_cat"].to_numpy())
        else: s += nc/tot * (1-f[pre+"macro_f1_cat"].to_numpy())/(1-f[P+"baseline/mean_mode/macro_f1_cat"].to_numpy())
    return s
def call(per_seed):
    d=np.concatenate(per_seed); m=d.mean(); h=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    sg=[x.mean() for x in per_seed]
    if m+h<0 and all(x<0 for x in sg): return "M", m
    if m-h>0 and all(x>0 for x in sg): return "B", m
    return "-", m
modes=[("rmse","acc"),("mae","acc"),("rmse","f1"),("mae","f1")]
tally={m:{"M":0,"B":0,"-":0,"below":0,"mf_M":0,"mf_B":0} for m in modes}
rows=[]
for v in VARIANTS:
    big = v.startswith("pendigits") or v.startswith("letter")
    model_frames=[]; base_frames=[]
    for s in SEEDS:
        if big:
            mf = fold_frame(E38[(v,s)][0]); mpre=P
        else:
            mf = fold_frame(E34[(v,s)][0]); mpre=P+"induced_checkpoint/"
        bf = fold_frame(E38[(v,s)][0])
        # sanity: shared baselines equal
        a = mf[P+"baseline/hgb/impute_score"].to_numpy(); b = bf[P+"baseline/hgb/impute_score"].to_numpy()
        assert np.allclose(a,b), (v,s)
        model_frames.append((mf,mpre)); base_frames.append(bf)
    line=[f"{v:18s}"]
    for mode in modes:
        per_seed=[]; per_mf=[]; bars=[]; mm=[]; bb=[]
        for (mf,mpre),bf in zip(model_frames, base_frames):
            m = parts(mf, mpre, *mode)
            sc = {b: parts(bf, P+f"baseline/{b}/", *mode) for b in BAR if P+f"baseline/{b}/rmse_num_z" in bf.columns or P+f"baseline/{b}/acc_cat" in bf.columns}
            best=min(sc,key=lambda b: sc[b].mean()); bars.append(best)
            per_seed.append(m-sc[best]); mm.append(m.mean()); bb.append(sc[best].mean())
            per_mf.append(m-parts(bf,P+"baseline/hgb/",*mode))
        c,dm = call(per_seed); cm,_ = call(per_mf)
        tally[mode][c]+=1; tally[mode]["below"] += np.mean(mm)<np.mean(bb)
        tally[mode]["mf_"+cm] = tally[mode].get("mf_"+cm,0)+1
        line.append(f"{mode[0]}/{mode[1]}: {c} {dm:+.3f} [{bars[0]}] mf:{cm}")
    print("  ".join(line))
for m,tl in tally.items(): print(m, tl)
