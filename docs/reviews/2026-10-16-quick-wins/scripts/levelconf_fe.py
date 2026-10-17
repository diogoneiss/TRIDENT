import re, numpy as np
T = r"C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/docs/tickets/imputation-token-shape/"
def grab(fn, col):
    out = {}
    for line in open(T+fn, encoding="utf-8"):
        m = re.match(r"\| `([a-z0-9-]+_\d+nan)`", line)
        if not m: continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        v = cells[col].replace("\u2212","-").split()[0].rstrip(",")
        out[m[1]] = float(v)
    return out
studies = {
 "E31 cw-mask": grab("01-column-wise-scoring.md", 4),
 "E32 G-M": grab("02-gaps-as-mask-in-training.md", 5),
 "E33 G-M": grab("03-replication-on-fresh-seeds.md", 4),
 "E34 K-H": grab("04-checkpoint-and-calibration.md", 8),
 "E35 K-H": grab("05-checkpoint-replication.md", 6),
}
LADDER = {"credit-g","kr-vs-kp","spambase"}
for name, d in studies.items():
    assert len(d)==21, (name,len(d))
    full = {}; lad = {}
    for v,x in d.items():
        t, l = v.rsplit("_",1); l=int(l[:-3])
        full.setdefault(l,[]).append(x)
        if t in LADDER: lad.setdefault(l,[]).append(x)
    # fixed effects OLS: x ~ level dummies + table dummies (ref level 20)
    vs = sorted(d); tabs = sorted({v.rsplit('_',1)[0] for v in vs}); levs=[20,40,60,80]
    X=[];y=[]
    for v in vs:
        t,l=v.rsplit("_",1); l=int(l[:-3])
        row=[1.0]+[1.0 if l==L else 0.0 for L in levs[1:]]+[1.0 if t==tt else 0.0 for tt in tabs[1:]]
        X.append(row); y.append(d[v])
    X=np.array(X); y=np.array(y)
    b,*_=np.linalg.lstsq(X,y,rcond=None)
    print(name)
    print("  full   :", ", ".join(f"{l}:{np.mean(full[l]):+.4f}(n={len(full[l])})" for l in levs))
    print("  ladder3:", ", ".join(f"{l}:{np.mean(lad[l]):+.4f}" for l in levs))
    print("  FE level effects vs 20nan:", ", ".join(f"{L}:{b[i+1]:+.4f}" for i,L in enumerate(levs[1:])))
