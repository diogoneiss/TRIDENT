"""cvdepskeptic_ label: counts under a balanced one-way random-effects (seed) model, read-only on scratch pickle."""
import importlib.util, sys, io, contextlib
from pathlib import Path
import numpy as np
from scipy import stats
here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("rr", here / "cvdepskeptic_rerun.py")
rr = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rr)
def call(lo, hi, signs):
    if hi < 0 and all(x < 0 for x in signs): return -1
    if lo > 0 and all(x > 0 for x in signs): return +1
    return 0
print(f"{'contrast':12s} {'naive':>6s} {'mixedREML':>9s} {'mixedSatt':>9s}  nF>1")
flipdetail = {}
for name, (variants, seeds, fn) in rr.CONTRASTS.items():
    cn, cm, cs = [], [], []; nF = 0
    for v in variants:
        per = [fn(v, s) for s in seeds]
        d = np.concatenate(per); n = len(d); m = d.mean()
        signs = [x.mean() for x in per]
        sm = np.array(signs)
        msb = 5 * sm.var(ddof=1); msw = np.mean([x.var(ddof=1) for x in per])
        se_n = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
        cn.append(call(m - tq * se_n, m + tq * se_n, signs))
        # REML truncation: if MSB<=MSW, sigma_b=0 -> pooled within, df=12; else var=MSB/15, df=2
        if msb <= msw:
            se = np.sqrt(msw / n); df = 12
        else:
            nF += 1; se = np.sqrt(msb / n); df = 2
        q = stats.t.ppf(0.975, df); cm.append(call(m - q * se, m + q * se, signs))
        # Satterthwaite-ish: var = MSB/15 always when F>1 else MSW/15, df by Satterthwaite of combination (use df=2 if F>1) -> same; instead
        # an alternative: var = (MSB/15) but floor at MSW/15 with df = Satterthwaite(MSB,MSW)
        s2b = max(0.0, (msb - msw) / 5); var = (msw + 5 * s2b) / n
        if s2b > 0:
            df_s = (msw / 5 + s2b) ** 2 / ((msb / 5) ** 2 / 2)  # = MSB^2 / (MSB^2/2) -> 2; keep simple
        else:
            df_s = 12
        q = stats.t.ppf(0.975, df_s); cs.append(call(m - q * np.sqrt(var), m + q * np.sqrt(var), signs))
        if cn[-1] != cm[-1]:
            flipdetail.setdefault(name, []).append((v, cn[-1], cm[-1], round(msb / msw, 2)))
    t = lambda c: f"{c.count(-1)}/{c.count(1)}"
    print(f"{name:12s} {t(cn):>6s} {t(cm):>9s} {t(cs):>9s}  {nF}")
print()
for k, v in flipdetail.items():
    print(k, v)
