"""stats_ label: re-run the E30-E38 verdicts under alternative tests (read-only, scratch)."""
import pickle
from pathlib import Path
import numpy as np
from scipy import stats

D = pickle.load(open(Path(__file__).with_name("stats_cells.pkl"), "rb"))
V21 = [*(f"{b}_{l}nan" for b in ("credit-g", "kr-vs-kp", "spambase") for l in (20, 40, 60, 80)),
       *(f"{b}_{l}nan" for b in ("vehicle", "biodeg", "kc2") for l in (20, 60)),
       "pendigits_20nan", "letter_20nan", "electricity_20nan"]
V5 = ["credit-g_20nan", "credit-g_80nan", "kr-vs-kp_40nan", "biodeg_60nan", "pendigits_20nan"]
S1, S2 = [42, 7, 13], [101, 202, 303]
IND = "impute/induced/impute_score"; CW = "impute/induced/column_wise/impute_score"
K = "impute/induced/induced_checkpoint/impute_score"; C = "impute/induced/calibrated/impute_score"

def g(exp, v, s, arm, key):
    return D[(exp, v, s, arm)][key].to_numpy()

CONTRASTS = {
 "E30 M-A": (V21, S1, lambda v, s: g("confirmatory-2026-10-01", v, s, "M", IND) - g("confirmatory-2026-10-01", v, s, "A", IND)),
 "E30 P-A": (V21, S1, lambda v, s: g("confirmatory-2026-10-01", v, s, "P", IND) - g("confirmatory-2026-10-01", v, s, "A", IND)),
 "E30 P-M": (V21, S1, lambda v, s: g("confirmatory-2026-10-01", v, s, "P", IND) - g("confirmatory-2026-10-01", v, s, "M", IND)),
 "E31 CW-mask": (V21, S1, lambda v, s: g("column-wise-2026-10-02", v, s, "", CW) - g("column-wise-2026-10-02", v, s, "", IND)),
 "E32 G-M": (V21, S1, lambda v, s: g("gap-token-2026-10-02", v, s, "", IND) - g("column-wise-2026-10-02", v, s, "", IND)),
 "E33 G-M": (V21, S2, lambda v, s: g("gap-token-replication-2026-10-03", v, s, "G", IND) - g("gap-token-replication-2026-10-03", v, s, "M", IND)),
 "E34 K-H": (V21, S2, lambda v, s: g("selection-2026-10-03", v, s, "", K) - g("selection-2026-10-03", v, s, "", IND)),
 "E34 C-H": (V21, S2, lambda v, s: g("selection-2026-10-03", v, s, "", C) - g("selection-2026-10-03", v, s, "", IND)),
 "E35 K-H": (V21, S1, lambda v, s: g("selection-replication-2026-10-04", v, s, "", K) - g("selection-replication-2026-10-04", v, s, "", IND)),
 "E35 C-H": (V21, S1, lambda v, s: g("selection-replication-2026-10-04", v, s, "", C) - g("selection-replication-2026-10-04", v, s, "", IND)),
 "E36 PG-R": (V21, S1, lambda v, s: g("pretrain-gap-token-2026-10-05", v, s, "", IND) - g("selection-replication-2026-10-04", v, s, "", K)),
 **{f"E37 {a}-R": (V5, S1, (lambda a: lambda v, s: g("architecture-2026-10-05", v, s, a, IND) - g("selection-replication-2026-10-04", v, s, "", K))(a)) for a in ("L4", "F256", "LN", "ALL")},
 "E38 ALL-R": (V21, S2, lambda v, s: g("architecture-large-2026-10-06", v, s, "", IND) - g("selection-2026-10-03", v, s, "", K)),
}

def call(lo, hi, signs):
    if hi < 0 and all(x < 0 for x in signs): return -1
    if lo > 0 and all(x > 0 for x in signs): return +1
    return 0

def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1, run)
    return adj

RATIO = {"nb25": 0.25, "nb29": 0.2 / 0.7}
summary = {}
anova = []
for name, (variants, seeds, fn) in CONTRASTS.items():
    rows = []
    for v in variants:
        per = [fn(v, s) for s in seeds]
        d = np.concatenate(per); n = len(d); m = d.mean(); sd = d.std(ddof=1)
        signs = [x.mean() for x in per]
        se_naive = sd / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        r = {"v": v, "mean": m, "signs": signs}
        r["naive"] = call(m - tq * se_naive, m + tq * se_naive, signs)
        r["p_naive"] = 2 * stats.t.sf(abs(m / se_naive), n - 1)
        for lab, ratio in RATIO.items():
            se = sd * np.sqrt(1 / n + ratio)
            r[lab] = call(m - tq * se, m + tq * se, signs)
            r["p_" + lab] = 2 * stats.t.sf(abs(m / se), n - 1)
        sm = np.array(signs); se3 = sm.std(ddof=1) / np.sqrt(3); t2 = stats.t.ppf(0.975, 2)
        r["seed3"] = call(sm.mean() - t2 * se3, sm.mean() + t2 * se3, signs)
        r["p_seed3"] = 2 * stats.t.sf(abs(sm.mean() / se3), 2)
        # within/between seed variance (one-way ANOVA, seeds as groups)
        F, pF = stats.f_oneway(*per)
        anova.append((name, v, F, pF))
        r["naive_noSign"] = call(m - tq * se_naive, m + tq * se_naive, [np.sign(m)] * 3)
        rows.append(r)
    for lab in ("naive", "nb25"):
        adj = holm([r["p_" + lab] for r in rows])
        for r, a in zip(rows, adj):
            r["holm_" + lab] = int(np.sign(r["mean"])) if (a < 0.05 and all(np.sign(x) == np.sign(r["mean"]) for x in r["signs"])) else 0
    summary[name] = rows

def tally(rows, key):
    c = [r[key] for r in rows]
    return f"{c.count(-1):2d}/{c.count(1):d}"

print("better/worse counts (arm better / reference better); columns:")
cols = ["naive", "naive_noSign", "holm_naive", "nb25", "nb29", "holm_nb25", "seed3"]
print(f"{'contrast':12s} " + " ".join(f"{c:>12s}" for c in cols))
for name, rows in summary.items():
    print(f"{name:12s} " + " ".join(f"{tally(rows, c):>12s}" for c in cols))

print()
for name in ["E30 P-A", "E30 P-M", "E30 M-A", "E31 CW-mask", "E34 K-H", "E34 C-H", "E35 C-H", "E36 PG-R", "E37 ALL-R", "E37 F256-R", "E38 ALL-R", "E33 G-M", "E32 G-M", "E35 K-H"]:
    rows = summary[name]
    print("==", name)
    for r in rows:
        if r["naive"] != 0 or r["nb25"] != 0 or r["seed3"] != 0:
            print(f"   {r['v']:18s} mean {r['mean']:+.4f} p_naive {r['p_naive']:.4f} p_nb {r['p_nb25']:.4f} p_seed3 {r['p_seed3']:.3f}  naive {r['naive']:+d} nb25 {r['nb25']:+d} nb29 {r['nb29']:+d} holmN {r['holm_naive']:+d} holmNB {r['holm_nb25']:+d} seed3 {r['seed3']:+d}")
Fs = np.array([a[2] for a in anova if np.isfinite(a[2])]); ps = np.array([a[3] for a in anova if np.isfinite(a[3])])
print("\nANOVA seeds-as-groups over", len(Fs), "variant-contrasts: median F", np.median(Fs).round(2), "mean F", Fs.mean().round(2), "share p<0.05", (ps < 0.05).mean().round(3), "share p<0.01", (ps < 0.01).mean().round(3))
for exp in ["E30", "E32", "E33", "E34", "E35", "E36", "E38"]:
    sel = [a for a in anova if a[0].startswith(exp) and np.isfinite(a[2])]
    print(f"  {exp}: median F {np.median([a[2] for a in sel]):.2f}, share p<0.05 {np.mean([a[3] < 0.05 for a in sel]):.2f}, n {len(sel)}")
pickle.dump(summary, open(Path(__file__).with_name("stats_summary.pkl"), "wb"))
