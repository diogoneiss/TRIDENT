"""E39 arms T and D vs missForest and lgbm/hgb on their own artifacts (stdlib only)."""
import runpy, os, sys, io, contextlib
here = os.path.dirname(os.path.abspath(__file__))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(os.path.join(here, "barcheck_verdicts.py"))
folds, verdict, fmt, mean3 = g["folds"], g["verdict"], g["fmt"], g["mean3"]
SB, E39T, E39D = g["SB"], g["E39T"], g["E39D"]
for v in ["credit-g_40nan", "kr-vs-kp_20nan", "kr-vs-kp_40nan", "spambase_20nan", "spambase_40nan"]:
    for pop in ("induced", "masked"):
        for name, arm in (("T", E39T), ("D", E39D)):
            m = {s: folds(arm[(v, s)], f"impute/{pop}/impute_score") for s in SB}
            mf = {s: folds(arm[(v, s)], f"impute/{pop}/baseline/missforest/impute_score") for s in SB}
            bars = {b: {s: folds(arm[(v, s)], f"impute/{pop}/baseline/{b}/impute_score") for s in SB}
                    for b in ("mean_mode", "knn5", "knn10", "hgb", "lgbm")}
            bn = min(bars, key=lambda b: mean3(bars[b], SB))
            print(f"{v:16s} {pop:7s} {name} {mean3(m,SB):.4f} | bar {bn} {mean3(bars[bn],SB):.3f} m-bar {fmt(verdict(m, bars[bn], SB))}"
                  f" | mF {mean3(mf,SB):.3f} m-mF {fmt(verdict(m, mf, SB))}")
