"""Read-only: paired 15-fold CIs, model K readout (E35) minus (a) the logged bar of five and
(b) the tuned baseline values measured by budgetskeptic_hgb2.py / budgetskeptic_knn.py."""
import re, sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np, pandas as pd
from scipy.stats import t

ROOT = Path(r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT")
STORE = sqlite3.connect(f"file:{(ROOT / 'mlflow.db').as_posix()}?mode=ro", uri=True, timeout=30)
BAR = ["mean_mode", "knn5", "knn10", "hgb"]
LGBM = {("kr-vs-kp_40nan", 42): [0.7776, 0.7943, 0.7837, 0.7263, 0.7496], ("kr-vs-kp_40nan", 7): [0.773, 0.7658, 0.7465, 0.7362, 0.8162], ("kr-vs-kp_40nan", 13): [0.7744, 0.7506, 0.7848, 0.7514, 0.7304]}
K = "impute/induced/induced_checkpoint/impute_score"
TUNED = {  # per seed, five fold values, from this session's scoring-only runs
    "kr-vs-kp_40nan": {42: [0.7695, 0.7706, 0.75, 0.7092, 0.7254], 7: [0.7483, 0.7319, 0.7297, 0.7231, 0.7658], 13: [0.7648, 0.733, 0.7603, 0.7555, 0.7316]},
    "kr-vs-kp_60nan": {42: [0.8528, 0.8608, 0.8806, 0.8742, 0.8557], 7: [0.8735, 0.8649, 0.8495, 0.8711, 0.8862], 13: [0.8889, 0.8696, 0.8546, 0.888, 0.8604]},
    "credit-g_40nan": {42: [0.9662, 0.9382, 0.9411, 0.9726, 0.9233], 7: [0.9879, 0.9775, 0.9632, 0.9403, 0.9043], 13: [0.9911, 0.9387, 0.95, 0.9911, 0.9866]},
    "spambase_20nan": {42: [0.9015, 0.8386, 0.8651, 0.849, 0.8384], 7: [0.8344, 0.8229, 0.8703, 0.8324, 0.8968], 13: [0.8895, 0.8449, 0.8159, 0.8449, 0.8776]},
    "credit-g_60nan": {42: [0.9845, 0.9878, 0.9862, 0.9928, 0.9671], 7: [0.9791, 1.0064, 0.9953, 0.9773, 0.9907], 13: [0.9916, 0.9766, 0.9799, 0.9923, 0.9906]},
}


def path(uri):
    p = unquote(urlparse(uri).path)
    return Path(p[1:] if re.match(r"^/[A-Za-z]:", p) else p)


rows = STORE.execute(
    """select d.value, s.value, r.artifact_uri from runs r
    join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = 'selection-replication-2026-10-04'
    join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
    join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
    join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
    where r.status = 'FINISHED' and r.lifecycle_stage = 'active'
    and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')""").fetchall()
cells = {(v, int(s)): pd.read_csv(path(u) / "metrics" / "raw_fold_metrics.csv").sort_values("fold") for v, s, u in rows}


def verdict(diffs):
    d = np.concatenate(diffs)
    m = d.mean(); h = t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    signs = [x.mean() for x in diffs]
    call = "model better" if m + h < 0 and all(s < 0 for s in signs) else (
        "baseline better" if m - h > 0 and all(s > 0 for s in signs) else "no verdict")
    return f"mean {m:+.4f} CI [{m - h:+.4f}, {m + h:+.4f}] seeds {[round(s, 4) for s in signs]} -> {call}"


for variant, tuned in TUNED.items():
    exact = variant in {v for v, _ in LGBM} or variant == "credit-g_60nan"
    vs_bar, vs_tuned, model_means, bar_means, tuned_means = [], [], [], [], []
    for seed in (42, 7, 13):
        c = cells[(variant, seed)]
        options = {b: c[f"impute/induced/baseline/{b}/impute_score"].to_numpy() for b in BAR}
        if (variant, seed) in LGBM:
            options["lgbm"] = np.array(LGBM[(variant, seed)])
        name = min(options, key=lambda b: options[b].mean())
        bar = options[name]
        model = c[K].to_numpy()
        vs_bar.append(model - bar)
        tv = np.array(tuned[seed])
        # tuned bar = per-run min of the logged bar and the tuned baseline, chosen on the run's mean (as summary.py does)
        alt = tv if tv.mean() < bar.mean() else bar
        vs_tuned.append(model - alt)
        model_means.append(model.mean()); bar_means.append(bar.mean()); tuned_means.append(alt.mean())
    print(variant, "(bar5 exact)" if exact else "(bar5 without lgbm folds: approximate)", f"model {np.mean(model_means):.4f} bar5 {np.mean(bar_means):.4f} bar5+tuned {np.mean(tuned_means):.4f}")
    print("   model - bar5      :", verdict(vs_bar))
    print("   model - bar5+tuned:", verdict(vs_tuned))
