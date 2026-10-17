"""Read-only: paired verdicts of the model vs the baseline bar and missForest, from mlruns
artifacts (raw_fold_metrics.csv). Stdlib only; never imports project code."""
import csv, json, os, datetime, math, statistics as st, collections

ROOT = r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT\mlruns"
TZ = datetime.timezone(datetime.timedelta(hours=-3))
T14 = 2.144786687917  # t_{0.975, 14}
VARIANTS = ["biodeg_20nan", "biodeg_60nan", "credit-g_20nan", "credit-g_40nan", "credit-g_60nan",
            "credit-g_80nan", "electricity_20nan", "kc2_20nan", "kc2_60nan", "kr-vs-kp_20nan",
            "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan", "letter_20nan", "pendigits_20nan",
            "spambase_20nan", "spambase_40nan", "spambase_60nan", "spambase_80nan", "vehicle_20nan",
            "vehicle_60nan"]
BAR = ["mean_mode", "knn5", "knn10", "hgb", "lgbm"]
POPS = ["induced", "masked"]


def ts(s):
    return datetime.datetime.fromisoformat(s).replace(tzinfo=TZ)


def scan():
    out = []
    for exp in os.listdir(ROOT):
        ed = os.path.join(ROOT, exp)
        if not os.path.isdir(ed):
            continue
        for run in os.listdir(ed):
            art = os.path.join(ed, run, "artifacts")
            raw = os.path.join(art, "metrics", "raw_fold_metrics.csv")
            if not os.path.exists(raw):
                continue
            prov = json.load(open(os.path.join(art, "data", "provenance.json"), encoding="utf-8"))
            hp = json.load(open(os.path.join(art, "parameters", "hyperparameters.json"), encoding="utf-8"))
            with open(raw, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            out.append(dict(run=f"{exp}/{run}", ds=prov["dataset_name"], seed=int(prov["seed"]), hp=hp,
                            mt=datetime.datetime.fromtimestamp(os.path.getmtime(raw), TZ), rows=rows))
    return out


ALL = scan()


def pick(lo, hi, seeds, pred=lambda r: True):
    sel = {}
    for r in ALL:
        if ts(lo) <= r["mt"] <= ts(hi) and r["seed"] in seeds and r["ds"] in VARIANTS and pred(r):
            key = (r["ds"], r["seed"])
            assert key not in sel, ("duplicate", key, r["run"], sel[key]["run"])
            sel[key] = r
    return sel


def folds(run, col):
    vals = {}
    for row in run["rows"]:
        v = row.get(col, "")
        if v == "" or v is None:
            return None
        vals[int(row["fold"])] = float(v)
    return vals


def verdict(model, base, seeds):
    """model/base: {seed: {fold: value}} -> mean diff, CI, per-seed means, verdict."""
    diffs, per_seed = [], []
    for s in seeds:
        d = [model[s][f] - base[s][f] for f in sorted(model[s])]
        diffs += d
        per_seed.append(st.mean(d))
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    lo, hi = m - T14 * se, m + T14 * se
    if hi < 0 and all(x < 0 for x in per_seed):
        v = "better"
    elif lo > 0 and all(x > 0 for x in per_seed):
        v = "worse"
    else:
        v = "nodiff"
    return m, lo, hi, per_seed, v, len(diffs)


def fmt(res):
    m, lo, hi, ps, v, n = res
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] seeds({','.join(f'{x:+.4f}' for x in ps)}) n={n} -> {v}"


def mean3(d, seeds):
    return st.mean(st.mean(d[s].values()) for s in seeds)


# ---------------------------------------------------------------- A: E35, seeds 42/7/13
SA = [42, 7, 13]
E35 = pick("2026-10-04T00:15", "2026-10-04T06:30", SA, lambda r: r["hp"].get("DECODE_CHECKPOINT") == "loss")
assert len(E35) == 63, len(E35)
# ADR 0017 table (bar of 5 by 3-seed mean, as named there) for the reproduction check
ADR17 = {  # variant: (ind model, ind bar name, ind bar, lgbm, mf, msk model, msk bar name, msk bar, msk mf)
    "biodeg_20nan": (0.587, "lgbm", 0.633, 0.633, 0.618, 0.544, "hgb", 0.627, 0.570),
    "biodeg_60nan": (0.761, "hgb", 0.866, 0.868, 0.762, 0.809, "hgb", 0.873, 0.786),
    "credit-g_20nan": (0.882, "hgb", 0.918, 0.922, 0.885, 0.880, "lgbm", 0.927, 0.896),
    "credit-g_40nan": (0.956, "hgb", 0.964, 0.968, 0.931, 0.982, "hgb", 0.997, 0.950),
    "credit-g_60nan": (0.992, "mean_mode", 1.000, 1.079, 1.021, 0.980, "mean_mode", 1.000, 1.024),
    "credit-g_80nan": (1.008, "mean_mode", 1.000, 1.112, 1.093, 1.008, "mean_mode", 1.000, 1.081),
    "electricity_20nan": (0.618, "hgb", 0.606, 0.612, 0.619, 0.686, "hgb", 0.690, 0.710),
    "kc2_20nan": (0.541, "knn5", 0.567, 0.741, 0.605, 0.530, "knn5", 0.559, 0.592),
    "kc2_60nan": (0.572, "knn5", 0.717, 0.819, 0.675, 0.552, "knn10", 0.717, 0.626),
    "kr-vs-kp_20nan": (0.586, "lgbm", 0.608, 0.608, 0.658, 0.649, "lgbm", 0.674, 0.716),
    "kr-vs-kp_40nan": (0.758, "lgbm", 0.764, 0.764, 0.801, 0.781, "lgbm", 0.777, 0.824),
    "kr-vs-kp_60nan": (0.865, "hgb", 0.903, 0.905, 0.935, 0.887, "lgbm", 0.911, 0.941),
    "kr-vs-kp_80nan": (0.976, "mean_mode", 1.000, 1.074, 1.021, 1.000, "mean_mode", 1.000, 1.017),
    "letter_20nan": (0.459, "knn10", 0.473, 0.499, 0.462, 0.496, "knn10", 0.542, 0.535),
    "pendigits_20nan": (0.332, "knn10", 0.350, 0.363, 0.334, 0.369, "knn10", 0.401, 0.409),
    "spambase_20nan": (0.848, "lgbm", 0.865, 0.865, 0.859, 0.865, "lgbm", 0.886, 0.923),
    "spambase_40nan": (0.903, "lgbm", 0.937, 0.937, 0.942, 0.897, "lgbm", 0.936, 0.972),
    "spambase_60nan": (0.942, "hgb", 0.984, 0.985, 0.981, 0.904, "lgbm", 0.982, 0.974),
    "spambase_80nan": (0.987, "mean_mode", 1.000, 1.043, 1.033, 0.993, "mean_mode", 1.000, 1.058),
    "vehicle_20nan": (0.418, "lgbm", 0.473, 0.473, 0.452, 0.420, "knn5", 0.505, 0.472),
    "vehicle_60nan": (0.645, "hgb", 0.701, 0.702, 0.647, 0.616, "lgbm", 0.668, 0.616),
}

print("=" * 100)
print("A. E35 (seeds 42/7/13, K readout, 5 folds) vs ADR 0017's named bar of 5; lgbm bars -> hgb proxy")
print("=" * 100)
mism = []
countsA = collections.Counter()
for v in VARIANTS:
    a = ADR17[v]
    for pi, pop in enumerate(POPS):
        mcol = f"impute/{pop}/induced_checkpoint/impute_score"
        model = {s: folds(E35[(v, s)], mcol) for s in SA}
        mm = mean3(model, SA)
        adr_model = a[0] if pop == "induced" else a[5]
        barname = a[1] if pop == "induced" else a[6]
        barval = a[2] if pop == "induced" else a[7]
        if abs(round(mm, 3) - adr_model) > 0.0011:
            mism.append((v, pop, "model", mm, adr_model))
        used = "hgb" if barname == "lgbm" else barname
        base = {s: folds(E35[(v, s)], f"impute/{pop}/baseline/{used}/impute_score") for s in SA}
        bm = mean3(base, SA)
        if barname != "lgbm" and abs(round(bm, 3) - barval) > 0.0011:
            mism.append((v, pop, used, bm, barval))
        res = verdict(model, base, SA)
        tag = f"{barname}(proxy hgb {bm:.3f})" if barname == "lgbm" else f"{barname} {bm:.3f}"
        print(f"{v:18s} {pop:7s} model {mm:.4f} | bar {tag:24s} | m-bar {fmt(res)}")
        countsA[(pop, res[4], barname == "lgbm")] += 1
print("reproduction mismatches (3 dp):", mism if mism else "none")
print("counts A:", dict(countsA))

# ---------------------------------------------------------------- B: seeds 101/202/303
SB = [101, 202, 303]
E34 = pick("2026-10-03T12:00", "2026-10-03T19:30", SB, lambda r: "impute/induced/induced_checkpoint/impute_score" in r["rows"][0] and r["hp"].get("DECODE_CHECKPOINT") in (None, "loss"))
E38 = pick("2026-10-06T14:40", "2026-10-07T07:30", SB, lambda r: r["hp"].get("LAYERS") == 4)
assert len(E34) == 63 and len(E38) == 63, (len(E34), len(E38))

print()
print("=" * 100)
print("B. today's default on seeds 101/202/303: E34 K readout (19 variants) / E38 ALL (pendigits, letter)")
print("   baselines per fold from E34 (ADR 0007 four) and E38 (lgbm where logged, missForest)")
print("=" * 100)
maxdiff = collections.defaultdict(float)
countsB = collections.Counter()
lines = []
for v in VARIANTS:
    big = v in ("pendigits_20nan", "letter_20nan")
    for pop in POPS:
        if big:
            model = {s: folds(E38[(v, s)], f"impute/{pop}/impute_score") for s in SB}
        else:
            model = {s: folds(E34[(v, s)], f"impute/{pop}/induced_checkpoint/impute_score") for s in SB}
        bases = {}
        for b in BAR + ["missforest"]:
            src = E34 if b in ("mean_mode", "knn5", "knn10", "hgb") else E38
            vals = {s: folds(src[(v, s)], f"impute/{pop}/baseline/{b}/impute_score") for s in SB}
            if all(vals[s] is not None for s in SB):
                bases[b] = vals
            if b in ("mean_mode", "knn5", "knn10", "hgb"):  # integrity: E34 == E38 cells
                other = {s: folds(E38[(v, s)], f"impute/{pop}/baseline/{b}/impute_score") for s in SB}
                for s in SB:
                    for f in vals[s]:
                        maxdiff[b] = max(maxdiff[b], abs(vals[s][f] - other[s][f]))
        means = {b: mean3(bases[b], SB) for b in bases}
        cand = {b: means[b] for b in BAR if b in bases}
        barname = min(cand, key=cand.get)
        mm = mean3(model, SB)
        rb = verdict(model, bases[barname], SB)
        rmf = verdict(model, bases["missforest"], SB)
        lg = f"{means['lgbm']:.3f}" if "lgbm" in means else "n/a"
        print(f"{v:18s} {pop:7s} model {mm:.4f}{' [ALL]' if big else ''} | bar {barname} {means[barname]:.3f}"
              f" (lgbm {lg}) | m-bar {fmt(rb)} | mF {means['missforest']:.3f} m-mF {fmt(rmf)}")
        countsB[(pop, "bar", rb[4])] += 1
        countsB[(pop, "mf", rmf[4])] += 1
        lines.append((v, pop, mm, barname, means[barname], rb[4], means["missforest"], rmf[4], "lgbm" in means))
print("integrity, max |E34 - E38| per ADR 0007 baseline fold value:", dict(maxdiff))
print("counts B:", dict(countsB))
print("variants with no lgbm fold values (E38's first 15 cells, ADR 0016 code):",
      sorted({l[0] for l in lines if not l[8]}))

# ---------------------------------------------------------------- C: E39 provisional tally
print()
print("=" * 100)
print("C. E39 (T, D from 2026-10-16 artifacts; F = E34 K readout), induced and masked, unofficial")
print("=" * 100)
def is_default(r):
    h = r["hp"]
    return h.get("LR_DECODE") == 0.001 and h.get("WEIGHT_DECAY_DECODE") == 0.0019 and h.get("DROPOUT") == 0.2
E39D = pick("2026-10-16T00:00", "2026-10-16T12:00", SB, is_default)
E39T = pick("2026-10-16T00:00", "2026-10-16T12:00", SB, lambda r: not is_default(r))
print("cells: T", len(E39T), "D", len(E39D))
md39 = 0.0
for v in ["credit-g_40nan", "kr-vs-kp_20nan", "kr-vs-kp_40nan", "spambase_20nan", "spambase_40nan"]:
    for pop in POPS:
        F = {s: folds(E34[(v, s)], f"impute/{pop}/induced_checkpoint/impute_score") for s in SB}
        T = {s: folds(E39T[(v, s)], f"impute/{pop}/impute_score") for s in SB}
        D = {s: folds(E39D[(v, s)], f"impute/{pop}/impute_score") for s in SB}
        for s in SB:
            for b in ("knn5", "hgb"):
                e = folds(E34[(v, s)], f"impute/{pop}/baseline/{b}/impute_score")
                for arm in (E39T, E39D):
                    o = folds(arm[(v, s)], f"impute/{pop}/baseline/{b}/impute_score")
                    md39 = max(md39, max(abs(e[f] - o[f]) for f in e))
        print(f"{v:16s} {pop:7s} F {mean3(F,SB):.4f} T {mean3(T,SB):.4f} D {mean3(D,SB):.4f}")
        print(f"    T-F {fmt(verdict(T, F, SB))}")
        print(f"    D-F {fmt(verdict(D, F, SB))}")
        print(f"    T-D {fmt(verdict(T, D, SB))}")
print("E39 integrity, max |E34 - E39| knn5/hgb fold value:", md39)
