"""Tally verdicts by missing level and by table from barcheck_out.txt (stdlib only)."""
import re, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "barcheck_out.txt"), encoding="utf-8").read()
secA = text.split("\nB. ")[0]
secB = text.split("\nB. ")[1].split("\nC. ")[0]

ADR17 = {  # ind model, ind bar, ind mf, msk model, msk bar, msk mf   (ADR 0017:145-165)
    "biodeg_20nan": (0.587, 0.633, 0.618, 0.544, 0.627, 0.570),
    "biodeg_60nan": (0.761, 0.866, 0.762, 0.809, 0.873, 0.786),
    "credit-g_20nan": (0.882, 0.918, 0.885, 0.880, 0.927, 0.896),
    "credit-g_40nan": (0.956, 0.964, 0.931, 0.982, 0.997, 0.950),
    "credit-g_60nan": (0.992, 1.000, 1.021, 0.980, 1.000, 1.024),
    "credit-g_80nan": (1.008, 1.000, 1.093, 1.008, 1.000, 1.081),
    "electricity_20nan": (0.618, 0.606, 0.619, 0.686, 0.690, 0.710),
    "kc2_20nan": (0.541, 0.567, 0.605, 0.530, 0.559, 0.592),
    "kc2_60nan": (0.572, 0.717, 0.675, 0.552, 0.717, 0.626),
    "kr-vs-kp_20nan": (0.586, 0.608, 0.658, 0.649, 0.674, 0.716),
    "kr-vs-kp_40nan": (0.758, 0.764, 0.801, 0.781, 0.777, 0.824),
    "kr-vs-kp_60nan": (0.865, 0.903, 0.935, 0.887, 0.911, 0.941),
    "kr-vs-kp_80nan": (0.976, 1.000, 1.021, 1.000, 1.000, 1.017),
    "letter_20nan": (0.459, 0.473, 0.462, 0.496, 0.542, 0.535),
    "pendigits_20nan": (0.332, 0.350, 0.334, 0.369, 0.401, 0.409),
    "spambase_20nan": (0.848, 0.865, 0.859, 0.865, 0.886, 0.923),
    "spambase_40nan": (0.903, 0.937, 0.942, 0.897, 0.936, 0.972),
    "spambase_60nan": (0.942, 0.984, 0.981, 0.904, 0.982, 0.974),
    "spambase_80nan": (0.987, 1.000, 1.033, 0.993, 1.000, 1.058),
    "vehicle_20nan": (0.418, 0.473, 0.452, 0.420, 0.505, 0.472),
    "vehicle_60nan": (0.645, 0.701, 0.647, 0.616, 0.668, 0.616),
}


def lvl(v):
    return v.split("_")[1]


def tbl(v):
    return v.split("_")[0]


def show(title, recs):
    """recs: list of (variant, pop, outcome)."""
    print(f"--- {title}")
    for pop in ("induced", "masked"):
        tot = collections.Counter(o for v, p, o in recs if p == pop)
        print(f"  {pop}: {dict(tot)}")
        for keyf, name in ((lvl, "level"), (tbl, "table")):
            groups = collections.defaultdict(collections.Counter)
            for v, p, o in recs:
                if p == pop:
                    groups[keyf(v)][o] += 1
            print(f"    by {name}: " + "; ".join(f"{g} {dict(c)}" for g, c in sorted(groups.items())))


def mean_cmp(m, b):
    return "below" if m < b else ("tie" if m == b else "above")


recs_bar, recs_mf = [], []
for v, (im, ib, imf, mm, mb, mmf) in ADR17.items():
    recs_bar += [(v, "induced", mean_cmp(im, ib)), (v, "masked", mean_cmp(mm, mb))]
    recs_mf += [(v, "induced", mean_cmp(im, imf)), (v, "masked", mean_cmp(mm, mmf))]
show("E35 means (ADR 0017 table, 3 dp): model vs bar of 5", recs_bar)
show("E35 means (ADR 0017 table, 3 dp): model vs missForest", recs_mf)

pat = re.compile(r"^(\S+)\s+(induced|masked)\s+model.*?m-bar .*?-> (\w+)", re.M)
recsA = [(v, p, o) for v, p, o in pat.findall(secA)]
show("E35 paired, model vs bar (lgbm bars proxied by hgb)", recsA)

patB = re.compile(r"^(\S+)\s+(induced|masked)\s+model.*?m-bar .*?-> (\w+) \| mF .*?-> (\w+)", re.M)
fb = patB.findall(secB)
show("seeds 101/202/303 paired, model vs bar of 5", [(v, p, o) for v, p, o, _ in fb])
show("seeds 101/202/303 paired, model vs missForest", [(v, p, o) for v, p, _, o in fb])
print("non-'better' lists:")
for name, recs in (("A bar", recsA), ("B bar", [(v, p, o) for v, p, o, _ in fb]),
                   ("B mF", [(v, p, o) for v, p, _, o in fb])):
    print(" ", name, [(v, p, o) for v, p, o in recs if o != "better"])
