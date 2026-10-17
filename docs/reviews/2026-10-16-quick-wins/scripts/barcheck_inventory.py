"""Read-only inventory of mlruns parent artifacts (stdlib only)."""
import csv, json, os, sys, datetime, collections

ROOT = r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT\mlruns"
TZ = datetime.timezone(datetime.timedelta(hours=-3))

rows = []
for exp in sorted(os.listdir(ROOT)):
    ed = os.path.join(ROOT, exp)
    if not os.path.isdir(ed):
        continue
    for run in os.listdir(ed):
        art = os.path.join(ed, run, "artifacts")
        raw = os.path.join(art, "metrics", "raw_fold_metrics.csv")
        if not os.path.exists(raw):
            continue
        try:
            prov = json.load(open(os.path.join(art, "data", "provenance.json"), encoding="utf-8"))
        except Exception:
            prov = {}
        try:
            hp = json.load(open(os.path.join(art, "parameters", "hyperparameters.json"), encoding="utf-8"))
        except Exception:
            hp = {}
        with open(raw, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(raw), TZ)
        rows.append(dict(exp=exp, run=run, ds=prov.get("dataset_name"), seed=prov.get("seed"),
                         ckpt=hp.get("DECODE_CHECKPOINT"), layers=hp.get("LAYERS"), dimfeed=hp.get("DIM_FEED"),
                         mt=mt, has_k="impute/induced/induced_checkpoint/impute_score" in header,
                         has_mf="impute/induced/baseline/missforest/impute_score" in header,
                         has_lgbm="impute/induced/baseline/lgbm/impute_score" in header))

lo = datetime.datetime.fromisoformat(sys.argv[1]).replace(tzinfo=TZ)
hi = datetime.datetime.fromisoformat(sys.argv[2]).replace(tzinfo=TZ)
sel = [r for r in rows if lo <= r["mt"] <= hi]
print("total parents with raw_fold_metrics:", len(rows), "in window:", len(sel))
for r in sorted(sel, key=lambda r: (str(r["ds"]), str(r["seed"]), r["mt"])):
    print(r["exp"], r["run"][:8], r["ds"], r["seed"], r["ckpt"], r["layers"], r["dimfeed"],
          r["mt"].strftime("%m-%d %H:%M"), "K" if r["has_k"] else "-", "MF" if r["has_mf"] else "-",
          "LGBM" if r["has_lgbm"] else "-")
