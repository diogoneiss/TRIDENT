import sqlite3, sys, collections, statistics
db = r"C:/Users/Diogo Neiss/Documents/Mestrado/TRIDENT/mlflow.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cur = con.cursor()
keys = {
 "H": "cv/test/impute/induced/impute_score/mean",
 "K": "cv/test/impute/induced/induced_checkpoint/impute_score/mean",
 "B": "cv/test/impute/induced/baseline/best/impute_score/mean",
}
for exp in ("selection-2026-10-03", "selection-replication-2026-10-04"):
    runs = cur.execute("""select t.run_uuid from tags t join tags r on r.run_uuid=t.run_uuid
        join runs ru on ru.run_uuid=t.run_uuid
        where t.key='experiment' and t.value=? and r.key='run_role' and r.value='parent' and ru.lifecycle_stage='active'""", (exp,)).fetchall()
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for (rid,) in runs:
        ds = cur.execute("select value from params where run_uuid=? and key='dataset_name'", (rid,)).fetchone()
        if ds is None:
            ds = cur.execute("select value from tags where run_uuid=? and key in ('dataset','dataset_name')", (rid,)).fetchone()
        name = ds[0] if ds else rid
        for lab, k in keys.items():
            v = cur.execute("select value from latest_metrics where run_uuid=? and key=?", (rid, k)).fetchone()
            if v: per[name][lab].append(v[0])
    print(exp, "parents", len(runs), "variants", len(per))
    nh = nk = 0
    for name in sorted(per):
        d = per[name]
        if not all(d[l] for l in "HKB"):
            print("  missing", name, {l: len(d[l]) for l in "HKB"}); continue
        h, k, b = (statistics.mean(d[l]) for l in "HKB")
        nh += h < b; nk += k < b
        flag = "K-only" if (k < b and not h < b) else ("H-only" if (h < b and not k < b) else "")
        print(f"  {name:22s} H={h:.4f} K={k:.4f} bar={b:.4f} K-H={k-h:+.4f} {flag}")
    print("  below bar: H", nh, "K", nk)
