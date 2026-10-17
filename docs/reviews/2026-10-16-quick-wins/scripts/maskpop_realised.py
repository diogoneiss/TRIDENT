import sqlite3, sys, statistics as st, pathlib
db = pathlib.Path(r"C:\Users\Diogo Neiss\Documents\Mestrado\TRIDENT\mlflow.db").as_uri() + "?mode=ro"
c = sqlite3.connect(db, uri=True)
q = """
select t.value as ds, m.value, r.start_time, coalesce(gt.value,'?') as gap
from latest_metrics m join runs r on r.run_uuid=m.run_uuid
join tags t on t.run_uuid=r.run_uuid and t.key in ('dataset','dataset_name')
left join tags gt on gt.run_uuid=r.run_uuid and gt.key='decode_gap_token'
where m.key='cv/test/impute/masked/realised_rate/mean' or m.key='impute/masked/realised_rate'
"""
try:
    rows = c.execute(q).fetchall()
except Exception as e:
    print("err", e); keys = c.execute("select distinct key from latest_metrics where key like '%realised%'").fetchall(); print(keys); sys.exit()
from collections import defaultdict
d = defaultdict(list)
for ds, v, t, g in rows:
    d[(ds, g)].append(v)
for k in sorted(d):
    vals = d[k]
    print(k, len(vals), round(min(vals),4), round(st.median(vals),4), round(max(vals),4))
