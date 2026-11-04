"""Per-epoch cv means (5 folds) of both stages for the c1_diag_e132 runs, from the leader's store, read-only -> JSON."""
import json, sqlite3, sys
c = sqlite3.connect("file:mlflow.db?mode=ro", uri=True)
runs = c.execute("""select r.run_uuid, (select value from tags where run_uuid=r.run_uuid and key='dataset'),
 (select value from tags where run_uuid=r.run_uuid and key='c1_arm') from runs r join tags t on t.run_uuid=r.run_uuid
 where t.key='study' and t.value='c1_diag_e132' and r.status='FINISHED'
 and (select value from tags where run_uuid=r.run_uuid and key='run_role')='parent'
 and (select value from tags where run_uuid=r.run_uuid and key='is_mirror')='false'""").fetchall()
out = {}
for uuid, dataset, arm in runs:
    d = {}
    for key in ("cv/pretrain/train_loss/mean", "cv/pretrain/val_loss/mean", "cv/finetune/train_loss/mean", "cv/finetune/val_loss/mean",
                "cv/finetune/val_loss/ci95_lower", "cv/finetune/val_loss/ci95_upper", "cv/pretrain/val_loss/ci95_lower", "cv/pretrain/val_loss/ci95_upper"):
        d[key] = [v for v, in c.execute("select value from metrics where run_uuid=? and key=? order by step", (uuid, key))]
    kids = c.execute("select run_uuid from tags where key='mlflow.parentRunId' and value=?", (uuid,)).fetchall()
    d["fold_val_f1"] = {k: [v for v, in c.execute("select value from metrics where run_uuid=? and key='finetune/val_f1_macro' order by step", (k,))] for k, in kids}
    out[f"{arm}|{dataset}"] = d
json.dump(out, sys.stdout)
