"""Build mfreplay_report.json: seed-42 rows (replay, sequential, arm C where run, mean_mode,
logged missForest, ADR 3-seed K, seed-42 K), pass counts, and the pre-registered verdicts."""
import glob
import json
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
TIE = 0.003
ADR = {  # ADR 0017 Outcome table, 3-seed means: induced K, induced MF, masked K, masked MF
    "biodeg_60nan": (0.761, 0.762, 0.809, 0.786),
    "credit-g_40nan": (0.956, 0.931, 0.982, 0.950),
    "electricity_20nan": (0.618, 0.619, 0.686, 0.710),
    "kc2_20nan": (0.541, 0.605, 0.530, 0.592),
    "kr-vs-kp_40nan": (0.758, 0.801, 0.781, 0.824),
    "pendigits_20nan": (0.332, 0.334, 0.369, 0.409),
    "spambase_40nan": (0.903, 0.942, 0.897, 0.972),
    "vehicle_60nan": (0.645, 0.647, 0.616, 0.616),
}
logged = json.load(open(os.path.join(SP, "mfreplay_logged.json")))
old = {}
for p in glob.glob(os.path.join(SP, "mfreplay_s42_p*.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        old.setdefault(r["variant"], {})[r["fold"]] = r
new = {}
for p in sorted(glob.glob(os.path.join(SP, "mfreplay_passes_s42*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        new.setdefault(r["variant"], {})[r["fold"]] = r


def verdict(x, k):
    d = x - k
    return "MF wins" if d < -TIE else ("MF loses" if d > TIE else "tie")


rows = []
for variant in ADR:
    folds = old[variant]
    assert sorted(folds) == [1, 2, 3, 4, 5]
    log = logged[f"{variant}|42"]
    fit = [(folds[f]["info"]["kept_rounds"], folds[f]["info"]["fitted_rounds"]) for f in range(1, 6)]
    for pi, pop in enumerate(("induced", "masked")):
        get = lambda arm, src=folds: np.array([src[f]["metrics"][f"impute/{pop}/baseline/{arm}/impute_score"] for f in range(1, 6)])
        rep, seq, mm = get("missforest"), get("missforest_seq"), get("mean_mode")
        k42 = log["parent_means"][f"{pop}/K"]
        lmf = log["parent_means"][f"{pop}/missforest"]
        row = dict(variant=variant, population=pop, seed=42, folds=5,
                   stock_replay=float(rep.mean()), sequential=float(seq.mean()), mean_mode=float(mm.mean()),
                   logged_missforest_seed42=lmf, replay_minus_logged=float(rep.mean() - lmf),
                   decoder_k_adr_3seed=ADR[variant][2 * pi], missforest_adr_3seed=ADR[variant][2 * pi + 1],
                   decoder_k_seed42=k42,
                   seq_minus_replay=float(seq.mean() - rep.mean()),
                   seq_minus_replay_per_fold=[round(float(x), 4) for x in seq - rep],
                   replay_vs_K42=verdict(rep.mean(), k42), seq_vs_K42=verdict(seq.mean(), k42),
                   replay_minus_K42=float(rep.mean() - k42), seq_minus_K42=float(seq.mean() - k42),
                   fit_kept_fitted_rounds=fit)
        if variant in new and sorted(new[variant]) == [1, 2, 3, 4, 5]:
            nf = new[variant]
            assert all(nf[f]["metrics"][f"impute/{pop}/baseline/{a}/impute_score"] == folds[f]["metrics"][f"impute/{pop}/baseline/{a}/impute_score"]
                       for f in range(1, 6) for a in ("missforest", "missforest_seq", "mean_mode")), variant
            c = get("missforest_seqstock", nf)
            row.update(seq_then_stock=float(c.mean()), seq_then_stock_minus_replay=float(c.mean() - rep.mean()),
                       seq_then_stock_vs_K42=verdict(c.mean(), k42),
                       stock_test_passes_computed_kept=[nf[f]["info"][f"passes_stock_{pop}"][0] for f in range(1, 6)],
                       seqstock_test_passes_computed_kept=[nf[f]["info"][f"passes_seqstock_{pop}"][0] for f in range(1, 6)],
                       S4=all(nf[f]["info"][f"S4_{pop}_verbatim_equals_stock"] for f in range(1, 6)),
                       rerun_bitwise_equal=True)
        rows.append(row)

summary = {}
for key in ("replay_vs_K42", "seq_vs_K42"):
    for pop in ("induced", "masked"):
        vs = [r[key] for r in rows if r["population"] == pop]
        summary[f"{key}/{pop}"] = {v: vs.count(v) for v in ("MF wins", "tie", "MF loses")}
d = [r["seq_minus_replay"] for r in rows]
summary["seq_better_than_replay_beyond_tie"] = sum(x < -TIE for x in d)
summary["seq_worse_than_replay_beyond_tie"] = sum(x > TIE for x in d)
summary["seq_replay_tie"] = sum(abs(x) <= TIE for x in d)
summary["flips_where_replay_loses_or_ties_and_seq_wins"] = [
    (r["variant"], r["population"]) for r in rows if r["replay_vs_K42"] != "MF wins" and r["seq_vs_K42"] == "MF wins"]
summary["flips_where_replay_wins_or_ties_and_seq_loses"] = [
    (r["variant"], r["population"]) for r in rows if r["replay_vs_K42"] != "MF loses" and r["seq_vs_K42"] == "MF loses"]
# Crude projection: ADR 3-seed missForest shifted by the seed-42 (seq - replay) delta, against ADR 3-seed K.
summary["crude_projection_3seed"] = [
    dict(variant=r["variant"], population=r["population"], adr_mf=r["missforest_adr_3seed"], adr_k=r["decoder_k_adr_3seed"],
         projected_seq=round(r["missforest_adr_3seed"] + r["seq_minus_replay"], 4),
         decoder_lower_now=r["decoder_k_adr_3seed"] < r["missforest_adr_3seed"],
         decoder_lower_projected=r["decoder_k_adr_3seed"] < r["missforest_adr_3seed"] + r["seq_minus_replay"])
    for r in rows]
json.dump(dict(rows=rows, summary=summary), open(os.path.join(SP, "mfreplay_report.json"), "w"), indent=1)
for r in rows:
    print(r["variant"], r["population"], "rep %.4f seq %.4f C %s K42 %.4f Kadr %.3f log %.4f | %s -> %s | passes %s" % (
        r["stock_replay"], r["sequential"], ("%.4f" % r["seq_then_stock"]) if "seq_then_stock" in r else "-",
        r["decoder_k_seed42"], r["decoder_k_adr_3seed"], r["logged_missforest_seed42"], r["replay_vs_K42"], r["seq_vs_K42"],
        r.get("stock_test_passes_computed_kept", "-")))
print(json.dumps(summary, indent=1))
