"""The numbers behind the 2026-10-27 advisor-meeting page, in one pass. Read-only.

    uv run --python 3.11 python -B scripts/experiments/meeting_figures.py <out_dir> [--reuse]

Runs each pre-registered reader unchanged (as ``results_digest.py`` does), keeps its stdout under
``<out_dir>/readers/``, parses the lines each figure needs into ``<out_dir>/meeting_figures.json``,
and adds two reads no reader prints:

- the design trajectory on seeds 42/7/13: the first decoder and normalised pre-training from E30's
  arms A and M, then E35's runs for gaps as ``[MASK]`` (H), the gap-chosen checkpoint (K) and the
  default output (KC from 40nan, K below), all from the store;
- E78's occlusion map on the synthetic table, averaged over folds and seeds from its fold CSVs.

A per-variant line that does not match its pattern stops the script: a silently dropped line would
become a missing cell. ``--reuse`` parses outputs already in ``<out_dir>/readers/`` instead of
rerunning them. Run from the checkout root.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")

E = "scripts/experiments"
READERS: dict[str, list[str]] = {
    "thesis_101": [f"{E}/thesis_table.py", "--seeds", "101/202/303"],
    "tabpfn": [f"{E}/tabpfn_report.py"],
    "tabpfn_seed_sets": [f"{E}/tabpfn_seed_sets_report.py"],
    "confirmation": [f"{E}/confirmation_report.py"],
    "default_refresh": [f"{E}/default_refresh_report.py", "--tally"],
    "mar_family": [f"{E}/mar_family_report.py", "--tally"],
    "mnar_bias": [f"{E}/mnar_bias_report.py"],
    "frozen": [f"{E}/frozen_encoder_report.py", "--tally"],
    "cls_stage2": [f"{E}/cls_stage2_run.py", "--tally", "/scratch2/diogoneiss/trident-2026-10-24-cls-stage2"],
    "occlusion_e78": [f"{E}/occlusion_map.py", "--tally"],
    "occlusion_e79": [f"{E}/occlusion_map.py", "--study", "e79", "--tally"],
    "gap_reliance": [f"{E}/gap_reliance.py", "--tally"],
    "null_gaps": [f"{E}/null_gaps_report.py"],
    "confirmatory": [f"{E}/confirmatory_report.py"],
    "gap_token": [f"{E}/gap_token_report.py"],
    "gap_token_rep": [f"{E}/gap_token_replication_report.py"],
    "selection": [f"{E}/selection_report.py"],
    "selection_rep": [f"{E}/selection_replication_report.py"],
    "calibration": [f"{E}/calibration_report.py"],
    "remask_e75": [f"{E}/remask_ensemble.py", "--tally"],
    "remask_e76": [f"{E}/remask_ensemble.py", "--study", "e76", "--tally"],
    "remask_e77": [f"{E}/remask_ensemble.py", "--conformal"],
    "spread_e80": [f"{E}/occlusion_spread.py"],
    "hybrid_e82": [f"{E}/deployable_hybrid.py", "--tally"],
    "tabpfn_cls_e83": [f"{E}/tabpfn_classification.py", "--tally"],
    "spread_e84": [f"{E}/spread_mechanisms.py", "--tally"],
    "occlusion_e85": [f"{E}/occlusion_map.py", "--study", "e85", "--tally"],
}

NUM = r"[+-]?\d+\.\d+"
CI = rf"({NUM}) \[({NUM}), ({NUM})\]"
CALL = r"(better|worse|none|equivalent)"
TALLY = re.compile(r"^\s*- (.+?): better (\d+) / no difference (\d+)(?: / equivalent (\d+))? / worse (\d+)")


def reader_output(out: Path, name: str, reuse: bool) -> str:
    path = out / "readers" / f"{name}.txt"
    if not (reuse and path.exists()):
        done = subprocess.run(["uv", "run", "--python", "3.11", "python", "-B", *READERS[name]],
                              capture_output=True, text=True, check=False)
        if done.returncode != 0:
            sys.exit(f"reader {name} failed ({done.returncode}):\n{done.stderr[-2000:]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(done.stdout)
    return path.read_text()


def contrast(text: str, label: str) -> dict[str, Any]:
    """``<label> x [lo, hi] call`` anywhere in a line, as one contrast."""
    m = re.search(re.escape(label) + rf"(?: \([^)]*\))?:? {CI} {CALL}", text)
    if m is None:
        raise ValueError(f"no contrast {label!r} in: {text[:200]}")
    return {"d": float(m[1]), "lo": float(m[2]), "hi": float(m[3]), "call": m[4]}


def sections(text: str, header: str = "## ") -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith(header):
            current = line[len(header):].strip()
            out[current] = []
        elif current:
            out[current].append(line)
    return out


def tallies(lines: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in lines:
        m = TALLY.match(line)
        if m:
            out[m[1].strip()] = {"better": int(m[2]), "none": int(m[3]), "equivalent": int(m[4] or 0),
                                 "worse": int(m[5])}
    return out


def variant_lines(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        m = re.match(r"^- `([^`]+)`(?: \(labelled[^)]*\))?: (.*)$", line)
        if m:
            out[m[1]] = m[2]
    return out


def find_section(secs: dict[str, list[str]], prefix: str) -> list[str]:
    hits = [v for k, v in secs.items() if k.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"expected one section starting {prefix!r}, found {len(hits)}: {list(secs)}")
    return hits[0]


def parse_tabpfn(text: str) -> dict[str, Any]:
    secs = sections(text)
    out: dict[str, Any] = {}
    for key, prefix in [("development", "Development set, seeds 101/202/303"),
                        ("confirmation", "Confirmation set (E44)"),
                        ("e57_confirmation", "E57: confirmation set")]:
        lines = find_section(secs, prefix)
        rows: dict[str, Any] = {}
        for variant, rest in variant_lines(lines).items():
            row: dict[str, Any] = {}
            labels = ["K − tabpfn", "KC − tabpfn", "H − tabpfn"] if key == "e57_confirmation" else \
                ["K − tabpfn", "K − strong_live", "K − strong (ADR 0028)"]
            for label in labels:
                row[label] = contrast(rest, label)
            rows[variant] = row
        if not rows:
            raise ValueError(f"no variant lines in {prefix}")
        out[key] = {"variants": rows, "tallies": tallies(lines)}
    return out


def parse_thesis(text: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    header: list[str] = []
    for line in text.splitlines():
        if line.startswith("| variant |"):
            header = [c.strip() for c in line.strip("|").split("|")]
        m = re.match(r"^\| `([^`]+)` \|(.*)\|$", line)
        if m and header:
            cells = [c.strip() for c in m[2].split("|")]
            row: dict[str, Any] = {"K": float(cells[0])}
            for name, cell in zip(header[2:], cells[1:]):
                c = re.match(rf"^([\w/-]+): {CI} {CALL}$", cell)
                if c is None:
                    raise ValueError(f"thesis cell {name} of {m[1]}: {cell!r}")
                row[name] = {"opponent": c[1], "d": float(c[2]), "lo": float(c[3]), "hi": float(c[4]),
                             "call": c[5]}
            rows[m[1]] = row
    if len(rows) != 21:
        raise ValueError(f"thesis table: {len(rows)} variants, expected 21")
    return rows


def parse_confirmation(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, lines in sections(text, "=== ").items():
        line = next((x for x in lines if x.strip().startswith("K - bar:")), None)
        if line is None:
            raise ValueError(f"confirmation: no K - bar for {variant}")
        m = re.search(rf"K - bar: {CI} \([^)]*\) -> (\w+)", line)
        if m is None:
            raise ValueError(f"confirmation line: {line}")
        out[variant] = {"d": float(m[1]), "lo": float(m[2]), "hi": float(m[3]), "call": m[4]}
    if len(out) != 15:
        raise ValueError(f"confirmation: {len(out)} variants, expected 15")
    return out


def parse_seed_sets(text: str) -> dict[str, Any]:
    return {name: tallies(lines) for name, lines in sections(text).items()}


def parse_mar(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lines in sections(text).items():
        m = re.match(r"^`([^`]+)`$", name)
        if m is None:
            continue
        row: dict[str, Any] = {}
        for label in ["(H − bar) MAR minus control", "(H − strong) MAR minus control"]:
            line = next((x for x in lines if x.strip().startswith(f"- {label}")), None)
            if line is None:
                raise ValueError(f"MAR {name}: no {label}")
            row[label] = contrast(line, label)
        out[m[1]] = row
    if len(out) != 12:
        raise ValueError(f"MAR family: {len(out)} variants, expected 12")
    return out


def parse_bias(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lines in sections(text, "### ").items():
        rows: dict[str, Any] = {}
        for line in lines:
            m = re.match(rf"^\| (\w+) \| ({NUM}) \| ({NUM}) \| {CI} \|", line)
            if m:
                rows[m[1]] = {"mnar": float(m[2]), "mcar": float(m[3]), "d": float(m[4]), "lo": float(m[5]),
                              "hi": float(m[6])}
        if len(rows) != 10:
            raise ValueError(f"bias {name}: {len(rows)} methods, expected 10")
        out[name.split(" against ")[0]] = rows
    return out


def parse_frozen(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, lines in sections(text, "=== ").items():
        arms = {m[1]: float(m[2]) for x in lines if (m := re.match(rf"^\s+(FN|FA|FR|C0|G): K ({NUM})$", x))}
        if set(arms) != {"FN", "FA", "FR", "C0", "G"}:
            raise ValueError(f"frozen {variant}: arms {sorted(arms)}")
        out[variant] = arms
    return out


def parse_cls(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"B − A": {}, "C − A": {}}
    for line in text.splitlines():
        m = re.match(rf"^- (\d\dnan) \(A's mean ({NUM})\): ([BC] − A) {CI} (.+?); tables from (.+)$", line)
        if m:
            out[m[3]][m[1]] = {"a_mean": float(m[2]), "d": float(m[4]), "lo": float(m[5]), "hi": float(m[6]),
                               "verdict": m[7], "range": m[8]}
    if len(out["B − A"]) != 5 or len(out["C − A"]) != 5:
        raise ValueError("E71: expected five levels per contrast")
    seeds = re.search(r"Complete seeds: (.+)", text)
    out["complete_seeds"] = seeds[1] if seeds else ""
    return out


def parse_gap_reliance(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, rest in variant_lines(text.splitlines()).items():
        s = re.match(rf"^S {CI} seeds .*?-> (above the SESOI|not above)", rest)
        if s is None:
            raise ValueError(f"E81 {variant}: {rest[:200]}")
        out[variant] = {"S": {"d": float(s[1]), "lo": float(s[2]), "hi": float(s[3]), "call": s[4]},
                        "K null − mask": contrast(rest, "K null − mask")}
    return out


def primary_lines(text: str) -> list[str]:
    return [x.strip() for x in text.splitlines() if x.startswith("Primary")]


def step_counts(text: str, pattern: str, better: tuple[str, ...], worse: tuple[str, ...],
                induced_only: bool = False) -> dict[str, int]:
    """Per-variant verdicts of one study's primary contrast, counted; an unknown verdict stops."""
    counts = {"better": 0, "none": 0, "worse": 0, "n": 0}
    population = "induced"
    for line in text.splitlines():
        if m := re.match(r"^  (induced|masked) ", line):
            population = m[1]
        if re.search(pattern, line) and not (induced_only and population != "induced"):
            verdict = line.rsplit("->", 1)[1].strip()
            counts["n"] += 1
            if verdict in better:
                counts["better"] += 1
            elif verdict in ("no detectable difference", "equivalent"):
                counts["none"] += 1
            elif verdict in worse:
                counts["worse"] += 1
            else:
                raise ValueError(f"unknown verdict {verdict!r} in: {line.strip()}")
    if counts["n"] == 0:
        raise ValueError(f"no primary lines for {pattern!r}")
    return counts


# --- the design trajectory, from the store -------------------------------------------------------

DEV = ["kr-vs-kp_20nan", "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan", "spambase_20nan",
       "spambase_40nan", "spambase_60nan", "spambase_80nan", "credit-g_20nan", "credit-g_40nan",
       "credit-g_60nan", "credit-g_80nan", "vehicle_20nan", "vehicle_60nan", "biodeg_20nan", "biodeg_60nan",
       "kc2_20nan", "kc2_60nan", "pendigits_20nan", "letter_20nan", "electricity_20nan"]
SEEDS = ["42", "7", "13"]
BASE = "cv/test/impute/induced/"
CHECKED = ["LR_PRE", "LR_DECODE", "DIM", "LAYERS", "DIM_FEED", "BATCH", "PROB_MASCARA", "EPOCHS_PRE"]
# (step, experiment tag, config filter, metric suffix or "default")
STEPS: list[tuple[str, str, dict[str, str], str]] = [
    ("first decoder", "confirmatory-2026-10-01",
     {"pretrain_objective": "embedding", "EPOCHS_DECODE": "150", "decode_gap_token": "null"}, "impute_score/mean"),
    ("normalised target, decode 450", "confirmatory-2026-10-01",
     {"pretrain_objective": "embedding_normalized", "EPOCHS_DECODE": "450", "decode_gap_token": "null",
      "decode_patience": "0"}, "impute_score/mean"),
    ("gaps as [MASK] (H)", "selection-replication-2026-10-04",
     {"decode_gap_token": "mask"}, "impute_score/mean"),
    ("checkpoint on validation gaps (K)", "selection-replication-2026-10-04",
     {"decode_gap_token": "mask"}, "induced_checkpoint/impute_score/mean"),
    ("default output (KC from 40nan)", "selection-replication-2026-10-04",
     {"decode_gap_token": "mask"}, "default"),
]


def trajectory() -> dict[str, Any]:
    con = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
    runs = [r for (r,) in con.execute(
        """select r.run_uuid from runs r
           join tags m on m.run_uuid = r.run_uuid and m.key = 'is_mirror' and m.value = 'false'
           join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
           join tags t on t.run_uuid = r.run_uuid and t.key = 'task' and t.value = 'imputation'
           where r.lifecycle_stage = 'active' and r.status = 'FINISHED'""")]
    info: dict[str, dict[str, str]] = defaultdict(dict)
    for r, k, v in con.execute("select run_uuid, key, value from params"):
        info[r][k] = v
    for r, k, v in con.execute("select run_uuid, key, value from tags"):
        info[r][k] = v
    metric: dict[str, dict[str, float]] = defaultdict(dict)
    for r, k, v in con.execute("select run_uuid, key, value from latest_metrics where key like ?",
                               (BASE + "%impute_score/mean",)):
        metric[r][k[len(BASE):]] = float(v)
    steps: list[dict[str, Any]] = []
    for label, experiment, filt, suffix in STEPS:
        cells: dict[str, dict[str, float]] = defaultdict(dict)
        config: dict[str, set[str]] = defaultdict(set)
        for r in runs:
            d = info[r]
            if d.get("experiment") != experiment or d.get("seed") not in SEEDS or d.get("dataset") not in DEV:
                continue
            if any(d.get(k) != v for k, v in filt.items()):
                continue
            rate = int(d["dataset"].rsplit("_", 1)[1].removesuffix("nan"))
            key = suffix
            if suffix == "default":
                key = "induced_checkpoint_calibrated/impute_score/mean" if rate >= 40 else \
                    "induced_checkpoint/impute_score/mean"
            if key not in metric[r]:
                continue
            if d["seed"] in cells[d["dataset"]]:
                raise ValueError(f"{label}: two runs for {d['dataset']} seed {d['seed']}")
            cells[d["dataset"]][d["seed"]] = metric[r][key]
            config[d["dataset"]].add(";".join(f"{k}={d.get(k, '-')}" for k in CHECKED))
        steps.append({"label": label, "experiment": experiment,
                      "variants": {v: mean(s.values()) for v, s in cells.items() if len(s) == 3},
                      "config": {v: sorted(c) for v, c in config.items()}})
    # a variant whose checked hyperparameters differ between steps is labelled, never pooled
    labelled = sorted({v for v in DEV if len({c for s in steps for c in s["config"].get(v, [])}) > 1})
    complete = [v for v in DEV if all(v in s["variants"] for s in steps)]
    pooled = [v for v in complete if v not in labelled]
    for s in steps:
        s["mean_pooled"] = mean(s["variants"][v] for v in pooled)
    return {"seeds": SEEDS, "complete": complete, "labelled": labelled, "pooled": pooled, "steps": steps}


def occlusion_synthetic() -> dict[str, Any]:
    base = Path("/scratch2/diogoneiss/trident-2026-10-26-occlusion/attribution")
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    files = sorted(base.glob("synthblock_40nan_s*/occlusion_fold_*.csv"))
    if len(files) != 15:
        raise ValueError(f"synthetic occlusion: {len(files)} fold files, expected 15")
    for f in files:
        with f.open() as fh:
            for row in csv.DictReader(fh):
                if row["split"] == "test":
                    sums[(row["target"], row["source"])].append(float(row["effect"]))
    columns = sorted({t for t, _ in sums} | {s for _, s in sums})
    return {"columns": columns, "effect": {f"{t}|{s}": mean(v) for (t, s), v in sums.items()}}


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    reuse = "--reuse" in sys.argv
    text = {name: reader_output(out, name, reuse) for name in READERS}
    data: dict[str, Any] = {
        "thesis_101": parse_thesis(text["thesis_101"]),
        "tabpfn": parse_tabpfn(text["tabpfn"]),
        "seed_sets": parse_seed_sets(text["tabpfn_seed_sets"]),
        "confirmation_bar": parse_confirmation(text["confirmation"]),
        "default_refresh": primary_lines(text["default_refresh"]),
        "mar": parse_mar(text["mar_family"]),
        "mnar_bias": parse_bias(text["mnar_bias"]),
        "frozen": parse_frozen(text["frozen"]),
        "cls_stage2": parse_cls(text["cls_stage2"]),
        "gap_reliance": parse_gap_reliance(text["gap_reliance"]),
        "steps_counts": {
            "normalised target, decode 450": step_counts(text["confirmatory"], r"^\s+M - A \(primary\)",
                                                         ("M better than A",), ("A better than M",), True),
            "gaps as [MASK]": step_counts(text["gap_token"], r"G - M, induced \(primary\)",
                                          ("G better than M", "G better"), ("M better than G", "M better")),
            "gaps as [MASK], fresh seeds": step_counts(text["gap_token_rep"], r"G - M, induced \(primary\)",
                                                       ("G better than M", "G better"),
                                                       ("M better than G", "M better")),
            "checkpoint on validation gaps": step_counts(text["selection_rep"], r"K - H \(primary\)",
                                                         ("K better",), ("H better",)),
            "calibration": step_counts(text["calibration"], r"KC - K over", ("KC better",), ("K better",)),
        },
        "primary": {name: primary_lines(text[name]) for name in
                    ["occlusion_e78", "occlusion_e79", "gap_reliance", "null_gaps", "frozen", "remask_e75",
                     "remask_e76", "remask_e77", "spread_e80", "default_refresh", "hybrid_e82",
                     "spread_e84", "occlusion_e85"]}
                   | {"tabpfn_cls_e83": [x.strip() for x in text["tabpfn_cls_e83"].splitlines() if re.match(r"^- \d+nan:", x)]},
        "trajectory": trajectory(),
        "occlusion_synthetic": occlusion_synthetic(),
    }
    (out / "meeting_figures.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    template = out / "page.template.html"
    if template.exists():
        built = subprocess.run(["date", "+%Y-%m-%d %H:%M GMT-3"], capture_output=True, text=True, check=True,
                               env={"TZ": "America/Sao_Paulo"}).stdout.strip()
        page = template.read_text().replace("__BUILT__", built)
        page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
        (out.parent / f"{out.name}.html").write_text(page)
    t = data["trajectory"]
    print(f"trajectory: {len(t['complete'])} complete, labelled {t['labelled']}")
    for s in t["steps"]:
        print(f"  {s['label']:40s} {s['mean_pooled']:.4f} ({len(s['variants'])} variants)")
    for k, v in data["steps_counts"].items():
        print(f"  counts {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
