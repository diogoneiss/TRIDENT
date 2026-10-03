"""Pre-registered analysis of the gap-token replication on fresh seeds (imputation-token-shape/03).

Read-only. Lower impute_score is better, so a negative (G - M) difference favours showing the
gaps as [MASK] in the decode stage. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/gap_token_replication_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/gap_token_replication_report.py --tally
    uv run --python 3.11 python scripts/experiments/gap_token_replication_report.py --todo 1|2|3|4

Both arms come from this study (tag experiment=gap-token-replication-2026-10-03, arm=G|M), on
seeds no earlier imputation study used. The pooled secondary adds E32's three seeds (G from
experiment=gap-token-2026-10-02, M from E31's re-read of E30's arm M). ``--todo`` prints a
queue's remaining cells: (variant, seed) pairs are assigned longest first, by twice E30's
measured arm-M cell time, to the least-loaded of four queues, and each pair runs G then M, so M
finds G's baselines in the cache.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from scipy.stats import t

VARIANTS = [
    *(f"{base}_{level}nan" for base in ("credit-g", "kr-vs-kp", "spambase") for level in (20, 40, 60, 80)),
    *(f"{base}_{level}nan" for base in ("vehicle", "biodeg", "kc2") for level in (20, 60)),
    "pendigits_20nan", "letter_20nan", "electricity_20nan",
]
SEEDS = [101, 202, 303]
E32_SEEDS = [42, 7, 13]
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "gap-token-replication-2026-10-03"
E32 = "gap-token-2026-10-02"
E32_REFERENCE = "column-wise-2026-10-02"
QUEUES = 4
CELL_MINUTES = {
    "credit-g_20nan": 5.3, "credit-g_40nan": 4.8, "credit-g_60nan": 3.8, "credit-g_80nan": 4.6,
    "kr-vs-kp_20nan": 12.6, "kr-vs-kp_40nan": 15.2, "kr-vs-kp_60nan": 14.5, "kr-vs-kp_80nan": 11.8,
    "spambase_20nan": 28.9, "spambase_40nan": 29.2, "spambase_60nan": 31.7, "spambase_80nan": 30.2,
    "vehicle_20nan": 4.5, "vehicle_60nan": 3.5, "biodeg_20nan": 6.3, "biodeg_60nan": 6.2,
    "kc2_20nan": 2.6, "kc2_60nan": 2.9, "pendigits_20nan": 41.9, "letter_20nan": 57.4,
    "electricity_20nan": 100.3,
}
HEAVY = [v for v in VARIANTS if int(re.search(r"_(\d+)nan", v)[1]) >= 60]  # type: ignore[index]
INDUCED = "impute/induced/impute_score"
MASKED = "impute/masked/impute_score"
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(experiment: str, with_arm: bool) -> dict[tuple[str, int, str], list[str]]:
    arm = "a.value" if with_arm else "''"
    arm_join = "join tags a on a.run_uuid = r.run_uuid and a.key = 'arm'" if with_arm else ""
    rows = STORE.execute(
        f"""
        select d.value, s.value, {arm}, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        {arm_join}
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        where {SOURCE}""",
        (experiment,),
    )
    found: dict[tuple[str, int, str], list[str]] = {}
    for variant, seed, arm_value, uri in rows:
        found.setdefault((variant, int(seed), arm_value), []).append(uri)
    return found


RUNS = _tagged(EXPERIMENT, with_arm=True)
E32_RUNS = {"G": _tagged(E32, with_arm=False), "M": _tagged(E32_REFERENCE, with_arm=False)}


def _read(uris: list[str], label: str) -> pd.DataFrame | None:
    if len(uris) > 1:
        sys.exit(f"{label}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    if seed in E32_SEEDS:
        return _read(E32_RUNS[arm].get((variant, seed, ""), []), f"E32 {variant} s{seed} {arm}")
    return _read(RUNS.get((variant, seed, arm), []), f"{variant} s{seed} {arm}")


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def verdict(variant: str, seeds: list[int], key: str = INDUCED) -> tuple[str, float, float, float, list[float]] | None:
    pairs = [(folds(variant, s, "G"), folds(variant, s, "M")) for s in seeds]
    if any(g is None or m is None for g, m in pairs):
        return None
    per_seed = [g[key].to_numpy() - m[key].to_numpy() for g, m in pairs]  # type: ignore[index]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = "G better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = "M better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    cells = {(s, a): folds(variant, s, a) for s in SEEDS for a in "GM"}
    missing = [f"s{s}{a}" for (s, a), f in cells.items() if f is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    g = [f for (s, a), f in cells.items() if a == "G" and f is not None]
    m = [f for (s, a), f in cells.items() if a == "M" and f is not None]
    if g and m:
        base = np.concatenate([_best_baseline(c) for c in m]).mean()
        print(f"  induced  G {pd.concat(g)[INDUCED].mean():.4f}  M {pd.concat(m)[INDUCED].mean():.4f}  best baseline {base:.4f}")
        print(f"  masked   G {pd.concat(g)[MASKED].mean():.4f}  M {pd.concat(m)[MASKED].mean():.4f}")
    for title, seeds, key in (
        ("G - M, induced (primary)", SEEDS, INDUCED),
        ("G - M, masked (secondary)", SEEDS, MASKED),
        ("G - M, induced, pooled with E32 (secondary)", E32_SEEDS + SEEDS, INDUCED),
    ):
        result = verdict(variant, seeds, key)
        if result is None:
            print(f"    {title}: incomplete")
            continue
        call, mean, lo, hi, signs = result
        text = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(seeds, signs))
        print(f"    {title} over {5 * len(seeds)} pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {text} -> {call}")
    cost = []
    for arm in "GM":
        sums = [float(f["total_seconds"].sum()) / 60 for s in SEEDS if (f := cells[(s, arm)]) is not None]
        if sums:
            cost.append(f"{arm} {np.mean(sums):.1f} min")
    print("  fold time per cell: " + ", ".join(cost))


def tally() -> None:
    for title, seeds in (("fresh seeds (primary)", SEEDS), ("pooled with E32, six seeds (secondary)", E32_SEEDS + SEEDS)):
        calls: Counter[str] = Counter()
        heavy: Counter[str] = Counter()
        by_level: dict[int, list[float]] = {}
        worse: list[str] = []
        for variant in VARIANTS:
            result = verdict(variant, seeds)
            if result is None:
                calls["incomplete"] += 1
                continue
            calls[result[0]] += 1
            if result[0] == "M better":
                worse.append(variant)
            if variant in HEAVY:
                heavy[result[0]] += 1
            level = int(re.search(r"_(\d+)nan", variant)[1])  # type: ignore[index]
            by_level.setdefault(level, []).append(result[1])
        print(f"G - M, {title}, all 21: " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
        print(f"  the {len(HEAVY)} variants at 60nan or more: " + ", ".join(f"{k} {v}" for k, v in sorted(heavy.items())))
        print("  mean difference by missing level: " + ", ".join(
            f"{level}nan {np.mean(values):+.4f} (n={len(values)})" for level, values in sorted(by_level.items())
        ))
        print(f"  variants where M is better: {worse or 'none'}")
    below = {"G": 0, "M": 0}
    means: dict[str, list[float]] = {"G": [], "M": []}
    for variant in VARIANTS:
        found = {a: [folds(variant, s, a) for s in SEEDS] for a in "GM"}
        cells = {a: [c for c in found[a] if c is not None] for a in "GM"}
        if any(len(cells[a]) < len(SEEDS) for a in "GM"):
            continue
        base = float(np.concatenate([_best_baseline(c) for c in cells["M"]]).mean())
        for a in "GM":
            mean = float(pd.concat(cells[a])[INDUCED].mean())
            means[a].append(mean)
            below[a] += mean < base
    if means["G"]:
        print(f"means over the complete variants: G {np.mean(means['G']):.4f}, M {np.mean(means['M']):.4f} (n={len(means['G'])})")
    print(f"variants whose mean is below the best baseline: G {below['G']}, M {below['M']}")


def queue(name: str) -> list[tuple[str, int, str]]:
    pairs = sorted(
        ((2 * CELL_MINUTES[v], v, s) for v in VARIANTS for s in SEEDS),
        key=lambda pair: (-pair[0], VARIANTS.index(pair[1]), SEEDS.index(pair[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in pairs:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].extend((variant, seed, arm) for arm in "GM")
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"]:
        tally()
    elif argv[:1] == ["--todo"]:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed, arm) not in RUNS:
                print(variant, seed, arm)
    else:
        for variant in argv or VARIANTS:
            report(variant)
