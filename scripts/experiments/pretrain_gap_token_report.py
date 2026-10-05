"""Pre-registered analysis of the pre-training gap-token study (imputation-token-shape/06).

Read-only. Lower impute_score is better, so a negative (PG - R) difference favours showing the
gaps as [MASK] to the pre-training stage too. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/pretrain_gap_token_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/pretrain_gap_token_report.py --tally
    uv run --python 3.11 python scripts/experiments/pretrain_gap_token_report.py --todo 1|2|3|4

Arm PG comes from this study (tag experiment=pretrain-gap-token-2026-10-05). The reference R is
today's default, read from E35's cells (tag experiment=selection-replication-2026-10-04) as their
K readout (``impute/*/induced_checkpoint/*``), which equals a default run to the last digit:
with no decode patience the trajectory is the same whichever checkpoint is kept. PG and R consume
the same random stream (showing a gap as [MASK] draws nothing), so a pair shares folds, masks and
initialisation; the integrity check holds each pair's baseline scores, which never depend on the
model, equal. ``--todo`` prints a queue's remaining cells, assigned as in E34 and E35.
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
SEEDS = [42, 7, 13]
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "pretrain-gap-token-2026-10-05"
REFERENCE = "selection-replication-2026-10-04"
QUEUES = 4
CELL_MINUTES = {
    "credit-g_20nan": 5.3, "credit-g_40nan": 4.8, "credit-g_60nan": 3.8, "credit-g_80nan": 4.6,
    "kr-vs-kp_20nan": 12.6, "kr-vs-kp_40nan": 15.2, "kr-vs-kp_60nan": 14.5, "kr-vs-kp_80nan": 11.8,
    "spambase_20nan": 28.9, "spambase_40nan": 29.2, "spambase_60nan": 31.7, "spambase_80nan": 30.2,
    "vehicle_20nan": 4.5, "vehicle_60nan": 3.5, "biodeg_20nan": 6.3, "biodeg_60nan": 6.2,
    "kc2_20nan": 2.6, "kc2_60nan": 2.9, "pendigits_20nan": 41.9, "letter_20nan": 57.4,
    "electricity_20nan": 100.3,
}
# The score of each arm, by population: PG's headline is the gap-chosen checkpoint under the
# default; R's is the same checkpoint read from E35's diagnostic family.
KEYS = {
    "induced": ("impute/induced/impute_score", "impute/induced/induced_checkpoint/impute_score"),
    "masked": ("impute/masked/impute_score", "impute/masked/induced_checkpoint/impute_score"),
}
EPOCH = "decode/induced_checkpoint_epoch"
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(experiment: str) -> dict[tuple[str, int], list[str]]:
    rows = STORE.execute(
        f"""
        select d.value, s.value, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        where {SOURCE}""",
        (experiment,),
    )
    found: dict[tuple[str, int], list[str]] = {}
    for variant, seed, uri in rows:
        found.setdefault((variant, int(seed)), []).append(uri)
    return found


RUNS = {"PG": _tagged(EXPERIMENT), "R": _tagged(REFERENCE)}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uris = RUNS[arm].get((variant, seed), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed} {arm}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def score(cell: pd.DataFrame, arm: str, population: str = "induced") -> np.ndarray:
    pg_key, r_key = KEYS[population]
    return cell[pg_key if arm == "PG" else r_key].to_numpy()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def verdict(variant: str, population: str = "induced") -> tuple[str, float, float, float, list[float]] | None:
    pairs = [(folds(variant, s, "PG"), folds(variant, s, "R")) for s in SEEDS]
    complete = [(g, r) for g, r in pairs if g is not None and r is not None]
    if len(complete) < len(SEEDS):
        return None
    per_seed = [score(g, "PG", population) - score(r, "R", population) for g, r in complete]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = "PG better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = "R better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    cells = {(s, a): folds(variant, s, a) for s in SEEDS for a in ("PG", "R")}
    missing = [f"s{s}{a}" for (s, a), f in cells.items() if f is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    pg = [f for (s, a), f in cells.items() if a == "PG" and f is not None]
    ref = [f for (s, a), f in cells.items() if a == "R" and f is not None]
    if pg and ref:
        base = float(np.concatenate([_best_baseline(c) for c in ref]).mean())
        print(
            f"  induced  PG {np.concatenate([score(c, 'PG') for c in pg]).mean():.4f}  "
            f"R {np.concatenate([score(c, 'R') for c in ref]).mean():.4f}  best baseline {base:.4f}"
        )
        print(f"  checkpoint epoch: PG {pd.concat(pg)[EPOCH].mean():.0f}, R {pd.concat(ref)[EPOCH].mean():.0f} (fold means)")
    for population, role in (("induced", "primary"), ("masked", "secondary")):
        result = verdict(variant, population)
        if result is None:
            print(f"    PG - R, {population} ({role}): incomplete")
            continue
        call, mean, lo, hi, signs = result
        seeds = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(SEEDS, signs))
        print(f"    PG - R, {population} ({role}) over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    times = [float(f["total_seconds"].sum()) / 60 for f in pg]
    if times:
        print(f"  fold time per PG cell: {np.mean(times):.1f} min")


def integrity() -> None:
    """Each pair's baseline scores never depend on the model, so they must be equal."""
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            pg, ref = folds(variant, seed, "PG"), folds(variant, seed, "R")
            if pg is None or ref is None:
                continue
            checked += 1
            for name in BASELINES:
                key = f"impute/induced/baseline/{name}/impute_score"
                if not np.array_equal(pg[key].to_numpy(), ref[key].to_numpy()):
                    differing.append(f"{variant} s{seed} {name}")
    print(f"integrity: {checked} pairs, baseline scores differing: {differing or 'none'}")


def tally() -> None:
    calls: Counter[str] = Counter()
    by_level: dict[int, list[float]] = {}
    r_better: list[str] = []
    for variant in VARIANTS:
        result = verdict(variant)
        if result is None:
            calls["incomplete"] += 1
            continue
        calls[result[0]] += 1
        if result[0] == "R better":
            r_better.append(variant)
        level = int(re.search(r"_(\d+)nan", variant)[1])  # type: ignore[index]
        by_level.setdefault(level, []).append(result[1])
    if calls["incomplete"]:
        reading = f"incomplete ({calls['incomplete']} variants lack a seed)"
    elif r_better:
        reading = "keep null (R better on " + ", ".join(r_better) + ")"
    elif calls["PG better"] >= 3:
        reading = "recommend PG as the default (an ADR, the user decides)"
    else:
        reading = "qualifies but no gain: keep null"
    print(f"reading (ticket 06): {reading}")
    print("PG - R, induced (primary): " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
    print("  mean difference by missing level: " + ", ".join(
        f"{level}nan {np.mean(values):+.4f}" for level, values in sorted(by_level.items())
    ))
    masked: Counter[str] = Counter()
    for variant in VARIANTS:
        result = verdict(variant, "masked")
        masked[result[0] if result else "incomplete"] += 1
    print("PG - R, masked (secondary): " + ", ".join(f"{k} {v}" for k, v in sorted(masked.items())))
    means: dict[str, list[float]] = {"PG": [], "R": []}
    below: Counter[str] = Counter()
    for variant in VARIANTS:
        read = {a: [folds(variant, s, a) for s in SEEDS] for a in ("PG", "R")}
        cells = {a: [c for c in read[a] if c is not None] for a in ("PG", "R")}
        if any(len(cells[a]) < len(SEEDS) for a in ("PG", "R")):
            continue
        base = float(np.concatenate([_best_baseline(c) for c in cells["R"]]).mean())
        for arm in ("PG", "R"):
            mean = float(np.concatenate([score(c, arm) for c in cells[arm]]).mean())
            means[arm].append(mean)
            below[arm] += mean < base
    if means["PG"]:
        print(f"means over the complete variants: PG {np.mean(means['PG']):.4f}, R {np.mean(means['R']):.4f} (n={len(means['PG'])})")
    print(f"variants whose mean is below the best baseline: PG {below['PG']}, R {below['R']}")
    integrity()


def queue(name: str) -> list[tuple[str, int]]:
    cells = sorted(
        ((CELL_MINUTES[v], v, s) for v in VARIANTS for s in SEEDS),
        key=lambda cell: (-cell[0], VARIANTS.index(cell[1]), SEEDS.index(cell[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in cells:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"]:
        tally()
    elif argv[:1] == ["--todo"]:
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in RUNS["PG"]:
                print(variant, seed)
    else:
        for variant in argv or VARIANTS:
            report(variant)
