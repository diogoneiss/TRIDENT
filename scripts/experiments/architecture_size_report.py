"""Pre-registered analysis of the bigger encoder by table size (imputation-architecture/02).

Read-only. Lower impute_score is better, so a negative (ALL - R) difference favours the bigger
encoder. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/architecture_size_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/architecture_size_report.py --tally
    uv run --python 3.11 python scripts/experiments/architecture_size_report.py --todo 1|2|3|4

Arm ALL (4 layers, feed-forward 256, a final LayerNorm) comes from this study (tag
experiment=architecture-large-2026-10-06) on seeds 101, 202 and 303, which E37 did not use. The
reference R is today's default on the same seeds, read from E34's cells (tag
experiment=selection-2026-10-03) as their K readout (``impute/*/induced_checkpoint/*``), which
equals a default run to the last digit. Variants are grouped by the rows of their table, fixed
before any score was seen: large (10,000 rows or more), medium, small. The integrity check holds
each pair's baseline scores, which never depend on the model, equal. ``--todo`` prints a queue's
remaining cells, assigned longest first by E30's arm-M cell time times 1.82, ALL's cost in E37.
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

GROUPS = {
    "large": ["pendigits_20nan", "letter_20nan", "electricity_20nan"],
    "medium": [f"{base}_{level}nan" for base in ("kr-vs-kp", "spambase") for level in (20, 40, 60, 80)],
    "small": [
        *(f"credit-g_{level}nan" for level in (20, 40, 60, 80)),
        *(f"{base}_{level}nan" for base in ("vehicle", "biodeg", "kc2") for level in (20, 60)),
    ],
}
VARIANTS = [v for group in GROUPS.values() for v in group]
SEEDS = [101, 202, 303]
ARM = "ALL"
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "architecture-large-2026-10-06"
REFERENCE = "selection-2026-10-03"
QUEUES = 4
ALL_FACTOR = 1.82
CELL_MINUTES = {
    "credit-g_20nan": 5.3, "credit-g_40nan": 4.8, "credit-g_60nan": 3.8, "credit-g_80nan": 4.6,
    "kr-vs-kp_20nan": 12.6, "kr-vs-kp_40nan": 15.2, "kr-vs-kp_60nan": 14.5, "kr-vs-kp_80nan": 11.8,
    "spambase_20nan": 28.9, "spambase_40nan": 29.2, "spambase_60nan": 31.7, "spambase_80nan": 30.2,
    "vehicle_20nan": 4.5, "vehicle_60nan": 3.5, "biodeg_20nan": 6.3, "biodeg_60nan": 6.2,
    "kc2_20nan": 2.6, "kc2_60nan": 2.9, "pendigits_20nan": 41.9, "letter_20nan": 57.4,
    "electricity_20nan": 100.3,
}
KEYS = {
    "induced": ("impute/induced/impute_score", "impute/induced/induced_checkpoint/impute_score"),
    "masked": ("impute/masked/impute_score", "impute/masked/induced_checkpoint/impute_score"),
}
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(experiment: str, label: str) -> dict[tuple[str, int, str], list[str]]:
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
    found: dict[tuple[str, int, str], list[str]] = {}
    for variant, seed, uri in rows:
        found.setdefault((variant, int(seed), label), []).append(uri)
    return found


RUNS = {**_tagged(EXPERIMENT, ARM), **_tagged(REFERENCE, "R")}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uris = RUNS.get((variant, seed, arm), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed} {arm}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def score(cell: pd.DataFrame, arm: str, population: str = "induced") -> np.ndarray:
    arm_key, r_key = KEYS[population]
    return cell[r_key if arm == "R" else arm_key].to_numpy()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def _pairs(variant: str) -> list[tuple[pd.DataFrame, pd.DataFrame]] | None:
    pairs = [(folds(variant, s, ARM), folds(variant, s, "R")) for s in SEEDS]
    complete = [(a, r) for a, r in pairs if a is not None and r is not None]
    return complete if len(complete) == len(SEEDS) else None


def verdict(variant: str, population: str = "induced") -> tuple[str, float, float, float, list[float]] | None:
    pairs = _pairs(variant)
    if pairs is None:
        return None
    per_seed = [score(a, ARM, population) - score(r, "R", population) for a, r in pairs]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = f"{ARM} better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = "R better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def relative(variant: str) -> list[float] | None:
    """The relative change, ALL against R, of each seed's five-fold mean."""
    pairs = _pairs(variant)
    if pairs is None:
        return None
    return [float(score(a, ARM).mean() / score(r, "R").mean() - 1) for a, r in pairs]


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def _minutes(variant: str, arm: str) -> float | None:
    cells = [c for s in SEEDS if (c := folds(variant, s, arm)) is not None]
    return float(np.mean([float(c["total_seconds"].sum()) / 60 for c in cells])) if cells else None


def report(variant: str) -> None:
    group = next(name for name, members in GROUPS.items() if variant in members)
    print(f"=== {variant} ({group})")
    pairs = _pairs(variant)
    if pairs is None:
        print("  incomplete")
        return
    ref_mean = float(np.concatenate([score(r, "R") for _, r in pairs]).mean())
    arm_mean = float(np.concatenate([score(a, ARM) for a, _ in pairs]).mean())
    base = float(np.concatenate([_best_baseline(r) for _, r in pairs]).mean())
    print(f"  induced  R {ref_mean:.4f}  {ARM} {arm_mean:.4f} ({100 * (arm_mean / ref_mean - 1):+.2f}%)  best baseline {base:.4f}")
    for population, role in (("induced", "primary"), ("masked", "secondary")):
        result = verdict(variant, population)
        assert result is not None
        call, mean, lo, hi, signs = result
        seeds = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(SEEDS, signs))
        print(f"    {ARM} - R, {population} ({role}) over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    print(f"  minutes a cell: {ARM} {_minutes(variant, ARM):.1f}, R {_minutes(variant, 'R'):.1f}")


def integrity() -> None:
    """Each pair's baseline scores never depend on the model, so they must be equal."""
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            cell, ref = folds(variant, seed, ARM), folds(variant, seed, "R")
            if cell is None or ref is None:
                continue
            checked += 1
            for name in BASELINES:
                key = f"impute/induced/baseline/{name}/impute_score"
                if not np.array_equal(cell[key].to_numpy(), ref[key].to_numpy()):
                    differing.append(f"{variant} s{seed} {name}")
    print(f"integrity: {checked} pairs, baseline scores differing: {differing or 'none'}")


def tally() -> None:
    large = {v: verdict(v) for v in GROUPS["large"]}
    if any(result is None for result in large.values()):
        print("reading (ticket 02): incomplete")
    else:
        calls = Counter(result[0] for result in large.values() if result is not None)
        if calls["R better"]:
            reading = "refuted (R better on a large table)"
        elif calls[f"{ARM} better"] >= 2:
            reading = "supported: recommend the bigger encoder on the large tables (size-dependent default, by ADR)"
        else:
            reading = "mixed"
        print(f"reading (ticket 02): {reading}")
    for name, members in GROUPS.items():
        calls = Counter()
        units: list[float] = []
        for variant in members:
            result = verdict(variant)
            calls[result[0] if result else "incomplete"] += 1
            units.extend(relative(variant) or [])
        line = f"{name} ({len(members)} variants): " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items()))
        if len(units) > 1:
            m, lo, hi = interval(np.array(units))
            line += f"; mean relative change {100 * m:+.2f}% [{100 * lo:+.2f}%, {100 * hi:+.2f}%] over {len(units)} (variant, seed) units"
        arm_minutes = [x for v in members if (x := _minutes(v, ARM)) is not None]
        ref_minutes = [x for v in members if (x := _minutes(v, "R")) is not None]
        if arm_minutes and len(arm_minutes) == len(ref_minutes):
            line += f"; cost {sum(arm_minutes) / sum(ref_minutes):.2f}x"
        print(line)
    masked = Counter((verdict(v, "masked") or ("incomplete",))[0] for v in VARIANTS)
    print("masked, all 21 (secondary): " + ", ".join(f"{k} {v}" for k, v in sorted(masked.items())))
    integrity()


def queue(name: str) -> list[tuple[str, int, str]]:
    cells = sorted(
        ((CELL_MINUTES[v] * ALL_FACTOR, v, s) for v in VARIANTS for s in SEEDS),
        key=lambda cell: (-cell[0], VARIANTS.index(cell[1]), SEEDS.index(cell[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in cells:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, ARM))
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
