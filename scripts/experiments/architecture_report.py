"""Pre-registered analysis of the architecture screening (imputation-architecture/01, task T03).

Read-only. Lower impute_score is better, so a negative (arm - R) difference favours the arm. Run
from the checkout root:

    uv run --python 3.11 python scripts/experiments/architecture_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/architecture_report.py --tally
    uv run --python 3.11 python scripts/experiments/architecture_report.py --todo 1|2|3|4

The arms (L4, F256, LN, ALL) come from this study (tag experiment=architecture-2026-10-05). The
reference R is today's default, read from E35's cells (tag
experiment=selection-replication-2026-10-04) as their K readout
(``impute/*/induced_checkpoint/*``), which equals a default run to the last digit. The integrity
check holds each pair's baseline scores, which never depend on the model, equal. ``--todo`` prints
a queue's remaining cells, assigned longest first by an estimated time (E30's arm-M cell time times
an arm factor) to the least-loaded of four queues.
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

VARIANTS = ["credit-g_20nan", "credit-g_80nan", "kr-vs-kp_40nan", "biodeg_60nan", "pendigits_20nan"]
SEEDS = [42, 7, 13]
ARMS = ["L4", "F256", "LN", "ALL"]
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "architecture-2026-10-05"
REFERENCE = "selection-replication-2026-10-04"
QUEUES = 4
# E30's arm-M minutes a cell (four trainers sharing the GPU), and a rough cost factor per arm.
CELL_MINUTES = {"credit-g_20nan": 5.3, "credit-g_80nan": 4.6, "kr-vs-kp_40nan": 15.2, "biodeg_60nan": 6.2, "pendigits_20nan": 41.9}
ARM_FACTOR = {"L4": 1.7, "F256": 1.2, "LN": 1.0, "ALL": 2.0}
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


def _tagged(experiment: str, with_arm: bool) -> dict[tuple[str, int, str], list[str]]:
    arm = "a.value" if with_arm else "'R'"
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


RUNS = {**_tagged(EXPERIMENT, with_arm=True), **_tagged(REFERENCE, with_arm=False)}


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


def verdict(variant: str, arm: str, population: str = "induced") -> tuple[str, float, float, float, list[float]] | None:
    pairs = [(folds(variant, s, arm), folds(variant, s, "R")) for s in SEEDS]
    complete = [(a, r) for a, r in pairs if a is not None and r is not None]
    if len(complete) < len(SEEDS):
        return None
    per_seed = [score(a, arm, population) - score(r, "R", population) for a, r in complete]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = f"{arm} better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = "R better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    print(f"=== {variant}")
    ref = [c for s in SEEDS if (c := folds(variant, s, "R")) is not None]
    if ref:
        base = float(np.concatenate([_best_baseline(c) for c in ref]).mean())
        print(f"  induced  R {np.concatenate([score(c, 'R') for c in ref]).mean():.4f}  best baseline {base:.4f}")
    for arm in ARMS:
        cells = [c for s in SEEDS if (c := folds(variant, s, arm)) is not None]
        if not cells:
            print(f"    {arm}: no cells")
            continue
        mean_score = np.concatenate([score(c, arm) for c in cells]).mean()
        minutes = np.mean([float(c["total_seconds"].sum()) / 60 for c in cells])
        line = f"    {arm} {mean_score:.4f} ({minutes:.1f} min a cell)"
        for population, role in (("induced", "primary"), ("masked", "secondary")):
            result = verdict(variant, arm, population)
            if result is None:
                line += f"; {population}: incomplete"
                continue
            call, mean, lo, hi, signs = result
            seeds = ", ".join(f"{x:+.4f}" for x in signs)
            line += f"; {arm} - R {population} ({role}) {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] ({seeds}) -> {call}"
        print(line)
    if ref:
        minutes = np.mean([float(c["total_seconds"].sum()) / 60 for c in ref])
        print(f"  R: {minutes:.1f} min a cell (E35)")


def integrity() -> None:
    """Each pair's baseline scores never depend on the model, so they must be equal."""
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            ref = folds(variant, seed, "R")
            for arm in ARMS:
                cell = folds(variant, seed, arm)
                if cell is None or ref is None:
                    continue
                checked += 1
                for name in BASELINES:
                    key = f"impute/induced/baseline/{name}/impute_score"
                    if not np.array_equal(cell[key].to_numpy(), ref[key].to_numpy()):
                        differing.append(f"{variant} s{seed} {arm} {name}")
    print(f"integrity: {checked} pairs, baseline scores differing: {differing or 'none'}")


def tally() -> None:
    advancing: list[tuple[int, float, str]] = []
    complete = True
    for arm in ARMS:
        calls: Counter[str] = Counter()
        means: list[float] = []
        for variant in VARIANTS:
            result = verdict(variant, arm)
            if result is None:
                calls["incomplete"] += 1
                complete = False
                continue
            calls[result[0]] += 1
            means.append(result[1])
        mean = float(np.mean(means)) if means else float("nan")
        print(f"{arm} - R, induced (primary): " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())) + f"; mean difference {mean:+.4f}")
        if not calls["incomplete"] and not calls["R better"] and calls[f"{arm} better"] >= 1:
            advancing.append((calls[f"{arm} better"], mean, arm))
    if not complete:
        print("reading (ticket 01): incomplete")
    elif not advancing:
        print("reading (ticket 01): no arm advances; T03 closes at screening scale")
    else:
        best = min(advancing, key=lambda a: (-a[0], a[1]))
        print(f"reading (ticket 01): advancing {[a[2] for a in advancing]}; to the 21-variant confirmation: {best[2]}")
    integrity()


def queue(name: str) -> list[tuple[str, int, str]]:
    cells = sorted(
        ((CELL_MINUTES[v] * ARM_FACTOR[a], v, s, a) for v in VARIANTS for s in SEEDS for a in ARMS),
        key=lambda cell: (-cell[0], VARIANTS.index(cell[1]), SEEDS.index(cell[2]), ARMS.index(cell[3])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed, arm in cells:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, arm))
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
