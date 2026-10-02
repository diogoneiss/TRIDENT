"""Pre-registered analysis of the column-wise scoring study (imputation-token-shape/01).

Read-only. Lower impute_score is better, so a negative (column-wise - mask) difference favours
asking one gap column at a time. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/column_wise_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/column_wise_report.py --tally
    uv run --python 3.11 python scripts/experiments/column_wise_report.py --todo 1|2|3|4

Every cell comes from this study (tag experiment=column-wise-2026-10-02). Both scores come from
the same trained model and the same cells, so each fold is one pair. ``--todo`` prints a
queue's remaining cells: (variant, seed) cells are assigned longest first, by E30's measured
time of arm M, to the least-loaded of four queues.
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
EXPERIMENT = "column-wise-2026-10-02"
REFERENCE = ("experiment", "confirmatory-2026-10-01")
QUEUES = 4
# Minutes of one arm-M cell in E30 (five folds, four trainers sharing the GPU).
CELL_MINUTES = {
    "credit-g_20nan": 5.3, "credit-g_40nan": 4.8, "credit-g_60nan": 3.8, "credit-g_80nan": 4.6,
    "kr-vs-kp_20nan": 12.6, "kr-vs-kp_40nan": 15.2, "kr-vs-kp_60nan": 14.5, "kr-vs-kp_80nan": 11.8,
    "spambase_20nan": 28.9, "spambase_40nan": 29.2, "spambase_60nan": 31.7, "spambase_80nan": 30.2,
    "vehicle_20nan": 4.5, "vehicle_60nan": 3.5, "biodeg_20nan": 6.3, "biodeg_60nan": 6.2,
    "kc2_20nan": 2.6, "kc2_60nan": 2.9, "pendigits_20nan": 41.9, "letter_20nan": 57.4,
    "electricity_20nan": 100.3,
}
# The variants where most of a row is gaps, on which the reading is fixed (ticket 01).
HEAVY = [v for v in VARIANTS if int(re.search(r"_(\d+)nan", v)[1]) >= 60]  # type: ignore[index]
MASK = "impute/induced/impute_score"
COLUMN_WISE = "impute/induced/column_wise/impute_score"
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(value: str, arm: str | None = None) -> dict[tuple[str, int], list[str]]:
    arm_join = "join tags a on a.run_uuid = r.run_uuid and a.key = 'arm' and a.value = ?" if arm else ""
    rows = STORE.execute(
        f"""
        select d.value, s.value, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        {arm_join}
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        where {SOURCE}""",
        (value, arm) if arm else (value,),
    )
    found: dict[tuple[str, int], list[str]] = {}
    for variant, seed, uri in rows:
        found.setdefault((variant, int(seed)), []).append(uri)
    return found


RUNS = _tagged(EXPERIMENT)


def _folds(uris: list[str], label: str) -> pd.DataFrame | None:
    if len(uris) > 1:
        sys.exit(f"{label}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def folds(variant: str, seed: int) -> pd.DataFrame | None:
    return _folds(RUNS.get((variant, seed), []), f"{variant} s{seed}")


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def verdict(variant: str) -> tuple[str, float, float, float, list[float]] | None:
    cells = [folds(variant, s) for s in SEEDS]
    if any(c is None for c in cells):
        return None
    per_seed = [c[COLUMN_WISE].to_numpy() - c[MASK].to_numpy() for c in cells]  # type: ignore[index]
    m, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = "column-wise better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = "mask better"
    else:
        call = "no detectable difference"
    return call, m, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    cells = {s: folds(variant, s) for s in SEEDS}
    missing = [f"s{s}" for s, c in cells.items() if c is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    done = [c for c in cells.values() if c is not None]
    if not done:
        return
    frame = pd.concat(done)
    base = np.concatenate([_best_baseline(c) for c in done])
    print(
        f"  mask {frame[MASK].mean():.4f}  column-wise {frame[COLUMN_WISE].mean():.4f}  "
        f"best baseline {base.mean():.4f}"
    )
    result = verdict(variant)
    if result is None:
        print(f"  incomplete ({len(done)} of {len(SEEDS)} seeds)")
        return
    call, m, lo, hi, signs = result
    seeds = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(SEEDS, signs))
    print(f"  column-wise - mask over 15 pairs: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")


def integrity() -> None:
    """Each cell's mask-path scores must equal E30's arm-M cell to the last digit."""
    reference = _tagged(REFERENCE[1], arm="M")
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            new = folds(variant, seed)
            old = _folds(reference.get((variant, seed), []), f"E30 {variant} s{seed}")
            if new is None or old is None:
                continue
            checked += 1
            for key in (MASK, "impute/masked/impute_score"):
                if not np.array_equal(new[key].to_numpy(), old[key].to_numpy()):
                    differing.append(f"{variant} s{seed} {key}")
    print(f"integrity: {checked} cells against E30's M; differing: {differing or 'none'}")


def tally() -> None:
    calls: Counter[str] = Counter()
    heavy: Counter[str] = Counter()
    by_level: dict[int, list[float]] = {}
    below = {"mask": 0, "column-wise": 0}
    for variant in VARIANTS:
        result = verdict(variant)
        if result is None:
            calls["incomplete"] += 1
            continue
        calls[result[0]] += 1
        if variant in HEAVY:
            heavy[result[0]] += 1
        level = int(re.search(r"_(\d+)nan", variant)[1])  # type: ignore[index]
        by_level.setdefault(level, []).append(result[1])
        done = [c for s in SEEDS if (c := folds(variant, s)) is not None]
        frame = pd.concat(done)
        base = float(np.concatenate([_best_baseline(c) for c in done]).mean())
        below["mask"] += frame[MASK].mean() < base
        below["column-wise"] += frame[COLUMN_WISE].mean() < base
    print("all 21: " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
    print(f"the {len(HEAVY)} variants at 60nan or more: " + ", ".join(f"{k} {v}" for k, v in sorted(heavy.items())))
    print("mean column-wise - mask by missing level: " + ", ".join(
        f"{level}nan {np.mean(values):+.4f} (n={len(values)})" for level, values in sorted(by_level.items())
    ))
    print(f"variants whose mean is below the best baseline: mask {below['mask']}, column-wise {below['column-wise']}")
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
            if (variant, seed) not in RUNS:
                print(variant, seed)
    else:
        for variant in argv or VARIANTS:
            report(variant)
