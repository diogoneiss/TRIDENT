"""Pre-registered analysis of the calibration at 40% missing and above (imputation-calibration/01).

Read-only. Lower impute_score is better, so a negative (KC - K) difference favours the
calibration. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/calibration_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/calibration_report.py --tally
    uv run --python 3.11 python scripts/experiments/calibration_report.py --todo 1|2|3|4

One run per (variant, seed), tag experiment=calibration-2026-10-16, seeds 404, 505 and 606, which
no earlier study used. Each run carries both arms: K, its headline (the checkpoint chosen on the
validation gaps, ADR 0015), and KC, the same checkpoint after the calibration fitted on the
validation rows' own gaps (``impute/induced/calibrated/*``). The pairs are within-run. NAMED are
the variants where both E34 and E35 found KC better than K. ``--todo`` prints a queue's remaining
cells, assigned longest first by E35's cell times.
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
    *(f"{base}_{level}nan" for base in ("kr-vs-kp", "spambase", "credit-g") for level in (40, 60, 80)),
    "vehicle_60nan", "biodeg_60nan", "kc2_60nan",
]
NAMED = ["kr-vs-kp_80nan", "spambase_60nan", "spambase_80nan", "credit-g_60nan", "credit-g_80nan", "vehicle_60nan"]
SEEDS = [404, 505, 606]
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "calibration-2026-10-16"
K_KEY = "impute/induced/impute_score"
KC_KEY = "impute/induced/calibrated/impute_score"
QUEUES = 4
CELL_MINUTES = {
    "kr-vs-kp_40nan": 19.3, "kr-vs-kp_60nan": 19.0, "kr-vs-kp_80nan": 17.3,
    "spambase_40nan": 36.0, "spambase_60nan": 33.3, "spambase_80nan": 37.0,
    "credit-g_40nan": 4.7, "credit-g_60nan": 4.3, "credit-g_80nan": 5.0,
    "vehicle_60nan": 3.7, "biodeg_60nan": 7.3, "kc2_60nan": 3.0,
}
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged() -> dict[tuple[str, int], list[str]]:
    rows = STORE.execute(
        f"""
        select d.value, s.value, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        where {SOURCE}""",
        (EXPERIMENT,),
    )
    found: dict[tuple[str, int], list[str]] = {}
    for variant, seed, uri in rows:
        found.setdefault((variant, int(seed)), []).append(uri)
    return found


RUNS = _tagged()


def folds(variant: str, seed: int) -> pd.DataFrame | None:
    uris = RUNS.get((variant, seed), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def _cells(variant: str) -> list[pd.DataFrame] | None:
    cells = [folds(variant, s) for s in SEEDS]
    complete = [c for c in cells if c is not None]
    return complete if len(complete) == len(SEEDS) else None


def verdict(variant: str) -> tuple[str, float, float, float, list[float]] | None:
    cells = _cells(variant)
    if cells is None:
        return None
    per_seed = [c[KC_KEY].to_numpy() - c[K_KEY].to_numpy() for c in cells]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(s < 0 for s in signs):
        call = "KC better"
    elif lo > 0 and all(s > 0 for s in signs):
        call = "K better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    named = " (named)" if variant in NAMED else ""
    print(f"=== {variant}{named}")
    cells = _cells(variant)
    result = verdict(variant)
    if cells is None or result is None:
        print("  incomplete")
        return
    k = float(np.concatenate([c[K_KEY].to_numpy() for c in cells]).mean())
    kc = float(np.concatenate([c[KC_KEY].to_numpy() for c in cells]).mean())
    base = float(np.concatenate([_best_baseline(c) for c in cells]).mean())
    print(f"  induced  K {k:.4f}  KC {kc:.4f} ({100 * (kc / k - 1):+.2f}%)  best baseline {base:.4f}")
    call, mean, lo, hi, signs = result
    seeds = ", ".join(f"s{s} {d:+.4f}" for s, d in zip(SEEDS, signs))
    print(f"    KC - K over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    frame = pd.concat(cells)
    parts = []
    if "impute/induced/calibrated/mean_alpha" in frame.columns:
        parts.append(f"mean alpha {frame['impute/induced/calibrated/mean_alpha'].mean():.3f}")
    parts.append(f"thresholded categorical columns per fold {frame['impute/induced/calibrated/n_thresholded_columns'].mean():.1f}")
    parts.append(f"minutes a cell {float(np.mean([c['total_seconds'].sum() / 60 for c in cells])):.1f}")
    print("  " + "; ".join(parts))


def tally() -> None:
    calls = {v: verdict(v) for v in VARIANTS}
    if any(result is None for result in calls.values()):
        missing = [v for v, result in calls.items() if result is None]
        print(f"reading (imputation-calibration/01): incomplete, missing {missing}")
    else:
        worse = [v for v, result in calls.items() if result is not None and result[0] == "K better"]
        named_better = [v for v in NAMED if (result := calls[v]) is not None and result[0] == "KC better"]
        if worse:
            reading = f"refuted (K better on {worse})"
        elif len(named_better) >= 4:
            reading = f"supported (KC better on {len(named_better)} of the 6 named, K better on none)"
        else:
            reading = f"mixed (KC better on {len(named_better)} of the 6 named, K better on none)"
        print(f"reading (imputation-calibration/01): {reading}")
    counts = Counter(result[0] if result else "incomplete" for result in calls.values())
    print("all 12: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for level in (40, 60, 80):
        diffs = [d for v in VARIANTS if v.endswith(f"_{level}nan") and (c := _cells(v)) for d in
                 np.concatenate([x[KC_KEY].to_numpy() - x[K_KEY].to_numpy() for x in c])]
        if diffs:
            print(f"{level}nan: mean KC - K {np.mean(diffs):+.4f} over {len(diffs)} fold pairs")
    below = {arm: 0 for arm in ("K", "KC")}
    for v in VARIANTS:
        cells = _cells(v)
        if cells is None:
            continue
        base = float(np.concatenate([_best_baseline(c) for c in cells]).mean())
        below["K"] += float(np.concatenate([c[K_KEY].to_numpy() for c in cells]).mean()) < base
        below["KC"] += float(np.concatenate([c[KC_KEY].to_numpy() for c in cells]).mean()) < base
    print(f"below the best baseline: K on {below['K']}, KC on {below['KC']} of {len(VARIANTS)}")


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
        for name in argv or VARIANTS:
            report(name)
