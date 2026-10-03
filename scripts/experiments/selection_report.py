"""Pre-registered analysis of the checkpoint and calibration study (imputation-token-shape/04).

Read-only. Lower impute_score is better, so a negative (X - H) difference favours X. Run from the
checkout root:

    uv run --python 3.11 python scripts/experiments/selection_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/selection_report.py --tally
    uv run --python 3.11 python scripts/experiments/selection_report.py --todo 1|2|3|4

Every cell comes from this study (tag experiment=selection-2026-10-03) and scores one decode
trajectory four ways: H the headline, C calibrated on the validation rows' own gaps, K at the
epoch those gaps score best, KC both. Each fold is one pair. The integrity check holds H to E33's
arm-G cells of the same variant and seed (tag experiment=gap-token-replication-2026-10-03).
``--todo`` prints a queue's remaining cells, assigned longest first by E30's measured arm-M cell
times to the least-loaded of four queues.
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
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
EXPERIMENT = "selection-2026-10-03"
REFERENCE = "gap-token-replication-2026-10-03"
QUEUES = 4
CELL_MINUTES = {
    "credit-g_20nan": 5.3, "credit-g_40nan": 4.8, "credit-g_60nan": 3.8, "credit-g_80nan": 4.6,
    "kr-vs-kp_20nan": 12.6, "kr-vs-kp_40nan": 15.2, "kr-vs-kp_60nan": 14.5, "kr-vs-kp_80nan": 11.8,
    "spambase_20nan": 28.9, "spambase_40nan": 29.2, "spambase_60nan": 31.7, "spambase_80nan": 30.2,
    "vehicle_20nan": 4.5, "vehicle_60nan": 3.5, "biodeg_20nan": 6.3, "biodeg_60nan": 6.2,
    "kc2_20nan": 2.6, "kc2_60nan": 2.9, "pendigits_20nan": 41.9, "letter_20nan": 57.4,
    "electricity_20nan": 100.3,
}
SCORE = {
    "H": "impute/induced/impute_score",
    "C": "impute/induced/calibrated/impute_score",
    "K": "impute/induced/induced_checkpoint/impute_score",
    "KC": "impute/induced/induced_checkpoint_calibrated/impute_score",
}
PRIMARY = [("C", "H"), ("K", "H")]
SECONDARY = [("KC", "H"), ("KC", "K"), ("KC", "C")]
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(experiment: str, arm: str | None) -> dict[tuple[str, int], list[str]]:
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
        (experiment, arm) if arm else (experiment,),
    )
    found: dict[tuple[str, int], list[str]] = {}
    for variant, seed, uri in rows:
        found.setdefault((variant, int(seed)), []).append(uri)
    return found


RUNS = _tagged(EXPERIMENT, None)


def _read(uris: list[str], label: str) -> pd.DataFrame | None:
    if len(uris) > 1:
        sys.exit(f"{label}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def folds(variant: str, seed: int) -> pd.DataFrame | None:
    return _read(RUNS.get((variant, seed), []), f"{variant} s{seed}")


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def verdict(variant: str, arm: str, reference: str) -> tuple[str, float, float, float, list[float]] | None:
    found = [folds(variant, s) for s in SEEDS]
    cells = [c for c in found if c is not None]
    if len(cells) < len(SEEDS):
        return None
    per_seed = [c[SCORE[arm]].to_numpy() - c[SCORE[reference]].to_numpy() for c in cells]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = f"{arm} better"
    elif lo > 0 and all(x > 0 for x in signs):
        call = f"{reference} better"
    else:
        call = "no detectable difference"
    return call, mean, lo, hi, signs


def _best_baseline(cell: pd.DataFrame) -> np.ndarray:
    name = min(BASELINES, key=lambda b: float(cell[f"impute/induced/baseline/{b}/impute_score"].mean()))
    return cell[f"impute/induced/baseline/{name}/impute_score"].to_numpy()


def report(variant: str) -> None:
    found = [folds(variant, s) for s in SEEDS]
    cells = [c for c in found if c is not None]
    missing = [f"s{s}" for s, c in zip(SEEDS, found) if c is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    if not cells:
        return
    frame = pd.concat(cells)
    base = float(np.concatenate([_best_baseline(c) for c in cells]).mean())
    print("  induced  " + "  ".join(f"{arm} {frame[key].mean():.4f}" for arm, key in SCORE.items()) + f"  best baseline {base:.4f}")
    parts = [
        f"checkpoint epoch by loss {frame['decode/loss_checkpoint_epoch'].mean():.0f}",
        f"by validation gaps {frame['decode/induced_checkpoint_epoch'].mean():.0f} (fold means)",
    ]
    if "impute/induced/calibrated/mean_alpha" in frame.columns:
        parts.append(f"mean alpha {frame['impute/induced/calibrated/mean_alpha'].mean():.3f}")
    parts.append(f"thresholded categorical columns per fold {frame['impute/induced/calibrated/n_thresholded_columns'].mean():.1f}")
    print("  " + "; ".join(parts))
    for role, comparisons in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for arm, reference in comparisons:
            result = verdict(variant, arm, reference)
            if result is None:
                print(f"    {arm} - {reference} ({role}): incomplete")
                continue
            call, mean, lo, hi, signs = result
            seeds = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(SEEDS, signs))
            print(f"    {arm} - {reference} ({role}) over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    masked = frame["impute/masked/induced_checkpoint/impute_score"] - frame["impute/masked/impute_score"]
    print(f"  masked K - H (secondary): {masked.mean():+.4f}")
    print(f"  fold time per cell: {np.mean([float(c['total_seconds'].sum()) / 60 for c in cells]):.1f} min")


def integrity() -> None:
    """H must equal E33's arm-G cell of the same variant and seed, to the last digit."""
    reference = _tagged(REFERENCE, "G")
    checked, differing = 0, []
    for variant in VARIANTS:
        for seed in SEEDS:
            new = folds(variant, seed)
            old = _read(reference.get((variant, seed), []), f"E33 {variant} s{seed}")
            if new is None or old is None:
                continue
            checked += 1
            for key in (SCORE["H"], "impute/masked/impute_score"):
                if not np.array_equal(new[key].to_numpy(), old[key].to_numpy()):
                    differing.append(f"{variant} s{seed} {key}")
    print(f"integrity: {checked} cells against E33's G; differing: {differing or 'none'}")


def tally() -> None:
    for role, comparisons in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for arm, reference in comparisons:
            calls: Counter[str] = Counter()
            worse: list[str] = []
            by_level: dict[int, list[float]] = {}
            for variant in VARIANTS:
                result = verdict(variant, arm, reference)
                if result is None:
                    calls["incomplete"] += 1
                    continue
                calls[result[0]] += 1
                if result[0] == f"{reference} better":
                    worse.append(variant)
                level = int(re.search(r"_(\d+)nan", variant)[1])  # type: ignore[index]
                by_level.setdefault(level, []).append(result[1])
            print(f"{arm} - {reference} ({role}): " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
            print("  mean difference by missing level: " + ", ".join(
                f"{level}nan {np.mean(values):+.4f}" for level, values in sorted(by_level.items())
            ))
            print(f"  variants where {reference} is better: {worse or 'none'}")
    means: dict[str, list[float]] = {arm: [] for arm in SCORE}
    below: Counter[str] = Counter()
    for variant in VARIANTS:
        found = [folds(variant, s) for s in SEEDS]
        cells = [c for c in found if c is not None]
        if len(cells) < len(SEEDS):
            continue
        frame = pd.concat(cells)
        base = float(np.concatenate([_best_baseline(c) for c in cells]).mean())
        for arm, key in SCORE.items():
            mean = float(frame[key].mean())
            means[arm].append(mean)
            below[arm] += mean < base
    if means["H"]:
        print("means over the complete variants: " + ", ".join(f"{arm} {np.mean(v):.4f}" for arm, v in means.items()))
        print("variants whose mean is below the best baseline: " + ", ".join(f"{arm} {below[arm]}" for arm in SCORE))
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
