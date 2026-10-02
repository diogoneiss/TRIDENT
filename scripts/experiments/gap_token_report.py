"""Pre-registered analysis of the gaps-as-mask training study (imputation-token-shape/02).

Read-only. Lower impute_score is better, so a negative (G - M) difference favours showing the
gaps as [MASK] in training. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/gap_token_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/gap_token_report.py --tally
    uv run --python 3.11 python scripts/experiments/gap_token_report.py --todo 1|2|3|4

Arm G comes from this study (tag experiment=gap-token-2026-10-02). The reference M is read
from E31's cells (tag experiment=column-wise-2026-10-02), which re-read E30's arm-M models and
hold both the headline mask-path score and the column-wise one; E31 verified their mask path
equals E30's arm M to the last digit. G and M are different trained models, so a pair is two
draws, as in E30. ``--todo`` prints a queue's remaining cells, assigned longest first by E30's
measured arm-M cell times to the least-loaded of four queues.
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
EXPERIMENT = "gap-token-2026-10-02"
REFERENCE = "column-wise-2026-10-02"
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
COLUMN_WISE = "impute/induced/column_wise/impute_score"
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


RUNS = {EXPERIMENT: _tagged(EXPERIMENT), REFERENCE: _tagged(REFERENCE)}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uris = RUNS[EXPERIMENT if arm == "G" else REFERENCE].get((variant, seed), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed} {arm}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def verdict(variant: str, key_g: str = INDUCED, key_m: str = INDUCED, label: str = "M") -> tuple[str, float, float, float, list[float]] | None:
    pairs = [(folds(variant, s, "G"), folds(variant, s, "M")) for s in SEEDS]
    if any(g is None or m is None for g, m in pairs):
        return None
    per_seed = [g[key_g].to_numpy() - m[key_m].to_numpy() for g, m in pairs]  # type: ignore[index]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(x < 0 for x in signs):
        call = f"G better than {label}"
    elif lo > 0 and all(x > 0 for x in signs):
        call = f"{label} better than G"
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
        print(
            f"  induced  G {pd.concat(g)[INDUCED].mean():.4f}  M {pd.concat(m)[INDUCED].mean():.4f}  "
            f"M column-wise {pd.concat(m)[COLUMN_WISE].mean():.4f}  best baseline {base:.4f}"
        )
        print(f"  masked   G {pd.concat(g)[MASKED].mean():.4f}  M {pd.concat(m)[MASKED].mean():.4f}")
    for title, key_g, key_m, label in (
        ("G - M, induced (primary)", INDUCED, INDUCED, "M"),
        ("G - M column-wise, induced (secondary)", INDUCED, COLUMN_WISE, "column-wise"),
        ("G - M, masked (secondary)", MASKED, MASKED, "M"),
    ):
        result = verdict(variant, key_g, key_m, label)
        if result is None:
            print(f"    {title}: incomplete")
            continue
        call, mean, lo, hi, signs = result
        seeds = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(SEEDS, signs))
        print(f"    {title} over 15 pairs: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    cost = []
    for arm in "GM":
        sums = [float(f["total_seconds"].sum()) / 60 for s in SEEDS if (f := cells[(s, arm)]) is not None]
        if sums:
            cost.append(f"{arm} {np.mean(sums):.1f} min")
    print("  fold time per cell: " + ", ".join(cost))


def tally() -> None:
    for title, key_g, key_m, label in (
        ("G - M, induced (primary)", INDUCED, INDUCED, "M"),
        ("G - M column-wise, induced (secondary)", INDUCED, COLUMN_WISE, "column-wise"),
    ):
        calls: Counter[str] = Counter()
        heavy: Counter[str] = Counter()
        by_level: dict[int, list[float]] = {}
        for variant in VARIANTS:
            result = verdict(variant, key_g, key_m, label)
            if result is None:
                calls["incomplete"] += 1
                continue
            calls[result[0]] += 1
            if variant in HEAVY:
                heavy[result[0]] += 1
            level = int(re.search(r"_(\d+)nan", variant)[1])  # type: ignore[index]
            by_level.setdefault(level, []).append(result[1])
        print(f"{title}, all 21: " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
        print(f"  the {len(HEAVY)} variants at 60nan or more: " + ", ".join(f"{k} {v}" for k, v in sorted(heavy.items())))
        print("  mean difference by missing level: " + ", ".join(
            f"{level}nan {np.mean(values):+.4f} (n={len(values)})" for level, values in sorted(by_level.items())
        ))
    below = {"G": 0, "M": 0}
    at_or_above = {"G": [], "M": []}
    for variant in VARIANTS:
        cells = {a: [folds(variant, s, a) for s in SEEDS] for a in "GM"}
        if any(c is None for a in "GM" for c in cells[a]):
            continue
        base = float(np.concatenate([_best_baseline(c) for c in cells["M"]]).mean())
        for a in "GM":
            mean = float(pd.concat(cells[a])[INDUCED].mean())
            below[a] += mean < base
            if mean >= 1.0:
                at_or_above[a].append(variant)
    print(f"variants whose mean is below the best baseline: G {below['G']}, M {below['M']}")
    print(f"variants at or above 1.0 (the mean/mode fill): G {at_or_above['G']}, M {at_or_above['M']}")


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
            if (variant, seed) not in RUNS[EXPERIMENT]:
                print(variant, seed)
    else:
        for variant in argv or VARIANTS:
            report(variant)
