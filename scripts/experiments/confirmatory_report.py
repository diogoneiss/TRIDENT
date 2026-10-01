"""Pre-registered analysis of the confirmatory study on 21 variants (imputation-pretraining/07).

Read-only. Lower impute_score is better, so a positive (X - A) difference favours A. Run from
the checkout root:

    uv run --python 3.11 python scripts/experiments/confirmatory_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/confirmatory_report.py --tally
    uv run --python 3.11 python scripts/experiments/confirmatory_report.py --todo 1|2|3|4

Every cell of A, M and P comes from this study (tag experiment=confirmatory-2026-10-01).
``--todo`` prints a queue's remaining cells: the (variant, seed) trios are assigned longest
first to the least-loaded of four queues, and each trio runs A, M, P in turn, so M and P find
A's baselines in the cache.
"""

from __future__ import annotations

import re
import sqlite3
import sys
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
SOURCES = {arm: ("experiment", "confirmatory-2026-10-01") for arm in "AMP"}
COMPARISONS = [("M", "A", "primary"), ("P", "A", "primary"), ("P", "M", "secondary")]
QUEUES = 4
# Estimated minutes of one (variant, seed) trio, A + M + P, with one trainer: what the
# longest-first assignment balances the queues by (ticket 07, "Running it").
TRIO_MINUTES = {"credit-g": 3.8, "kr-vs-kp": 9.9, "spambase": 44.4, "vehicle": 2.8, "biodeg": 6.1,
                "kc2": 2.3, "pendigits": 45.2, "letter": 81.8, "electricity": 97.0}
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(tag: tuple[str, str]) -> dict[tuple[str, int, str], list[str]]:
    rows = STORE.execute(
        f"""
        select d.value, s.value, a.value, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = ? and x.value = ?
        join tags a on a.run_uuid = r.run_uuid and a.key = 'arm'
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        where {SOURCE}""",
        tag,
    )
    found: dict[tuple[str, int, str], list[str]] = {}
    for variant, seed, arm, uri in rows:
        found.setdefault((variant, int(seed), arm), []).append(uri)
    return found


RUNS = {tag: _tagged(tag) for tag in set(SOURCES.values())}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uris = RUNS[SOURCES[arm]].get((variant, seed, arm), [])
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


def queue(name: str) -> list[tuple[str, int, str]]:
    trios = sorted(
        ((TRIO_MINUTES[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS),
        key=lambda trio: (-trio[0], VARIANTS.index(trio[1]), SEEDS.index(trio[2])),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in trios:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].extend((variant, seed, arm) for arm in "AMP")
    return assigned[int(name) - 1]


def tally() -> None:
    """How many variants each primary comparison calls each way."""
    from collections import Counter
    for arm, reference, role in COMPARISONS:
        calls: Counter[str] = Counter()
        for variant in VARIANTS:
            cells = {(s, a): folds(variant, s, a) for s in SEEDS for a in (arm, reference)}
            per_seed = [
                cells[(s, arm)]["impute/induced/impute_score"].to_numpy()  # type: ignore[index]
                - cells[(s, reference)]["impute/induced/impute_score"].to_numpy()  # type: ignore[index]
                for s in SEEDS if cells[(s, arm)] is not None and cells[(s, reference)] is not None
            ]
            if len(per_seed) < len(SEEDS):
                calls["incomplete"] += 1
                continue
            m, lo, hi = interval(np.concatenate(per_seed))
            signs = [float(d.mean()) for d in per_seed]
            if lo > 0 and all(x > 0 for x in signs):
                calls[f"{reference} better"] += 1
            elif hi < 0 and all(x < 0 for x in signs):
                calls[f"{arm} better"] += 1
            else:
                calls["no detectable difference"] += 1
        print(f"{arm} - {reference} ({role}): " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))


def report(variant: str) -> None:
    arms = ("A", "M", "P")
    cells = {(s, a): folds(variant, s, a) for s in SEEDS for a in arms}
    missing = [f"s{s}{a}" for (s, a), f in cells.items() if f is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    for population in ("induced", "masked"):
        key = f"impute/{population}/impute_score"
        line = [f"  {population:7s}"]
        for arm in arms:
            done = [f[key] for s in SEEDS if (f := cells[(s, arm)]) is not None]
            if done:
                m, lo, hi = interval(pd.concat(done).to_numpy())
                line.append(f"{arm} {m:.4f} [{lo:.4f}, {hi:.4f}]")
        bars = []
        for s in SEEDS:
            run = next((f for a in arms if (f := cells[(s, a)]) is not None), None)
            if run is not None:
                name = min(BASELINES, key=lambda b: float(run[f"impute/{population}/baseline/{b}/impute_score"].mean()))
                bars.append(run[f"impute/{population}/baseline/{name}/impute_score"])
        if bars:
            m, lo, hi = interval(pd.concat(bars).to_numpy())
            line.append(f"best baseline {m:.4f} [{lo:.4f}, {hi:.4f}]")
        print("  ".join(line))
        for arm, reference, role in COMPARISONS:
            seeds = [s for s in SEEDS if cells[(s, arm)] is not None and cells[(s, reference)] is not None]
            if not seeds:
                continue
            per_seed = [
                cells[(s, arm)][key].to_numpy() - cells[(s, reference)][key].to_numpy()  # type: ignore[index]
                for s in seeds
            ]
            diffs = np.concatenate(per_seed)
            m, lo, hi = interval(diffs)
            signs = [float(d.mean()) for d in per_seed]
            if len(seeds) < len(SEEDS):
                call = f"incomplete ({len(seeds)} of {len(SEEDS)} seeds)"
            elif lo > 0 and all(x > 0 for x in signs):
                call = f"{reference} better than {arm}"
            elif hi < 0 and all(x < 0 for x in signs):
                call = f"{arm} better than {reference}"
            else:
                call = "no detectable difference"
            seed_text = ", ".join(f"s{s} {x:+.4f}" for s, x in zip(seeds, signs))
            print(f"    {arm} - {reference} ({role}) over {len(diffs)} pairs: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seed_text} -> {call}")
    cost = []
    for arm in arms:
        sums = [float(f["total_seconds"].sum()) / 60 for s in SEEDS if (f := cells[(s, arm)]) is not None]
        if sums:
            cost.append(f"{arm} {np.mean(sums):.1f} min")
    print("  fold time per cell: " + ", ".join(cost))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--tally"]:
        tally()
    elif argv[:1] == ["--todo"]:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed, arm) not in RUNS[SOURCES[arm]]:
                print(variant, seed, arm)
    else:
        for variant in argv or VARIANTS:
            report(variant)
