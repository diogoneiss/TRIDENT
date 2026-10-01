"""Pre-registered analysis of the batched-heads check (imputation-pretraining/06), read-only.

Lower impute_score is better, so a positive (H - A) difference favours the per-column heads.
Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/batched_heads_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/batched_heads_report.py --todo h

A is the per-column reference already on gorgona8: study 02's A cells, and the pre-training
ablation's server A cells for pendigits and kr-vs-kp seed 13.
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

VARIANTS = ["credit-g_20nan", "credit-g_80nan", "kr-vs-kp_40nan", "pendigits_20nan"]
SEEDS = [42, 7, 13]
BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
STUDY_02 = ("experiment", "pretrain-objective-2026-09-30")
ABLATION = ("ablation", "pretraining-2026-09-29")
SOURCES = {"A": STUDY_02, "H": ("experiment", "batched-heads-2026-10-01")}
# Study 02 reused these server cells of the ablation as its A references.
REUSED_A = {("pendigits_20nan", s) for s in (42, 7, 13)} | {("kr-vs-kp_40nan", 13)}
COMPARISONS = [("H", "A", "primary, 06")]
ORDER = ["pendigits_20nan", "kr-vs-kp_40nan", "credit-g_80nan", "credit-g_20nan"]
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


RUNS = {tag: _tagged(tag) for tag in {*SOURCES.values(), ABLATION}}


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    source = ABLATION if arm == "A" and (variant, seed) in REUSED_A else SOURCES[arm]
    uris = RUNS[source].get((variant, seed, arm), [])
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
    assert name == "h"
    return [(v, s, "H") for v in ORDER for s in SEEDS]


def report(variant: str) -> None:
    arms = ("A", "H")
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
    if argv[:1] == ["--todo"]:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed, arm) not in RUNS[SOURCES[arm]]:
                print(variant, seed, arm)
    else:
        for variant in argv or VARIANTS:
            report(variant)
