"""Pre-registered analysis of the early-stopping (04) and larger-batch (05) studies, read-only.

docs/tickets/imputation-pretraining/04-decode-early-stopping.md and 05-larger-batch.md. Lower
impute_score is better, so a positive (X - M) difference favours M. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/decode_patience_batch_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/decode_patience_batch_report.py --todo p|b2|b4

M comes from study 03 and N from study 02; P, Q2 and Q4 from these two studies. All ran on
gorgona8.
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
SOURCES = {
    "M": ("experiment", "pretrain-decode-budget-2026-09-30"),
    "N": ("experiment", "pretrain-objective-2026-09-30"),
    "P": ("experiment", "decode-early-stopping-2026-09-30"),
    "Q2": ("experiment", "larger-batch-2026-09-30"),
    "Q4": ("experiment", "larger-batch-2026-09-30"),
}
# (arm, reference, role): the pre-registered comparisons of the two studies.
COMPARISONS = [("P", "M", "primary, 04"), ("P", "N", "secondary, 04"), ("Q2", "M", "primary, 05"), ("Q4", "M", "primary, 05"), ("Q4", "Q2", "secondary, 05")]
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
    arm = {"p": "P", "b2": "Q2", "b4": "Q4"}[name]
    return [(v, s, arm) for v in ORDER for s in SEEDS]


def report(variant: str) -> None:
    arms = ("M", "N", "P", "Q2", "Q4")
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
    trained = [f["decode/epochs_trained"] for s in SEEDS if (f := cells[(s, "P")]) is not None and "decode/epochs_trained" in f]
    if trained:
        epochs = pd.concat(trained)
        print(f"  P decode epochs trained: median {epochs.median():.0f}, range {epochs.min():.0f}-{epochs.max():.0f} of 450")
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
