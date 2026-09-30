"""Pre-registered analysis of the pre-training objective study, read from the store read-only.

docs/tickets/imputation-pretraining/02-pretraining-objectives.md. Lower impute_score is
better, so a positive (X - A) difference favours arm A. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/pretrain_objective_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/pretrain_objective_report.py --todo 1|2

``--todo`` prints the cells a queue still has to run, in its pre-registered order, one
"<variant> <seed> <arm>" per line. The server's A and B cells of the pre-training ablation
(every pendigits one, and kr-vs-kp seed 13 A) are reused as references rather than rerun.
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
EXPERIMENT = "pretrain-objective-2026-09-30"
ABLATION = "pretraining-2026-09-29"
# The ablation's cells that ran on gorgona8, reused as same-machine references.
REUSED = {(v, s, a) for v in ["pendigits_20nan"] for s in SEEDS for a in "AB"} | {
    ("kr-vs-kp_40nan", 13, "A")
}
PRIMARY = ("V", "N")
STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged(key: str, value: str) -> dict[tuple[str, int, str], list[str]]:
    rows = STORE.execute(
        f"""
        select d.value, s.value, a.value, r.artifact_uri from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = ? and x.value = ?
        join tags a on a.run_uuid = r.run_uuid and a.key = 'arm'
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        where {SOURCE}""",
        (key, value),
    )
    found: dict[tuple[str, int, str], list[str]] = {}
    for variant, seed, arm, uri in rows:
        found.setdefault((variant, int(seed), arm), []).append(uri)
    return found


RUNS = _tagged("experiment", EXPERIMENT)
ABLATION_RUNS = _tagged("ablation", ABLATION)


def uri_of(variant: str, seed: int, arm: str) -> str | None:
    key = (variant, seed, arm)
    uris = ABLATION_RUNS.get(key, []) if key in REUSED else RUNS.get(key, [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed} {arm}: {len(uris)} finished runs; the design allows one")
    return uris[0] if uris else None


def folds(variant: str, seed: int, arm: str) -> pd.DataFrame | None:
    uri = uri_of(variant, seed, arm)
    if uri is None:
        return None
    frame = pd.read_csv(_path(uri) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def queue(number: str) -> list[tuple[str, int, str]]:
    """The pre-registered order of each of the two concurrent queues."""
    cheap = ["credit-g_20nan", "credit-g_80nan", "kr-vs-kp_40nan"]
    references_a = [(v, s, "A") for v in cheap for s in SEEDS if (v, s, "A") not in REUSED]
    references_b = [(v, s, "B") for v in cheap for s in SEEDS]
    value = [(v, s, "V") for v in VARIANTS for s in SEEDS]
    normalised = [(v, s, "N") for v in VARIANTS for s in SEEDS]
    return references_a + value if number == "1" else normalised + references_b


def todo(number: str) -> None:
    for variant, seed, arm in queue(number):
        if (variant, seed, arm) not in RUNS:
            print(variant, seed, arm)


def report(variant: str) -> None:
    cells = {(s, a): folds(variant, s, a) for s in SEEDS for a in "ABVN"}
    missing = [f"s{s}{a}" for (s, a), f in cells.items() if f is None]
    print(f"=== {variant}" + (f"  (missing {missing})" if missing else ""))
    for population in ("induced", "masked"):
        key = f"impute/{population}/impute_score"
        line = [f"  {population:7s}"]
        for arm in "ABVN":
            done = [f[key] for s in SEEDS if (f := cells[(s, arm)]) is not None]
            if done:
                m, lo, hi = interval(pd.concat(done).to_numpy())
                line.append(f"{arm} {m:.4f} [{lo:.4f}, {hi:.4f}] n={sum(map(len, done))}")
        # A run's best baseline depends only on variant, seed and folds, so any arm's run
        # carries it; the first one present per seed is read.
        bars, names = [], []
        for s in SEEDS:
            run = next((f for a in "BAVN" if (f := cells[(s, a)]) is not None), None)
            if run is None:
                continue
            name = min(BASELINES, key=lambda b: float(run[f"impute/{population}/baseline/{b}/impute_score"].mean()))
            bars.append(run[f"impute/{population}/baseline/{name}/impute_score"])
            names.append(name)
        if bars:
            m, lo, hi = interval(pd.concat(bars).to_numpy())
            line.append(f"best baseline {m:.4f} [{lo:.4f}, {hi:.4f}] ({'/'.join(sorted(set(names)))})")
        print("  ".join(line))
        for arm in PRIMARY:
            for reference in "AB":
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
                role = "primary" if reference == "A" else "secondary"
                print(f"    {arm} - {reference} ({role}) over {len(diffs)} pairs: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seed_text} -> {call}")
    cost = []
    for arm in "ABVN":
        sums = [float(f["total_seconds"].sum()) / 60 for s in SEEDS if (f := cells[(s, arm)]) is not None]
        if sums:
            cost.append(f"{arm} {np.mean(sums):.1f} min")
    print("  fold time per cell: " + ", ".join(cost))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"]:
        todo(argv[1])
    else:
        for variant in argv or VARIANTS:
            report(variant)
