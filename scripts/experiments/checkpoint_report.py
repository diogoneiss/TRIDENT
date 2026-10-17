"""Pre-registered analysis of the deployable-checkpoint study (imputation-protocol/03, E43).

Read-only. Lower impute_score is better, so a negative (A - B) difference favours A. Run from the
checkout root:

    uv run --python 3.11 python scripts/experiments/checkpoint_report.py [variant ...]
    uv run --python 3.11 python scripts/experiments/checkpoint_report.py --tally
    uv run --python 3.11 python scripts/experiments/checkpoint_report.py --todo 1|2|3|4

One run per (variant, seed), tag experiment=deployable-checkpoint-2026-10-17, seeds 111, 222 and
333, which no earlier study used. Each run carries every arm on one trajectory (within-run pairs):

- K, the headline: the checkpoint chosen on the validation gaps (ADR 0015), an oracle;
- H, ``loss_checkpoint``: the epoch of the lowest validation loss;
- M, ``masked_checkpoint``: the epoch whose fixed validation mask scores best;
- KC, ``calibrated``: K after the calibration fitted on the validation gaps, an oracle too.

The bar is the best of the logged baselines outside ``summary.OUTSIDE_THE_BAR``, chosen on each
run's CV mean as ADR 0007 decision 8 chooses it. ``--todo`` prints a queue's remaining cells,
assigned longest first by E40's cell times.
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

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training.summary import OUTSIDE_THE_BAR  # noqa: E402

VARIANTS = [
    "kr-vs-kp_60nan", "kr-vs-kp_80nan", "spambase_60nan", "spambase_80nan",
    "credit-g_60nan", "credit-g_80nan", "vehicle_60nan", "biodeg_60nan", "kc2_60nan",
]
EIGHTY = ["kr-vs-kp_80nan", "spambase_80nan", "credit-g_80nan"]
SEEDS = [111, 222, 333]
EXPERIMENT = "deployable-checkpoint-2026-10-17"
POPULATION = "impute/induced"
ARMS = {
    "K": f"{POPULATION}/impute_score",
    "H": f"{POPULATION}/loss_checkpoint/impute_score",
    "M": f"{POPULATION}/masked_checkpoint/impute_score",
    "KC": f"{POPULATION}/calibrated/impute_score",
}
QUEUES = 4
# E40's minutes a cell on the same variants under today's default (calibration_report.py).
CELL_MINUTES = {
    "kr-vs-kp_60nan": 19.5, "kr-vs-kp_80nan": 16.0, "spambase_60nan": 39.6, "spambase_80nan": 43.9,
    "credit-g_60nan": 5.8, "credit-g_80nan": 5.7, "vehicle_60nan": 4.5, "biodeg_60nan": 9.9,
    "kc2_60nan": 3.6,
}
_BASELINE = re.compile(rf"^{re.escape(POPULATION)}/baseline/([^/]+)/impute_score$")
_STORE: sqlite3.Connection | None = None
SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""


def _path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def _tagged() -> dict[tuple[str, int], list[str]]:
    global _STORE
    if _STORE is None:
        _STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
    rows = _STORE.execute(
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
        found.setdefault((str(variant), int(seed)), []).append(str(uri))
    return found


_RUNS: dict[tuple[str, int], list[str]] | None = None


def runs() -> dict[tuple[str, int], list[str]]:
    global _RUNS
    if _RUNS is None:
        _RUNS = _tagged()
    return _RUNS


def folds(variant: str, seed: int) -> pd.DataFrame | None:
    uris = runs().get((variant, seed), [])
    if len(uris) > 1:
        sys.exit(f"{variant} s{seed}: {len(uris)} finished runs; the design allows one")
    if not uris:
        return None
    frame = pd.read_csv(_path(uris[0]) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def _cells(variant: str) -> list[pd.DataFrame] | None:
    cells = [folds(variant, seed) for seed in SEEDS]
    complete = [cell for cell in cells if cell is not None]
    return complete if len(complete) == len(SEEDS) else None


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def bar(cell: pd.DataFrame) -> np.ndarray:
    """The bar of this run: the bar baseline with the lowest CV mean, fold by fold."""
    names = [m.group(1) for column in cell.columns if (m := _BASELINE.match(column))]
    inside = [name for name in names if name not in OUTSIDE_THE_BAR]
    best = min(inside, key=lambda name: float(cell[f"{POPULATION}/baseline/{name}/impute_score"].mean()))
    return np.asarray(cell[f"{POPULATION}/baseline/{best}/impute_score"].to_numpy(dtype=float))


def _values(cell: pd.DataFrame, arm: str) -> np.ndarray:
    return bar(cell) if arm == "bar" else np.asarray(cell[ARMS[arm]].to_numpy(dtype=float))


def verdict(variant: str, first: str, second: str) -> tuple[str, float, float, float, list[float]] | None:
    """``first`` - ``second`` over 15 fold pairs: '<first> better', '<second> better' or 'none'."""
    cells = _cells(variant)
    if cells is None:
        return None
    per_seed = [_values(cell, first) - _values(cell, second) for cell in cells]
    mean, lo, hi = interval(np.concatenate(per_seed))
    signs = [float(d.mean()) for d in per_seed]
    if hi < 0 and all(s < 0 for s in signs):
        call = f"{first} better"
    elif lo > 0 and all(s > 0 for s in signs):
        call = f"{second} better"
    else:
        call = "none"
    return call, mean, lo, hi, signs


def report(variant: str) -> None:
    print(f"=== {variant}{' (80nan)' if variant in EIGHTY else ''}")
    cells = _cells(variant)
    if cells is None:
        print("  incomplete")
        return
    means = {arm: float(np.concatenate([_values(c, arm) for c in cells]).mean()) for arm in (*ARMS, "bar")}
    print("  " + "  ".join(f"{arm} {value:.4f}" for arm, value in means.items()))
    for first, second in (("M", "H"), ("M", "K"), ("K", "H"), ("KC", "K"), ("K", "bar"), ("H", "bar"), ("M", "bar")):
        result = verdict(variant, first, second)
        assert result is not None
        call, mean, lo, hi, signs = result
        seeds = ", ".join(f"s{s} {d:+.4f}" for s, d in zip(SEEDS, signs))
        print(f"    {first} - {second}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]; {seeds} -> {call}")
    epochs = pd.concat(cells)
    parts = [
        f"{name} epoch {epochs[f'decode/{name}_checkpoint_epoch'].mean():.1f}"
        for name in ("loss", "induced", "masked")
        if f"decode/{name}_checkpoint_epoch" in epochs.columns
    ]
    parts.append(f"minutes a cell {float(np.mean([c['total_seconds'].sum() / 60 for c in cells])):.1f}")
    print("  " + "; ".join(parts))


def tally() -> None:
    calls = {
        (variant, first, second): verdict(variant, first, second)
        for variant in VARIANTS for first, second in (("M", "H"), ("M", "K"), ("K", "H"), ("KC", "K"))
    }
    missing = sorted({variant for (variant, _, _), result in calls.items() if result is None})
    if missing:
        print(f"reading (imputation-protocol/03): incomplete, missing {missing}")
        return
    k_over_m = [v for v in VARIANTS if (r := calls[(v, "M", "K")]) is not None and r[0] == "K better"]
    m_over_h_80 = [v for v in EIGHTY if (r := calls[(v, "M", "H")]) is not None and r[0] == "M better"]
    h_over_m = [v for v in VARIANTS if (r := calls[(v, "M", "H")]) is not None and r[0] == "H better"]
    if not k_over_m:
        reading = "1. M suffices: K better than M on none of the 9; transfer claims report M"
    elif len(m_over_h_80) >= 2 and not h_over_m:
        reading = f"2. M helps: M better than H on {m_over_h_80}, worse on none; M is the deployable headline"
    else:
        reading = "3. otherwise: H is the deployable headline; 'beats the mean/mode fill at 80nan' is oracle-only"
    print(f"reading (imputation-protocol/03): {reading}")
    print(f"  K better than M on {k_over_m}; M better than H at 80nan on {m_over_h_80}; H better than M on {h_over_m}")
    for first, second in (("M", "H"), ("M", "K"), ("K", "H"), ("KC", "K")):
        counts: dict[str, int] = {}
        for variant in VARIANTS:
            result = calls[(variant, first, second)]
            assert result is not None
            counts[result[0]] = counts.get(result[0], 0) + 1
        print(f"  {first} - {second} (fold-pair t, seed agreement): " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    for arm in ("K", "H", "M"):
        below = [v for v in EIGHTY if (r := verdict(v, arm, "bar")) is not None and r[0] == f"{arm} better"]
        print(f"  {arm} better than the bar at 80nan (fold-pair t, seed agreement): {below}")


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
            if (variant, seed) not in runs():
                print(variant, seed)
    else:
        for name in argv or VARIANTS:
            report(name)
