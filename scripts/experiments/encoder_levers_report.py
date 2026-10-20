"""Pre-registered analysis of E50, the decode stage's levers against washing out the encoder
(imputation-pretraining/09).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/encoder_levers_report.py [--tally | --integrity | --todo 1|2|3|4]

Tag experiment=encoder-levers-2026-10-19, arms G, F100, L10 and C0, seeds 101/202/303, on
pendigits_20nan (where E49 found washout) and kr-vs-kp_40nan (where it found real transfer). Every
reading is on the induced ``impute_score`` over 15 fold pairs with ``report_common``'s verdict
rule and the 1% SESOI; K, H and M are read by the run's ``decode_checkpoint`` tag, K uncalibrated
(ADR 0027). Also each arm's minimum of the cross-fold decode validation loss.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "encoder-levers-2026-10-19"
VARIANTS = ("pendigits_20nan", "kr-vs-kp_40nan")
ARMS = ("G", "F100", "L10", "C0")
LEVERS = ("F100", "L10")
SEEDS = [101, 202, 303]
SESOI = 0.01
QUEUES = 4
CELL_MINUTES = {"pendigits_20nan": 73.0, "kr-vs-kp_40nan": 19.0}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[str, dict[tuple[str, int], list[common.Run]]] = {}


def _runs(arm: str) -> dict[tuple[str, int], list[common.Run]]:
    if arm not in _FOUND:
        _FOUND[arm] = common.parents(EXPERIMENT, arm)
    return _FOUND[arm]


def cells(variant: str, arm: str) -> Cells | None:
    found = []
    for seed in SEEDS:
        runs = _runs(arm).get((variant, seed), [])
        if len(runs) > 1:
            sys.exit(f"{variant} s{seed} {arm}: {len(runs)} finished runs; the design allows one")
        if not runs:
            return None
        found.append((runs[0], common.fold_table(runs[0])))
    return found


def contrast(a: Cells, b: Cells, readout: str = "K", population: str = "induced") -> common.Verdict:
    per_seed, scale = [], []
    for (run_a, frame_a), (run_b, frame_b) in zip(a, b):
        x = common.model(frame_a, run_a, readout, population)
        y = common.model(frame_b, run_b, readout, population)
        per_seed.append(x - y)
        scale.append(float(y.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def _decode_minimum(found: Cells) -> float:
    store = sqlite3.connect(f"file:{(ROOT / 'mlflow.db').resolve()}?mode=ro", uri=True)
    minima = []
    for run, _ in found:
        run_id = run.uri.rstrip("/").split("/")[-2]
        values = [v for (v,) in store.execute(
            "select value from metrics where run_uuid = ? and key = 'cv/decode/val_loss/mean'", (run_id,))]
        if values:
            minima.append(min(values))
    return float(np.mean(minima)) if minima else float("nan")


def line(label: str, v: common.Verdict) -> str:
    return f"{label} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}"


def tally() -> None:
    calls: dict[tuple[str, str], str] = {}
    for variant in VARIANTS:
        print(f"=== {variant}")
        base = cells(variant, "G")
        if base is None:
            print("  G incomplete")
            continue
        k = float(np.concatenate([common.model(f, r, "K") for r, f in base]).mean())
        print(f"  G: K {k:.4f}; decode validation loss at its minimum {_decode_minimum(base):.4f}")
        for arm in ARMS[1:]:
            found = cells(variant, arm)
            if found is None:
                print(f"  {arm}: incomplete")
                continue
            readings = {
                f"{arm} − G, K": contrast(found, base, "K"),
                f"{arm} − G, H": contrast(found, base, "H"),
                f"{arm} − G, masked K": contrast(found, base, "K", "masked"),
            }
            calls[variant, arm] = readings[f"{arm} − G, K"].call
            print("  " + "; ".join(line(k, v) for k, v in readings.items())
                  + f"; decode validation loss at its minimum {_decode_minimum(found):.4f}")
    if not all((v, a) in calls for v in VARIANTS for a in LEVERS):
        print("\nPrimary: incomplete")
        return
    helping = [a for a in LEVERS if calls["pendigits_20nan", a] == "better"]
    safe = [a for a in helping if calls["kr-vs-kp_40nan", a] != "worse"]
    if safe:
        reading = "supported"
    elif not helping:
        reading = "refuted"
    else:
        reading = "mixed"
    print()
    print(f"Primary: levers better than G on pendigits_20nan (K): {helping or 'none'}; of those, not worse on "
          f"kr-vs-kp_40nan: {safe or 'none'} -> {reading} ({common.RULE})")


def integrity() -> None:
    """G must reproduce the development runs of today's default on the same seeds (K and masked)."""
    for variant in VARIANTS:
        source = common.development_source(variant, "101/202/303")
        assert source is not None
        reference = common.parents(source.experiment, source.arm)
        for seed in SEEDS:
            ours, theirs = _runs("G").get((variant, seed)), reference.get((variant, seed))
            if not ours or not theirs:
                print(f"  {variant} s{seed}: missing")
                continue
            a, b = common.fold_table(ours[0]), common.fold_table(theirs[0])
            worst = 0.0
            for population in ("induced", "masked"):
                x = common.model(a, ours[0], "K", population)
                y = common.model(b, theirs[0], "K", population)
                worst = max(worst, float(np.abs(x - y).max()))
            print(f"  {variant} s{seed}: largest difference from {source.experiment} {source.arm or ''} {worst:g}")


def queue(name: str) -> list[tuple[str, int, str]]:
    work = sorted(((CELL_MINUTES[v] * (0.7 if a == "C0" else 1.0), v, s, a) for v in VARIANTS for s in SEEDS for a in ARMS),
                  key=lambda item: (-item[0], item[1], item[2], item[3]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed, arm in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, arm))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--integrity"]:
        integrity()
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed) not in _runs(arm):
                print(variant, seed, arm)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
