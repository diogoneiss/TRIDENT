"""Pre-registered analysis of E52, T28's D3: a frozen encoder, pre-trained against random
(imputation-pretraining/10).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/frozen_encoder_report.py [--tally | --todo 1|2|3|4]

Tag experiment=frozen-encoder-2026-10-20, arms FN, FA and FR on four variants and C0 on the two
credit-g ones, seeds 101/202/303. G (today's full decode) and, on pendigits_20nan and
kr-vs-kp_40nan, C0 are read from E50's runs on the same seeds and folds
(encoder-levers-2026-10-19, which reproduced the development runs exactly); G on credit-g is the
development run (``report_common.development_source``). Every reading is on the induced
``impute_score`` over 15 fold pairs with ``report_common``'s verdict rule and the 1% SESOI, K read
uncalibrated (ADR 0027).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "frozen-encoder-2026-10-20"
LEVERS = "encoder-levers-2026-10-19"
VARIANTS = ("pendigits_20nan", "kr-vs-kp_40nan", "credit-g_20nan", "credit-g_80nan")
OWN_C0 = ("credit-g_20nan", "credit-g_80nan")
SEEDS = [101, 202, 303]
SESOI = 0.01
QUEUES = 4
CELL_MINUTES = {
    ("pendigits_20nan", "FN"): 55.0, ("pendigits_20nan", "FA"): 55.0, ("pendigits_20nan", "FR"): 30.0,
    ("kr-vs-kp_40nan", "FN"): 15.0, ("kr-vs-kp_40nan", "FA"): 15.0, ("kr-vs-kp_40nan", "FR"): 9.0,
}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[tuple[str, str | None], dict[tuple[str, int], list[common.Run]]] = {}


def arms(variant: str) -> tuple[str, ...]:
    return ("FN", "FA", "FR", "C0") if variant in OWN_C0 else ("FN", "FA", "FR")


def _parents(experiment: str, arm: str | None) -> dict[tuple[str, int], list[common.Run]]:
    if (experiment, arm) not in _FOUND:
        _FOUND[experiment, arm] = common.parents(experiment, arm)
    return _FOUND[experiment, arm]


def _source(variant: str, arm: str) -> tuple[str, str | None]:
    """Where an arm's runs live: this study's, or E50's and the development runs for G and C0."""
    if arm in ("FN", "FA", "FR") or (arm == "C0" and variant in OWN_C0):
        return EXPERIMENT, arm
    if variant not in OWN_C0:
        return LEVERS, arm
    assert arm == "G"
    source = common.development_source(variant, "101/202/303")
    assert source is not None
    return source.experiment, source.arm


def cells(variant: str, arm: str) -> Cells | None:
    experiment, tag = _source(variant, arm)
    found = []
    for seed in SEEDS:
        runs = _parents(experiment, tag).get((variant, seed), [])
        if len(runs) > 1:
            sys.exit(f"{variant} s{seed} {arm}: {len(runs)} finished runs in {experiment}; one expected")
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


def _line(label: str, v: common.Verdict, base: float) -> str:
    return f"{label} {v.mean:+.4f} ({100 * v.mean / base:+.1f}%) [{v.low:+.4f}, {v.high:+.4f}] {v.call}"


PAIRS = (("FN", "FR"), ("FA", "FR"), ("FN", "FA"), ("FN", "G"), ("FR", "C0"), ("C0", "G"))


def tally() -> None:
    usable: dict[str, str] = {}
    for variant in VARIANTS:
        print(f"=== {variant}")
        found = {arm: cells(variant, arm) for arm in ("FN", "FA", "FR", "C0", "G")}
        for arm, value in found.items():
            if value is not None:
                k = float(np.concatenate([common.model(f, r, "K") for r, f in value]).mean())
                print(f"  {arm}: K {k:.4f}")
        for a, b in PAIRS:
            x, y = found[a], found[b]
            if x is None or y is None:
                print(f"  {a} − {b}: incomplete")
                continue
            base = float(np.concatenate([common.model(f, r, "K") for r, f in y]).mean())
            readings = [_line(f"{a} − {b}, K", contrast(x, y, "K"), base),
                        _line("H", contrast(x, y, "H"), base),
                        _line("masked K", contrast(x, y, "K", "masked"), base)]
            print("  " + "; ".join(readings))
            if (a, b) == ("FN", "FR"):
                usable[variant] = contrast(x, y, "K").call
    if len(usable) < len(VARIANTS):
        print("\nPrimary: incomplete")
        return
    better = [v for v, c in usable.items() if c == "better"]
    worse = [v for v, c in usable.items() if c == "worse"]
    print()
    print(f"Primary, FN − FR on K: a usable pre-trained representation (FN better) on {better or 'none'}; "
          f"FN worse on {worse or 'none'}; {common.tally(usable.values())}")
    reading = "usable" if len(better) >= 3 else "nothing usable" if not better else "mixed"
    print(f"Reading: {reading} ({len(better)} of {len(VARIANTS)}; usable needs 3 or more, nothing usable needs none)")


def queue(name: str) -> list[tuple[str, int, str]]:
    work = sorted(
        ((CELL_MINUTES.get((v, a), 5.0), v, s, a) for v in VARIANTS for s in SEEDS for a in arms(v)),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed, arm in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, arm))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed) not in _parents(EXPERIMENT, arm):
                print(variant, seed, arm)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
