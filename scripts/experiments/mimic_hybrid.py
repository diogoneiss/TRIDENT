"""E163: T39's deployable hybrid on E149's MIMIC-IV transplant variants, TabPFN or missForest as the other side (mimic/10).
E166 (mimic/11) is the same study on seeds 4545/5656/6767: put ``--study e166`` before any command (default ``e163``).

    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] train H <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] --todo <queue> <queues>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/mimic_hybrid.py [--study e166] cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py [--study e166] --tally

``train`` is ``deployable_hybrid.train`` (H, with the hidden observed validation cells read in place) under this
study's tag and state. ``cells`` is ``hybrid_selection_rules.cells`` (TabPFN refit, H's ledger asserted equal to the
run's H) followed by ``missforest_hybrid.add_missforest`` (missForest refit, asserted equal to the run's logged
missForest), so one cell table holds H's, TabPFN's and missForest's fills. Every step runs on medusa1 (ADR 0035: the
whole study on one machine; gorgona8's GPU is the other project's). ``--todo`` deals the cells for ``c15_queue.sh``
with ``QUEUE_RUNNER`` and ``QUEUE_STATE``. ``--tally`` prints every method's level and the fold-pair verdicts, then the
primary reading on the two ``tmatch`` variants. Run from the checkout root with ``TRIDENT_DATASETS_ROOT`` set.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import missforest_hybrid as mh  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "mimic-hybrid-2026-11-08"
STATE = Path("/scratch/diogoneiss/trident-2026-11-08-mimic-hybrid")
SEEDS = (42, 7, 13)
VARIANTS = tuple(f"mimic-{t}-donor_{v}_20nan" for t in ("cardio", "sofa") for v in ("tmatch", "trand", "mcar"))
PRIMARY = tuple(v for v in VARIANTS if "_tmatch_" in v)
STUDIES = {"e163": (EXPERIMENT, STATE, SEEDS),
           "e166": ("mimic-hybrid-seeds-2026-11-09", Path("/scratch/diogoneiss/trident-2026-11-09-mimic-hybrid-seeds"), (4545, 5656, 6767))}


def use(study: str) -> None:
    global EXPERIMENT, STATE, SEEDS
    EXPERIMENT, STATE, SEEDS = STUDIES[study]
    e82.EXPERIMENT, e82.STATE, e82.SEEDS, e82.VARIANTS = EXPERIMENT, STATE, SEEDS, VARIANTS
    rules.STATE = STATE


use("e163")


def todo(which: int, queues: int) -> None:
    if socket.gethostname() != "medusa":  # medusa1's hostname
        sys.exit("E163 runs on medusa1")
    for k, (variant, seed) in enumerate((v, s) for s in SEEDS for v in VARIANTS):
        if k % queues == which - 1 and not (STATE / "selection" / f"{variant}_s{seed}" / "selection_fold_5.csv").exists():
            print("H", variant, seed)


def cells(variant: str, seed: int) -> None:
    rules.cells(variant, seed)
    mh.add_missforest(EXPERIMENT, STATE / "cells", STATE / "cells-mf", variant, seed)


def tally() -> None:
    scope = [(v, s) for v in VARIANTS for s in SEEDS]
    missing = [p for p in scope if not (STATE / "cells-mf" / f"{p[0]}_s{p[1]}" / "selection_fold_5.csv").exists()]
    if missing:
        print(f"incomplete: {missing}")
        return
    pairs = mh.PAIRS + (("pick", "H"), ("pick", "table_tabpfn"), ("pick", "table_missforest"), ("crossfit", "H"), ("crossfit", "table_tabpfn"))
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        per = [mh.both_sides(STATE / "cells-mf" / f"{variant}_s{s}") for s in SEEDS]
        values = {m: [np.asarray(p[m]) for p in per] for m in per[0]}
        print(f"- `{variant}`{' (primary)' if variant in PRIMARY else ''}: "
              + ", ".join(f"{m} {np.concatenate(v).mean():.4f}" for m, v in values.items()))
        for first, second in pairs:
            scale = float(np.mean([v.mean() for v in values[second]]))
            v = common.verdict([a - b for a, b in zip(values[first], values[second])], sesoi=mh.SESOI * scale)
            print(f"  - {first} − {second} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            calls.setdefault(variant, {})[f"{first}-{second}"] = v.call
        print(f"  - the pick took missForest on {int(np.concatenate(values['picked_missforest']).sum())} of 15 folds, the cross-fit pick on "
              f"{int(np.concatenate(values['crossfit_missforest']).sum())}")
    over = [v for v in PRIMARY if calls[v]["table_tabpfn-H"] == "better"]
    under = [v for v in PRIMARY if calls[v]["table_tabpfn-H"] == "worse"]
    behind = [v for v in PRIMARY if calls[v]["table_tabpfn-tabpfn"] == "worse"]
    reading = ("refuted" if not over or under else "supported" if not behind else "mixed")
    print(f"\nPRIMARY `table` (TabPFN side) on {', '.join(PRIMARY)}: better than H on {over or 'none'}, worse than H on "
          f"{under or 'none'}, worse than TabPFN on {behind or 'none'} -> {reading}")
    print(f"({common.RULE}; SESOI {mh.SESOI:.0%} of the opponent's mean)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--study"] and len(argv) >= 2 and argv[1] in STUDIES:
        use(argv[1])
        argv = argv[2:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] == "H":
        e82.train(argv[2], int(argv[3]), False)
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv[:1] == ["cells"] and len(argv) == 3:
        cells(argv[1], int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
