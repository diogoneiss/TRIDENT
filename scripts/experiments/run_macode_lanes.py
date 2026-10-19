"""Start E48's offline MaCoDE pass in GPU lanes (imputation-protocol/07), detached.

    python3 scripts/experiments/run_macode_lanes.py [<lanes>]

Splits the 108 (variant, seed) pairs into lanes balanced by estimated minutes (the pre-launch smoke
and the tables' sizes), confirmation folds first in every lane, and starts each lane as one detached
shell running ``macode_run.py`` item by item. Progress goes to ``<out>/progress.log``; a relaunch
skips finished files. Run from the checkout root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path("/scratch2/diogoneiss/trident-2026-10-19-macode")
MINUTES = {"electricity": 15.0, "letter": 8.0, "pendigits": 4.6, "spambase": 6.0, "kr-vs-kp": 3.5,
           "credit-g": 1.0, "biodeg": 1.2, "vehicle": 0.8, "kc2": 0.6}
DEVELOPMENT = [
    *(f"{t}_{level}nan" for t in ("kr-vs-kp", "spambase", "credit-g") for level in (20, 40, 60, 80)),
    "vehicle_20nan", "vehicle_60nan", "biodeg_20nan", "biodeg_60nan", "kc2_20nan", "kc2_60nan",
    "pendigits_20nan", "letter_20nan", "electricity_20nan",
]
CONFIRMATION = [
    *(f"{t}_{level}nan" for t in ("electricity", "letter", "pendigits") for level in (40, 60, 80)),
    *(f"{t}_{level}nan" for t in ("vehicle", "biodeg", "kc2") for level in (40, 80)),
]


def main(argv: list[str]) -> int:
    lanes = int(argv[0]) if argv else 3
    if not Path("train.py").exists():
        sys.exit("run from the TRIDENT checkout root")
    items = [(v, s, "confirmation") for v in CONFIRMATION for s in (707, 808, 909)]
    items += [(v, s, "development") for v in DEVELOPMENT for s in (101, 202, 303)]
    loads = [0.0] * lanes
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(lanes)]
    for item in sorted(items, key=lambda i: -MINUTES[i[0].split("_")[0]]):
        k = loads.index(min(loads))
        loads[k] += MINUTES[item[0].split("_")[0]]
        assigned[k].append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    env = "export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
    for k, group in enumerate(assigned, start=1):
        group.sort(key=lambda i: (i[2] != "confirmation", -MINUTES[i[0].split("_")[0]]))
        steps = [
            f"echo \"$(date '+%F %T %z') start {v} s{s} ({label})\" >> {OUT}/progress.log; "
            f"uv run --python 3.11 python -B scripts/experiments/macode_run.py {OUT}/csv {v} {s} >> {OUT}/lane{k}.log 2>&1 "
            f"&& echo \"$(date '+%F %T %z') done {v} s{s} ({label})\" >> {OUT}/progress.log "
            f"|| echo \"$(date '+%F %T %z') FAILED {v} s{s} ({label})\" >> {OUT}/progress.log"
            for v, s, label in group
        ]
        script = f"{env}; " + "; ".join(steps) + f"; echo \"$(date '+%F %T %z') lane {k} complete\" >> {OUT}/progress.log"
        subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"lane {k}: {len(group)} items, about {loads[k - 1]:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
