"""Start E55's offline TabPFN v2 pass in GPU lanes (imputation-protocol/10), detached.

    python3 scripts/experiments/run_tabpfn_lanes.py [<lanes>]

Splits the 108 (variant, seed) pairs into lanes balanced by estimated minutes (the pre-launch
smoke, scaled by rows and columns), with confirmation pairs first in every lane, and starts each
lane as one detached shell running ``tabpfn_run.py`` item by item under the ``tabpfn`` extra.

- **The deadline:** a pair starts only if now + 1.3 × its estimate is before ``DEADLINE``; a
  skipped pair is logged.
- **Progress** goes to ``<out>/progress.log``; a relaunch skips finished files.

Run from the checkout root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn")
WEIGHTS = Path("/scratch2/diogoneiss/tabpfn-weights")
DEADLINE = "2026-10-21 08:00:00 -0300"
# Minutes a (variant, seed) pair takes, five folds, with two lanes sharing the GPU. From the smoke
# of 2026-10-20 at seed 7: electricity_20nan 121 s alone with the one-pass fill; spambase_20nan
# 1,129 s and credit-g_60nan 40 s before it. The others are scaled by columns, features and
# context rows.
MINUTES: dict[str, float] = {"electricity": 6.0, "letter": 18.0, "pendigits": 10.0, "spambase": 20.0, "kr-vs-kp": 5.0,
                             "credit-g": 1.0, "biodeg": 2.0, "vehicle": 1.0, "kc2": 1.0}
DEVELOPMENT = [
    *(f"{t}_{level}nan" for t in ("kr-vs-kp", "spambase", "credit-g") for level in (20, 40, 60, 80)),
    "vehicle_20nan", "vehicle_60nan", "biodeg_20nan", "biodeg_60nan", "kc2_20nan", "kc2_60nan",
    "pendigits_20nan", "letter_20nan", "electricity_20nan",
]
CONFIRMATION = [
    *(f"{t}_{level}nan" for t in ("electricity", "letter", "pendigits") for level in (40, 60, 80)),
    *(f"{t}_{level}nan" for t in ("vehicle", "biodeg", "kc2") for level in (40, 80)),
]


def minutes(variant: str) -> float:
    return MINUTES[variant.split("_")[0]]


def main(argv: list[str]) -> int:
    lanes = int(argv[0]) if argv else 2
    if not Path("train.py").exists():
        sys.exit("run from the checkout root")
    if not MINUTES:
        sys.exit("fill MINUTES from the smoke first")
    items = [(v, s, "confirmation") for v in CONFIRMATION for s in (707, 808, 909)]
    items += [(v, s, "development") for v in DEVELOPMENT for s in (101, 202, 303)]
    loads = [0.0] * lanes
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(lanes)]
    for item in sorted(items, key=lambda i: (i[2] != "confirmation", -minutes(i[0]))):
        k = loads.index(min(loads))
        loads[k] += minutes(item[0])
        assigned[k].append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    env = (f"export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
           f"TABPFN_MODEL_CACHE_DIR={WEIGHTS}; DEADLINE=$(date -d '{DEADLINE}' +%s)")
    for k, group in enumerate(assigned, start=1):
        steps = [
            f"if [ $(( $(date +%s) + {int(minutes(v) * 78)} )) -ge $DEADLINE ]; then "
            f"echo \"$(date '+%F %T %z') skipped {v} s{s} ({label}): would not end by the deadline\" >> {OUT}/progress.log; else "
            f"echo \"$(date '+%F %T %z') start {v} s{s} ({label})\" >> {OUT}/progress.log; "
            f"uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_run.py {OUT}/csv {v} {s} >> {OUT}/lane{k}.log 2>&1 "
            f"&& echo \"$(date '+%F %T %z') done {v} s{s} ({label})\" >> {OUT}/progress.log "
            f"|| echo \"$(date '+%F %T %z') FAILED {v} s{s} ({label})\" >> {OUT}/progress.log; fi"
            for v, s, label in group
        ]
        script = f"{env}; " + "; ".join(steps) + f"; echo \"$(date '+%F %T %z') lane {k} complete\" >> {OUT}/progress.log"
        subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"lane {k}: {len(group)} items, about {loads[k - 1]:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
