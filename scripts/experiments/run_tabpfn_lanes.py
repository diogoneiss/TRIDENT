"""Start E55's offline TabPFN v2 pass in GPU lanes (imputation-protocol/10), detached.

    python3 scripts/experiments/run_tabpfn_lanes.py [<lanes>] [e58]

Splits the 108 (variant, seed) pairs into lanes balanced by estimated minutes (the pre-launch
smoke, scaled by rows and columns), with confirmation pairs first in every lane, and starts each
lane as one detached shell running ``tabpfn_run.py`` item by item under the ``tabpfn`` extra.

- **The deadline:** a pair starts only if now + 1.3 × its estimate is before ``DEADLINE``; a
  skipped pair is logged.
- **Progress** goes to ``<out>/progress.log``; a relaunch skips finished files.

``e58`` runs E58's pairs instead (imputation-protocol/13): the seed sets E55 did not cover, logging
to their own folder and writing their CSVs beside E55's, which every reader merges. Run from the
checkout root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn")
E58_OUT = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn-e58")
E58_DEADLINE = "2026-10-21 23:00:00 -0300"
E40 = [*(f"{t}_{level}nan" for t in ("kr-vs-kp", "spambase", "credit-g") for level in (40, 60, 80)), "vehicle_60nan", "biodeg_60nan", "kc2_60nan"]
E51 = [f"{t}_20nan" for t in ("electricity", "letter", "pendigits", "spambase", "kr-vs-kp", "biodeg", "credit-g", "vehicle", "kc2")]
MNAR = ["biodeg_probit_60nan", "kc2_probit_60nan", "vehicle_probit_60nan", "spambase_probit_60nan", "biodeg_probit_20nan",
        "kc2_probit_20nan", "vehicle_probit_20nan", "biodeg_probit0p25_60nan", "biodeg_probit1p0_60nan", "biodeg_probit2p0_60nan"]
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
    e58 = argv[1:2] == ["e58"]
    if e58:
        items = [(v, s, "42/7/13") for v in DEVELOPMENT for s in (42, 7, 13)]
        items += [(v, s, "404/505/606") for v in E40 for s in (404, 505, 606)]
        items += [(v, s, "1001/2002/3003") for v in E51 for s in (1001, 2002, 3003)]
        items += [(v, s, "mnar") for v in MNAR for s in (42, 7, 13)]
    else:
        items = [(v, s, "confirmation") for v in CONFIRMATION for s in (707, 808, 909)]
        items += [(v, s, "development") for v in DEVELOPMENT for s in (101, 202, 303)]
    log_dir, deadline = (E58_OUT, E58_DEADLINE) if e58 else (OUT, DEADLINE)
    loads = [0.0] * lanes
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(lanes)]
    for item in sorted(items, key=lambda i: (i[2] not in ("confirmation", "mnar"), -minutes(i[0]))):
        k = loads.index(min(loads))
        loads[k] += minutes(item[0])
        assigned[k].append(item)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = (f"export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
           f"TABPFN_MODEL_CACHE_DIR={WEIGHTS}; DEADLINE=$(date -d '{deadline}' +%s)")
    for k, group in enumerate(assigned, start=1):
        steps = [
            f"if [ $(( $(date +%s) + {int(minutes(v) * 78)} )) -ge $DEADLINE ]; then "
            f"echo \"$(date '+%F %T %z') skipped {v} s{s} ({label}): would not end by the deadline\" >> {log_dir}/progress.log; else "
            f"echo \"$(date '+%F %T %z') start {v} s{s} ({label})\" >> {log_dir}/progress.log; "
            f"uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_run.py {OUT}/csv {v} {s} >> {log_dir}/lane{k}.log 2>&1 "
            f"&& echo \"$(date '+%F %T %z') done {v} s{s} ({label})\" >> {log_dir}/progress.log "
            f"|| echo \"$(date '+%F %T %z') FAILED {v} s{s} ({label})\" >> {log_dir}/progress.log; fi"
            for v, s, label in group
        ]
        script = f"{env}; " + "; ".join(steps) + f"; echo \"$(date '+%F %T %z') lane {k} complete\" >> {log_dir}/progress.log"
        subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"lane {k}: {len(group)} items, about {loads[k - 1]:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
