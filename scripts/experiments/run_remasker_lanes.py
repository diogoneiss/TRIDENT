"""Start E157's offline ReMasker pass in GPU lanes (imputation-protocol/55), detached.

    python3 scripts/experiments/run_remasker_lanes.py [<lanes>] [--out DIR] [--rule paper] [--host gorgona8|medusa1]
        [--variants V,V,...] [--gpus N]

With no option it is E157's launch. E158 (imputation-protocol/56) passes ``--rule paper``, its own ``--out``, and
``--host``, which keeps that machine's share of the pairs (``MEDUSA1``; every arm of a pair on one machine, ADR 0035).
E159 (imputation-protocol/57) passes ``--variants`` (the letter variants each machine runs, all three seeds) and
``--gpus 2`` on medusa1, which spreads its lanes over two GPUs. E160 (imputation-protocol/58) passes ``--seeds``.

Splits the 27 (variant, seed) pairs into lanes balanced by estimated minutes (the pre-launch smoke), largest first,
and starts each lane as one detached shell running ``remasker_run.py`` item by item. Progress goes to
``<out>/progress.log``; a relaunch skips finished files. To stop, kill the lanes' shells. Run from the checkout root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OUT = Path("/scratch2/diogoneiss/trident-2026-11-06-remasker")
# Minutes a seed (five folds) from the pre-launch timing smoke, beside E156 on the same GPU.
MINUTES = {"pendigits": 90.0, "kc2": 15.0, "vehicle": 12.0, "biodeg": 10.0, "letter": 165.0}
CONFIRMATION = [
    *(f"pendigits_{level}nan" for level in (40, 60, 80)),
    *(f"{t}_{level}nan" for t in ("vehicle", "biodeg", "kc2") for level in (40, 80)),
]

# E158's pairs on medusa1 (its two GPUs, one lane each); the rest stay on gorgona8.
MEDUSA1 = {("pendigits_80nan", 707), ("pendigits_80nan", 808), ("pendigits_80nan", 909), ("pendigits_60nan", 909)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lanes", nargs="?", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rule", choices=("reference", "paper"), default="reference")
    parser.add_argument("--host", choices=("gorgona8", "medusa1"))
    parser.add_argument("--variants", help="comma-separated variants in place of E157's nine, all three seeds")
    parser.add_argument("--gpus", type=int, help="spread the lanes over this many GPUs")
    parser.add_argument("--seeds", default="707,808,909", help="comma-separated seeds (E160: the families' own)")
    options = parser.parse_args(argv)
    lanes, out = options.lanes, options.out
    if not Path("train.py").exists():
        sys.exit("run from the TRIDENT checkout root")
    variants = options.variants.split(",") if options.variants else CONFIRMATION
    seeds = [int(x) for x in options.seeds.split(",")]
    items = [(v, s, "confirmation") for v in variants for s in seeds]
    if options.host is not None:
        items = [i for i in items if ((i[0], i[1]) in MEDUSA1) == (options.host == "medusa1")]
    loads = [0.0] * lanes
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(lanes)]
    for item in sorted(items, key=lambda i: -MINUTES[i[0].split("_")[0]]):
        k = loads.index(min(loads))
        loads[k] += MINUTES[item[0].split("_")[0]]
        assigned[k].append(item)
    out.mkdir(parents=True, exist_ok=True)
    env = f"export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 REMASKER_REMASK={options.rule}"
    for k, group in enumerate(assigned, start=1):
        group.sort(key=lambda i: (i[2] != "confirmation", -MINUTES[i[0].split("_")[0]]))
        steps = [
            f"echo \"$(date '+%F %T %z') start {v} s{s} ({label})\" >> {out}/progress.log; "
            f"uv run --python 3.11 python -B scripts/experiments/remasker_run.py {out}/csv {v} {s} >> {out}/lane{k}.log 2>&1 "
            f"&& echo \"$(date '+%F %T %z') done {v} s{s} ({label})\" >> {out}/progress.log "
            f"|| echo \"$(date '+%F %T %z') FAILED {v} s{s} ({label})\" >> {out}/progress.log"
            for v, s, label in group
        ]
        gpus = options.gpus or (2 if options.host == "medusa1" else None)
        if gpus:
            env_k = f"{env} CUDA_VISIBLE_DEVICES={(k - 1) % gpus}"
        else:
            env_k = env
        script = f"{env_k}; " + "; ".join(steps) + f"; echo \"$(date '+%F %T %z') lane {k} complete\" >> {out}/progress.log"
        subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"lane {k}: {len(group)} items, about {loads[k - 1]:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
