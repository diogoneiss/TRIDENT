"""E188: the imputation task's default run with ``--score_remask`` on fresh seeds (arm ``KR``), over the 17 MCAR
variants of E184 and E185 at 40nan and above (docs/tickets/imputation-protocol/77), on gorgona8 only. With the
re-masking diagnostic a fold ledger holds the default's calibrated output (``induced``) and the same model's
uncalibrated one (``induced_uncalibrated``), so calibration is read with no retraining noise.

    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_run.py train KR <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/uncalibrated_utility_run.py --todo <queue> <queues>

The trainer is ``sliced_loss_run.py``'s, run here with this study's variants, seeds, state, tag and arm. ``--todo``
deals the cells for ``c15_queue.sh`` (``QUEUE_RUNNER``, ``QUEUE_STATE``): the core (``credit-g`` and ``kr-vs-kp``)
first, then the second batch, shortest variants first. The reads are ``uncalibrated_utility_read.py``. Importing this
module repoints ``sliced_loss_run``: no other script may import it. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import sliced_loss_run as e184  # noqa: E402
from scripts.experiments.hybrid_fresh_seeds import MINUTES  # noqa: E402

EXPERIMENT = "uncalibrated-utility-2026-11-23"
STATE = Path("/scratch/diogoneiss/trident-2026-11-23-uncalibrated-utility")
ARM = "KR"  # K, the default run, with the re-masking diagnostic
CORE = ("credit-g_40nan", "credit-g_60nan", "credit-g_80nan", "kr-vs-kp_40nan", "kr-vs-kp_60nan", "kr-vs-kp_80nan")
REST = ("kc2_40nan", "kc2_80nan", "vehicle_40nan", "vehicle_80nan", "biodeg_40nan", "biodeg_80nan",
        "pendigits_40nan", "pendigits_60nan", "pendigits_80nan", "electricity_40nan", "electricity_80nan")
SEEDS: tuple[int, ...] = (2718, 6464, 7272)
assert len({*CORE, *REST}) == 17 and not any(v.endswith("_20nan") for v in (*CORE, *REST))
assert not set(SEEDS) & {*e184.SEEDS, 1717, 8484, 9595}

e184.VARIANTS, e184.SEEDS, e184.EXPERIMENT, e184.STATE = (*CORE, *REST), SEEDS, EXPERIMENT, STATE
e184.ARMS = {ARM: {"score_remask": True}}  # not in its TERM: no run carries the loss term


def todo(which: int, queues: int) -> None:
    cells = [(v, s) for group in (CORE, REST) for v in sorted(group, key=lambda v: (MINUTES[common.table_of(v)], v)) for s in SEEDS]
    for k, (variant, seed) in enumerate(cells):
        if k % queues == which - 1 and e184._run(ARM, variant, seed) is None:
            print(ARM, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] == ARM:
        e184.train(ARM, argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
