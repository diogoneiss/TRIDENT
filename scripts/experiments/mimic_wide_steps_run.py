"""E180's cells: on ``mimic-wide``, the models trained on 10,000 rows of each fold for as many optimizer steps as
the full table's epochs take (docs/tickets/mimic/17-wide-table-matched-steps.md). ``mimic_wide_run.py``'s cells
under three arms a variant, all trained on one machine: E176's (``M``, ``D``), E177's (``M10k``, ``D10k``) and the
capped arm with its epochs scaled (``M10kS``, ``D10kS``). The batch stays 256, so only the row pool differs from
the full arm at the same number of steps.

    uv run --python 3.11 python -B scripts/experiments/mimic_wide_steps_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_wide_steps_run.py --todo <queue> <queues>

Queues through ``c15_queue.sh`` with ``QUEUE_RUNNER`` and ``QUEUE_STATE``. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import mimic_run as run  # noqa: E402
from scripts.experiments import mimic_wide_run as wide  # noqa: E402  (sets the variants and E176's and E177's arms)

run.EXPERIMENT = "mimic-wide-steps-2026-11-14"
# The defaults train 300 pre-training and 450 decode epochs. At batch 256 an epoch is 179 steps on the natural
# table's 45,755 training rows and 128 on the donor table's 32,550, against 40 on 10,000 rows:
# 300 * 179 / 40 = 1342.5, 450 * 179 / 40 = 2013.75, 300 * 128 / 40 = 960, 450 * 128 / 40 = 1440.
run.ARMS |= {
    "M10kS": {**run.ARMS["M10k"], "hyperparams_override": {"EPOCHS_PRE": 1343, "EPOCHS_DECODE": 2014}},
    "D10kS": {**run.ARMS["D10k"], "hyperparams_override": {"EPOCHS_PRE": 960, "EPOCHS_DECODE": 1440}},
}
SUFFIXES = ("10k", "10kS", "")  # capped, capped at the full table's steps, full: the cheap arms first within a seed


def todo(which: int, queues: int) -> None:
    cells = [(arm + suffix, variant, seed) for seed in run.SEEDS for arm, variant in wide.CELLS for suffix in SUFFIXES]
    for k, cell in enumerate(cells):
        if k % queues == which - 1 and not run.done(*cell):
            print(*cell)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        run.train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
