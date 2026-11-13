"""E176's cells: the MIMIC-IV table past TabPFN's context, natural gaps and transplanted patterns
(docs/tickets/mimic/14-wide-table.md). ``mimic_run.py`` with ``mimic-wide``'s variants: the natural table under
E148's arm (``M``, the masked checkpoint), the donor table's two variants under E149's (``D``, the defaults).
E177 (mimic/15) is the same cells with the models trained on 10,000 rows of each fold (``--train_rows``), arms
``M10k`` and ``D10k``, in the same state directory; ``--todo`` deals those, E176's being finished.

    uv run --python 3.11 python -B scripts/experiments/mimic_wide_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_wide_run.py --todo <queue> <queues>

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

run.EXPERIMENT = "mimic-wide-2026-11-12"
ROWS = 10_000  # TabPFN v2's context (``tabpfn_baseline.CONTEXT_ROWS``), as the cap on TRIDENT's training rows
run.ARMS = {"M": {"decode_checkpoint": "masked"}, "D": {},
            "M10k": {"decode_checkpoint": "masked", "train_rows": ROWS}, "D10k": {"train_rows": ROWS}}
CELLS = (
    ("M", "mimic-wide_natural"),
    ("D", "mimic-wide-donor_tmatch_20nan"),
    ("D", "mimic-wide-donor_mcar_20nan"),
)
CAPPED = tuple((f"{arm}10k", variant) for arm, variant in CELLS)
run.VARIANTS = tuple(variant for _, variant in CELLS)


def todo(which: int, queues: int) -> None:
    cells = [(arm, variant, seed) for seed in run.SEEDS for arm, variant in CAPPED]
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
