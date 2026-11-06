"""E149's cells: real gap patterns transplanted onto the complete rows of the MIMIC-IV tables
(docs/tickets/mimic/04-pattern-transplant.md). ``mimic_run.py`` with other variants and one arm, the imputation
defaults: the donor table is the complete sibling, so ADR 0015's ``induced`` checkpoint applies.

    uv run --python 3.11 python -B scripts/experiments/mimic_transplant_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_transplant_run.py --todo <queue> <queues>

The variants are the ones ``datasets/make_mimic_variants.py`` cut under ``$TRIDENT_DATASETS_ROOT``. Queues through
``c15_queue.sh`` with ``QUEUE_RUNNER`` and ``QUEUE_STATE``, a state directory of its own. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import mimic_run as run  # noqa: E402

run.EXPERIMENT = "mimic-pattern-transplant-2026-11-06"
run.VARIANTS = tuple(
    f"{table}-donor_{variant}_20nan" for table in ("mimic-cardio", "mimic-sofa") for variant in ("tmatch", "trand", "mcar")
)
run.ARMS = {"D": {}}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        run.train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        run.todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
