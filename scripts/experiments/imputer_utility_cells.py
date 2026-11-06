"""E147's GPU tier (pre-registered and run as E149): E145's two arms on MAR and MNAR tables at 60nan (ticket
imputation-protocol/53). A wrapper around
``scripts/experiments/decode_metric_run.py``, which is not changed: other variants, every seed on this machine.

    DECODE_METRIC_STATE=<state dir> uv run --python 3.11 python -B <this file> train <arm> <variant> <seed>
    DECODE_METRIC_STATE=<state dir> uv run --python 3.11 python -B <this file> --todo <queue> <queues>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))
if not os.environ.get("DECODE_METRIC_STATE"):
    sys.exit("set DECODE_METRIC_STATE: this study never writes to E145's state directory")

from scripts.experiments import decode_metric_run as run  # noqa: E402

MAR = tuple(f"{table}_marf_60nan" for table in ("kr-vs-kp", "spambase", "credit-g", "vehicle", "biodeg", "kc2"))
MNAR = tuple(f"{table}_probit_60nan" for table in ("spambase", "vehicle", "biodeg", "kc2"))
run.VARIANTS = MAR + MNAR
run.HERE = run.SEEDS  # a pair's two arms go to one queue, so the whole study sits on the machine that runs it

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        run.train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        run.todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
