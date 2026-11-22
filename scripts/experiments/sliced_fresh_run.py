"""E187: the default run (K) and the run whose checkpoint is chosen by ``sw_score`` (S) on fresh seeds, over the 26
MCAR variants of E184 and E185 (docs/tickets/imputation-protocol/76), on gorgona8 only. The trainer, the queue order
and the checks are ``sliced_loss_run.py``'s, run here with this study's variants, seeds, state, tag and the two arms.

    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_run.py train <K|S> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_run.py --gate
    uv run --python 3.11 python -B scripts/experiments/sliced_fresh_run.py --pairing

The reads are not here (``sliced_loss_run.py``'s tally needs E182's baseline files, which these seeds do not have).
Importing this module repoints ``sliced_loss_run``: no other study's script may import it. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import sliced_loss_run as e184  # noqa: E402

EXPERIMENT = "sliced-fresh-2026-11-22"
STATE = Path("/scratch/diogoneiss/trident-2026-11-22-sliced-fresh")
E185 = ("pendigits_20nan", "pendigits_40nan", "pendigits_60nan", "pendigits_80nan",
        "electricity_20nan", "electricity_40nan", "electricity_80nan", "letter_20nan")
VARIANTS: tuple[str, ...] = (*e184.VARIANTS, *E185)
SEEDS: tuple[int, ...] = (1717, 8484, 9595)
ARMS: dict[str, dict[str, object]] = {"K": {}, "S": {"decode_metric": "sw_score"}}
assert len(VARIANTS) == len(set(VARIANTS)) == 26 and not set(SEEDS) & set(e184.SEEDS)

e184.VARIANTS, e184.SEEDS, e184.EXPERIMENT, e184.STATE = VARIANTS, SEEDS, EXPERIMENT, STATE
e184.ARMS, e184.PAIRS = ARMS, (("K", "S"),)  # neither arm is in its TERM: no run carries the loss term

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] in ARMS:
        e184.train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        e184.todo(int(argv[1]), int(argv[2]))
    elif argv == ["--gate"]:
        e184.gate()
    elif argv == ["--pairing"]:
        e184.pairing()
    else:
        sys.exit(__doc__)
