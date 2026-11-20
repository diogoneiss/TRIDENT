"""E185: E184's sliced Wasserstein decode-loss term (``--decode_sw_weight 0.8``) on the eight larger MCAR variants
E184 left out (docs/tickets/imputation-protocol/74), on gorgona8 only. Same arms, weight, checks and read as
``sliced_loss_run.py``, whose functions run here with this study's variants, state, tag and a rule scaled to 8.

    uv run --python 3.11 python -B scripts/experiments/sliced_loss_large_run.py train <S|LS|K|L> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_large_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_large_run.py --gate
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_large_run.py --pairing
    uv run --python 3.11 python -B scripts/experiments/sliced_loss_large_run.py --tally [core]

Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import classcond_run as e181  # noqa: E402
from scripts.experiments import sliced_loss_run as e184  # noqa: E402
from scripts.experiments import sliced_run as e182  # noqa: E402

VARIANTS: tuple[str, ...] = (
    "pendigits_20nan", "pendigits_40nan", "pendigits_60nan", "pendigits_80nan",
    "electricity_20nan", "electricity_40nan", "electricity_80nan", "letter_20nan",
)
assert set(VARIANTS) <= set(e181.VARIANTS) and set(VARIANTS) <= set(e182.VARIANTS)
assert not set(VARIANTS) & set(e184.VARIANTS)


def _rule(better: int, worse: int, counted: int) -> str:
    """E184's rule with its worse allowance scaled from 2 of 18 to 1 of 8 (ticket 74)."""
    if not counted:
        return "no read (the control passes on no variant)"
    return "supported" if 2 * better >= counted and worse <= 1 else "refuted" if 6 * better <= counted else "mixed"


assert (_rule(4, 1, 8), _rule(4, 2, 8), _rule(1, 0, 8), _rule(2, 0, 8)) == ("supported", "mixed", "refuted", "mixed")

e184.VARIANTS, e184.EXPERIMENT = VARIANTS, "sliced-loss-large-2026-11-20"
e184.STATE = Path("/scratch/diogoneiss/trident-2026-11-20-sliced-loss-large")
e184._rule = _rule

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] in e184.ARMS:
        e184.train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        e184.todo(int(argv[1]), int(argv[2]))
    elif argv == ["--gate"]:
        e184.gate()
    elif argv == ["--pairing"]:
        e184.pairing()
    elif argv[:1] == ["--tally"] and argv[1:] in ([], ["core"]):
        e184.tally(core=argv[1:] == ["core"])
    else:
        sys.exit(__doc__)
