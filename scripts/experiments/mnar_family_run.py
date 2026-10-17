"""One cell of the full MNAR family (imputation-protocol/06, E46).

    uv run --python 3.11 python scripts/experiments/mnar_family_run.py \
        <variant> <seed> <G|Mx> <output_dir> <metrics_dir> [--no-mlflow]

The arms are E45's (``mnar_smoke_run.py``), unchanged; only the experiment tag differs, so the two
families stay apart in the store. Variants: ``spambase_probit_60nan`` and ``spambase_60nan``; the
20nan rung (``<base>_probit_20nan`` and ``<base>_20nan`` of biodeg, kc2 and vehicle); and biodeg's
tilt dial (``biodeg_probit0p25_60nan``, ``biodeg_probit1p0_60nan``, ``biodeg_probit2p0_60nan``).
Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import mnar_smoke_run as smoke  # noqa: E402

EXPERIMENT = "mnar-family-2026-10-17"

if __name__ == "__main__":
    smoke.EXPERIMENT = EXPERIMENT
    raise SystemExit(smoke.main(sys.argv[1:]))
