"""One cell of the checkpoint and calibration study (imputation-token-shape/04).

    uv run --python 3.11 python scripts/experiments/selection_run.py \
        <variant> <seed> <output_dir> <metrics_dir>

A plain imputation run under today's defaults (ADR 0014), reading the variant's promoted file
or the defaults exactly as any run does, with ``--score_induced_checkpoint`` and
``--score_calibrated`` on: the same decode trajectory is scored four ways on the test split,
the headline (H), H calibrated on the validation rows' own gaps (C), the epoch those gaps score
best (K), and K calibrated (KC). Five folds. Run from the checkout root.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from train import main as train_main  # noqa: E402

EXPERIMENT = "selection-2026-10-03"


def main(argv: list[str]) -> int:
    variant, seed, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3]
    print("cell", variant, seed, flush=True)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        disable_mlflow=False,
        plot_losses=False,
        save_model=False,
        seed=seed,
        cv_folds=5,
        task="imputation",
        score_induced_checkpoint=True,
        score_calibrated=True,
        mlflow_tags={"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
