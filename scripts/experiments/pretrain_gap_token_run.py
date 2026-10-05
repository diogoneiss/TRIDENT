"""One cell of the pre-training gap-token study (imputation-token-shape/06).

    uv run --python 3.11 python scripts/experiments/pretrain_gap_token_run.py \
        <variant> <seed> <output_dir> <metrics_dir>

Arm PG: a plain imputation run under today's defaults (ADRs 0013 to 0015), reading the variant's
promoted file or the defaults exactly as any run does, with ``--pretrain_gap_token mask``: the
pre-training stage, too, shows every real gap as [MASK] instead of [NULL]. The reference is E35's
K readout of the same variant and seed, which equals a default run to the last digit. Five folds.
Run from the checkout root.
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

EXPERIMENT = "pretrain-gap-token-2026-10-05"


def main(argv: list[str]) -> int:
    variant, seed, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3]
    print("cell", variant, seed, "PG", flush=True)
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
        pretrain_gap_token="mask",
        mlflow_tags={"experiment": EXPERIMENT, "arm": "PG"},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
