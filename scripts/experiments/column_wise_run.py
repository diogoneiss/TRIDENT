"""One cell of the column-wise scoring study (imputation-token-shape/01, task T02 step 1).

    uv run --python 3.11 python scripts/experiments/column_wise_run.py \
        <variant> <seed> <output_dir> <metrics_dir>

A plain imputation run under today's defaults (ADR 0013: the configuration E30 called M),
reading the variant's promoted file or the defaults exactly as any run does, with
``--score_column_wise`` on: the same model then scores the induced cells both ways, every gap
as [MASK] at once (the headline) and one gap column at a time with the other gaps as [NULL].
Five folds. Run from the checkout root.
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

EXPERIMENT = "column-wise-2026-10-02"


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
        score_column_wise=True,
        mlflow_tags={"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
