"""One cell of the deployable-checkpoint study (imputation-protocol/03, E43).

    uv run --python 3.11 python scripts/experiments/checkpoint_run.py \
        <variant> <seed> <output_dir> <metrics_dir> [--no-mlflow]

A run of today's imputation configuration, read as a default run reads it, with
``--score_checkpoints`` and ``--score_calibrated``. One trajectory gives every arm:

- K, the headline ``impute/*``: the checkpoint chosen on the validation gaps (ADR 0015), an oracle;
- H, ``impute/*/loss_checkpoint/*``: the epoch of the lowest validation loss;
- M, ``impute/*/masked_checkpoint/*``: the epoch whose fixed validation mask scores best;
- KC, ``impute/induced/calibrated/*``: K after the calibration fitted on the validation gaps.

A trailing ``--no-mlflow`` runs the cell without MLflow and without the study's tags. Five folds.
Run from the checkout root.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training.config import complete_configuration, load_hyperparameters  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENT = "deployable-checkpoint-2026-10-17"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("metrics_dir")
    parser.add_argument("--no-mlflow", action="store_true")
    options = parser.parse_args(argv)
    resolved, source = load_hyperparameters(Namespace(dataset_name=options.variant, task="imputation"))
    print(
        "cell", options.variant, options.seed, "source", source,
        "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True,
    )
    args = Namespace(
        dataset_name=options.variant,
        label_column="class",
        output_dir=options.output_dir,
        metrics_dir=options.metrics_dir,
        disable_mlflow=options.no_mlflow,
        plot_losses=False,
        save_model=False,
        seed=options.seed,
        cv_folds=5,
        task="imputation",
        score_checkpoints=True,
        score_calibrated=True,
        mlflow_tags={} if options.no_mlflow else {"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
