"""One cell of E51, today's imputation default on fresh seeds at 20nan (imputation-protocol/08).

    uv run --python 3.11 python scripts/experiments/default_refresh_run.py \
        <variant> <seed> <output_dir> <metrics_dir> [--no-mlflow]

Today's configuration, read as a default run of the checkout reads it (ADRs 0013 to 0015, 0018
on pendigits_20nan and letter_20nan, 0019, 0021's live strong tier, 0027's automatic
calibration, which resolves to none at 20nan), with ``--score_checkpoints`` and
``--score_calibrated`` as E44 ran them. One trajectory gives:

- K, the headline ``impute/*``: the checkpoint chosen on the validation gaps (ADR 0015);
- H, ``impute/*/loss_checkpoint/*``, and M, ``impute/*/masked_checkpoint/*``;
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

EXPERIMENT = "default-refresh-2026-10-20"


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
