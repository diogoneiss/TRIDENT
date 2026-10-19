"""One cell of the deployable calibration study (imputation-calibration/02, E47).

    uv run --python 3.11 python scripts/experiments/deployable_calibration_run.py \
        <variant> <seed> <output_dir> <metrics_dir> [--no-mlflow]

Today's imputation configuration as a default run reads it, with four changes, none of which
touches training:

- ``--decode_checkpoint loss``: the headline is H, the loss's checkpoint, which needs no complete
  table (E43);
- ``--decode_calibration none``: the headline stays uncalibrated (ADR 0027 would calibrate it);
- ``--score_masked_calibration``: H after a calibration fitted on the fixed validation mask's
  cells, which are observed (HC_m, ``impute/*/masked_calibrated/*``), the deployable candidate;
- ``--score_calibrated`` and ``--score_induced_checkpoint``: H after the calibration fitted on the
  validation gaps (HC_g, ``impute/induced/calibrated/*``, an oracle), and K and KC
  (``induced_checkpoint`` and ``induced_checkpoint_calibrated``), which must equal E40's cells.

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

EXPERIMENT = "deployable-calibration-2026-10-18"
EXTRA: dict[str, object] = {
    "decode_checkpoint": "loss",
    "decode_calibration": "none",
    "score_masked_calibration": True,
    "score_calibrated": True,
    "score_induced_checkpoint": True,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("output_dir")
    parser.add_argument("metrics_dir")
    parser.add_argument("--no-mlflow", action="store_true")
    options = parser.parse_args(argv)
    resolved, source = load_hyperparameters(Namespace(dataset_name=options.variant, task="imputation", **EXTRA))
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
        mlflow_tags={} if options.no_mlflow else {"experiment": EXPERIMENT},
        **EXTRA,
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
