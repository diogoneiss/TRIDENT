"""One cell of E67: the gaps shown as ``[NULL]`` in the decode stage, alone (imputation-protocol/22, T33).

    uv run --python 3.11 python scripts/experiments/null_gaps_run.py \
        <variant> <seed> <G|N> <output_dir> <metrics_dir> [--no-mlflow]

Arms:

- G: today's imputation configuration as a default run reads it, with ``--score_checkpoints``,
  exactly as E45 and E46 ran their G (``mnar_smoke_run.py``). It is run here only for the gate; the
  study reads E45's and E46's G.
- N: G with ``DECODE_GAP_TOKEN null``. A real gap is shown as the stored ``[NULL]`` instead of
  ``[MASK]`` in the decode stage, in training and in scoring alike.
  - This is the one factor E45's Mx changed beside the loss checkpoint and column-wise scoring.
  - N keeps today's checkpoint (K) and all-at-once scoring.

Five folds. A trailing ``--no-mlflow`` runs the cell without MLflow and its tags. Run from the
checkout root.
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

EXPERIMENT = "null-gaps-2026-10-23"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("arm", choices=("G", "N"))
    parser.add_argument("output_dir")
    parser.add_argument("metrics_dir")
    parser.add_argument("--no-mlflow", action="store_true")
    options = parser.parse_args(argv)
    extra: dict[str, object] = {"score_checkpoints": True}
    if options.arm == "N":
        extra["decode_gap_token"] = "null"
    resolved, source = load_hyperparameters(Namespace(dataset_name=options.variant, task="imputation", **extra))
    print(
        "cell", options.variant, options.seed, options.arm, "source", source,
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
        mlflow_tags={} if options.no_mlflow else {"experiment": EXPERIMENT, "arm": options.arm},
        **extra,
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
