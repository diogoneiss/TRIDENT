"""One cell of the MNAR smoke family (imputation-protocol/05, E45).

    uv run --python 3.11 python scripts/experiments/mnar_smoke_run.py \
        <variant> <seed> <G|Mx> <output_dir> <metrics_dir> [--no-mlflow]

Arms:

- G: today's imputation configuration as a default run reads it, with ``--score_checkpoints``: the
  headline is K (an oracle, more so under MNAR), with H and M on the same trajectory. H is the
  headline this study reads (E43).
- Mx: the gaps shown as ``[NULL]`` in the decode stage and the loss checkpoint (E31's shape), with
  ``--score_column_wise`` (each gap column asked with the others as ``[NULL]``) and
  ``--score_checkpoints``.

Variants: ``<base>_probit_60nan`` (``datasets/generate_mnar.py``) and the MCAR ``<base>_60nan`` of
biodeg, kc2 and vehicle. A trailing ``--no-mlflow`` runs the cell without MLflow and its tags.
Five folds. Run from the checkout root.
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

EXPERIMENT = "mnar-smoke-2026-10-17"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("arm", choices=("G", "Mx"))
    parser.add_argument("output_dir")
    parser.add_argument("metrics_dir")
    parser.add_argument("--no-mlflow", action="store_true")
    options = parser.parse_args(argv)
    extra: dict[str, object] = {"score_checkpoints": True}
    if options.arm == "Mx":
        extra |= {"decode_gap_token": "null", "decode_checkpoint": "loss", "score_column_wise": True}
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
