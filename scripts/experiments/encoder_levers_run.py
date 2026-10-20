"""One cell of E50, the decode stage's levers against washing out the encoder (imputation-pretraining/09).

    uv run --python 3.11 python scripts/experiments/encoder_levers_run.py \
        <variant> <seed> <G|F100|L10|C0> <output_dir> <metrics_dir> [--no-mlflow]

Today's imputation configuration as a default run reads it (the variant's promoted file, if it
has one, over the task defaults), with ``--score_checkpoints`` so that H and M are logged beside
K, plus one change per arm:

- G: none;
- F100: ``DECODE_FREEZE_EPOCHS 100``, the first 100 decode epochs train the heads only;
- L10: ``DECODE_ENCODER_LR_SCALE 0.1``, the encoder at a tenth of the decode learning rate;
- C0: ``EPOCHS_PRE 0``, no pre-training.

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

from src.training.config import hyperparameter_file  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENT = "encoder-levers-2026-10-19"
CHANGES: dict[str, dict[str, object]] = {
    "G": {},
    "F100": {"DECODE_FREEZE_EPOCHS": 100},
    "L10": {"DECODE_ENCODER_LR_SCALE": 0.1},
    "C0": {"EPOCHS_PRE": 0},
}


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    check_only = "--no-mlflow" in argv[5:]
    if arm not in CHANGES:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(CHANGES)}")
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    config.update(CHANGES[arm])
    print("cell", variant, seed, arm, "config from", promoted if promoted.exists() else "defaults", json.dumps(config), flush=True)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        disable_mlflow=check_only,
        plot_losses=False,
        save_model=False,
        seed=seed,
        cv_folds=5,
        task="imputation",
        hyperparams_override=config,
        score_checkpoints=True,
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
