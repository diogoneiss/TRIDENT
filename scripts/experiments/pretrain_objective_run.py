"""One cell of the pre-registered pre-training objective study.

docs/tickets/imputation-pretraining/02-pretraining-objectives.md. Run from the checkout root
(ADR 0009 requires server runs to start there):

    uv run --python 3.11 python scripts/experiments/pretrain_objective_run.py \
        <variant> <seed> <arm A|B|V|N> <output_dir> <metrics_dir>

A = the embedding objective as configured, B = no pre-training, V = the value objective,
N = the normalised embedding objective; every arm keeps the variant's configured epochs,
except B's zero pre-training epochs. The same runner as the pre-training ablation's
otherwise: the promoted configuration where one exists, cosine, five folds.
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

EXPERIMENT = "pretrain-objective-2026-09-30"
OBJECTIVES = {"A": "embedding", "B": "embedding", "V": "value", "N": "embedding_normalized"}


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    if arm not in OBJECTIVES:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(OBJECTIVES)}")
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    pre = int(str(config.get("EPOCHS_PRE", 300)))
    config.update(
        {
            "EPOCHS_PRE": 0 if arm == "B" else pre,
            "LR_SCHEDULER": "cosine",
            "PRETRAIN_OBJECTIVE": OBJECTIVES[arm],
        }
    )
    source = promoted if promoted.exists() else "defaults"
    print("cell", variant, seed, arm, "config from", source, "objective", OBJECTIVES[arm], flush=True)
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
        hyperparams_override=config,
        lr_scheduler="cosine",
        mlflow_tags={"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
