"""One cell of the pre-registered decode-budget study (imputation-pretraining/03).

docs/tickets/imputation-pretraining/03-normalised-pretraining-with-a-long-decode.md. Run
from the checkout root:

    uv run --python 3.11 python scripts/experiments/pretrain_decode_budget_run.py \
        <variant> <seed> M <output_dir> <metrics_dir>

M = the normalised embedding objective for the configured pre-training epochs (300), then a
decode stage of the configured decode epochs plus the pre-training epochs (450), the
length arm C gave the decoder. Otherwise the runner of study 02.
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

EXPERIMENT = "pretrain-decode-budget-2026-09-30"


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    if arm != "M":
        sys.exit(f"unknown arm {arm!r}; this study runs only M")
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    pre = int(str(config.get("EPOCHS_PRE", 300)))
    decode = int(str(config.get("EPOCHS_DECODE", 150)))
    config.update(
        {
            "EPOCHS_PRE": pre,
            "EPOCHS_DECODE": decode + pre,
            "LR_SCHEDULER": "cosine",
            "PRETRAIN_OBJECTIVE": "embedding_normalized",
        }
    )
    source = promoted if promoted.exists() else "defaults"
    print("cell", variant, seed, arm, "config from", source, "epochs", (pre, decode + pre), flush=True)
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
