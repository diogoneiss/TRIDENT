"""One cell of the early-stopping (imputation-pretraining/04) or larger-batch (05) study.

    uv run --python 3.11 python scripts/experiments/decode_patience_batch_run.py \
        <variant> <seed> <arm P|Q2|Q4> <output_dir> <metrics_dir>

Every arm starts from study 03's arm M: the normalised embedding objective for the configured
pre-training epochs (300), then a decode stage of 450. P adds a decode patience of 50 epochs;
Q2 and Q4 train with batch 1024 and both learning rates doubled or quadrupled. The
configuration is the one M's cells read, which for ``credit-g_20nan`` is the promoted file
removed on 2026-09-30 (``fcaadfb``), pinned here so its pairs compare one configuration.
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

from src.training.config import hyperparameter_file  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENTS = {"P": "decode-early-stopping-2026-09-30", "Q2": "larger-batch-2026-09-30", "Q4": "larger-batch-2026-09-30"}
# datasets/hiperparams/credit-g/credit-g_20nan.imputation.json at fcaadfb, as study 03 read it.
PINNED = {
    "credit-g_20nan": {
        "DIM": 128, "HIDDEN_DIM": 16, "HEADS": 16, "LAYERS": 2, "DIM_FEED": 32,
        "DROPOUT": 0.30000000000000004, "EPOCHS_PRE": 300, "BATCH": 256, "LR_PRE": 0.00034,
        "WEIGHT_DECAY_PRE": 0.005, "PROB_MASCARA": 0.4, "LR_SCHEDULER": "cosine",
        "EPOCHS_DECODE": 150, "LR_DECODE": 0.002538672990101232,
        "WEIGHT_DECAY_DECODE": 0.0023168563407433367, "LAMBDA_NUM": 2.738287642961479,
        "EVAL_MASK_RATE": 0.2,
    }
}
DEFAULT_LR_PRE, DEFAULT_LR_DECODE = 0.00034, 0.001


def configuration(variant: str) -> tuple[dict[str, object], str]:
    if variant in PINNED:
        return dict(PINNED[variant]), "pinned fcaadfb"
    promoted = hyperparameter_file(variant, "imputation")
    if promoted.exists():
        return json.loads(promoted.read_text()), str(promoted)
    return {}, "defaults"


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    if arm not in EXPERIMENTS:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(EXPERIMENTS)}")
    config, source = configuration(variant)
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
    if arm == "P":
        config["DECODE_PATIENCE"] = 50
    else:
        scale = 2.0 if arm == "Q2" else 4.0
        config["BATCH"] = 1024
        config["LR_PRE"] = scale * float(str(config.get("LR_PRE", DEFAULT_LR_PRE)))
        config["LR_DECODE"] = scale * float(str(config.get("LR_DECODE", DEFAULT_LR_DECODE)))
    print("cell", variant, seed, arm, "config from", source, json.dumps(config), flush=True)
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
        mlflow_tags={"experiment": EXPERIMENTS[arm], "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
