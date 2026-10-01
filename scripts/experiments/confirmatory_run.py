"""One cell of the confirmatory study on the 21 imputation variants (imputation-pretraining/07).

    uv run --python 3.11 python scripts/experiments/confirmatory_run.py \
        <variant> <seed> <arm A|M|P> <output_dir> <metrics_dir>

A = the current default: the ``embedding`` objective for the configured pre-training epochs,
then the configured decode epochs. M = the ``embedding_normalized`` objective, then a decode
stage of the configured decode plus pre-training epochs (450). P = M with a decode patience of
50 epochs. Each variant uses the configuration a run of it loads today (its promoted
``*.imputation.json`` or the defaults), cosine, five folds, the default (batched) decoder heads.
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

EXPERIMENT = "confirmatory-2026-10-01"
ARMS = ("A", "M", "P")


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    pre = int(str(config.get("EPOCHS_PRE", 300)))
    decode = int(str(config.get("EPOCHS_DECODE", 150)))
    config["LR_SCHEDULER"] = "cosine"
    if arm == "A":
        config["PRETRAIN_OBJECTIVE"] = "embedding"
    else:
        config.update({"PRETRAIN_OBJECTIVE": "embedding_normalized", "EPOCHS_DECODE": decode + pre})
        if arm == "P":
            config["DECODE_PATIENCE"] = 50
    source = promoted if promoted.exists() else "defaults"
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
        mlflow_tags={"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
