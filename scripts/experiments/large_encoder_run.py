"""One cell of E59: ADR 0018's bigger encoder on the large tables at 40-80nan (imputation-protocol/14).

    uv run --python 3.11 python scripts/experiments/large_encoder_run.py \
        <variant> <seed> ALL <output_dir> <metrics_dir> [--no-mlflow]

Today's imputation configuration as a default run reads it (no promoted file applies to these
variants), with ``--score_checkpoints`` and ``--score_calibrated`` as E44 ran, plus E37's ALL arm,
the bigger encoder ADR 0018 adopted on pendigits_20nan and letter_20nan: ``LAYERS 4``,
``DIM_FEED 256``, ``ENCODER_FINAL_NORM layer``. G, today's default, is E44's run on the same seed.
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

EXPERIMENT = "large-encoder-2026-10-21"
ARMS: dict[str, dict[str, object]] = {"ALL": {"LAYERS": 4, "DIM_FEED": 256, "ENCODER_FINAL_NORM": "layer"}}


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    check_only = "--no-mlflow" in argv[5:]
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    config.update(ARMS[arm])
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
        score_calibrated=True,
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
