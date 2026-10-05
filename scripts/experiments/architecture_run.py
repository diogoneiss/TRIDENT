"""One cell of the architecture screening (imputation-architecture/01, task T03).

    uv run --python 3.11 python scripts/experiments/architecture_run.py \
        <variant> <seed> <arm L4|F256|LN|ALL|R0> <output_dir> <metrics_dir> [--no-mlflow]

Each arm is today's imputation configuration (ADRs 0013 to 0015), read from the variant's
promoted file or the defaults exactly as a run reads it, with one architectural change:
L4 = ``LAYERS`` 4 (default 2), F256 = ``DIM_FEED`` 256 (default 32), LN = a final LayerNorm on the
encoder (``ENCODER_FINAL_NORM layer``), ALL = the three together. R0 changes nothing: it is the
pre-launch check that this path reproduces a default run, never a study cell. A trailing
``--no-mlflow`` runs the cell without MLflow and without the study's tags. Five folds. Run from
the checkout root.
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

EXPERIMENT = "architecture-2026-10-05"
CHANGES: dict[str, dict[str, object]] = {
    "L4": {"LAYERS": 4},
    "F256": {"DIM_FEED": 256},
    "LN": {"ENCODER_FINAL_NORM": "layer"},
    "ALL": {"LAYERS": 4, "DIM_FEED": 256, "ENCODER_FINAL_NORM": "layer"},
    "R0": {},
}


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    check_only = argv[5:] == ["--no-mlflow"]
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
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
