"""E148's cells: observed cells hidden on the MIMIC-IV tables with natural gaps
(docs/tickets/mimic/03-masked-cells-natural-gaps.md).

    uv run --python 3.11 python -B scripts/experiments/mimic_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mimic_run.py --todo <queue> <queues>

Each cell is a default imputation run of the checkout, five folds, tracked in MLflow, with
``--decode_checkpoint masked`` (the tables have no complete sibling, so the default would keep the epoch of the lowest
loss) and ``--export_gap_fills`` (the fills of the natural gaps, which E150 reads; it changes no score). The tables are
read from ``$TRIDENT_DATASETS_ROOT``. A cell is done when its cross-validation summary is written, so relaunching
resumes. Queues through ``c15_queue.sh`` with ``QUEUE_RUNNER`` and ``QUEUE_STATE``. Run from the checkout root.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))
if not os.environ.get("QUEUE_STATE") or not os.environ.get("TRIDENT_DATASETS_ROOT"):
    sys.exit("set QUEUE_STATE (the study's state directory) and TRIDENT_DATASETS_ROOT (where the tables are)")

STATE = Path(os.environ["QUEUE_STATE"])
EXPERIMENT = "mimic-natural-gaps-2026-11-06"
VARIANTS: tuple[str, ...] = ("mimic-cardio_natural", "mimic-sofa_natural")
SEEDS = (42, 7, 13)
ARMS: dict[str, dict[str, object]] = {"M": {"decode_checkpoint": "masked", "export_gap_fills": True}}


def cell_dir(arm: str, variant: str, seed: int) -> Path:
    return STATE / "results" / f"{variant}_s{seed}_{arm}"


def done(arm: str, variant: str, seed: int) -> bool:
    return any((cell_dir(arm, variant, seed) / variant).glob("*/metrics/cv_summary.json"))


def train(arm: str, variant: str, seed: int) -> None:
    from train import main as train_main

    out = cell_dir(arm, variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out), metrics_dir=str(out / "metrics"),
                     disable_mlflow=False, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
                     mlflow_tags={"experiment": EXPERIMENT, "arm": arm}, **ARMS[arm])
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    cells = [(arm, variant, seed) for seed in SEEDS for variant in VARIANTS for arm in ARMS]
    for k, (arm, variant, seed) in enumerate(cells):
        if k % queues == which - 1 and not done(arm, variant, seed):
            print(arm, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
