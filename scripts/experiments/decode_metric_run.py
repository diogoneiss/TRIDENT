"""E145's cells: the decode checkpoint chosen by ``impute_score`` (A) or by the distributional score (B)
(docs/tickets/imputation-protocol/51-decode-metric.md).

    uv run --python 3.11 python -B scripts/experiments/decode_metric_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/decode_metric_run.py --todo <queue> <queues>

Each cell is a default imputation run of the checkout, five folds, with ``--decode_calibration none`` so
the fold ledgers hold K, the fills of the kept checkpoint, and arm B adds ``--decode_metric dist_score``.
MLflow is off; a cell is done when its five ledgers are written. Seed 42 runs on gorgona8 and seeds 7 and
13 on medusa1; both arms of a (variant, seed) pair go to one queue, so they train on one GPU. Queues through ``c15_queue.sh`` with ``QUEUE_RUNNER`` and
``QUEUE_STATE``. Run from the checkout root.
"""

from __future__ import annotations

import os
import socket
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

HOST = "gorgona8" if socket.gethostname() == "gorgona8" else "medusa"
STATE = Path(os.environ.get("DECODE_METRIC_STATE") or (
    ("/scratch2/diogoneiss" if HOST == "gorgona8" else "/scratch/diogoneiss") + "/trident-2026-11-05-decode-metric"))
# E40's development variants at 40nan and above, where ADR 0025's panel saw the fills shrink.
VARIANTS = tuple(f"{table}_{level}nan" for table in ("kr-vs-kp", "spambase", "credit-g") for level in (40, 60, 80)) + (
    "vehicle_60nan", "biodeg_60nan", "kc2_60nan")
SEEDS = (42, 7, 13)
HERE = {"gorgona8": (42,), "medusa": (7, 13)}[HOST]  # ADR 0035: split by seed, both arms of a pair on one machine
ARMS: dict[str, dict[str, str]] = {"A": {}, "B": {"decode_metric": "dist_score"}}


def cell_dir(arm: str, variant: str, seed: int) -> Path:
    return STATE / "results" / f"{variant}_s{seed}_{arm}"


def done(arm: str, variant: str, seed: int) -> bool:
    return len(list((cell_dir(arm, variant, seed) / variant).glob("*/imputation/imputation_fold_*_cells.csv"))) == 5


def train(arm: str, variant: str, seed: int) -> None:
    from train import main as train_main

    out = cell_dir(arm, variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out), metrics_dir=str(out / "metrics"),
                     disable_mlflow=True, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
                     mlflow_tags={}, decode_calibration="none", **ARMS[arm])
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    pairs = [(variant, seed) for variant in VARIANTS for seed in HERE]
    big = ("spambase", "kr-vs-kp")
    pairs.sort(key=lambda p: (p[0].split("_")[0] not in big, p[1], p[0]))  # dearest first, so the queues end together
    for k, (variant, seed) in enumerate(pairs):
        if k % queues == which - 1:
            for arm in ARMS:
                if not done(arm, variant, seed):
                    print(arm, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
