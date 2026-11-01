"""ADR 0036's thesis cells: today's default on the four variants it changed, at the seeds the thesis
tables read (imputation-protocol/43).

    uv run --python 3.11 python -B scripts/experiments/adr36_thesis_runs.py train <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/adr36_thesis_runs.py --todo <n> <queues>

``train`` runs one cell of today's default (the variant's promoted file, ADR 0036), five folds,
``--score_checkpoints --score_calibrated`` so every readout the tables use is logged, MLflow tag
``experiment=adr36-thesis-2026-11-01``. ``--todo`` deals this machine's cells (CELLS) over the queues,
skipping any the store already holds.
"""

from __future__ import annotations

import socket
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "adr36-thesis-2026-11-01"
# The development tables read pendigits_20nan and letter_20nan at 101/202/303 and 42/7/13; the
# confirmation set (E44) reads pendigits_40nan and letter_40nan at 707/808/909.
CELLS = [*((v, s) for v in ("pendigits_20nan", "letter_20nan") for s in (101, 202, 303, 42, 7, 13)),
         *((v, s) for v in ("pendigits_40nan", "letter_40nan") for s in (707, 808, 909))]
MACHINE = {"gorgona8": CELLS[::3], "medusa": [c for k, c in enumerate(CELLS) if k % 3]}  # medusa1 has two GPUs


def train(variant: str, seed: int) -> None:
    from train import main as train_main

    out = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-11-01-adr36-thesis"
    train_main(Namespace(
        dataset_name=variant, label_column="class", output_dir=str(out / "results" / f"{variant}_s{seed}"),
        metrics_dir=str(out / "metrics" / f"{variant}_s{seed}"), disable_mlflow=False, plot_losses=False,
        save_model=False, seed=seed, cv_folds=5, task="imputation", score_checkpoints=True, score_calibrated=True,
        mlflow_tags={"experiment": EXPERIMENT},
    ), return_metrics=True)
    print("CELL_DONE", variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    done = common.parents(EXPERIMENT)
    mine = [c for c in MACHINE[socket.gethostname()] if c not in done]
    for k, (variant, seed) in enumerate(mine):
        if k % queues == which - 1:
            print(variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 3:
        train(argv[1], int(argv[2]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
