"""E190's cells: the imputation default on the wide development table, ``gas-drift``, at 20 to 80nan
(docs/tickets/imputation-protocol/79-wide-table-standing.md). With ``WIDE_TABLE=nomao``, E191's: the same on the
confirmation table (docs/tickets/imputation-protocol/80-wide-table-nomao.md).

    uv run --python 3.11 python -B scripts/experiments/wide_table_run.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/wide_table_run.py --todo <queue> <queues>

Each cell is a default imputation run of the checkout, five folds, tracked in MLflow, with no flag beside the
defaults. The table is read from the checkout (``datasets/build_wide_tables.py``). A cell is done when its
cross-validation summary is written, so relaunching resumes. Queues through ``c15_queue.sh`` with ``QUEUE_RUNNER``
and ``QUEUE_STATE``; the cells are listed level by level, so three queues hold one seed each. Run from the
checkout root.
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
if not os.environ.get("QUEUE_STATE"):
    sys.exit("set QUEUE_STATE (the study's state directory)")

STATE = Path(os.environ["QUEUE_STATE"])
TABLE = os.environ.get("WIDE_TABLE", "gas-drift")  # E191 sets nomao
EXPERIMENT = {"gas-drift": "wide-table-standing-2026-11-25", "nomao": "wide-table-nomao-2026-11-27"}[TABLE]
VARIANTS: tuple[str, ...] = tuple(f"{TABLE}_{level}nan" for level in (20, 40, 60, 80))
SEEDS = (42, 7, 13)
ARM = "K"


def cell_dir(variant: str, seed: int) -> Path:
    return STATE / "results" / f"{variant}_s{seed}_{ARM}"


def finished(variant: str, seed: int) -> list[Path]:
    """The cell's run directories that reached the cross-validation summary; a run that died leaves none."""
    return [path.parents[1] for path in (cell_dir(variant, seed) / variant).glob("*/metrics/cv_summary.json")]


def train(variant: str, seed: int) -> None:
    from train import main as train_main

    out = cell_dir(variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out), metrics_dir=str(out / "metrics"),
                     disable_mlflow=False, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
                     mlflow_tags={"experiment": EXPERIMENT, "arm": ARM})
    train_main(args, return_metrics=True)
    print("CELL_DONE", ARM, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    cells = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    for k, (variant, seed) in enumerate(cells):
        if k % queues == which - 1 and not finished(variant, seed):
            print(ARM, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:2] == ["train", ARM] and len(argv) == 4:
        train(argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
