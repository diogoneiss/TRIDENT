"""C1's cells (docs/plans/2026-11-04-c1-split.md; tickets docs/tickets/c1-split/), MLflow off, five folds a cell.

    uv run --python 3.11 python -B scripts/experiments/c1_run.py train <arm> <variant> <seed>
    C1_CUT=<2|3> uv run --python 3.11 python -B scripts/experiments/c1_run.py --todo <n> <queues>

Arms, over each variant's resolved classification configuration with ``PRETRAIN_OBJECTIVE embedding_normalized``
and ``C15_ARM A1`` (the shipped classifier plus its per-fold readouts; ``src/training/c15_inputs.py``):
S the shipped split; T the tied arm (``PRETRAIN_HOLE_TOKEN null``); B no pre-training (``EPOCHS_PRE 0``); BT B run
through T's code path (the smoke's integrity check 1); ANCHOR S with the paper's ``embedding`` objective; S_GEOM S
again, for the token dump the reused S cells lack (seed 42; its scores enter nothing).

S's decision-mode cells on cut 3's files are E124's A1 cells (same configuration, folds and seeds; the protocol's
cost table shares them), read from C15's state directory and never rerun. Every other cell lands on the machine
that holds its pair's S cell (ADR 0035, through ``c15_run.machine``). Cut 2 (the smoke) runs with deterministic
algorithms forced (``C1_DETERMINISTIC=1``) under ``<state>/smoke``.
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

from scripts.experiments import c15_run  # noqa: E402

HOST = c15_run.HOST
STATE = Path("/scratch2/diogoneiss" if HOST == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-11-04-c1"
TABLES = ("biodeg", "kc2", "vehicle", "spambase")
MNAR = tuple(f"{b}_probit_60nan" for b in TABLES)
MCAR = tuple(f"{b}_60nan" for b in TABLES)
SEEDS = (42, 7, 13)
ARMS: dict[str, dict[str, object]] = {
    "S": {},
    "T": {"PRETRAIN_HOLE_TOKEN": "null"},
    "B": {"EPOCHS_PRE": 0},
    "BT": {"EPOCHS_PRE": 0, "PRETRAIN_HOLE_TOKEN": "null"},
    "ANCHOR": {"PRETRAIN_OBJECTIVE": "embedding"},
    "S_GEOM": {},
}
SMOKE_VARIANT = "biodeg_probit_60nan"
CUTS: dict[str, dict[tuple[str, int], tuple[str, ...]]] = {
    "2": {(SMOKE_VARIANT, 42): ("S", "T", "B", "BT")},
    "3": {
        **{(v, s): ("T", "B") for v in MCAR for s in SEEDS},
        **{(v, s): ("T", "B", "ANCHOR") + (("S_GEOM",) if s == 42 else ()) for v in MNAR for s in SEEDS},
    },
}


def cut() -> str:
    return os.environ.get("C1_CUT", "3")


def cell_dir(arm: str, variant: str, seed: int) -> Path:
    if arm == "S" and cut() != "2":
        return c15_run.cell_dir("A1", variant, seed, c15_run.STATE)
    return (STATE / "smoke" if cut() == "2" else STATE) / arm / f"{variant}_s{seed}"


def done(arm: str, variant: str, seed: int) -> bool:
    d = cell_dir(arm, variant, seed) / "metrics"
    return (d / f"{variant}_metrics.csv").exists() and len(list(d.glob("c15_readout_fold_*.pt"))) == 5


def train(arm: str, variant: str, seed: int) -> None:
    if os.environ.get("C1_DETERMINISTIC") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", variant])))
    override = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": "A1", **ARMS[arm]}
    out = cell_dir(arm, variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False, save_model=False,
                     seed=seed, cv_folds=5, task="classification", mlflow_tags={}, hyperparams_override=override)
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    pairs = CUTS[cut()]
    # The smoke's cells pair with nothing reused, so they take gorgona8, idle otherwise.
    mine = [pair for pair in pairs if (HOST == "gorgona8" if cut() == "2" else c15_run.machine(pair) == HOST)]
    mine.sort(key=lambda p: (not p[0].startswith("spambase"), p[1], p[0]))  # dearest first, so the queues end together
    for k, (variant, seed) in enumerate(mine):
        if k % queues == which - 1:
            for arm in pairs[(variant, seed)]:
                if (arm != "S" or cut() == "2") and not done(arm, variant, seed):
                    print(arm, variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
