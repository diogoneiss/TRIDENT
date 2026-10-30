"""E98: step zero Stage 4, the TRIDENT arm tuned on validation (classification-step-zero/07, H1c).

    uv run --python 3.11 python -B scripts/experiments/stage4_trident.py study <variant>
    uv run --python 3.11 python -B scripts/experiments/stage4_trident.py eval <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/stage4_trident.py --todo <n> <queues>

``study`` runs ADR 0029's validation-ranked Optuna study on one variant: 20 trials of the ``full``
profile (the paper's 20–60 epoch budget), each scored as the mean validation macro-F1 at the kept
epoch over ten folds at seed 42, ``cosine_legacy``. Its retrain (seed 42, the same ten folds) is the
seed-42 evaluation. ``eval`` trains the study's best configuration at another seed on ten folds,
MLflow tag ``experiment=step-zero-trident-2026-10-30``. ``--todo`` deals each variant's study, then its
two evaluations, over the queues, the largest tables first. Every output is under ``STATE``.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments.cls_stage2_run import ROWS, VARIANTS, fold_metrics  # noqa: E402

EXPERIMENT = "step-zero-trident-2026-10-30"
# gorgona8 and medusa1 both see the other's state only through these copies; each writes its own.
STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-10-30-step-zero"
TUNING_SEED = 42
EVAL_SEEDS = (7, 13)
TRIALS = 20
FOLDS = 10


def study_dir(variant: str) -> Path:
    return STATE / "optuna" / variant


def best_config(variant: str) -> dict[str, object] | None:
    found = sorted(study_dir(variant).glob(f"{variant}/optuna_*/best_hyperparameters.json"))
    return json.loads(found[-1].read_text()) if found else None


def retrained(variant: str) -> bool:
    return any(fold_metrics(d) is not None for d in study_dir(variant).glob(f"{variant}/optuna_*/best_model"))


def eval_dir(variant: str, seed: int) -> Path:
    return STATE / "eval" / f"{variant}_s{seed}"


def claim(item: str) -> bool:
    """Reserve an item for this process; False if another queue holds or finished it (amendment
    2026-10-30: helper queues share the work of the six)."""
    path = STATE / "claims" / item
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def study(variant: str) -> None:
    import opt

    if retrained(variant) or not claim(f"study_{variant}"):
        print("CELL_DONE study", variant, "skipped: finished or claimed by another queue", flush=True)
        return

    opt.run_hyperparameter_optimization(Namespace(
        dataset_name=variant, label_column="class", n_trials=TRIALS, seed=TUNING_SEED,
        output_dir=str(study_dir(variant)), metrics_dir=str(STATE / "metrics" / f"{variant}_study"),
        retrain_best=True, promote_best=False, task="classification", cv_folds=FOLDS,
        classification_search_objective="validation", lr_scheduler="cosine_legacy", search_space="full",
        disable_mlflow=False,
    ))
    print("CELL_DONE study", variant, json.dumps(best_config(variant)), flush=True)


def evaluate(variant: str, seed: int) -> None:
    from train import main as train_main

    config = best_config(variant)
    if config is None or not retrained(variant):
        sys.exit(f"{variant}: no finished study yet")
    if fold_metrics(eval_dir(variant, seed)) is not None or not claim(f"eval_{variant}_s{seed}"):
        print("CELL_DONE eval", variant, seed, "skipped: finished or claimed by another queue", flush=True)
        return
    train_main(Namespace(
        dataset_name=variant, label_column="class", output_dir=str(eval_dir(variant, seed)),
        metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"), disable_mlflow=False, plot_losses=False,
        save_model=False, seed=seed, cv_folds=FOLDS, task="classification", lr_scheduler="cosine_legacy",
        hyperparams_override=config, mlflow_tags={"experiment": EXPERIMENT, "arm": "trident_tuned"},
    ), return_metrics=True)
    frame = fold_metrics(eval_dir(variant, seed))
    print("CELL_DONE eval", variant, seed, [] if frame is None else [round(float(x), 4) for x in frame["f1_macro"]], flush=True)


def todo(which: int, queues: int) -> None:
    """Variant by variant, the largest tables first, dealt round-robin: a queue runs its variant's
    study, then that variant's evaluations, so a configuration is evaluated on the machine that
    tuned it."""
    work = sorted(VARIANTS, key=lambda v: (-ROWS[v.split("_")[0]], v))
    for k, variant in enumerate(work):
        if k % queues != which - 1:
            continue
        if not retrained(variant):
            print("study", variant)
        for seed in EVAL_SEEDS:
            if fold_metrics(eval_dir(variant, seed)) is None:
                print("eval", variant, seed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["study"] and len(argv) == 2:
        study(argv[1])
    elif argv[:1] == ["eval"] and len(argv) == 3:
        evaluate(argv[1], int(argv[2]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    else:
        sys.exit(__doc__)
