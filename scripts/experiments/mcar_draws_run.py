"""One cell of E62: today's imputation default on fresh MCAR draws (imputation-protocol/17, T31).

    uv run --python 3.11 python scripts/experiments/mcar_draws_run.py \
        <variant> <seed> <output_dir> <metrics_dir> [--no-mlflow]

``<variant>`` is a shipped variant (draw 1, ``<base>_<pp>nan``) or a fresh draw of one
(``<base>_d<d>_<pp>nan``, from ``datasets/generate_mcar_draws.py``). Every draw trains with the
configuration a default run of the shipped variant resolves, passed as an override so that a
promoted file keyed by the shipped name (ADR 0018's on pendigits_20nan) reaches its draws too.
``--score_checkpoints`` and ``--score_calibrated`` as E44 and E51 ran; ADR 0027's ``auto``
calibration resolves on each draw's measured rate, the same as the original's. Five folds. A
trailing ``--no-mlflow`` runs the cell without MLflow and without the study's tags. Run from the
checkout root.
"""

from __future__ import annotations

import json
import re
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training.config import complete_configuration, load_hyperparameters  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENT = "mcar-draws-2026-10-22"
DRAW = re.compile(r"^(?P<base>[^_]+)_d(?P<draw>\d+)_(?P<pp>\d{2})nan$")


def original_of(variant: str) -> tuple[str, int]:
    """The shipped variant a draw was made from, and the draw's number (1 for the shipped one)."""
    found = DRAW.match(variant)
    if found is None:
        return variant, 1
    return f"{found['base']}_{found['pp']}nan", int(found["draw"])


def main(argv: list[str]) -> int:
    variant, seed, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3]
    check_only = "--no-mlflow" in argv[4:]
    original, draw = original_of(variant)
    resolved, source = load_hyperparameters(Namespace(dataset_name=original, task="imputation"))
    config = complete_configuration(resolved, "imputation")
    again, _ = load_hyperparameters(Namespace(dataset_name=variant, task="imputation", hyperparams_override=config))
    if again != resolved:
        sys.exit(f"{variant}: the override does not resolve to {original}'s configuration")
    print("cell", variant, seed, "draw", draw, "of", original, "config from", source, json.dumps(config), flush=True)
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
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "mcar_draw": str(draw), "draw_of": original},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
