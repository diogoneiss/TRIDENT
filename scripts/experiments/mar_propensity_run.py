"""One cell of E65: the propensity-mask arm (P) on the MAR family
(imputation-protocol/20, T30).

    uv run --python 3.11 python scripts/experiments/mar_propensity_run.py \
        <variant> <seed> <output_dir> <metrics_dir> [--no-mlflow]

``<variant>`` is ``<base>_marf[<slope>]_<pp>nan`` or ``<base>_marfctl_<pp>nan`` (``datasets/generate_mar_family.py``).
Both train with the configuration a default run of ``<base>_<pp>nan`` resolves plus
``DECODE_MASK_WEIGHTING propensity`` (``src/training/mask_propensity.py``), passed as an
override, as E62 and E63 did; ``--score_checkpoints`` and ``--score_calibrated`` as E45 ran. Five folds. A
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

EXPERIMENT = "mar-propensity-2026-10-22"
NAME = re.compile(r"^(?P<base>[^_]+)_(?P<mechanism>marfctl|marf(?P<slope>\d+(?:p\d+)?)?)_(?P<pp>\d{2})nan$")


def main(argv: list[str]) -> int:
    variant, seed, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3]
    check_only = "--no-mlflow" in argv[4:]
    found = NAME.match(variant)
    if found is None:
        sys.exit(f"{variant} is not a MAR variant or control")
    sibling = f"{found['base']}_{found['pp']}nan"
    resolved, source = load_hyperparameters(Namespace(dataset_name=sibling, task="imputation"))
    config = {**complete_configuration(resolved, "imputation"), "DECODE_MASK_WEIGHTING": "propensity"}
    mechanism = "mcar" if found["mechanism"] == "marfctl" else "mar"
    slope = (found["slope"] or "1p5") if mechanism == "mar" else "none"
    print("cell", variant, seed, mechanism, "config of", sibling, "from", source, json.dumps(config), flush=True)
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
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "mechanism": mechanism, "mar_slope": slope, "arm": "P"},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
