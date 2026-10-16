"""One cell of the re-tuning of the promoted imputation files (imputation-retune/01, E39).

    uv run --python 3.11 python scripts/experiments/retune_run.py \
        <variant> <seed> <arm T|D|F> <output_dir> <metrics_dir> [--config <json>] [--no-mlflow]

T = the variant's study winner: the sampled knobs in ``--config`` (the study's
``best_hyperparameters.json``) are the override, so every other key takes today's imputation
defaults (ADRs 0013 to 0015), exactly as the study's trials trained and as a promoted file of that
winner would resolve. D = today's defaults, no promoted file (an empty override). F = the
variant's current promoted file, read as a default run reads it; the study reads its F cells from
E34, so this arm only serves pre-launch checks. A trailing ``--no-mlflow`` runs the cell without
MLflow and without the study's tags. Five folds. Run from the checkout root.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training.config import complete_configuration, hyperparameter_file, load_hyperparameters  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENT = "retune-2026-10-15"
ARMS = ("T", "D", "F")


def override(variant: str, arm: str, config: str | None) -> dict[str, object] | None:
    """What the cell passes as ``hyperparams_override``; None reads the promoted file."""
    if arm == "T":
        if config is None:
            sys.exit("arm T needs --config <the study's best_hyperparameters.json>")
        values = json.loads(Path(config).read_text())
        if not isinstance(values, dict):
            sys.exit(f"{config}: expected a JSON object")
        return dict(values)
    if arm == "D":
        return {}
    if not hyperparameter_file(variant, "imputation").exists():
        sys.exit(f"arm F: {variant} has no promoted file")
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("output_dir")
    parser.add_argument("metrics_dir")
    parser.add_argument("--config")
    parser.add_argument("--no-mlflow", action="store_true")
    options = parser.parse_args(argv)
    values = override(options.variant, options.arm, options.config)
    resolved, source = load_hyperparameters(
        Namespace(dataset_name=options.variant, task="imputation", hyperparams_override=values)
    )
    print(
        "cell", options.variant, options.seed, options.arm, "override", json.dumps(values),
        "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")),
        flush=True,
    )
    args = Namespace(
        dataset_name=options.variant,
        label_column="class",
        output_dir=options.output_dir,
        metrics_dir=options.metrics_dir,
        disable_mlflow=options.no_mlflow,
        plot_losses=False,
        save_model=False,
        seed=options.seed,
        cv_folds=5,
        task="imputation",
        hyperparams_override=values,
        mlflow_tags={} if options.no_mlflow else {"experiment": EXPERIMENT, "arm": options.arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
