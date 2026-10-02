"""One cell of the gaps-as-mask training study (imputation-token-shape/02, task T02 step 2).

    uv run --python 3.11 python scripts/experiments/gap_token_run.py \
        <variant> <seed> <output_dir> <metrics_dir>

Arm G: a plain imputation run under today's defaults (ADR 0013, E30's arm M), reading the
variant's promoted file or the defaults exactly as any run does, with ``DECODE_GAP_TOKEN``
``mask``: the decode stage shows every real gap as [MASK], the row shape the induced headline
scores in, instead of [NULL]. Nothing else changes, so each cell pairs with E30's arm-M cell of
the same variant and seed (and with E31's, which re-read the same models). Five folds. Run from
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

from train import main as train_main  # noqa: E402

EXPERIMENT = "gap-token-2026-10-02"


def main(argv: list[str]) -> int:
    variant, seed, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3]
    print("cell", variant, seed, "G", flush=True)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        disable_mlflow=False,
        plot_losses=False,
        save_model=False,
        seed=seed,
        cv_folds=5,
        task="imputation",
        decode_gap_token="mask",
        mlflow_tags={"experiment": EXPERIMENT, "arm": "G"},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
