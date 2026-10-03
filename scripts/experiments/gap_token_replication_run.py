"""One cell of the gap-token replication on fresh seeds (imputation-token-shape/03).

    uv run --python 3.11 python scripts/experiments/gap_token_replication_run.py \
        <variant> <seed> <arm G|M> <output_dir> <metrics_dir>

G = a plain imputation run under today's defaults (ADR 0014: the gaps shown as [MASK] in the
decode stage), naming nothing. M = the same run with ``--decode_gap_token null``, the default
before ADR 0014 (ADR 0013's configuration, E30's arm M). Each variant reads its promoted file or
the defaults exactly as any run does. Five folds. Run from the checkout root.
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

EXPERIMENT = "gap-token-replication-2026-10-03"
ARMS = {"G": None, "M": "null"}


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    print("cell", variant, seed, arm, flush=True)
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
        decode_gap_token=ARMS[arm],
        mlflow_tags={"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
