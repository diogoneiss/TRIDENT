"""E74: the per-column K + TabPFN hybrid as an output, on fresh seeds (imputation-protocol/27, T37).

    uv run --python 3.11 python -B scripts/experiments/hybrid_blend.py train <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/hybrid_blend.py --todo <n>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_blend.py blend <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_blend.py --tally

``train`` runs one cell of today's default imputation configuration, read as a default run of the
checkout reads it, five folds, MLflow tag ``experiment=hybrid-blend-2026-10-24``. At these rates
(40nan and above) ADR 0027 calibrates the induced output, so the headline, and the fold ledger's
fills, are KC. ``blend`` is E69's cross-fitted per-column blend (``blend_weights.collect`` and
``score_pair``) with the model side KC, asserted fold by fold. ``--tally`` applies the rule fixed in
the ticket. Run from the checkout root.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import blend_weights as weights  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

EXPERIMENT = "hybrid-blend-2026-10-24"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-24-hybrid-blend")
SEEDS = (1212, 2323, 3434)
VARIANTS = weights.HARD
SESOI = 0.01
QUEUES = 4
ROWS = {"electricity": 45312, "kr-vs-kp": 3196, "biodeg": 1055, "credit-g": 1000, "vehicle": 846}


def train(variant: str, seed: int, no_mlflow: bool) -> None:
    from src.training.config import complete_configuration, load_hyperparameters
    from train import main as train_main

    resolved, source = load_hyperparameters(Namespace(dataset_name=variant, task="imputation"))
    print("cell", variant, seed, "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"),
        disable_mlflow=no_mlflow, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT},
    )
    metrics = train_main(args, return_metrics=True) or {}
    print("CELL_DONE", json.dumps({k: v for k, v in metrics.items() if k.startswith("impute/induced/") and k.endswith("/impute_score") and "/baseline/" not in k}), flush=True)


def todo(which: int) -> None:
    done = common.parents(EXPERIMENT)
    work = sorted(((ROWS[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS), key=lambda w: (-w[0], w[1], w[2]))
    loads = [0.0] * QUEUES
    for cost, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and (variant, seed) not in done:
            print(variant, seed)


def blend(variant: str, seed: int) -> None:
    started = time.time()
    weights.score_pair(weights.collect(STATE, EXPERIMENT, "KC", variant, seed), variant, seed, STATE / "blend")
    print(f"DONE {variant} s{seed}: model equals the run's KC on 5 folds, {time.time() - started:.0f} s", flush=True)


def tally() -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted((STATE / "blend" / "csv").glob("*.csv"))], ignore_index=True)
    calls: dict[tuple[str, str], str] = {}
    counts: dict[str, list[str]] = {}
    for variant in VARIANTS:
        rows = table[table.variant == variant]
        if sorted(rows["seed"].unique()) != sorted(SEEDS):
            print(f"- `{variant}`: incomplete")
            continue
        values = {m: weights._values(table, variant, list(SEEDS), m) for m in ("model", "tabpfn", "equal", "weighted")}
        parts = []
        for ours, other in (("weighted", "model"), ("weighted", "tabpfn"), ("weighted", "equal"), ("tabpfn", "model")):
            diffs = [a - b for a, b in zip(values[ours], values[other])]
            v = common.verdict(diffs, sesoi=SESOI * float(np.mean([o.mean() for o in values[other]])))
            calls[variant, f"{ours} − {other}"] = v.call
            counts.setdefault(f"{ours} − {other}", []).append(v.call)
            parts.append(f"{ours} − {other} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print(f"- `{variant}` (numeric share {rows['numeric_share'].mean():.2f}; KC {np.concatenate(values['model']).mean():.4f}): " + "; ".join(parts))
    print()
    for key, tallied in counts.items():
        print(f"- {key}: {common.tally(tallied)}")
    both = [v for v in VARIANTS if calls.get((v, "weighted − model")) == "better" and calls.get((v, "weighted − tabpfn")) == "better"]
    print(f"- the hybrid beats both KC and TabPFN on {len(both)}: {both or 'none'}")
    if len(counts.get("weighted − model", [])) < len(VARIANTS):
        print(f"\nPrimary: incomplete ({len(counts.get('weighted − model', []))} of {len(VARIANTS)})")
        return
    over_kc = [v for v in VARIANTS if calls[v, "weighted − model"] == "better"]
    behind = [v for v in VARIANTS if calls[v, "weighted − tabpfn"] == "worse"]
    if len(over_kc) >= 8 and len(behind) <= 1:
        reading = "supported"
    elif len(over_kc) <= 5 or len(behind) >= 3:
        reading = "refuted"
    else:
        reading = "mixed"
    print(f"\nPrimary: the hybrid is better than KC on {len(over_kc)} of {len(VARIANTS)} and worse than TabPFN on {len(behind)} "
          f"({behind or 'none'}) -> {reading} ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) in (3, 4):
        train(argv[1], int(argv[2]), argv[3:] == ["--no-mlflow"])
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv[:1] == ["blend"] and len(argv) == 3:
        blend(argv[1], int(argv[2]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
