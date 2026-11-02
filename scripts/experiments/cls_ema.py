"""E112: does an EMA target make classification pre-training help? (classification-step-zero/12, T13).

    uv run --python 3.11 python -B scripts/experiments/cls_ema.py train <variant> <seed> <E|B>
    uv run --python 3.11 python -B scripts/experiments/cls_ema.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/cls_ema.py --tally

E87's protocol (``cls_tenfold.py``: the old sweep's ``hyperparameters.json``, ``cosine_legacy``, ten
folds) with arm E, ``PRETRAIN_OBJECTIVE embedding_ema``. Each pair runs on one machine (ADR 0035):
medusa1 runs E at seeds 42/7/13/101 against E87's B there; gorgona8 runs E and B at seed 202. MLflow
tag ``experiment=cls-ema-v2-2026-11-02``. ``--tally`` (on gorgona8, with medusa1's cells and E87's copied
under ``STATE/medusa`` and ``STATE/e87``) reads E − B per level and E − A (medusa1's seeds) descriptively.
"""

from __future__ import annotations

import json
import socket
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.cls_determinism_run import sweep_run  # noqa: E402
from scripts.experiments.cls_stage2_run import ARM_COST, BASES, LEVELS, ROWS, cell_dir, fold_metrics  # noqa: E402

# The first launch (tag cls-ema-2026-11-02) ran before the NaN fix; its runs stay in the store, unread.
EXPERIMENT = "cls-ema-v2-2026-11-02"
STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-11-02-cls-ema"
VARIANTS = tuple(f"{b}_{lv}nan" for b in BASES for lv in LEVELS)
ARMS = {"E": {"PRETRAIN_OBJECTIVE": "embedding_ema"}, "B": {"EPOCHS_PRE": 0}}
WORK = {"medusa": ((42, 7, 13, 101), ("E",)), "gorgona8": ((202,), ("E", "B"))}
FOLDS = 10
SESOI = 0.01


def train(variant: str, seed: int, arm: str) -> None:
    from train import main as train_main

    run_id, artifacts = sweep_run(variant)
    override = json.loads((artifacts / "parameters" / "hyperparameters.json").read_text()) | ARMS[arm]
    print("cell", variant, seed, arm, "sweep run", run_id, "override", json.dumps(override), flush=True)
    train_main(Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(cell_dir(STATE, variant, seed, arm)), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}_{arm}"),
        disable_mlflow=False, plot_losses=False, save_model=False, seed=seed, cv_folds=FOLDS,
        task="classification", lr_scheduler="cosine_legacy", hyperparams_override=override,
        mlflow_tags={"experiment": EXPERIMENT, "arm": arm},
    ), return_metrics=True)
    frame = fold_metrics(cell_dir(STATE, variant, seed, arm))
    print("CELL_DONE", variant, seed, arm, "folds", 0 if frame is None else len(frame), flush=True)


def todo(which: int, queues: int) -> None:
    """This machine's cells, seed by seed, the largest tables first, dealt greedily by cost."""
    seeds, arms = WORK[socket.gethostname()]
    for seed in seeds:
        work = sorted(((ROWS[v.split("_")[0]] * ARM_COST["A" if a == "E" else a], v, a) for v in VARIANTS for a in arms), key=lambda w: (-w[0], w[1], w[2]))
        loads = [0.0] * queues
        for cost, variant, arm in work:
            target = loads.index(min(loads))
            loads[target] += cost
            if target == which - 1 and fold_metrics(cell_dir(STATE, variant, seed, arm)) is None:
                print(variant, seed, arm)


def _cell(variant: str, seed: int, arm: str):  # type: ignore[no-untyped-def]
    """Where a cell lives on gorgona8 once copied: E87's A and B (medusa1), medusa1's E, or here."""
    if arm in ("A", "B") and seed in WORK["medusa"][0]:
        return fold_metrics(cell_dir(STATE / "e87", variant, seed, arm))
    if seed in WORK["medusa"][0]:
        return fold_metrics(cell_dir(STATE / "medusa", variant, seed, arm))
    return fold_metrics(cell_dir(STATE, variant, seed, arm))


def _contrast(seeds: tuple[int, ...], first: str, second: str) -> dict[str, common.Verdict] | None:
    out = {}
    for level in LEVELS:
        per_seed = []
        for seed in seeds:
            diffs = []
            for base in BASES:
                a, b = _cell(f"{base}_{level}nan", seed, first), _cell(f"{base}_{level}nan", seed, second)
                if a is None or b is None:
                    return None
                diffs.append((a["f1_macro"] - b["f1_macro"]).to_numpy(dtype=float))
            per_seed.append(np.concatenate(diffs))
        out[level] = common.verdict(per_seed, sesoi=SESOI)
    return out


def tally() -> None:
    seeds = (*WORK["medusa"][0], *WORK["gorgona8"][0])
    primary = _contrast(seeds, "E", "B")
    if primary is None:
        print("Primary: incomplete")
        return
    read = {"better": "B higher", "worse": "E higher", "none": "no verdict", "equivalent": "equivalent"}
    for level, v in primary.items():
        print(f"- {level}nan: E − B {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds {', '.join(f'{s:+.4f}' for s in v.seeds)} -> {read[v.call]}")
    descriptive = _contrast(WORK["medusa"][0], "E", "A")
    if descriptive:
        for level, v in descriptive.items():
            print(f"  - {level}nan, descriptive (42/7/13/101): E − A {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] -> {read[v.call]}")
    helps = [lv for lv, v in primary.items() if v.call == "worse"]
    hurts = [lv for lv, v in primary.items() if v.call == "better"]
    reading = "supported" if len(helps) >= 3 and not hurts else "refuted" if len(helps) <= 1 else "mixed"
    print(f"\nPrimary (E112, E − B): E higher at {helps or 'no level'}, B higher at {hurts or 'no level'} of {len(LEVELS)} -> {reading} ({common.RULE}, SESOI {SESOI})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[3] in ARMS:
        train(argv[1], int(argv[2]), argv[3])
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv in (["--tally"], []):
        tally()
    else:
        sys.exit(__doc__)
