"""E181: TRIDENT against its baselines when ``impute_score`` is measured against the class-conditional mean/mode
fill and the decode checkpoint is chosen by that score (docs/tickets/imputation-protocol/70-class-conditional-score.md).

    uv run --python 3.11 python -B scripts/experiments/classcond_run.py train <Kc|K> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/classcond_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/classcond_run.py --tally

``Kc`` is the imputation task's default run with ``--decode_metric impute_score_classcond``: it keeps the epoch
that score names and logs it for the model and every in-run baseline. ``K`` is the default run, trained only on the
tables with both kinds of column (``MIXED``): on a table of one kind the two scores differ by a constant factor
within a fold, so they keep the same epoch. ``--todo`` deals the cells for ``c15_queue.sh`` (``QUEUE_RUNNER``,
``QUEUE_STATE``), shortest first. ``--tally`` refuses a partial read. Run from the checkout root, on medusa1.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.hybrid_fresh_seeds import MCAR, MINUTES  # noqa: E402
from src.training.imputation_metrics import CLASS_SCORE, class_conditional_score  # noqa: E402

EXPERIMENT = "classcond-score-2026-11-17"
STATE = Path("/scratch/diogoneiss/trident-2026-11-17-classcond")
VARIANTS: tuple[str, ...] = MCAR
SEEDS: tuple[int, ...] = (101, 202, 303)
MIXED = ("credit-g", "electricity")
ARMS: dict[str, dict[str, object]] = {"Kc": {"decode_metric": CLASS_SCORE}, "K": {}}
SESOI = 0.01
P = "impute/induced"


def cells() -> list[tuple[str, str, int]]:
    return [(arm, v, s) for v in VARIANTS for s in SEEDS for arm in ARMS if arm == "Kc" or common.table_of(v) in MIXED]


def cell_dir(arm: str, variant: str, seed: int) -> Path:
    return STATE / "results" / f"{variant}_s{seed}_{arm}"


def folds(arm: str, variant: str, seed: int) -> pd.DataFrame | None:
    """The fold metrics of the cell's run that reached the cross-validation summary; a run that died leaves none."""
    found = sorted((cell_dir(arm, variant, seed) / variant).glob("*/metrics/cv_summary.json"))
    return pd.read_csv(found[-1].parent / "raw_fold_metrics.csv").sort_values("fold").reset_index(drop=True) if found else None


def train(arm: str, variant: str, seed: int) -> None:
    import torch

    from train import main as train_main

    if not torch.cuda.is_available():  # a lost GPU would otherwise train the cell on the CPU, silently
        sys.exit("no CUDA device: the cell is not run")
    out = cell_dir(arm, variant, seed)
    args = Namespace(dataset_name=variant, label_column="class", output_dir=str(out), metrics_dir=str(out / "metrics"),
                     disable_mlflow=False, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
                     mlflow_tags={"experiment": EXPERIMENT, "arm": arm}, **ARMS[arm])
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, variant, seed, flush=True)


def todo(which: int, queues: int) -> None:
    loads = [0.0] * queues
    for cost, variant, seed, arm in sorted(((MINUTES[common.table_of(v)], v, s, a) for a, v, s in cells()),
                                           key=lambda w: (w[0], w[1], w[2], w[3])):
        target = loads.index(min(loads))
        loads[target] += cost
        if target == which - 1 and folds(arm, variant, seed) is None:
            print(arm, variant, seed)


def on_class_scale(frame: pd.DataFrame, naive: pd.DataFrame) -> np.ndarray:
    """A run's induced score per fold on the class-conditional scale, from its own errors and the fill's errors
    ``naive`` logged for the same cells: how ``K``, which does not log the score, is read on it."""
    scores = []
    for (_, own), (_, fill) in zip(frame.iterrows(), naive.iterrows()):
        n_num, n_cat = int(own[f"{P}/n_num_cells"]), int(own[f"{P}/n_cat_cells"])
        errors = {name: float(own[f"{P}/{name}"]) for name in ("rmse_num_z", "acc_cat") if f"{P}/{name}" in own}
        scores.append(class_conditional_score(errors, n_num, n_cat, float(fill.get(f"{P}/classcond_naive_rmse_num_z", 0.0)),
                                              float(fill.get(f"{P}/classcond_naive_error_cat", 0.0))))
    return np.asarray(scores)


def _best(frames: list[pd.DataFrame], names: tuple[str, ...], metric: str) -> tuple[str, list[np.ndarray]] | None:
    """Among ``names``, the baseline with the lowest mean ``metric`` over the variant's folds, and its fold scores."""
    have = [n for n in names if all(f"{P}/baseline/{n}/{metric}" in f.columns for f in frames)]
    if not have:
        return None
    name = min(have, key=lambda n: float(np.mean([f[f"{P}/baseline/{n}/{metric}"].mean() for f in frames])))
    return name, [f[f"{P}/baseline/{name}/{metric}"].to_numpy(dtype=float) for f in frames]


def tally() -> None:
    missing = [c for c in cells() if folds(*c) is None]
    if missing:
        print(f"incomplete: {len(missing)} of {len(cells())} cells missing, e.g. {missing[:5]}")
        return
    weak = tuple(n for n in common.WEAK if n != "mean_mode")
    calls: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        frames = [f for s in SEEDS if (f := folds("Kc", variant, s)) is not None]
        own = {m: [f[f"{P}/{m}"].to_numpy(dtype=float) for f in frames] for m in (CLASS_SCORE, "impute_score")}
        # The rescaling K is read by reproduces what Kc logged for itself.
        assert all(np.allclose(on_class_scale(f, f), o, atol=1e-9) for f, o in zip(frames, own[CLASS_SCORE]))
        whole = [f[f"{P}/baseline/mean_mode/{CLASS_SCORE}"].to_numpy(dtype=float) for f in frames]
        print(f"- `{variant}`: Kc {np.concatenate(own[CLASS_SCORE]).mean():.4f} on the class scale "
              f"({np.concatenate(own['impute_score']).mean():.4f} on impute_score); the whole-fold mean/mode fill {np.concatenate(whole).mean():.4f}")
        call = calls.setdefault(variant, {})
        v = common.verdict([o - 1.0 for o in own[CLASS_SCORE]], sesoi=SESOI)
        call["fill"] = v.call
        print(f"  - Kc − class fill {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        for label, names in (("bar", weak), ("strong", common.STRONG)):
            for metric in (CLASS_SCORE, "impute_score"):
                best = _best(frames, names, metric)
                if best is None:
                    continue
                name, scores = best
                v = common.verdict([o - b for o, b in zip(own[metric], scores)], sesoi=SESOI * float(np.mean([b.mean() for b in scores])))
                call[f"{label}/{metric}"] = v.call
                print(f"  - Kc − {label} ({name}) on {metric} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        if common.table_of(variant) in MIXED:
            plain = [f for s in SEEDS if (f := folds("K", variant, s)) is not None]
            for metric, theirs in ((CLASS_SCORE, [on_class_scale(k, f) for k, f in zip(plain, frames)]),
                                   ("impute_score", [k[f"{P}/impute_score"].to_numpy(dtype=float) for k in plain])):
                v = common.verdict([o - k for o, k in zip(own[metric], theirs)], sesoi=SESOI * float(np.mean([k.mean() for k in theirs])))
                call[f"K/{metric}"] = v.call
                print(f"  - Kc − K on {metric} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")

    def count(key: str, wanted: str) -> list[str]:
        return [v for v in VARIANTS if calls[v].get(key) == wanted]

    better, worse = count("fill", "better"), count("fill", "worse")
    reading = "supported" if len(better) >= 18 and len(worse) <= 2 else "refuted" if len(worse) >= 6 else "mixed"
    print(f"\nPrimary, Kc against the class-conditional fill: better on {len(better)} of {len(VARIANTS)}, worse on {len(worse)} {worse} -> {reading}")
    for label in ("bar", "strong"):
        moved = [v for v in VARIANTS if calls[v].get(f"{label}/{CLASS_SCORE}") != calls[v].get(f"{label}/impute_score")]
        print(f"Secondary, Kc against the {label} on the class scale: {common.tally(calls[v][f'{label}/{CLASS_SCORE}'] for v in VARIANTS if f'{label}/{CLASS_SCORE}' in calls[v])}; "
              f"calls that differ from impute_score's: {moved}")
    for metric in (CLASS_SCORE, "impute_score"):
        print(f"Secondary, Kc − K on {metric}, mixed tables: {common.tally(calls[v][f'K/{metric}'] for v in VARIANTS if f'K/{metric}' in calls[v])}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--tally"]:
        tally()
    else:
        sys.exit(__doc__)
