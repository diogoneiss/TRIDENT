"""E156 (docs/tickets/c15-mnar-prediction/12-mechanism-split.md): A1 against A2 on MIMIC-IV complete tables with known
mechanisms injected. Its cells, E123's G2 on its files, and the study's reader.

    uv run --python 3.11 python -B scripts/experiments/mechanism_split.py train <arm> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/mechanism_split.py --todo <n> <queues>
    uv run --python 3.11 python -B scripts/experiments/mechanism_split.py --g2 [jobs]
    uv run --python 3.11 python -B scripts/experiments/mechanism_split.py --read

A cell is C15's (A1 ``[NULL]``, A2 missForest + indicator), as ``h2_natural.train``, on a file under ``DATA`` written by
``datasets/make_mimic_complete.py`` and ``datasets/mnar_family.py --root``. ``--read`` refuses until all 36 cells and
the 90 G2 cells are in.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

STATE = Path("/scratch2/diogoneiss/trident-2026-11-06-mechanism-split")
DATA = STATE / "data"
os.environ["TRIDENT_DATASETS_ROOT"] = str(DATA)  # before any dataset is resolved

from scripts.experiments import c15_report as report  # noqa: E402
from scripts.experiments import c15_run as run  # noqa: E402

TABLES = ("mimic-sofa-semi", "mimic-cardio-semi")
MECHANISMS = ("outcome3p0", "probit0p5", "mcar")
ARMS = ("A1", "A2")
SEEDS = (2525, 4747, 6969)
G2_SEEDS = (42, 7, 13)  # E123's


def variant(table: str, mechanism: str) -> str:
    return f"{table}_{mechanism}_s5858_60nan"


FILES = tuple(variant(t, m) for t in TABLES for m in MECHANISMS)


def train(arm: str, name: str, seed: int) -> None:
    from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
    from train import main as train_main

    base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", name])))
    override = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": arm}
    out = run.cell_dir(arm, name, seed, STATE)
    args = Namespace(dataset_name=name, label_column="class", output_dir=str(out / "results"),
                     metrics_dir=str(out / "metrics"), disable_mlflow=True, plot_losses=False, save_model=False,
                     seed=seed, cv_folds=5, task="classification", mlflow_tags={}, hyperparams_override=override)
    train_main(args, return_metrics=True)
    print("CELL_DONE", arm, name, seed, flush=True)


def todo(which: int, queues: int) -> None:
    """(file, seed) pairs dealt over the queues, both arms of a pair in one queue."""
    for k, (name, seed) in enumerate((f, s) for s in SEEDS for f in FILES):
        if k % queues == which - 1:
            for arm in ARMS:
                if not run.done(arm, name, seed, STATE):
                    print(arm, name, seed)


def g2(jobs: int) -> None:
    """E123's out-of-fold G2 on the six files, cells written under STATE/g2_oof."""
    from joblib import Parallel, delayed  # type: ignore[import-untyped]

    from scripts.experiments import c15_stage0 as stage0
    stage0.DATA = DATA
    os.environ["PYTHONPATH"] = str(ROOT)  # the workers import scripts.experiments
    out = STATE / "g2_oof"
    out.mkdir(parents=True, exist_ok=True)
    cells = [(v, s, f) for v in FILES for s in G2_SEEDS for f in range(1, 6) if not (out / f"{v}_s{s}_f{f}.json").exists()]
    print(f"{len(cells)} of {len(FILES) * len(G2_SEEDS) * 5} G2 cells to run", flush=True)

    def one(v: str, s: int, f: int) -> None:
        stage0.DATA = DATA  # in the worker too
        (out / f"{v}_s{s}_f{f}.json").write_text(json.dumps(stage0.g2_oof_cell(v, s, f)))

    Parallel(n_jobs=jobs, verbose=5)(delayed(one)(*c) for c in cells)


def g2_calls() -> dict[str, tuple[bool, str]]:
    """E123's rule per file: informative if the mean drop >= 0.02 with its 95% t interval above 0."""
    from scripts.experiments import c15_stage0 as stage0
    out: dict[str, tuple[bool, str]] = {}
    for v in FILES:
        drops = np.array([json.loads((STATE / "g2_oof" / f"{v}_s{s}_f{f}.json").read_text())["drop"]
                          for s in G2_SEEDS for f in range(1, 6)])
        m, lo, hi = stage0.interval(drops)
        out[v] = (m >= stage0.THRESHOLD and lo > 0, f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return out


def differences(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A1 − A2 per (seed, fold), seeds in order: macro-F1 and test log loss."""
    f1, ll = [], []
    for seed in SEEDS:
        (f1_1, ll_1), (f1_2, ll_2) = (report.scores([torch.load(run.cell_dir(arm, name, seed, STATE) / "metrics"
                                                                 / f"c15_readout_fold_{k}.pt", weights_only=False)
                                                      for k in range(1, 6)]) for arm in ARMS)
        f1.append(f1_1 - f1_2)
        ll.append(ll_1 - ll_2)
    return np.concatenate(f1), np.concatenate(ll)


def call(values: np.ndarray) -> tuple[str, str]:
    """Gains / loses / holds within ±0.01 / no verdict, on the Nadeau–Bengio corrected 95% interval."""
    m, lo, hi = report.interval(values)
    _, plo, phi = report.interval(values, False)
    verdict = "gains" if lo > 0 else "loses" if hi < 0 else "holds" if lo > -0.01 and hi < 0.01 else "no verdict"
    seeds = ", ".join(f"{s.mean():+.4f}" for s in values.reshape(len(SEEDS), 5))
    return verdict, f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (plain [{plo:+.4f}, {phi:+.4f}]; seeds {seeds})"


def read() -> None:
    missing = [(a, f, s) for f in FILES for s in SEEDS for a in ARMS if not run.done(a, f, s, STATE)]
    g2_missing = [(v, s, f) for v in FILES for s in G2_SEEDS for f in range(1, 6)
                  if not (STATE / "g2_oof" / f"{v}_s{s}_f{f}.json").exists()]
    if missing or g2_missing:
        sys.exit(f"E156 incomplete: {len(missing)} cells and {len(g2_missing)} G2 cells missing; read only with all in")
    gate = g2_calls()
    print("## E156 G2 (E123's rule, out of fold, seeds 42/7/13): relative log-loss drop from the indicators")
    for v, (ok, text) in gate.items():
        print(f"- {v}: {text} -> {'informative' if ok else 'not informative'}")
    d = {f: differences(f) for f in FILES}

    print("\n## Per file, A1 − A2 (descriptive): macro-F1, absolute test log loss")
    for f, (f1, ll) in d.items():
        print(f"- {f}: F1 {f1.mean():+.4f}, LL {ll.mean():+.4f}")

    def contrast(mechanism: str, against: str) -> None:
        tables = [t for t in TABLES if gate[variant(t, mechanism)][0]] if mechanism == "outcome3p0" else list(TABLES)
        per = {t: d[variant(t, mechanism)][0] - d[variant(t, against)][0] for t in TABLES}
        for t in TABLES:
            print(f"- {t}: {call(per[t])[1]}{'' if t in tables else ' (G2 fails: left out of the pool)'}")
        if not tables:
            print("- pooled: not read, G2 fails on every table")
            return
        verdict, text = call(np.mean([per[t] for t in tables], axis=0))
        print(f"- pooled over {len(tables)} table(s): {text} -> {verdict}")

    print("\n## Primary: (A1 − A2) on outcome3p0 minus on its MCAR sibling, macro-F1, NB-corrected 95% over 15 cells; "
          "predicted gains")
    contrast("outcome3p0", "mcar")
    print("\n## Secondary 1: the same for probit0p5 minus MCAR; predicted not gains")
    contrast("probit0p5", "mcar")
    print("\n## Secondary 2: the split, (A1 − A2) on outcome3p0 minus on probit0p5; predicted gains")
    contrast("outcome3p0", "probit0p5")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv[:1] == ["--g2"] and len(argv) <= 2:
        g2(int(argv[1]) if len(argv) == 2 else 8)
    elif argv == ["--read"]:
        read()
    else:
        sys.exit(__doc__)
