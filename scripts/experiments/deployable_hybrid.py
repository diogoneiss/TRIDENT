"""E82: the H + TabPFN hybrid with weights chosen without the complete table (imputation-protocol/31, T39).

    uv run --python 3.11 python -B scripts/experiments/deployable_hybrid.py train <variant> <seed> [--no-mlflow]
    uv run --python 3.11 python -B scripts/experiments/deployable_hybrid.py --todo <n>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/deployable_hybrid.py blend <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/deployable_hybrid.py --tally

``train`` runs one cell with ``--decode_checkpoint loss --decode_calibration none``: the run keeps H
(ADR 0024's deployable readout) and nothing in training reads the complete table, which only scores.
MLflow tag ``experiment=deployable-hybrid-2026-10-27``. After each fold's decode stage, the run's model
(H) is read in place, random states restored, as E78 does. The validation split's *observed* cells are
dealt into 5 groups at random (stream ``[seed, fold, 82]``). Each group in turn is hidden as
``[MASK]``, with the real gaps, and H fills it. ``selection_fold_<k>.csv`` holds every observed
validation cell with its truth (the observed value) and H's fill.

``blend`` refits TabPFN v2 on each training fold, as E74 does. It fills the test split's induced cells
and, group by group, the same hidden validation cells. Then it scores five fills of the test induced
cells: H; TabPFN; the equal blend; the *deployable* hybrid, whose weights come from the validation
cells (E69's grid and rule, no complete table); and the *oracle* hybrid with H as the model side,
whose weights come from the other four folds' induced cells (E69/E74's procedure). It asserts that
the ledger's score equals the run's H, fold by fold.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import blend_weights as weights  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import tabpfn_blend as tb  # noqa: E402

EXPERIMENT = "deployable-hybrid-2026-10-27"
STATE = Path("/scratch2/diogoneiss/trident-2026-10-27-deployable-hybrid")
E74 = Path("/scratch2/diogoneiss/trident-2026-10-24-hybrid-blend/blend/csv")
SEEDS = (1212, 2323, 3434)
VARIANTS = weights.HARD
GROUPS = 5
SESOI = 0.01
QUEUES = 4
ROWS = {"electricity": 45312, "kr-vs-kp": 3196, "biodeg": 1055, "credit-g": 1000, "vehicle": 846}
METHODS = ("model", "tabpfn", "equal", "deployable", "oracle")


def _selection_cells(model: Any, frame: pd.DataFrame, device: torch.device, seed: int, fold_ordinal: int) -> pd.DataFrame:
    from src.embedder import EpochMasker, as_category_strings
    from src.training.imputation_metrics import CATEGORICAL, NUMERICAL

    embedder = model.embedder
    nulls = frame.isna().to_numpy()
    rows, cols = np.nonzero(~nulls)
    group = np.random.default_rng([seed, fold_ordinal, 82]).integers(0, GROUPS, size=len(rows))
    masker = EpochMasker(embedder, frame, device)
    categorical, numerical = list(embedder.categorical_columns), list(embedder.numerical_columns)
    labels = {name: as_category_strings(frame[name]) for name in categorical}
    numbers = {name: frame[name].to_numpy(dtype=float) for name in numerical}
    records = []
    for g in range(GROUPS):
        hide = np.zeros_like(nulls)
        hide[rows[group == g], cols[group == g]] = True
        with torch.no_grad():
            predicted = model.predict(embedder.gaps_as_mask(masker.apply(hide)))
        for r, c in zip(rows[group == g], cols[group == g]):
            name = str(frame.columns[c])
            fill: object
            truth: object
            if name in categorical:
                i = categorical.index(name)
                fill = str(embedder.label_encoders[name].inverse_transform([int(predicted.categorical_ids[i, r])])[0])
                truth = str(labels[name][int(r)])
                kind = CATEGORICAL
            else:
                fill = float(predicted.numerical_values[numerical.index(name), r])
                truth = float(numbers[name][int(r)])
                kind = NUMERICAL
            records.append({"row": int(r), "column": name, "kind": kind, "actual": truth, "model": fill, "group": g})
    return pd.DataFrame(records)


def train(variant: str, seed: int, no_mlflow: bool) -> None:
    import src.training.runner as runner
    from src.training.config import complete_configuration, load_hyperparameters
    from train import main as train_main

    out = STATE / "selection" / f"{variant}_s{seed}"
    original = getattr(runner, "train_and_evaluate_decoder")

    def read_after(**kwargs: Any) -> Any:
        outcome = original(**kwargs)
        model, dataset, fold = outcome.model, kwargs["dataset"], kwargs["fold"]
        states = (np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        was_training = model.training
        model.eval()
        try:
            features = dataset.frame.drop(columns=[dataset.label_column])
            validation = features.iloc[fold.validation_indices].reset_index(drop=True)
            out.mkdir(parents=True, exist_ok=True)
            _selection_cells(model, validation, kwargs["device"], kwargs["seed"], kwargs["fold_ordinal"]).to_csv(
                out / f"selection_fold_{kwargs['fold_ordinal']}.csv", index=False)
        finally:
            np.random.set_state(states[0])
            torch.set_rng_state(states[1])
            if states[2] is not None:
                torch.cuda.set_rng_state_all(states[2])
            model.train(was_training)
        return outcome

    setattr(runner, "train_and_evaluate_decoder", read_after)
    flags = {"decode_checkpoint": "loss", "decode_calibration": "none"}
    resolved, source = load_hyperparameters(Namespace(dataset_name=variant, task="imputation", **flags))
    print("cell", variant, seed, "source", source, "resolved", json.dumps(complete_configuration(resolved, "imputation")), flush=True)
    args = Namespace(
        dataset_name=variant, label_column="class",
        output_dir=str(STATE / "results" / f"{variant}_s{seed}"), metrics_dir=str(STATE / "metrics" / f"{variant}_s{seed}"),
        disable_mlflow=no_mlflow, plot_losses=False, save_model=False, seed=seed, cv_folds=5, task="imputation",
        mlflow_tags={} if no_mlflow else {"experiment": EXPERIMENT}, **flags,
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


def _typed(cells: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    from src.training.imputation_metrics import NUMERICAL

    numeric = (cells["kind"] == NUMERICAL).to_numpy()
    for c in columns:
        cells[c] = [float(v) if n else str(v) for v, n in zip(cells[c], numeric)]
    return cells


def blend(variant: str, seed: int) -> None:
    from src.training.data import build_folds, prepare_dataset
    from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines
    from src.training.tabpfn_baseline import fit_tabpfn_imputers
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    started = time.time()
    found = common.parents(EXPERIMENT)[(variant, seed)]
    if len(found) != 1:
        raise RuntimeError(f"{variant} s{seed}: {len(found)} runs")
    run = found[0]
    logged = common.model(common.fold_table(run), run, "H")
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, tb.FOLDS, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        validation = features.iloc[np.asarray(fold.validation_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = dict(mean_mode_baselines(train, numerical, categorical))
        cells = _typed(tb._ledger(variant, seed, ordinal, STATE), ("actual", "imputed"))
        assert abs(tb._score(cells, naive) - float(logged[ordinal - 1])) < 1e-9, f"{variant} s{seed} fold {ordinal}: ledger is not the run's H"
        imputer = fit_tabpfn_imputers(train, numerical, categorical)[0]
        positions, columns = cells["row"].to_numpy(dtype=int), test.columns.get_indexer(pd.Index(cells["column"]))
        hidden = np.zeros(test.shape, dtype=bool)
        hidden[positions, columns] = True
        filled = imputer.impute(test, hidden).to_numpy(dtype=object)[positions, columns]
        frame = cells[["row", "column", "kind", "actual"]].assign(
            model=cells["imputed"], tabpfn=[float(g) if k == NUMERICAL else str(g) for g, k in zip(filled, cells["kind"])], fold=ordinal)
        selection = pd.read_csv(STATE / "selection" / f"{variant}_s{seed}" / f"selection_fold_{ordinal}.csv", dtype={"actual": object, "model": object})
        selection = _typed(selection, ("actual", "model"))
        tabpfn: list[object] = [None] * len(selection)
        for g in range(GROUPS):
            part = np.flatnonzero(selection["group"].to_numpy() == g)
            r = selection["row"].to_numpy(dtype=int)[part]
            c = validation.columns.get_indexer(pd.Index(selection["column"].iloc[part]))
            hide = np.zeros(validation.shape, dtype=bool)
            hide[r, c] = True
            guesses = imputer.impute(validation, hide).to_numpy(dtype=object)[r, c]
            for i, guess, kind in zip(part, guesses, selection["kind"].iloc[part]):
                tabpfn[i] = float(guess) if kind == NUMERICAL else str(guess)
        selection["tabpfn"] = pd.Series(tabpfn, index=selection.index, dtype=object)
        folds.append((frame, naive, selection))
    pooled = pd.concat([f for f, _, _ in folds], ignore_index=True)
    rows: list[dict[str, object]] = []
    chosen_rows: list[dict[str, object]] = []
    for frame, naive, selection in folds:
        k = int(frame["fold"].iloc[0])
        deployable = weights.choose(selection)
        oracle = weights.choose(pooled[pooled["fold"] != k])
        fills = {"model": list(frame["model"]), "tabpfn": list(frame["tabpfn"]), "equal": weights.apply(frame, {}, weights.EQUAL),
                 "deployable": weights.apply(frame, deployable, weights.EQUAL), "oracle": weights.apply(frame, oracle, weights.EQUAL)}
        for method, values in fills.items():
            rows.append({"variant": variant, "seed": seed, "fold": k, "method": method,
                         "impute_score": tb._score(frame.assign(imputed=np.array(values, dtype=object)), naive)})
        counts = selection.groupby("column").size()
        for column in sorted(set(frame["column"])):
            chosen_rows.append({"variant": variant, "seed": seed, "fold": k, "column": column,
                                "deployable": deployable.get(column), "oracle": oracle.get(column), "selection_cells": int(counts.get(column, 0))})
    for sub in ("csv", "weights"):
        (STATE / "blend" / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATE / "blend" / "csv" / f"{variant}_s{seed}.csv", index=False)
    pd.DataFrame(chosen_rows).to_csv(STATE / "blend" / "weights" / f"{variant}_s{seed}.csv", index=False)
    print(f"DONE {variant} s{seed}: ledger equals the run's H on 5 folds, {time.time() - started:.0f} s", flush=True)


def _values(table: pd.DataFrame, variant: str, method: str) -> list[np.ndarray]:
    return weights._values(table, variant, list(SEEDS), method)


def tally() -> None:
    table = pd.concat([pd.read_csv(p) for p in sorted((STATE / "blend" / "csv").glob("*.csv"))], ignore_index=True)
    e74 = pd.concat([pd.read_csv(p) for p in sorted(E74.glob("*.csv"))], ignore_index=True)
    chosen = pd.concat([pd.read_csv(p) for p in sorted((STATE / "blend" / "weights").glob("*.csv"))], ignore_index=True)
    calls: dict[tuple[str, str], str] = {}
    counts: dict[str, list[str]] = {}
    for variant in VARIANTS:
        if sorted(table.loc[table.variant == variant, "seed"].unique()) != sorted(SEEDS):
            print(f"- `{variant}`: incomplete")
            continue
        values = {m: _values(table, variant, m) for m in METHODS}
        values["KC"] = weights._values(e74, variant, list(SEEDS), "model")
        values["E74 hybrid"] = weights._values(e74, variant, list(SEEDS), "weighted")
        parts = []
        for ours, other in (("deployable", "model"), ("deployable", "tabpfn"), ("deployable", "oracle"), ("oracle", "model"),
                            ("deployable", "KC"), ("deployable", "E74 hybrid")):
            v = common.verdict([a - b for a, b in zip(values[ours], values[other])], sesoi=SESOI * float(np.mean([o.mean() for o in values[other]])))
            calls[variant, f"{ours} − {other}"] = v.call
            counts.setdefault(f"{ours} − {other}", []).append(v.call)
            parts.append(f"{ours} − {other} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        w = chosen[chosen.variant == variant]
        agree = float((w["deployable"] == w["oracle"]).mean())
        print(f"- `{variant}` (H {np.concatenate(values['model']).mean():.4f}, TabPFN {np.concatenate(values['tabpfn']).mean():.4f}; "
              f"selection cells a column a fold: median {w['selection_cells'].median():.0f}, min {w['selection_cells'].min()}; "
              f"deployable weight = oracle weight on {agree:.0%}): " + "; ".join(parts))
    print()
    for key, tallied in counts.items():
        print(f"- {key}: {common.tally(tallied)}")
    both = [v for v in VARIANTS if calls.get((v, "deployable − model")) == "better" and calls.get((v, "deployable − tabpfn")) == "better"]
    print(f"- the deployable hybrid beats both H and TabPFN on {len(both)}: {both or 'none'}")
    if len(counts.get("deployable − model", [])) < len(VARIANTS):
        print(f"\nPrimary: incomplete ({len(counts.get('deployable − model', []))} of {len(VARIANTS)})")
        return
    over = [v for v in VARIANTS if calls[v, "deployable − model"] == "better"]
    under = [v for v in VARIANTS if calls[v, "deployable − model"] == "worse"]
    behind = [v for v in VARIANTS if calls[v, "deployable − tabpfn"] == "worse"]
    if len(over) >= 8 and not under and len(behind) <= 2:
        reading = "supported"
    elif len(over) <= 5 or len(under) >= 2 or len(behind) >= 5:
        reading = "refuted"
    else:
        reading = "mixed"
    print(f"\nPrimary: the deployable hybrid is better than H on {len(over)} of {len(VARIANTS)}, worse on {under or 'none'}, "
          f"and worse than TabPFN on {len(behind)} ({behind or 'none'}) -> {reading} ({common.RULE})")


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
