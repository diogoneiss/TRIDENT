"""E98: step zero Stage 4, the tuned GBDT arms (docs/tickets/classification-step-zero/07-step-zero.md).

    uv run --python 3.11 python -B scripts/experiments/stage4_gbdt.py [--state <dir>] run [<variant> ...]
    uv run --python 3.11 python -B scripts/experiments/stage4_gbdt.py [--state <dir>] --tally

``run`` tunes three arms per variant on seed 42's ten folds (``build_folds(frame, 'class', 10, 42)``)
and evaluates the tuned parameters at seeds 42, 7 and 13 on each seed's ten folds:

- ``native/hgb``, ``native/lgbm``: E97's native-NaN arms (``cls_stage1_bar``: training-split
  vocabulary codes, NaN for a gap or an unseen category, categories declared), the bar candidates;
- ``missforest+ind/hgb``: E97's missForest fill plus one indicator per column with a training gap,
  then HGB; descriptive. missForest is fit once per fold on the training split, before any trial.

Each arm is an Optuna TPE study (sampler seed 42, 20 trials, maximise) whose objective is the mean
over the ten folds of the validation split's macro-F1. Every fit stops early on the fold's validation
split, its loss weighted by the training split's balanced class weights as in Stage 1: patience 20,
at most 500 rounds; ``class_weight='balanced'``, ``random_state`` 42. The bar of a variant is the
native arm with the higher best validation score, chosen before any test number is read.

Per variant, ``<state>/json/<variant>.json`` holds each arm's best parameters, best score and every
trial (written after tuning; a rerun reuses it and goes straight to the evaluation), and
``<state>/csv/<variant>.csv`` one row per seed, fold and arm (written last; a variant with it is
skipped). Both are written through a temporary name. Nothing touches the store.

``--tally`` prints, per level, each arm's mean test macro-F1 (and the bar's), and the arm each
variant picked as its bar. The H1c comparison against TRIDENT is another script's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

# First, before numpy, scikit-learn and LightGBM load: it sets OMP_NUM_THREADS.
from scripts.experiments.cls_stage1_bar import BASES, LEVELS, THREADS, _balanced, _codes, _impute  # noqa: E402
from scripts.experiments.cls_stage2_run import VARIANTS  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

assert os.environ["OMP_NUM_THREADS"] == str(THREADS)

STATE = Path("/scratch2/diogoneiss/trident-2026-10-30-step-zero/gbdt")
TUNING_SEED = 42
SEEDS = (42, 7, 13)
FOLDS = 10
TRIALS = 20
PATIENCE = 20
ROUNDS = 500
RANDOM_STATE = 42
ARMS = ("native/hgb", "native/lgbm", "missforest+ind/hgb")
BAR_ARMS = ARMS[:2]

Matrix = NDArray[np.float64]
Labels = NDArray[np.int64]


@dataclass
class Fold:
    native: list[Matrix]  # train, validation, test
    filled: list[Matrix]  # missForest fill + indicators
    y: list[Labels]


def _json(state: Path, variant: str) -> Path:
    return state / "json" / f"{variant}.json"


def _csv(state: Path, variant: str) -> Path:
    return state / "csv" / f"{variant}.csv"


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(text)
    partial.rename(path)


def _space(arm: str, trial: Any) -> dict[str, Any]:
    if arm == "native/lgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0)}
    return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0)}


def _fit(arm: str, params: dict[str, Any], x: list[Matrix], y: list[Labels], declared: list[int]) -> tuple[Any, int]:
    """Fit on the training split, stopping on the validation split as Stage 1's ``_learn`` does;
    the model and its kept rounds."""
    import lightgbm
    from sklearn.ensemble import HistGradientBoostingClassifier

    weights = _balanced(y[0], y[1])
    if arm == "native/lgbm":
        booster = lightgbm.LGBMClassifier(n_estimators=ROUNDS, subsample_freq=1, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=THREADS, deterministic=True,
                                          force_row_wise=True, verbose=-1, **params)
        booster.fit(x[0], y[0], categorical_feature=declared or "auto", eval_set=[(x[1], y[1])],
                    eval_sample_weight=[weights], callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)])
        return booster, int(booster.best_iteration_)
    mask = np.zeros(x[0].shape[1], dtype=bool)
    mask[declared] = True
    hgb = HistGradientBoostingClassifier(
        max_iter=ROUNDS, categorical_features=mask if mask.any() else None, early_stopping=True,
        n_iter_no_change=PATIENCE, class_weight="balanced", random_state=RANDOM_STATE, **params)
    hgb.fit(x[0], y[0], X_val=x[1], y_val=y[1], sample_weight_val=weights)
    return hgb, int(hgb.n_iter_)


def _folds(dataset: Any, seed: int) -> list[Fold]:
    """The ten folds of ``seed``, coded and filled once, as Stage 1 codes and fills them."""
    from src.embedder import as_category_strings
    from src.training.data import build_folds
    from src.training.imputation_baselines import _NOT_A_CLASS

    label = dataset.label_column
    features = dataset.frame.drop(columns=[label])
    categorical = list(dataset.categorical_columns)
    numerical = list(dataset.numerical_columns)
    labels = dataset.frame[label].to_numpy().astype(np.int64)
    folds = []
    for ordinal, fold in enumerate(build_folds(dataset.frame, label, FOLDS, seed), start=1):
        parts = [np.asarray(fold.train_indices), np.asarray(fold.validation_indices), np.asarray(fold.test_indices)]
        splits = [features.iloc[p].reset_index(drop=True) for p in parts]
        vocabulary = {c: sorted(set(as_category_strings(splits[0][c]).tolist()) - _NOT_A_CLASS) for c in categorical}
        native = [_codes(s, categorical, vocabulary) for s in splits]
        hidden = [np.isnan(m) for m in native]
        clock = time.perf_counter()
        filled, rounds = (_impute("missforest", splits, hidden, numerical, categorical)
                          if any(h.any() for h in hidden) else (splits, 0))
        plain = [_codes(f, categorical, vocabulary) for f in filled]
        assert not any(np.isnan(m).any() for m in plain), f"missforest left a gap, seed {seed} fold {ordinal}"
        gapped = [j for j in range(len(features.columns)) if hidden[0][:, j].any()]
        folds.append(Fold(native, [np.hstack([m, h[:, gapped].astype(float)]) for m, h in zip(plain, hidden)],
                          [labels[p] for p in parts]))
        print(f"  seed {seed} fold {ordinal}: missforest {time.perf_counter() - clock:.1f} s, {rounds} rounds", flush=True)
    return folds


def _scores(arm: str, params: dict[str, Any], folds: list[Fold], declared: list[int]) -> list[dict[str, float]]:
    """Per fold: validation and test metrics of ``params``."""
    from src.training.finetuning import build_fold_result

    out = []
    for ordinal, fold in enumerate(folds, start=1):
        x = fold.filled if arm == "missforest+ind/hgb" else fold.native
        clock = time.perf_counter()
        model, rounds = _fit(arm, params, x, fold.y, declared)
        val = build_fold_result(ordinal, "", fold.y[1], np.asarray(model.predict(x[1]))).metrics
        test = build_fold_result(ordinal, "", fold.y[2], np.asarray(model.predict(x[2]))).metrics
        out.append({"val_f1_macro": float(val["f1_macro"]), "f1_macro": float(test["f1_macro"]),
                    "f1_micro": float(test["f1_micro"]), "accuracy": float(test["accuracy"]),
                    "rounds": rounds, "seconds": round(time.perf_counter() - clock, 3)})
    return out


def _tune(variant: str, arm: str, folds: list[Fold], declared: list[int]) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.time()

    def objective(trial: optuna.Trial) -> float:
        # Only the validation metrics are read here; the test predictions are discarded unread.
        value = float(np.mean([s["val_f1_macro"] for s in _scores(arm, _space(arm, trial), folds, declared)]))
        print(f"  {variant} {arm} trial {trial.number}: {value:.4f} ({time.time() - started:.0f} s)", flush=True)
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
    study.optimize(objective, n_trials=TRIALS)
    return {"best_params": study.best_params, "best_value": study.best_value, "seconds": round(time.time() - started, 1),
            "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]}


def run_variant(variant: str, state: Path) -> None:
    from src.training import lightgbm_baseline, missforest
    from src.training.data import prepare_dataset
    from src.training.types import DatasetSpec

    if _csv(state, variant).exists():
        return
    lightgbm_baseline.THREADS = THREADS  # hard-coded at 8 in src/, read at call time
    missforest.THREADS = THREADS
    warnings.filterwarnings("ignore")
    started = time.time()
    dataset = prepare_dataset(DatasetSpec.from_name(variant, None))
    columns = list(dataset.frame.drop(columns=[dataset.label_column]).columns)
    declared = [columns.index(c) for c in dataset.categorical_columns]
    print(f"{variant}: {len(dataset.frame)} rows", flush=True)
    folds = {TUNING_SEED: _folds(dataset, TUNING_SEED)}
    if _json(state, variant).exists():
        tuned = json.loads(_json(state, variant).read_text())
        print(f"{variant}: tuned parameters from {_json(state, variant)}", flush=True)
    else:
        arms = {arm: _tune(variant, arm, folds[TUNING_SEED], declared) for arm in ARMS}
        # max keeps the first on an exact tie: HGB.
        bar = max(BAR_ARMS, key=lambda a: float(arms[a]["best_value"]))
        tuned = {"variant": variant, "bar": bar, "arms": arms}
        _atomic(_json(state, variant), json.dumps(tuned, indent=1))
    rows = []
    for seed in SEEDS:
        if seed not in folds:
            folds[seed] = _folds(dataset, seed)
        for arm in ARMS:
            clock = time.time()
            scores = _scores(arm, tuned["arms"][arm]["best_params"], folds[seed], declared)
            rows += [{"variant": variant, "seed": seed, "fold": k, "arm": arm} | s for k, s in enumerate(scores, start=1)]
            if seed == TUNING_SEED:  # the refit of the best trial: the same fits, so the same score
                value = float(np.mean([s["val_f1_macro"] for s in scores]))
                if abs(value - float(tuned["arms"][arm]["best_value"])) > 1e-9:
                    print(f"  WARNING {variant} {arm}: refit validation {value:.6f}, study {tuned['arms'][arm]['best_value']:.6f}", flush=True)
            print(f"  {variant} {arm} seed {seed}: eval {time.time() - clock:.0f} s", flush=True)
        folds.pop(seed)
    frame = pd.DataFrame(rows)
    _atomic(_csv(state, variant), frame.to_csv(index=False))
    table = frame.groupby("arm", sort=False)["f1_macro"].mean()
    print(f"DONE {variant}: {time.time() - started:.0f} s; bar {tuned['bar']}; "
          + "; ".join(f"{a} {v:.4f}" for a, v in table.items()), flush=True)


def tally(state: Path) -> None:
    found = {v: pd.read_csv(_csv(state, v)) for v in VARIANTS if _csv(state, v).exists()}
    missing = [v for v in VARIANTS if v not in found]
    if missing:
        print(f"incomplete: {len(missing)} of {len(VARIANTS)} variants missing, e.g. {', '.join(missing[:5])}")
    bars = {v: str(json.loads(_json(state, v).read_text())["bar"]) for v in found}
    for level in LEVELS:
        variants = [f"{b}_{level}nan" for b in BASES if f"{b}_{level}nan" in found]
        if not variants:
            continue
        frame = pd.concat(found[v] for v in variants)
        bar = pd.concat(found[v][found[v].arm == bars[v]] for v in variants)
        means = frame.groupby("arm")["f1_macro"].mean()
        print(f"## {level}nan ({len(variants)} of {len(BASES)} tables), mean test macro-F1 over seeds and folds")
        for arm in ARMS:
            print(f"- {arm:<20} {means[arm]:.4f}")
        print(f"- {'bar':<20} {bar['f1_macro'].mean():.4f}")
        print("- bar picked: " + ", ".join(f"{v.split('_')[0]} {bars[v].split('/')[1]}" for v in variants) + "\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--state"] and len(argv) >= 2:
        STATE = Path(argv[1])
        argv = argv[2:]
    if argv[:1] == ["run"]:
        unknown = [v for v in argv[1:] if v not in VARIANTS]
        if unknown:
            sys.exit(f"unknown variants: {', '.join(unknown)}")
        for name in argv[1:] or list(VARIANTS):
            run_variant(name, STATE)
    elif argv == ["--tally"]:
        tally(STATE)
    else:
        sys.exit(__doc__)
