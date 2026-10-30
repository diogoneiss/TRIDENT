"""E99: the deep baselines MLP, SAINT and TabNet, tuned as E98 tuned TRIDENT (classification-step-zero/08).

    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py [--state <dir>] run <model> <variant>
    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py [--state <dir>] probe <model> <variant>

``run`` tunes ``<model>`` (``mlp``, ``saint`` or ``tabnet``, from pytorch-widedeep) on seed 42's ten
folds (``build_folds(frame, 'class', 10, 42)``), then evaluates the tuned parameters at seeds 42, 7
and 13 on each seed's ten folds. The study is Optuna TPE (sampler seed 42, 20 trials, maximise) on
the mean over the ten folds of the validation split's macro-F1, as E98's TRIDENT studies.

Every fit is TRIDENT's recipe: cross-entropy weighted by the training split's balanced class weights,
AdamW, at most ``EPOCHS`` epochs, and the epoch of the lowest validation loss kept (early stopping,
patience ``PATIENCE``, best weights restored). Inputs, fit on the training split alone: a numerical
gap filled with the column's training mean, plus one 0/1 indicator per numerical column with a
training gap; a categorical gap as its own level ``[NULL]``, an unseen category as widedeep's
unknown index; numerical columns standardised.

``<state>/json/<model>_<variant>.json`` holds the best parameters, best score and every trial
(written after tuning; a rerun reuses it), ``<state>/csv/<model>_<variant>.csv`` one row per seed
and fold (written last; a unit with it is skipped), both through a temporary name. ``probe`` fits
one mid-space configuration on seed 42's ten folds and prints only the seconds, never a score.
Nothing touches the store.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# shortcut: pytorch-widedeep 1.5.0's callbacks use np.Inf, removed in NumPy 2; drop once widedeep is bumped.
np.Inf = np.inf  # type: ignore[attr-defined]

from scripts.experiments.cls_stage2_run import VARIANTS  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-30-step-zero/deep")
TUNING_SEED = 42
SEEDS = (42, 7, 13)
FOLDS = 10
TRIALS = 20
EPOCHS = 200
PATIENCE = 20
MODELS = ("mlp", "saint", "tabnet")
NULL = "[NULL]"
# One configuration from the middle of each space, for the cost probe only.
MIDDLE: dict[str, dict[str, Any]] = {
    "mlp": {"hidden": "256-128", "dropout": 0.25, "lr": 1e-3, "weight_decay": 3e-5},
    "saint": {"input_dim": 32, "n_blocks": 2, "n_heads": 4, "attn_dropout": 0.15, "ff_dropout": 0.15, "lr": 5e-4, "weight_decay": 3e-5},
    "tabnet": {"n_steps": 5, "step_dim": 16, "gamma": 1.5, "lambda_sparse": 3e-5, "lr": 1.5e-2, "weight_decay": 3e-5},
}
BATCH = {"mlp": 256, "saint": 256, "tabnet": 1024}


def _space(model: str, trial: Any) -> dict[str, Any]:
    lr_low, lr_high = {"mlp": (1e-4, 1e-2), "saint": (1e-4, 3e-3), "tabnet": (5e-3, 5e-2)}[model]
    common = {"lr": trial.suggest_float("lr", lr_low, lr_high, log=True),
              "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)}
    if model == "mlp":
        return {"hidden": trial.suggest_categorical("hidden", ["128-64", "256-128", "256-256-128", "512-256"]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5)} | common
    if model == "saint":
        return {"input_dim": trial.suggest_categorical("input_dim", [16, 32]),
                "n_blocks": trial.suggest_int("n_blocks", 1, 3),
                "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
                "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
                "ff_dropout": trial.suggest_float("ff_dropout", 0.0, 0.3)} | common
    return {"n_steps": trial.suggest_int("n_steps", 3, 7),
            "step_dim": trial.suggest_categorical("step_dim", [8, 16, 32, 64]),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)} | common


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(text)
    partial.rename(path)


def _folds(dataset: Any, seed: int) -> list[list[pd.DataFrame]]:
    """The ten folds of ``seed`` as train/validation/test frames with the label, gaps already handled
    from the training split (module docstring)."""
    from src.embedder import as_category_strings
    from src.training.data import build_folds
    from src.training.imputation_baselines import _NOT_A_CLASS

    label = dataset.label_column
    categorical, numerical = list(dataset.categorical_columns), list(dataset.numerical_columns)
    out = []
    for fold in build_folds(dataset.frame, label, FOLDS, seed):
        parts = [dataset.frame.iloc[np.asarray(p)].reset_index(drop=True)
                 for p in (fold.train_indices, fold.validation_indices, fold.test_indices)]
        gapped = [c for c in numerical if parts[0][c].isna().any()]
        means = {c: float(parts[0][c].mean()) for c in numerical}
        frames = []
        for part in parts:
            frame = pd.DataFrame({label: part[label].to_numpy().astype(np.int64)})
            for c in categorical:
                values = as_category_strings(part[c])
                frame[c] = np.where(np.isin(values, list(_NOT_A_CLASS | {"nan"})), NULL, values)
            for c in numerical:
                frame[c] = part[c].astype(float).fillna(means[c] if np.isfinite(means[c]) else 0.0)
            for c in gapped:
                frame[f"{c}__gap"] = part[c].isna().astype(float)
            frames.append(frame)
        out.append(frames)
    return out


def _fit(model: str, params: dict[str, Any], frames: list[pd.DataFrame], label: str, categorical: list[str],
         seed: int) -> dict[str, float]:
    """Fit on the training frame, stop on the validation frame; validation and test metrics."""
    import torch
    from pytorch_widedeep import Trainer
    from pytorch_widedeep.callbacks import EarlyStopping
    from pytorch_widedeep.models import SAINT, TabMlp, TabNet, WideDeep
    from pytorch_widedeep.preprocessing import TabPreprocessor

    from src.training.finetuning import build_fold_result

    continuous = [c for c in frames[0].columns if c not in categorical and c != label]
    prep = TabPreprocessor(cat_embed_cols=categorical or None, continuous_cols=continuous or None,
                           cols_to_scale=continuous or None, with_attention=model == "saint", verbose=0,
                           **({"cat_embed_dim": params["input_dim"]} if model == "saint" else {}))
    x = [prep.fit_transform(frames[0])] + [prep.transform(f) for f in frames[1:]]
    y = [f[label].to_numpy() for f in frames]
    classes = int(max(int(v.max()) for v in y)) + 1
    present, counts = np.unique(y[0], return_counts=True)
    weights = np.ones(classes)
    weights[present] = len(y[0]) / (len(present) * counts)
    shared = {"column_idx": prep.column_idx, "cat_embed_input": prep.cat_embed_input,
              "continuous_cols": continuous or None}
    if model == "mlp":
        net = TabMlp(mlp_hidden_dims=[int(h) for h in params["hidden"].split("-")], mlp_dropout=params["dropout"], **shared)
    elif model == "saint":
        net = SAINT(input_dim=params["input_dim"], n_blocks=params["n_blocks"], n_heads=params["n_heads"],
                    attn_dropout=params["attn_dropout"], ff_dropout=params["ff_dropout"], **shared)
    else:
        net = TabNet(n_steps=params["n_steps"], step_dim=params["step_dim"], attn_dim=params["step_dim"],
                     gamma=params["gamma"], **shared)
    wide = WideDeep(deeptabular=net, pred_dim=classes)
    stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    trainer = Trainer(wide, objective="multiclass", seed=seed, verbose=0, callbacks=[stopper],
                      optimizers=torch.optim.AdamW(wide.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]),
                      class_weight=weights.tolist(),
                      **({"lambda_sparse": params["lambda_sparse"]} if model == "tabnet" else {}))
    clock = time.perf_counter()
    trainer.fit(X_train={"X_tab": x[0], "target": y[0]}, X_val={"X_tab": x[1], "target": y[1]},
                n_epochs=EPOCHS, batch_size=BATCH[model])
    seconds = time.perf_counter() - clock
    val = build_fold_result(1, "", y[1], np.asarray(trainer.predict(X_tab=x[1], batch_size=BATCH[model]))).metrics
    test = build_fold_result(1, "", y[2], np.asarray(trainer.predict(X_tab=x[2], batch_size=BATCH[model]))).metrics
    return {"val_f1_macro": float(val["f1_macro"]), "f1_macro": float(test["f1_macro"]),
            "f1_micro": float(test["f1_micro"]), "accuracy": float(test["accuracy"]),
            "kept_epoch": int(stopper.best_epoch) + 1, "seconds": round(seconds, 3)}


def _scores(model: str, params: dict[str, Any], folds: list[list[pd.DataFrame]], dataset: Any, seed: int) -> list[dict[str, float]]:
    categorical = list(dataset.categorical_columns)
    return [_fit(model, params, frames, dataset.label_column, categorical, seed) for frames in folds]


def _dataset(variant: str) -> Any:
    from src.training.data import prepare_dataset
    from src.training.types import DatasetSpec

    warnings.filterwarnings("ignore")
    return prepare_dataset(DatasetSpec.from_name(variant, None))


def probe(model: str, variant: str) -> None:
    dataset = _dataset(variant)
    clock = time.time()
    scores = _scores(model, MIDDLE[model], _folds(dataset, TUNING_SEED), dataset, TUNING_SEED)
    # Only the cost is printed: the probe's scores are never read.
    print(f"PROBE {model} {variant}: {time.time() - clock:.0f} s for {FOLDS} folds, "
          f"kept epochs {[int(s['kept_epoch']) for s in scores]}", flush=True)


def run(model: str, variant: str, state: Path) -> None:
    import optuna

    js, csv = state / "json" / f"{model}_{variant}.json", state / "csv" / f"{model}_{variant}.csv"
    if csv.exists():
        print(f"DONE {model} {variant}: skipped, finished", flush=True)
        return
    dataset = _dataset(variant)
    started = time.time()
    folds = _folds(dataset, TUNING_SEED)
    if js.exists():
        tuned = json.loads(js.read_text())
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            # Only the validation metrics are read here; the test predictions are discarded unread.
            value = float(np.mean([s["val_f1_macro"] for s in _scores(model, _space(model, trial), folds, dataset, TUNING_SEED)]))
            print(f"  {model} {variant} trial {trial.number}: ({time.time() - started:.0f} s)", flush=True)
            return value

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNING_SEED))
        study.optimize(objective, n_trials=TRIALS)
        tuned = {"model": model, "variant": variant, "best_params": study.best_params, "best_value": study.best_value,
                 "seconds": round(time.time() - started, 1),
                 "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]}
        _atomic(js, json.dumps(tuned, indent=1))
    rows = []
    for seed in SEEDS:
        scores = _scores(model, tuned["best_params"], folds if seed == TUNING_SEED else _folds(dataset, seed), dataset, seed)
        rows += [{"variant": variant, "model": model, "seed": seed, "fold": k} | s for k, s in enumerate(scores, start=1)]
        print(f"  {model} {variant} seed {seed}: evaluated ({time.time() - started:.0f} s)", flush=True)
    _atomic(csv, pd.DataFrame(rows).to_csv(index=False))
    print(f"DONE {model} {variant}: {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--state"] and len(argv) >= 2:
        STATE = Path(argv[1])
        argv = argv[2:]
    if len(argv) == 3 and argv[0] in ("run", "probe") and argv[1] in MODELS and argv[2] in VARIANTS:
        probe(argv[1], argv[2]) if argv[0] == "probe" else run(argv[1], argv[2], STATE)
    else:
        sys.exit(__doc__)
