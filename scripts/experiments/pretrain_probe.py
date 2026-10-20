"""E53, T28's D2: collapse and linear probes on the pre-training stage, and the encoder's drift
in the decode stage (imputation-pretraining/11). Offline: nothing is written to the store.

    uv run --python 3.11 python scripts/experiments/pretrain_probe.py probe <variant> <seed> <N|A|N_lr4|N_b64> <out_dir> [--device cpu]
    uv run --python 3.11 python scripts/experiments/pretrain_probe.py drift <variant> <seed> <N|R> <out_dir> [--device cpu]
    uv run --python 3.11 python scripts/experiments/pretrain_probe.py fidelity <variant> <seed> [--device cpu]

The first fold of a five-fold run of the variant, with today's configuration as a default run
reads it (the promoted file over the task defaults) plus the setting's change: N none (the
normalised objective), A the shipped ``embedding`` objective, N_lr4 a quarter of ``LR_PRE``, N_b64
``BATCH 64``. ``train_pretrainer`` itself runs; the script only watches it:

- ``probe``: the stage is run twice from the same seed, with ``EPOCHS_PRE 0`` (the Xavier start,
  the same as E52's FR and E50's C0) and in full. A tracker computes the diagnostics every
  ``EVERY`` epochs and at the last, after the epoch's own validation pass, in eval mode, with
  every random stream saved and restored, so the stage's draws do not move (``fidelity``
  checks it). On a fixed mask of the validation split, at its hidden cells, by kind:
  - E04's diagnostics in the loss's own space (the prediction after the predictor under N, the
    layer-normalised target): NMSE against the column-mean target, the centred cosine and its
    standard error 1/sqrt(d·n), the within-column share of the target's second moment, and the
    same share of the raw embedder target (the embedder's erosion);
  - the effective rank of each column's centred targets and predictions.
  - Linear probes, at the start and the end: from the transformer's output at the hidden cells
    (what the decode stage reads), per column, a ``BayesianRidge`` for numbers and a
    ``LogisticRegression`` for categories (C by 3-fold CV), on standardised features. They are
    fitted on the training split's hidden cells under ``TRAIN_MASKS`` fixed masks and scored on
    the validation split's. The
    score is ``impute_score``'s shape: the pooled z-space RMSE over the mean fill's, and the error
    over the mode's, weighted by cells (1.0 = no information).
- ``drift``: N or R (the Xavier start), then today's full decode stage
  (``train_and_evaluate_decoder``). It logs the linear CKA between the encoder's outputs on the
  fixed masked validation input before and after, and ``‖ΔW‖/‖W‖`` per module. "After" is the
  checkpoint the decode stage keeps.
- ``fidelity``: a plain ``train_pretrainer`` call and the watched one must give the same
  validation losses, element for element.

Writes CSVs under ``out_dir``. Run from the checkout root.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import random
import sys
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import BayesianRidge, LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.embedder import EncodedTable, EpochMasker  # noqa: E402
from src.training import pretraining  # noqa: E402
from src.training.config import complete_configuration, hyperparameter_file, resolve_training_request  # noqa: E402
from src.training.data import build_folds, load_complete_sibling, prepare_dataset  # noqa: E402
from src.training.decoding import train_and_evaluate_decoder  # noqa: E402
from src.training.tracking import DisabledTracker  # noqa: E402
from src.training.types import DEFAULT_BASELINE_CACHE_DIR, FoldSplit, PreparedDataset, PretrainingOutcome, TrainingRequest  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

SETTINGS = ("N", "A", "N_lr4", "N_b64")
PROBE_SEED = 20261020
EVERY = 10
# The probes' training cells: this many fixed masks of the training split, and the logistic
# probe's inverse regularisation, chosen per column by 3-fold CV on those cells.
TRAIN_MASKS = 4
PROBE_C = (0.01, 0.1, 1.0)
BATCH_ROWS = 1024
Float = NDArray[np.float64]


def _config(variant: str, setting: str) -> dict[str, object]:
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    if setting == "A":
        config["PRETRAIN_OBJECTIVE"] = "embedding"
    elif setting == "N_lr4":
        base = _request(variant, 0, config).hyperparameters.pretraining_learning_rate
        config["LR_PRE"] = base / 4
    elif setting == "N_b64":
        config["BATCH"] = 64
    elif setting not in ("N", "R"):
        sys.exit(f"unknown setting {setting!r}")
    return config


def _request(variant: str, seed: int, config: dict[str, object]) -> TrainingRequest:
    return resolve_training_request(Namespace(
        dataset_name=variant, label_column="class", task="imputation", seed=seed, cv_folds=5,
        hyperparams_override=config, disable_mlflow=True,
    ))


@contextlib.contextmanager
def _isolated(seed: int) -> Iterator[None]:
    """Run a block on its own numpy stream, leaving every random stream as it found it."""
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Only a CUDA context that already exists is saved, so a CPU run never creates one.
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None  # type: ignore[no-untyped-call]
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@dataclasses.dataclass
class _Split:
    masked: EncodedTable
    clean: EncodedTable


def _split(model: Any, frame: pd.DataFrame, p: float, seed: int, device: torch.device) -> _Split:
    with _isolated(seed):
        masked = EpochMasker(model.embedder, frame, device).draw(p)
        clean = model.embedder.encode(frame, device)
    return _Split(masked, clean)


def _forward(model: Any, split: _Split) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """At the hidden cells: the transformer's output, the loss's prediction, the raw and the
    loss's target, and each cell's (row, column)."""
    outputs: list[list[torch.Tensor]] = [[], [], [], [], []]
    with torch.no_grad():
        for start in range(0, len(split.masked), BATCH_ROWS):
            rows = slice(start, start + BATCH_ROWS)
            masked, clean = split.masked[rows], split.clean[rows]
            encoded = model.transformer(model.embedder(masked))[:, 1:, :]
            positions = masked.masked_positions
            enc = encoded[positions]
            raw = model.embedder(clean)[:, 1:, :][positions]
            where = positions.nonzero()
            where[:, 0] += start
            for bucket, value in zip(outputs, (enc, model._prediction(enc), raw, model._target(raw), where)):
                bucket.append(value)
    enc, prediction, raw, target, where = (torch.cat(bucket) for bucket in outputs)
    return enc, prediction, raw, target, where[:, 0], where[:, 1]


def _centred(values: torch.Tensor, columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each cell's column mean, and the values minus it."""
    means = torch.zeros_like(values)
    for column in sorted(set(columns.tolist())):
        chosen = columns == column
        means[chosen] = values[chosen].mean(dim=0)
    return means, values - means


def _effective_rank(values: torch.Tensor) -> float:
    if values.shape[0] < 2:
        return float("nan")
    singular = torch.linalg.svdvals(values.double())
    share = singular / singular.sum()
    share = share[share > 0]
    return float(torch.exp(-(share * share.log()).sum()))


def _collapse(prediction: torch.Tensor, raw: torch.Tensor, target: torch.Tensor, columns: torch.Tensor) -> dict[str, float]:
    base, within = _centred(target, columns)
    raw_base, raw_within = _centred(raw, columns)
    mse_model = float(((prediction - target) ** 2).mean())
    mse_base = float((within ** 2).mean())
    cosine = torch.nn.functional.cosine_similarity(prediction - base, within, dim=1)
    second = float((target ** 2).mean())
    ranks_target, ranks_prediction = [], []
    for column in sorted(set(columns.tolist())):
        chosen = columns == column
        ranks_target.append(_effective_rank(within[chosen]))
        ranks_prediction.append(_effective_rank(_centred(prediction[chosen], columns[chosen])[1]))
    return {
        "n_cells": float(len(columns)),
        "mse_model": mse_model,
        "mse_base": mse_base,
        "nmse": mse_model / mse_base if mse_base > 0 else float("nan"),
        "centred_cosine": float(cosine.mean()),
        "cosine_se": 1.0 / float(np.sqrt(target.shape[1] * len(columns))),
        "second_moment": second,
        "within_share": mse_base / second if second > 0 else float("nan"),
        "raw_within_share": float((raw_within ** 2).mean()) / float((raw ** 2).mean()),
        "erank_target": float(np.nanmean(ranks_target)),
        "erank_prediction": float(np.nanmean(ranks_prediction)),
    }


def _diagnostics(model: Any, split: _Split, n_categorical: int) -> list[dict[str, Any]]:
    _, prediction, raw, target, _, columns = _forward(model, split)
    rows = []
    for kind, chosen in (("all", columns >= 0), ("cat", columns < n_categorical), ("num", columns >= n_categorical)):
        if bool(chosen.any()):
            rows.append({"kind": kind, **_collapse(prediction[chosen], raw[chosen], target[chosen], columns[chosen])})
    return rows


def _truth(split: _Split, rows: torch.Tensor, columns: torch.Tensor, n_categorical: int) -> Float:
    values = np.empty(len(rows))
    cat = columns < n_categorical
    values[cat.cpu().numpy()] = split.clean.cat_indices[columns[cat], rows[cat]].cpu().numpy()
    num = ~cat
    values[num.cpu().numpy()] = split.clean.num_values[columns[num] - n_categorical, rows[num]].cpu().numpy()
    return values


def _cells(model: Any, split: _Split, n_categorical: int) -> tuple[Float, Float, Float]:
    enc, _, _, _, rows, columns = _forward(model, split)
    return enc.double().cpu().numpy(), _truth(split, rows, columns, n_categorical), columns.cpu().numpy().astype(float)


def _logistic(x: Float, y: NDArray[np.int64], seed: int) -> LogisticRegression:
    """The logistic probe, its C chosen by 3-fold CV accuracy on the training cells."""
    best, best_accuracy = PROBE_C[0], -1.0
    for c in PROBE_C:
        hits = 0
        for fit, held in KFold(n_splits=3, shuffle=True, random_state=seed).split(x):
            if len(np.unique(y[fit])) < 2:
                hits += int((y[held] == y[fit][0]).sum())
                continue
            hits += int((LogisticRegression(C=c, max_iter=2000).fit(x[fit], y[fit]).predict(x[held]) == y[held]).sum())
        if hits / len(y) > best_accuracy:
            best, best_accuracy = c, hits / len(y)
    return LogisticRegression(C=best, max_iter=2000).fit(x, y)


def _probe(model: Any, trains: list[_Split], validation: _Split, n_categorical: int) -> dict[str, float]:
    """Linear probes from the transformer's output at hidden cells to the hidden values."""
    parts = [_cells(model, split, n_categorical) for split in trains]
    x_train, y_train, c_train = (np.concatenate([part[i] for part in parts]) for i in range(3))
    x_val, y_val, c_val = _cells(model, validation, n_categorical)
    sse_probe = sse_mean = 0.0
    n_num = errors_probe = errors_mode = n_cat = 0
    for column in np.unique(c_val):
        tr, va = c_train == column, c_val == column
        if not tr.any() or not va.any():
            continue
        scaler = StandardScaler().fit(x_train[tr])
        features_train, features_val = scaler.transform(x_train[tr]), scaler.transform(x_val[va])
        if column >= n_categorical:
            mean, scale = float(y_train[tr].mean()), float(y_train[tr].std()) or 1.0
            guess = BayesianRidge().fit(features_train, y_train[tr]).predict(features_val)
            sse_probe += float((((guess - y_val[va]) / scale) ** 2).sum())
            sse_mean += float((((mean - y_val[va]) / scale) ** 2).sum())
            n_num += int(va.sum())
        else:
            labels, counts = np.unique(y_train[tr].astype(np.int64), return_counts=True)
            mode = labels[int(np.argmax(counts))]
            if len(labels) < 2:
                guess = np.full(int(va.sum()), mode)
            else:
                guess = _logistic(features_train, y_train[tr].astype(np.int64), PROBE_SEED).predict(features_val)
            errors_probe += int((guess != y_val[va]).sum())
            errors_mode += int((mode != y_val[va]).sum())
            n_cat += int(va.sum())
    num_ratio = float(np.sqrt(sse_probe / sse_mean)) if n_num and sse_mean > 0 else float("nan")
    cat_ratio = errors_probe / errors_mode if n_cat and errors_mode > 0 else float("nan")
    weighted = [(n, r) for n, r in ((n_num, num_ratio), (n_cat, cat_ratio)) if n and not np.isnan(r)]
    score = sum(n * r for n, r in weighted) / sum(n for n, _ in weighted) if weighted else float("nan")
    return {"probe_score": score, "probe_num_ratio": num_ratio, "probe_cat_ratio": cat_ratio,
            "probe_n_num": float(n_num), "probe_n_cat": float(n_cat)}


class _Watch:
    """A tracker that runs the diagnostics after an epoch's validation pass."""

    def __init__(self, on_epoch: Any) -> None:
        self.on_epoch = on_epoch

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if step is not None:
            self.on_epoch(step)


@dataclasses.dataclass
class _Fold:
    request: TrainingRequest
    dataset: PreparedDataset
    fold: FoldSplit
    train: pd.DataFrame
    validation: pd.DataFrame


def _fold(variant: str, seed: int, config: dict[str, object]) -> _Fold:
    request = _request(variant, seed, config)
    set_global_seed(seed)
    dataset = prepare_dataset(request.dataset)
    fold = build_folds(dataset.frame, dataset.label_column, 5, seed)[0]
    features = dataset.frame.drop(columns=[dataset.label_column])
    return _Fold(
        request, dataset, fold,
        features.iloc[fold.train_indices].reset_index(drop=True),
        features.iloc[fold.validation_indices].reset_index(drop=True),
    )


def _run_stage(state: _Fold, epochs: int | None, device: torch.device, on_epoch: Any = None) -> tuple[Any, list[float]]:
    """``train_pretrainer`` from the fold's seed, returning the model it trained."""
    captured: dict[str, Any] = {}
    original = getattr(pretraining, "_stage_model")

    def capture(*args: Any, **kwargs: Any) -> Any:
        captured["model"] = original(*args, **kwargs)
        return captured["model"]

    hyperparameters = state.request.hyperparameters
    if epochs is not None:
        hyperparameters = dataclasses.replace(hyperparameters, pretraining_epochs=epochs)
    setattr(pretraining, "_stage_model", capture)
    try:
        set_global_seed(state.request.seed)
        tracker = _Watch(lambda epoch: on_epoch(captured["model"], epoch)) if on_epoch else DisabledTracker()
        outcome = pretraining.train_pretrainer(state.dataset, state.fold, hyperparameters, device, tracker)
    finally:
        setattr(pretraining, "_stage_model", original)
    return outcome.model, list(outcome.validation_losses)


def probe(variant: str, seed: int, setting: str, out: Path, device: torch.device) -> None:
    state = _fold(variant, seed, _config(variant, setting))
    hp = state.request.hyperparameters
    n_categorical = len(state.dataset.categorical_columns)
    rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    splits: dict[str, _Split] = {}
    trains: list[_Split] = []

    def fixed(model: Any) -> tuple[list[_Split], _Split]:
        if not splits:
            splits["validation"] = _split(model, state.validation, hp.mask_probability, PROBE_SEED + seed, device)
            trains.extend(_split(model, state.train, hp.mask_probability, PROBE_SEED + seed + 1 + i, device) for i in range(TRAIN_MASKS))
        return trains, splits["validation"]

    def watch(model: Any, epoch: int) -> None:
        if epoch % EVERY and epoch != hp.pretraining_epochs - 1:
            return
        with _isolated(PROBE_SEED):
            train, validation = fixed(model)
            for row in _diagnostics(model, validation, n_categorical):
                rows.append({"stage": "train", "epoch": epoch + 1, **row})
            if epoch == hp.pretraining_epochs - 1:
                probes.append({"stage": "final", **_probe(model, train, validation, n_categorical)})

    start, _ = _run_stage(state, 0, device)
    start.eval()
    with _isolated(PROBE_SEED):
        train, validation = fixed(start)
        rows.extend({"stage": "start", "epoch": 0, **row} for row in _diagnostics(start, validation, n_categorical))
        probes.append({"stage": "start", **_probe(start, train, validation, n_categorical)})
    _run_stage(state, None, device, watch)
    meta = {"variant": variant, "seed": seed, "setting": setting, "device": device.type}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**meta, **row} for row in rows]).to_csv(out / f"{variant}_s{seed}_{setting}_collapse.csv", index=False)
    pd.DataFrame([{**meta, **row} for row in probes]).to_csv(out / f"{variant}_s{seed}_{setting}_probe.csv", index=False)
    print("PROBE_DONE", json.dumps({**meta, **{p["stage"]: round(p["probe_score"], 4) for p in probes}}), flush=True)


def _outputs(model: Any, split: _Split) -> torch.Tensor:
    with torch.no_grad():
        chunks = [model.transformer(model.embedder(split.masked[start : start + BATCH_ROWS]))[:, 1:, :].reshape(-1, model.embedder.dimensao)
                  for start in range(0, len(split.masked), BATCH_ROWS)]
    return torch.cat(chunks).double()


def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x - x.mean(dim=0), y - y.mean(dim=0)
    return float(torch.linalg.matrix_norm(y.T @ x) ** 2 / (torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)))


def _modules(model: Any) -> dict[str, list[torch.Tensor]]:
    found: dict[str, list[torch.Tensor]] = {"embedder": [p.detach().clone() for p in model.embedder.parameters()]}
    for name, child in model.transformer.named_children():
        parameters = [p.detach().clone() for p in child.parameters()]
        if parameters:
            found[f"transformer.{name}"] = parameters
    return found


def drift(variant: str, seed: int, start: str, out: Path, device: torch.device) -> None:
    state = _fold(variant, seed, _config(variant, start))
    hp = dataclasses.replace(state.request.hyperparameters, decode_calibration="none")
    model, _ = _run_stage(state, 0 if start == "R" else None, device)
    model.eval()
    with _isolated(PROBE_SEED):
        validation = _split(model, state.validation, hp.mask_probability, PROBE_SEED + seed, device)
        before = _outputs(model, validation)
    weights = _modules(model)
    pre = copy.deepcopy(model)
    outcome = train_and_evaluate_decoder(
        dataset=state.dataset, fold=state.fold, pretraining=PretrainingOutcome(model=model, train_losses=[], validation_losses=[]),
        hyperparameters=hp, device=device, tracker=DisabledTracker(), seed=seed, fold_ordinal=1,
        complete_sibling=load_complete_sibling(state.request.dataset), baseline_cache_dir=DEFAULT_BASELINE_CACHE_DIR,
    )
    after_model = outcome.model
    after_model.eval()
    after = _outputs(after_model, validation)
    moved = _modules(after_model)
    row: dict[str, Any] = {"variant": variant, "seed": seed, "start": start, "device": device.type, "linear_cka": _cka(before, after),
                           "induced_impute_score": float(outcome.result.metrics.get("impute/induced/impute_score", float("nan")))}
    for name, old in weights.items():
        change = sum(float(((n - o) ** 2).sum()) for n, o in zip(moved[name], old))
        size = sum(float((o ** 2).sum()) for o in old)
        row[f"rel_change/{name}"] = float(np.sqrt(change / size)) if size > 0 else float("nan")
    assert _cka(before, _outputs(pre.eval(), validation)) > 1 - 1e-9, "the snapshot moved before decoding"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / f"{variant}_s{seed}_{start}_drift.csv", index=False)
    print("DRIFT_DONE", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def fidelity(variant: str, seed: int, device: torch.device) -> None:
    state = _fold(variant, seed, _config(variant, "N"))
    _, plain = _run_stage(state, None, device)
    calls: list[int] = []
    n_categorical = len(state.dataset.categorical_columns)

    def watch(model: Any, epoch: int) -> None:
        calls.append(epoch)
        with _isolated(PROBE_SEED):
            _diagnostics(model, _split(model, state.validation, 0.5, PROBE_SEED, device), n_categorical)

    _, watched = _run_stage(state, None, device, watch)
    worst = max(abs(a - b) for a, b in zip(plain, watched))
    print(f"FIDELITY {variant} s{seed} on {device.type}: {len(plain)} epochs, diagnostics at {len(calls)}, largest difference {worst:g}")
    assert len(plain) == len(watched) and worst == 0.0, "the watched stage moved"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "drift", "fidelity"))
    parser.add_argument("variant")
    parser.add_argument("seed", type=int)
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    options = parser.parse_args(argv)
    device = torch.device(options.device)
    if options.mode == "fidelity":
        fidelity(options.variant, options.seed, device)
    elif options.mode == "probe":
        setting, out = options.rest
        probe(options.variant, options.seed, setting, Path(out), device)
    else:
        start, out = options.rest
        drift(options.variant, options.seed, start, Path(out), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
