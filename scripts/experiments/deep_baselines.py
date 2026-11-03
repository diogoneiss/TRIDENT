"""E99: the deep baselines MLP, SAINT and TabNet, tuned as E98 tuned TRIDENT (classification-step-zero/08).

    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py [--state <dir>] run <model> <variant>
    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py [--state <dir>] probe <model> <variant>
    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py --deal <machine>
    uv run --python 3.11 --extra deep-baselines python -B scripts/experiments/deep_baselines.py --tally

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
Nothing touches the store. ``--deal`` prints a machine's (model, variant) units, dearest first (E99's
split by the probe's cost); ``--tally`` reads E99's rule against E98's tuned TRIDENT.
``--study e106`` first: TabNet alone with a bigger budget (batch 256, 400 epochs, patience 50) at
40–80nan (classification-step-zero/09). ``--study e107`` first: the three models at 00 and 20nan,
descriptive (classification-step-zero/10).
"""

from __future__ import annotations

import json
import socket
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

STATE = Path("/scratch2/diogoneiss" if socket.gethostname() == "gorgona8" else "/scratch/diogoneiss") / "trident-2026-10-31-deep-baselines"
LEVELS: tuple[str, ...] = ("40", "60", "80")  # E99's scope (the user, 2026-10-31)
# The cost probe's seconds for ten folds at 40nan (ticket 08); SAINT's two out-of-memory tables are guesses.
COST = {"mlp": {"kc2": 4, "credit-g": 7, "vehicle": 6, "biodeg": 6, "spambase": 12, "kr-vs-kp": 20, "pendigits": 63, "letter": 211, "electricity": 223},
        "saint": {"kc2": 14, "credit-g": 12, "vehicle": 19, "biodeg": 25, "spambase": 60, "kr-vs-kp": 28, "pendigits": 117, "letter": 140, "electricity": 286},
        "tabnet": {"kc2": 18, "credit-g": 19, "vehicle": 21, "biodeg": 20, "spambase": 110, "kr-vs-kp": 70, "pendigits": 317, "letter": 717, "electricity": 683}}
CAPACITY = {"medusa": 2.0, "gorgona8": 0.7}  # two 5090s against one shared 3090 Ti
TUNING_SEED = 42
SEEDS = (42, 7, 13)
FOLDS = 10
TRIALS = 20
EPOCHS = 200
PATIENCE = 20
MODELS: tuple[str, ...] = ("mlp", "saint", "tabnet")
STUDY = "e99"
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
    shared = {"column_idx": prep.column_idx, "cat_embed_input": getattr(prep, "cat_embed_input", None),  # unset on an all-numerical table
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
                      num_workers=0,  # the default, one worker per core, respawns every epoch: 38 s a kc2 fold
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
    import gc

    import torch

    categorical = list(dataset.categorical_columns)
    out = []
    for frames in folds:
        out.append(_fit(model, params, frames, dataset.label_column, categorical, seed))
        # widedeep's trainer sits in reference cycles; without a collection each fit's model stays on
        # the GPU, and SAINT ran out of memory by the eighth fold on biodeg (23 GiB, 0.02 with this).
        gc.collect()
        torch.cuda.empty_cache()
    return out


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
    if STUDY == "e115":
        tuned = json.loads((STATE / "e99_json" / f"{model}_{sibling(variant)}.json").read_text())
    elif STUDY in ("e106", "e111"):
        # E106 retrains E99's tuned configuration with the bigger budget; no new search (the user, 2026-11-01).
        tuned = json.loads((STATE / "e99_json" / f"{model}_{variant}.json").read_text())
    elif js.exists():
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


# E115: the MAR and MNAR variants at 60nan; each borrows its MCAR sibling's tuned configuration.
E115_VARIANTS = (*(f"{t}_probit_60nan" for t in ("biodeg", "kc2", "vehicle", "spambase")),
                 *(f"{t}_marf_60nan" for t in ("biodeg", "kc2", "vehicle", "spambase", "credit-g", "kr-vs-kp")))


def sibling(variant: str) -> str:
    """The MCAR variant whose study gives an E115 variant its configuration."""
    base, *_, level = variant.split("_")
    return f"{base}_{level}"


def _use(study: str) -> None:
    """E106 and E107: the same driver with another scope or budget, in their own state."""
    global STATE, LEVELS, MODELS, EPOCHS, PATIENCE, STUDY, SEEDS, VARIANTS
    STATE, STUDY = STATE.parent / f"trident-2026-11-01-deep-{study}", study
    if study == "e106":
        MODELS, EPOCHS, PATIENCE = ("tabnet",), 400, 50
        BATCH["tabnet"] = 256
    elif study == "e111":
        MODELS, SEEDS = ("mlp", "saint"), (5151, 6262, 7373)
    elif study == "e115":
        MODELS, LEVELS, VARIANTS = ("mlp", "saint"), ("60",), E115_VARIANTS
    else:
        LEVELS = ("00", "20")


def deal(machine: str) -> None:
    """Dearest units first, each to the machine with the least load per capacity; print this one's."""
    units = sorted(((COST[m][v.split("_")[0]], m, v) for m in MODELS for v in VARIANTS if v.rsplit("_", 1)[1][:2] in LEVELS),
                   key=lambda u: (-u[0], u[1], u[2]))
    load = {k: 0.0 for k in CAPACITY}
    if STUDY == "e115":  # all on medusa1, beside TRIDENT's evaluations and its E98 studies
        load["gorgona8"] = float("inf")
    for cost, model, variant in units:
        target = min(load, key=lambda k: load[k] / CAPACITY[k])
        load[target] += cost
        if target == machine:
            print(model, variant)


def tally_e115() -> None:
    """E115 (descriptive): per mechanism, TRIDENT − baseline pooled over its variants, one array per
    seed. Runs on medusa1, where both sides' folds live."""
    from scripts.experiments import report_common as common
    from scripts.experiments import stage4_trident as t
    from scripts.experiments.cls_stage2_run import fold_metrics

    groups = {"MNAR (probit)": [v for v in VARIANTS if "_probit_" in v], "MAR (marf)": [v for v in VARIANTS if "_marf_" in v]}
    for name, variants in groups.items():
        print(f"## {name}: {', '.join(v.split('_')[0] for v in variants)}")
        for model in MODELS:
            per_seed: list[np.ndarray] = []
            missing = []
            for seed in SEEDS:
                diffs = []
                for v in variants:
                    ours = fold_metrics(t.eval_dir(v, seed))
                    path = STATE / "csv" / f"{model}_{v}.csv"
                    if ours is None or not path.exists():
                        missing.append(f"{v} s{seed}")
                        continue
                    theirs = pd.read_csv(path)
                    theirs = theirs[theirs.seed == seed].sort_values("fold")
                    diffs.append(ours.sort_values("fold")["f1_macro"].to_numpy(dtype=float) - theirs["f1_macro"].to_numpy(dtype=float))
                if diffs:
                    per_seed.append(np.concatenate(diffs))
            if missing:
                print(f"- TRIDENT − {model}: incomplete ({len(missing)} missing, e.g. {missing[0]})")
                continue
            vd = common.verdict(per_seed)
            mark = {"worse": "TRIDENT higher", "better": "baseline higher", "none": "no verdict"}[vd.call]
            print(f"- TRIDENT − {model}: {vd.mean:+.4f} [{vd.low:+.4f}, {vd.high:+.4f}] seeds "
                  + ", ".join(f"{x:+.4f}" for x in vd.seeds) + f", {sum(len(d) for d in per_seed)} fold pairs -> {mark}")
        print()


def tally() -> None:
    """E99's rule: per level and baseline, TRIDENT − baseline test macro-F1 over 270 fold pairs."""
    from scripts.experiments import report_common as common
    from scripts.experiments.cls_stage2_run import BASES
    from scripts.experiments.stage4_tally import SEEDS as T_SEEDS
    from scripts.experiments.stage4_tally import trident

    assert STUDY == "e111" or tuple(T_SEEDS) == SEEDS
    missing: list[str] = []
    marks: dict[str, dict[str, str]] = {m: {} for m in MODELS}
    for level in LEVELS:
        print(f"## {level}nan")
        for model in MODELS:
            per_seed: dict[int, list[np.ndarray]] = {s: [] for s in SEEDS}
            for base in BASES:
                variant = f"{base}_{level}nan"
                found = [p for p in (STATE / "csv" / f"{model}_{variant}.csv", STATE / "medusa" / "csv" / f"{model}_{variant}.csv") if p.exists()]
                if not found:
                    missing.append(f"{model} {variant}")
                    continue
                theirs = pd.read_csv(found[0])
                for seed in SEEDS:
                    ours = trident(variant, seed)
                    if ours is None:
                        missing.append(f"TRIDENT {variant} s{seed}")
                        continue
                    rows = theirs[theirs.seed == seed].sort_values("fold")
                    per_seed[seed].append(ours.sort_values("fold")["f1_macro"].to_numpy(dtype=float) - rows["f1_macro"].to_numpy(dtype=float))
            if any(len(per_seed[s]) < len(BASES) for s in SEEDS):
                print(f"- TRIDENT − {model}: incomplete")
                continue
            v = common.verdict([np.concatenate(per_seed[s]) for s in SEEDS])
            # The rule reads lower as better; a positive difference favours TRIDENT.
            mark = {"worse": "TRIDENT higher", "better": "baseline higher", "none": "no verdict"}[v.call]
            marks[model][level] = mark
            print(f"- TRIDENT − {model}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds "
                  + ", ".join(f"{x:+.4f}" for x in v.seeds) + f" -> {mark}")
        print()
    if missing:
        print(f"- missing: {len(missing)}: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}\n\nPrimary: incomplete")
        return
    if STUDY == "e107":
        print("Descriptive (E107, 00/20nan): " + "; ".join(f"{m} " + ", ".join(f"{lv}nan {marks[m][lv]}" for lv in LEVELS) for m in MODELS))
        return
    leads = {m: sum(x == "TRIDENT higher" for x in marks[m].values()) >= 2 and "baseline higher" not in marks[m].values() for m in MODELS}
    if STUDY == "e106":
        print(f"\nPrimary (E106, TabNet with the bigger budget, 40/60/80nan): TRIDENT {'leads' if leads['tabnet'] else 'does not lead'}"
              f" -> E99's TabNet reading {'holds' if leads['tabnet'] else 'does not hold'} ({common.RULE})")
        return
    beaten = [m for m in MODELS if sum(x == "baseline higher" for x in marks[m].values()) >= 2]
    if STUDY == "e111":
        reading = "replicated" if all(leads.values()) else "not replicated"
        print("- leads: " + ", ".join(f"{m} {'yes' if leads[m] else 'no'}" for m in MODELS)
              + f"\n\nPrimary (E111, E99 on seeds 5151/6262/7373, MLP and SAINT): {reading} ({common.RULE})")
        return
    reading = "supported" if all(leads.values()) else "refuted" if beaten else "mixed"
    print("- leads: " + ", ".join(f"{m} {'yes' if leads[m] else 'no'}" for m in MODELS)
          + f"\n\nPrimary (E99, 40/60/80nan): {reading}" + (f" (higher than TRIDENT at 2 or more levels: {', '.join(beaten)})" if beaten else "")
          + f" ({common.RULE})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--study"] and len(argv) >= 2 and argv[1] in ("e106", "e107", "e111", "e115"):
        _use(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--state"] and len(argv) >= 2:
        STATE = Path(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--deal"] and len(argv) == 2:
        deal(argv[1])
    elif argv == ["--tally"]:
        tally_e115() if STUDY == "e115" else tally()
    elif len(argv) == 3 and argv[0] in ("run", "probe") and argv[1] in MODELS and argv[2] in VARIANTS:
        probe(argv[1], argv[2]) if argv[0] == "probe" else run(argv[1], argv[2], STATE)
    else:
        sys.exit(__doc__)
