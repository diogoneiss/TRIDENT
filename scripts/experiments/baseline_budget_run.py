"""E42's offline pass: the baselines with a validation-only tuning budget (imputation-protocol/02).

    uv run --python 3.11 python -B scripts/experiments/baseline_budget_run.py [--families knn,hgb,lgbm] <out_dir> <variant> <seed> [<seed> ...]

For each seed the variant's five folds are rebuilt as a run built them (``build_folds``, the
evaluation mask at 0.2 drawn per fold as the decode stage draws it). Per fold and family (knn,
hgb, lgbm, missforest), every setting of the family's grid (``src/training/tuned_baselines.py``)
is fitted on the training fold and scored on the validation rows:

- on their own gaps against the complete sibling, the signal K chooses on: the setting chosen
  there is ``<family>_tuned``;
- on the fixed validation mask, the deployable signal: the sensitivity reading's
  ``<family>_tuned_masked``.

At 60% missing and above each setting is also read shrunk toward the training mean by
alpha in ``SHRINKAGE``. Ties keep the setting met first, the family's current one. Only the chosen
settings are scored on the test split, on both populations. Nothing refits on train plus
validation, as the model does not.

``--families`` picks the families, by default the weak tier's (knn, hgb, lgbm), which the tuned bar
needs; ``--families missforest`` is the strong tier's pass, written to ``csv_missforest`` and
``choices_missforest`` so the two passes never share a file.

One CSV per (variant, seed), ``<out_dir>/csv/<variant>_s<seed>.csv``, one row per fold, keys as in
a run's ``raw_fold_metrics.csv``; ``impute/<population>/baseline/tuned_bar/impute_score`` is the
tuned weak tier's best (mean_mode, knn, hgb, lgbm) chosen on the run's CV mean, as ADR 0007
decision 8 chooses the bar. The choices go to ``<out_dir>/choices/<variant>_s<seed>.csv``. A
finished file is skipped. Nothing is written to the store. Threads are capped at 4.
"""

from __future__ import annotations

import csv
import re
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.backfill_baseline_imputers import _cells, _frame_values  # noqa: E402
from src.training import lightgbm_baseline, missforest  # noqa: E402
from src.training.data import build_folds, evaluation_mask, in_variant_dtypes, load_complete_sibling, prepare_dataset  # noqa: E402
from src.training.imputation_baselines import BaselineImputer, MeanModeImputer  # noqa: E402
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines, score_cells  # noqa: E402
from src.training.tuned_baselines import SHRINKAGE, build, grid, shrink  # noqa: E402
from src.training.types import DatasetSpec, FoldSplit, PreparedDataset  # noqa: E402

THREADS = 4
FOLDS = 5
RATE = 0.2
FAMILIES = ("knn", "hgb", "lgbm")
WEAK_TIER = ("mean_mode", "knn_tuned", "hgb_tuned", "lgbm_tuned")
Population = tuple[pd.DataFrame, pd.DataFrame]


def _populations(
    dataset: PreparedDataset, sibling: pd.DataFrame | None, frame: pd.DataFrame, indices: np.ndarray, seed: int, ordinal: int
) -> dict[str, Population]:
    """A split's two scored populations, built as ``recompute_folds`` builds the test split's."""
    columns = list(frame.columns)
    categorical = list(dataset.categorical_columns)
    numbers, categories = _frame_values(frame, dataset)
    drawn = evaluation_mask(frame, RATE, seed, ordinal)
    found = {"masked": (frame, _cells((drawn.astype(object) == "[MASK]").to_numpy(), columns, numbers, categories, "masked"))}
    if sibling is not None:
        truth = in_variant_dtypes(sibling.iloc[indices].reset_index(drop=True)[columns].copy(), frame, categorical)
        if dataset.scaler is not None and len(dataset.numerical_columns):
            truth[list(dataset.numerical_columns)] = dataset.scaler.transform(truth[list(dataset.numerical_columns)])
        gaps = (frame.isna() & truth.notna()).to_numpy()
        truth_numbers, truth_categories = _frame_values(truth, dataset)
        found["induced"] = (frame, _cells(gaps, columns, truth_numbers, truth_categories, "induced"))
    return found


def _guesses(imputer: BaselineImputer, population: Population) -> np.ndarray:
    frame, cells = population
    rows = cells["row"].to_numpy(dtype=int)
    positions = frame.columns.get_indexer(pd.Index(cells["column"]))
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[rows, positions] = True
    return np.asarray(imputer.impute(frame, hidden).to_numpy(dtype=object)[rows, positions])


def _score(population: Population, guesses: np.ndarray, naive: Mapping[str, float | str], alpha: float) -> dict[str, float]:
    _, cells = population
    kinds = cells["kind"].to_numpy()
    imputed: list[float | str] = []
    for guess, kind, column in zip(guesses, kinds, cells["column"]):
        if kind == NUMERICAL:
            imputed.append(float(shrink(np.array(float(guess)), np.array(float(naive[column])), alpha)))
        else:
            imputed.append(str(guess))
    scored = score_cells(cells.assign(imputed=imputed), naive).metrics
    return {name: float(value) for name, value in scored.items() if not name.startswith("n_")}


def _fold(
    dataset: PreparedDataset, sibling: pd.DataFrame | None, seed: int, ordinal: int, fold: FoldSplit, level: int,
    families: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    split = {}
    for name, indices in (("train", fold.train_indices), ("validation", fold.validation_indices), ("test", fold.test_indices)):
        split[name] = (features.iloc[np.asarray(indices)].reset_index(drop=True), np.asarray(indices))
    train = split["train"][0]
    naive = mean_mode_baselines(train, numerical, categorical)
    validation = _populations(dataset, sibling, *split["validation"], seed, ordinal)
    test = _populations(dataset, sibling, *split["test"], seed, ordinal)
    alphas = SHRINKAGE if level >= 60 else (0.0,)
    metrics: dict[str, float] = {}
    choices: dict[str, str] = {}

    mean_mode = MeanModeImputer(numerical, categorical)
    mean_mode.fit(train)
    for population, cells in test.items():
        for name, value in _score(cells, _guesses(mean_mode, cells), naive, 0.0).items():
            metrics[f"impute/{population}/baseline/mean_mode/{name}"] = value

    for family in families:
        best: dict[str, tuple[float, str, float, BaselineImputer]] = {}
        for setting in grid(family):
            imputer = build(setting, numerical, categorical)
            imputer.fit(train)
            guesses = {population: _guesses(imputer, cells) for population, cells in validation.items()}
            for alpha in alphas:
                for signal, population in (("induced", "induced"), ("masked", "masked")):
                    if population not in validation:
                        continue
                    value = _score(validation[population], guesses[population], naive, alpha)["impute_score"]
                    if signal not in best or value < best[signal][0]:
                        best[signal] = (value, setting.label, alpha, imputer)
        for signal, suffix in (("induced", "_tuned"), ("masked", "_tuned_masked")):
            if signal not in best:
                continue
            value, label, alpha, imputer = best[signal]
            choices[f"{family}{suffix}"] = f"{label}_a{alpha}"
            for population, cells in test.items():
                for name, score in _score(cells, _guesses(imputer, cells), naive, alpha).items():
                    metrics[f"impute/{population}/baseline/{family}{suffix}/{name}"] = score
    return metrics, choices


def main(argv: list[str]) -> int:
    families: tuple[str, ...] = FAMILIES
    if argv[:1] == ["--families"]:
        families, argv = tuple(argv[1].split(",")), argv[2:]
    if len(argv) < 3 or not set(families) <= {"knn", "hgb", "lgbm", "missforest"}:
        sys.exit(__doc__)
    suffix = "" if families == FAMILIES else "_" + "_".join(families)
    out_dir, variant, seeds = Path(argv[0]), argv[1], [int(seed) for seed in argv[2:]]
    (out_dir / f"csv{suffix}").mkdir(parents=True, exist_ok=True)
    (out_dir / f"choices{suffix}").mkdir(parents=True, exist_ok=True)
    missforest.THREADS = THREADS
    lightgbm_baseline.THREADS = THREADS
    warnings.filterwarnings("ignore")
    match = re.search(r"_(\d+)nan$", variant)
    level = int(match.group(1)) if match else 0
    spec = DatasetSpec.from_name(variant, None)
    dataset = prepare_dataset(spec)
    sibling = load_complete_sibling(spec)
    for seed in seeds:
        path = out_dir / f"csv{suffix}" / f"{variant}_s{seed}.csv"
        if path.exists():
            print(f"SKIP {path.name}: done", flush=True)
            continue
        started = time.time()
        folds: dict[int, dict[str, float]] = {}
        chosen: dict[int, dict[str, str]] = {}
        for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, FOLDS, seed), start=1):
            folds[ordinal], chosen[ordinal] = _fold(dataset, sibling, seed, ordinal, fold, level, families)
        for population in ("masked", "induced"):
            means = {
                name: float(np.mean([folds[o][f"impute/{population}/baseline/{name}/impute_score"] for o in folds]))
                for name in WEAK_TIER
                if all(f"impute/{population}/baseline/{name}/impute_score" in folds[o] for o in folds)
            }
            if not means:
                continue
            winner = min(means, key=lambda name: (means[name], name))
            for ordinal in folds:
                folds[ordinal][f"impute/{population}/baseline/tuned_bar/impute_score"] = folds[ordinal][f"impute/{population}/baseline/{winner}/impute_score"]
                chosen[ordinal][f"tuned_bar/{population}"] = winner
        columns = sorted({key for metrics in folds.values() for key in metrics})
        partial = path.with_suffix(".partial")
        with partial.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", "seed", "fold", *columns])
            for ordinal, metrics in sorted(folds.items()):
                writer.writerow([variant, seed, ordinal, *(metrics.get(column, "") for column in columns)])
        partial.rename(path)
        labels = sorted({key for choice in chosen.values() for key in choice})
        with (out_dir / f"choices{suffix}" / f"{variant}_s{seed}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", "seed", "fold", *labels])
            for ordinal, choice in sorted(chosen.items()):
                writer.writerow([variant, seed, ordinal, *(choice.get(label, "") for label in labels)])
        print(f"DONE {variant} s{seed}: {len(folds)} folds, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
