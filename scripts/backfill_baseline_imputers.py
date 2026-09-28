#!/usr/bin/env python3
"""Backfill the baseline imputers of ADR 0007 onto runs that predate the current set.

The baselines depend only on a run's dataset, seed, fold count and evaluation mask
rates, never on its model, so they can be recomputed without training: the same folds,
the same hidden cells, the same truths, the same imputers the decode stage fits. This
script does that for every finished imputation parent that logged baselines but lacks
some of the current ones (``mean_mode``, ``knn5``, ``knn10``, ``hgb``), and writes the
missing ones where a live run would have: the seven cross-validation statistics on the
parent at step 0, and each diagnostic child's own fold value.

A run is written only if the recomputation reproduces, within 1e-6, every baseline mean
the run already logged (a legacy ``knn`` is checked as ``knn5``). A run whose own numbers
cannot be reproduced is refused and reported: its cells were not the ones recomputed
here, as with the electricity run scored before the spelling fix of 2026-09-27. Every
run written is tagged ``baselines_backfilled=true``, nothing already logged is written
again, and each written tree is re-mirrored (ADR 0006).

Usage (dry run by default, honours ``MLFLOW_TRACKING_URI``):

    uv run --python 3.11 python scripts/backfill_baseline_imputers.py
    uv run --python 3.11 python scripts/backfill_baseline_imputers.py --cache <dir> --apply

``--cache`` keeps each run's recomputed folds as JSON so a dry run and the ``--apply``
after it pay for the imputers once; ``--run <id>`` (repeatable) restricts the pass. Runs
are taken smallest dataset first, so on a machine short of memory the cache fills as far
as it can before the large tables, and a rerun resumes from it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import Metric, Run  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.embedder import as_category_strings  # noqa: E402
from src.mlflow_utils import BASELINES_BACKFILLED_TAG, IS_MIRROR_TAG, TASK_TAG, setup_mlflow  # noqa: E402
from src.training.data import (  # noqa: E402
    PROCESSED_DATASETS,
    build_folds,
    evaluation_mask,
    in_variant_dtypes,
    load_complete_sibling,
    prepare_dataset,
)
from src.training.imputation_baselines import fit_baseline_imputers, score_baselines  # noqa: E402
from src.training.imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines  # noqa: E402
from src.training.mirroring import mirror_run_tree  # noqa: E402

# The summariser's and the tracker's own helpers, so a backfilled statistic is computed and
# named exactly as a live run's is.
from src.training.summary import _summarize_metric  # noqa: E402
from src.training.tracking import _summary_metrics  # noqa: E402
from src.training.types import DatasetSpec, PreparedDataset  # noqa: E402

CURRENT_BASELINES = ("mean_mode", "knn5", "knn10", "hgb")
# A name a run logged before the first amendment, and what it is called now.
LEGACY_NAMES = {"knn": "knn5"}
TOLERANCE = 1e-6
_BASELINE_MEAN = re.compile(r"cv/test/(impute/.+)/baseline/([^/]+)/([^/]+)/mean")
_EXTRA_RATE = re.compile(r"cv/test/impute/masked/rate_(\d+)/impute_score/mean")
_METRICS_PER_BATCH = 1000
_CELL_COLUMNS = ["row", "column", "kind", "population", "actual", "imputed", "confidence", "actual_original"]

FoldMetrics = dict[int, dict[str, float]]


@dataclass
class RunPlan:
    """What the backfill would write to one run tree, or why it will not."""

    run_id: str
    name: str
    parent: dict[str, float] = field(default_factory=dict)
    children: dict[str, dict[str, float]] = field(default_factory=dict)
    refused: str | None = None
    checked: int = 0


def plan_run(
    run_id: str,
    name: str,
    logged: Mapping[str, float],
    folds: Mapping[int, Mapping[str, float]],
    children: Mapping[str, int],
) -> RunPlan:
    """Check the recomputation against the run's own baselines, then list what is missing.

    ``logged`` is the parent's latest metrics, ``folds`` the recomputed per-fold metrics
    keyed ``impute/<population>/baseline/<name>/<metric>``, and ``children`` each
    diagnostic child's run id with the fold it replays.
    """
    plan = RunPlan(run_id, name)
    ordered = [folds[fold] for fold in sorted(folds)]
    for key, value in sorted(logged.items()):
        match = _BASELINE_MEAN.fullmatch(key)
        if match is None:
            continue
        population, baseline, metric = match.groups()
        fold_key = f"{population}/baseline/{LEGACY_NAMES.get(baseline, baseline)}/{metric}"
        values = [fold.get(fold_key) for fold in ordered]
        if not values or any(entry is None for entry in values):
            plan.refused = f"{key} is logged but was not recomputed"
            return plan
        recomputed = float(np.mean([float(entry) for entry in values if entry is not None]))
        if abs(recomputed - value) > TOLERANCE:
            plan.refused = f"{key}: logged {value:.6f}, recomputed {recomputed:.6f}"
            return plan
        plan.checked += 1
    if plan.checked == 0:
        plan.refused = "no logged baseline to check the recomputation against"
        return plan

    for fold_key in sorted(ordered[0]):
        if f"cv/test/{fold_key}/mean" in logged:
            continue
        plan.parent.update(
            _summary_metrics(f"cv/test/{fold_key}", _summarize_metric([fold[fold_key] for fold in ordered]))
        )
        for child_id, fold in children.items():
            plan.children.setdefault(child_id, {})[f"test/{fold_key}"] = float(folds[fold][fold_key])
    return plan


def apply_plan(client: MlflowClient, plan: RunPlan) -> None:
    """Write a plan where a live run would have: step 0, the run's own id, tagged."""
    if plan.refused is not None or not plan.parent:
        return
    stamp = int(time.time() * 1000)
    for run_id, metrics in [(plan.run_id, plan.parent), *plan.children.items()]:
        entries = [Metric(key, float(value), stamp, 0) for key, value in sorted(metrics.items())]
        for start in range(0, len(entries), _METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=entries[start : start + _METRICS_PER_BATCH])
        client.set_tag(run_id, BASELINES_BACKFILLED_TAG, "true")


def _cells(
    positions: np.ndarray,
    columns: Sequence[str],
    numbers: Mapping[str, np.ndarray],
    categories: Mapping[str, np.ndarray],
    population: str,
) -> pd.DataFrame:
    """The scored-cell table the decode stage builds, without the model's guesses.

    Numerical truths go through float32, as the embedder's tensors do, so a recomputed
    error matches a logged one to the last bit a float32 carries.
    """
    rows = []
    for row, position in zip(*np.nonzero(positions)):
        column = columns[position]
        if column in categories:
            actual: float | str = str(categories[column][row])
            kind = CATEGORICAL
        else:
            actual = float(np.float32(numbers[column][row]))
            kind = NUMERICAL
        rows.append(
            {"row": int(row), "column": column, "kind": kind, "population": population,
             "actual": actual, "imputed": actual, "confidence": float("nan"), "actual_original": actual}
        )
    return pd.DataFrame(rows, columns=_CELL_COLUMNS)


def _frame_values(frame: pd.DataFrame, dataset: PreparedDataset) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    numbers = {column: frame[column].to_numpy(dtype=float) for column in dataset.numerical_columns}
    categories = {column: as_category_strings(frame[column]) for column in dataset.categorical_columns}
    return numbers, categories


def recompute_folds(
    dataset_name: str, seed: int, cv_folds: int, rate: float, extra_rates: Sequence[float]
) -> FoldMetrics:
    """Every current baseline's metrics on each fold's populations, as the run scored them."""
    spec = DatasetSpec.from_name(dataset_name, None)
    dataset = prepare_dataset(spec)
    sibling = load_complete_sibling(spec)
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    folds: FoldMetrics = {}
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, cv_folds, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = mean_mode_baselines(train, numerical, categorical)
        imputers = fit_baseline_imputers(train, numerical, categorical)
        columns = list(test.columns)
        numbers, categories = _frame_values(test, dataset)
        metrics: dict[str, float] = {}
        for at_rate, prefix in [(rate, "impute/masked")] + [
            (extra, f"impute/masked/rate_{round(extra * 100)}") for extra in extra_rates
        ]:
            drawn = evaluation_mask(test, at_rate, seed, ordinal)
            hidden = (drawn.astype(object) == "[MASK]").to_numpy()
            cells = _cells(hidden, columns, numbers, categories, "masked")
            scored = score_baselines(imputers, test, cells, naive).metrics
            metrics.update({f"{prefix}/{key}": value for key, value in scored.items()})
        if sibling is not None:
            truth = in_variant_dtypes(
                sibling.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)[columns].copy(), test, categorical
            )
            if dataset.scaler is not None and numerical:
                truth[numerical] = dataset.scaler.transform(truth[numerical])
            gaps = (test.isna() & truth.notna()).to_numpy()
            truth_numbers, truth_categories = _frame_values(truth, dataset)
            cells = _cells(gaps, columns, truth_numbers, truth_categories, "induced")
            scored = score_baselines(imputers, test, cells, naive).metrics
            metrics.update({f"impute/induced/{key}": value for key, value in scored.items()})
        folds[ordinal] = metrics
    return folds


def _needs_backfill(metrics: Mapping[str, float]) -> bool:
    """A run that logged ADR 0007 baselines but not every current one."""
    populations = {
        match.group(1)
        for key in metrics
        if (match := _BASELINE_MEAN.fullmatch(key)) is not None
    }
    return bool(populations) and any(
        f"cv/test/{population}/baseline/{name}/impute_score/mean" not in metrics
        for population in populations
        for name in CURRENT_BASELINES
    )


def candidate_runs(client: MlflowClient, only: Sequence[str] | None) -> list[Run]:
    """Finished imputation parents in the experiment families, never mirrors."""
    experiments = [
        experiment.experiment_id
        for experiment in client.search_experiments()
        if experiment.name.startswith("TRIDENT/") and not experiment.name.startswith("TRIDENT/mirror/")
    ]
    runs = client.search_runs(
        experiments,
        filter_string=(
            f"tags.{TASK_TAG} = 'imputation' and tags.run_role = 'parent' "
            f"and tags.{IS_MIRROR_TAG} = 'false' and attributes.status = 'FINISHED'"
        ),
        max_results=50_000,
    )
    return [run for run in runs if (not only or run.info.run_id in only) and _needs_backfill(run.data.metrics)]


def _dataset_bytes(run: Run) -> int:
    """How large a run's table is on disk, to take the cheap recomputations first."""
    name = run.data.params.get("dataset_name", "")
    path = PROCESSED_DATASETS / name.split("_", 1)[0] / f"{name}.csv"
    return path.stat().st_size if path.exists() else 0


def _children(client: MlflowClient, run: Run) -> dict[str, int]:
    kids = client.search_runs(
        [run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'"
    )
    return {kid.info.run_id: int(kid.data.tags["fold"]) for kid in kids if kid.data.tags.get("fold", "").isdigit()}


def _recompute_cached(run: Run, cache: Path | None) -> FoldMetrics:
    params = run.data.params
    extra = sorted(
        int(match.group(1)) / 100
        for key in run.data.metrics
        if (match := _EXTRA_RATE.fullmatch(key)) is not None
    )
    path = None if cache is None else cache / f"{run.info.run_id}.json"
    if path is not None and path.exists():
        return {int(fold): metrics for fold, metrics in json.loads(path.read_text()).items()}
    folds = recompute_folds(
        params["dataset_name"], int(params["seed"]), int(params["cv_folds"]), float(params["EVAL_MASK_RATE"]), extra
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(folds))
    return folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--run", action="append", default=None, help="Restrict to this parent run id (repeatable).")
    parser.add_argument("--cache", type=Path, default=None, help="Keep recomputed folds here as JSON, one file per run.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    args = parser.parse_args(argv)

    mode = "APPLIED" if args.apply else "DRY RUN"
    # KNN's distances are computed in chunks of this many megabytes; the default 1024 is
    # what a 45,000-row table on a 16 GB machine cannot always spare. Chunking never
    # changes a distance, only how many are held at once.
    sklearn.set_config(working_memory=128)
    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = sorted(candidate_runs(client, args.run), key=_dataset_bytes)
    print(f"[{mode}] tracking URI: {tracking_uri}; {len(runs)} run(s) lack part of the current baseline set")
    written = refused = 0
    for run in runs:
        name = run.info.run_name or run.info.run_id
        started = time.perf_counter()
        folds = _recompute_cached(run, args.cache)
        children = _children(client, run)
        plan = plan_run(run.info.run_id, name, run.data.metrics, folds, children)
        seconds = time.perf_counter() - started
        if plan.refused is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {plan.refused}", flush=True)
            continue
        per_child = sorted({len(metrics) for metrics in plan.children.values()})
        print(
            f"[{mode}] {name}: reproduced {plan.checked} logged baseline means; "
            f"+{len(plan.parent)} parent metrics, +{per_child} per child on {len(plan.children)} children "
            f"({seconds:.0f} s)",
            flush=True,
        )
        if args.apply:
            apply_plan(client, plan)
            mirror_run_tree(client, run.info.run_id)
            written += 1
    print(f"[{mode}] {written} run tree(s) written and re-mirrored, {refused} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
