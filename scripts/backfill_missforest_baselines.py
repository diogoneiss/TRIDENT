#!/usr/bin/env python3
"""Backfill the missForest baselines of ADR 0016 onto runs recorded before them.

Like every baseline, ``missforest`` and ``missforest_lgbm`` depend only on a run's dataset,
seed, fold count and evaluation mask rates, never on its model, so they are recomputed
without training, on the same folds and cells as ``scripts/backfill_baseline_imputers.py``
recomputes the ADR 0007 baselines, and written where a live run would have: the seven
cross-validation statistics on the parent at step 0, and each diagnostic child's fold value.

Only the two missForests and the mean/mode baseline are recomputed. The mean/mode baseline
is the check: a run is written only if its recomputed mean/mode reproduces, within 1e-9,
every mean/mode mean the run logged, on every population and metric, which says these are
the run's own cells, truths and naive fills. The KNN and gradient-boosting baselines are
not recomputed (an hour of KNN transforms on electricity alone) and so not checked again.

Where a missForest beats the run's best baseline on a population, the best baseline moves
with it: every statistic under ``baseline/best`` is rewritten as the new best's, the tag
``best_baseline/<population>`` renamed, and the gap to it (``gap_to_best_baseline`` and its
percentage) taken again fold by fold, from the model's values in the run's
``metrics/raw_fold_metrics.csv`` and the recomputed ones. They are logged again at step 0;
MLflow's latest value is the newer one, and the history keeps both. The run's
``cv_summary.json`` artifact is not rewritten, as no backfill before this one rewrote it.
Every tree written is tagged ``missforest_backfilled=true`` and re-mirrored (ADR 0006).

The scores are cached where a live run caches them (``results/baseline_cache/``, ADR 0007
decision 10), so runs that share a fold pay once, a rerun after a kill resumes, and a later
live run on the same fold finds them. Dry run by default; runs are taken smallest dataset
first. ``--dataset`` (repeatable) restricts a pass to some tables, so dry runs can fill the
cache in parallel before a single ``--apply`` writes the store.

    uv run --python 3.11 python scripts/backfill_missforest_baselines.py [--dataset kc2_20nan]
    uv run --python 3.11 python scripts/backfill_missforest_baselines.py --apply
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import Run  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

import scripts.backfill_baseline_imputers as adr0007  # noqa: E402
from src.mlflow_utils import MISSFOREST_BACKFILLED_TAG, setup_mlflow  # noqa: E402
from src.training.baseline_cache import BaselineScorer  # noqa: E402
from src.training.imputation_baselines import MeanModeImputer, score_baselines  # noqa: E402
from src.training.mirroring import mirror_run_tree  # noqa: E402
from src.training.summary import GAP_TO_BEST_BASELINE  # noqa: E402
from src.training.types import DEFAULT_BASELINE_CACHE_DIR  # noqa: E402

NEW_BASELINES = ("missforest", "missforest_lgbm")
_CHECKED = frozenset({"mean_mode"})
_BEST_TAG = "best_baseline/"


@dataclass
class Plan:
    """What the backfill would write to one run tree, or why it will not.

    ``moved`` names, per population whose best baseline changes, the old and the new.
    """

    run_id: str
    name: str
    parent: dict[str, float] = field(default_factory=dict)
    children: dict[str, dict[str, float]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    moved: dict[str, tuple[str, str]] = field(default_factory=dict)
    refused: str | None = None


def plan_missforest(
    run_id: str,
    name: str,
    logged: Mapping[str, float],
    tags: Mapping[str, str],
    folds: Mapping[int, Mapping[str, float]],
    children: Mapping[str, int],
    rows: Mapping[int, Mapping[str, float]],
) -> Plan:
    """The missForest statistics a run lacks, and its best baseline and gap where they move.

    ``logged`` and ``tags`` are the parent's, ``folds`` the recomputed mean/mode and
    missForest fold values, ``children`` each diagnostic child with the fold it replays, and
    ``rows`` the run's own fold values as ``raw_fold_metrics.csv`` keeps them.
    """
    plan = Plan(run_id, name)
    base = adr0007.plan_run(run_id, name, logged, folds, children, checked=_CHECKED)
    if base.refused is not None:
        plan.refused = base.refused
        return plan

    after = {**logged, **base.parent}
    current = {key[len(_BEST_TAG):]: value for key, value in tags.items() if key.startswith(_BEST_TAG)}
    _, chosen = adr0007.best_statistics(after, adr0007._baseline_populations(after))
    moved = {
        tag[len(_BEST_TAG):]: (current.get(tag[len(_BEST_TAG):], ""), best)
        for tag, best in chosen.items()
        if current.get(tag[len(_BEST_TAG):]) != best
    }
    best_metrics, best_tags = adr0007.best_statistics(after, moved)
    gap: dict[str, float] = {}
    if moved:
        merged = {fold: {**folds.get(fold, {}), **values} for fold, values in rows.items()}
        gap, refusal = adr0007.plan_gap(
            {**after, **best_metrics},
            {population: new for population, (_, new) in moved.items()},
            merged,
            redo=moved,
        )
        if refusal is not None:
            plan.refused = f"best baseline moved but the gap cannot be taken again: {refusal}"
            return plan

    plan.parent = {**base.parent, **best_metrics, **gap}
    plan.children = base.children
    plan.tags = best_tags
    plan.moved = moved
    return plan


def apply_missforest_plan(client: MlflowClient, plan: Plan) -> None:
    """Write a plan where a live run would have (step 0, each run's own id), mark every run
    of the tree, rename the moved best baselines, and refresh the tree's mirror."""
    if plan.refused is not None:
        return
    for run_id, metrics in [(plan.run_id, plan.parent), *plan.children.items()]:
        adr0007._write_at_step_zero(client, run_id, metrics)
        client.set_tag(run_id, MISSFOREST_BACKFILLED_TAG, "true")
    for key, value in plan.tags.items():
        client.set_tag(plan.run_id, key, value)
    mirror_run_tree(client, plan.run_id)


def _missforest_scorer(cache_dir: Path | None) -> adr0007.FoldScorerFactory:
    """A fold's mean/mode baseline, computed (seconds), and its missForests, from the cache
    a live run reads when it can; never the ADR 0007 learned baselines."""

    def make(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> adr0007.FoldScorer:
        naive = MeanModeImputer(numerical, categorical)
        naive.fit(train)
        scorer = BaselineScorer(train, numerical, categorical, cache_dir, families=("missforest",))

        def score(test: pd.DataFrame, cells: pd.DataFrame, fills: Mapping[str, float | str]) -> Mapping[str, float]:
            return {
                **score_baselines([naive], test, cells, fills).metrics,
                **scorer.score(test, cells, fills).metrics,
            }

        return score

    return make


def _lacks_missforest(run: Run) -> bool:
    populations = {
        match.group(1)
        for key in run.data.metrics
        if (match := adr0007._BASELINE_MEAN.fullmatch(key)) is not None
    }
    return any(
        f"cv/test/{population}/baseline/{baseline}/impute_score/mean" not in run.data.metrics
        for population in populations
        for baseline in NEW_BASELINES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--run", action="append", default=None, help="Restrict to this parent run id (repeatable).")
    parser.add_argument("--dataset", action="append", default=None, help="Restrict to this variant (repeatable).")
    parser.add_argument("--no_cache", action="store_true", help="Neither read nor fill results/baseline_cache/.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    args = parser.parse_args(argv)

    mode = "APPLIED" if args.apply else "DRY RUN"
    cache_dir = None if args.no_cache else Path(__file__).resolve().parents[1] / DEFAULT_BASELINE_CACHE_DIR
    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = sorted(
        (
            run
            for run in adr0007._imputation_parents(client, args.run)
            if _lacks_missforest(run)
            and (not args.dataset or run.data.params.get("dataset_name") in args.dataset)
        ),
        key=adr0007._dataset_bytes,
    )
    print(f"[{mode}] tracking URI: {tracking_uri}; {len(runs)} run(s) lack the missForest baselines", flush=True)
    written = refused = moved = 0
    for run in runs:
        name = run.info.run_name or run.info.run_id
        started = time.perf_counter()
        naive_refusal = adr0007._naive_refusal(run, name)
        if naive_refusal is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {naive_refusal}", flush=True)
            continue
        folds = adr0007.recompute_folds(*adr0007._arguments(run), fold_scorer=_missforest_scorer(cache_dir))
        rows = adr0007._fold_rows(client, run.info.run_id)
        plan = plan_missforest(
            run.info.run_id, name, run.data.metrics, run.data.tags, folds, adr0007._children(client, run), rows
        )
        seconds = time.perf_counter() - started
        if plan.refused is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {plan.refused}", flush=True)
            continue
        shifts = "; ".join(
            f"{population} best {old} -> {new}, gap "
            f"{plan.parent[f'cv/test/{population}/{GAP_TO_BEST_BASELINE}/impute_score/mean']:+.4f}"
            for population, (old, new) in sorted(plan.moved.items())
        )
        print(
            f"[{mode}] {name}: +{len(plan.parent)} parent metrics on {len(plan.children)} children"
            f"{'; ' + shifts if shifts else ''} ({seconds:.0f} s)",
            flush=True,
        )
        moved += bool(plan.moved)
        if args.apply:
            apply_missforest_plan(client, plan)
            written += 1
    print(
        f"[{mode}] {written} run tree(s) written and re-mirrored, {refused} refused, "
        f"{moved} with a best baseline that moved to a missForest",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
