#!/usr/bin/env python3
"""Backfill ADR 0021 onto runs recorded before it: the linear arms and the strong tier's best.

ADR 0021 logs every baseline in two tiers.

- **The weak tier** is the bar of ADR 0007 and ADR 0017, unchanged.
- **The strong tier** is missForest plus condexp, SoftImpute and ICE
  (``src/training/linear_baselines.py``), with its own best and gap.

Like every baseline, the arms depend only on a run's dataset, seed, fold count and evaluation
mask rates, never on its model. So they are recomputed without training, on the same folds and
cells as ``scripts/backfill_baseline_imputers.py`` recomputes, and written where a live run would
have written them:

- the seven cross-validation statistics on the parent, at step 0;
- each diagnostic child's fold value.

With them, per run and population:

- the strong tier's best, chosen among the strong baselines the run then carries by the lowest
  mean ``impute_score``, a tie to the name that sorts first (``summary._best_strong_baselines``),
  copied under ``baseline/best_strong``, with the tag ``best_strong_baseline/<population>``;
- ``gap_to_best_strong_baseline`` and its ``_pct``, the model's headline minus that baseline, fold
  by fold (``summary.best_strong_baseline_gaps``). The model's fold values come from the run's
  ``raw_fold_metrics.csv`` and the baseline's from the recomputation. Both must average to the
  run's own means within the tolerance.

**The check that these are the run's own cells.** Every family the gap needs (ADR 0007's, for the
check, missForest and the arms) is recomputed through the live cache. A run is written only if the
recomputation reproduces every mean/mode mean it logged within 1e-9.

**Nothing moves.** The bar, `baseline/best` and the gap to it are untouched.

**Marking and mirroring.** A tree is marked ``strong_tier_backfilled=true``, the parent last, and
re-mirrored (ADR 0006).

**How to run it.** Dry run by default; runs are taken smallest dataset first. ``--dataset``
(repeatable) splits a pass, so dry runs fill the cache in parallel before a single ``--apply``.
``--check_against`` compares the planned arms with E41's offline per-fold CSVs, which must agree to
the last digit on the same machine. A backup through SQLite's API comes first, in a gap between
studies: the pass holds the store for about 2 h.

    uv run --python 3.11 python scripts/backfill_strong_tier.py [--dataset kc2_20nan] [--check_against DIR]
    uv run --python 3.11 python scripts/backfill_strong_tier.py --apply
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.entities import Run  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

import scripts.backfill_baseline_imputers as adr0007  # noqa: E402
from src.mlflow_utils import STRONG_TIER_BACKFILLED_TAG, setup_mlflow  # noqa: E402
from src.training.baseline_cache import BaselineScorer  # noqa: E402
from src.training.mirroring import MirrorIndex, mirror_run_tree  # noqa: E402
from src.training.summary import (  # noqa: E402
    BEST_STRONG_BASELINE,
    GAP_TO_BEST_STRONG_BASELINE,
    STRONG_BASELINES,
    best_strong_baseline_gaps,
)
from src.training.tracking import _summary_metrics  # noqa: E402
from src.training.types import DEFAULT_BASELINE_CACHE_DIR  # noqa: E402

NEW_BASELINES = ("condexp", "softimpute", "ice")
_FAMILIES = ("adr0007", "missforest", "linear")
_CHECKED = frozenset({"mean_mode"})
_STRONG_TAG = "best_strong_baseline/"


@dataclass
class Plan:
    """What the backfill would write to one run tree, or why it will not."""

    run_id: str
    name: str
    parent: dict[str, float] = field(default_factory=dict)
    children: dict[str, dict[str, float]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    refused: str | None = None


def plan_strong_tier(
    run_id: str,
    name: str,
    logged: Mapping[str, float],
    folds: Mapping[int, Mapping[str, float]],
    children: Mapping[str, int],
    rows: Mapping[int, Mapping[str, float]],
    fold_tolerance: float = adr0007.TOLERANCE,
) -> Plan:
    """What ADR 0021 writes to one run: the arms, the strong tier's best, and the gap to it.

    ``logged`` is the parent's latest metrics, ``folds`` every recomputed baseline's fold values,
    ``children`` each diagnostic child with the fold it replays, and ``rows`` the run's own fold
    values as ``raw_fold_metrics.csv`` keeps them.
    """
    plan = Plan(run_id, name)
    base = adr0007.plan_run(run_id, name, logged, folds, children, checked=_CHECKED)
    if base.refused is not None:
        plan.refused = base.refused
        return plan
    # Only the arms are written; a family recomputed for the gap's sake but already logged
    # (missForest) keeps the run's own numbers.
    arms = tuple(f"/baseline/{arm}/" for arm in NEW_BASELINES)
    plan.parent = {key: value for key, value in base.parent.items() if any(part in key for part in arms)}
    plan.children = {
        child: {key: value for key, value in metrics.items() if any(part in key for part in arms)}
        for child, metrics in base.children.items()
    }
    after = {**logged, **plan.parent}
    ordered = [folds[fold] for fold in sorted(folds)]
    for population in sorted(adr0007._baseline_populations(after)):
        scored = sorted(
            (after[f"cv/test/{population}/baseline/{arm}/impute_score/mean"], arm)
            for arm in STRONG_BASELINES
            if f"cv/test/{population}/baseline/{arm}/impute_score/mean" in after
        )
        if not scored:
            continue
        best = scored[0][1]
        source = f"cv/test/{population}/baseline/{best}/"
        target = f"cv/test/{population}/baseline/{BEST_STRONG_BASELINE}/"
        plan.parent.update({target + key[len(source):]: value for key, value in after.items() if key.startswith(source)})
        plan.tags[f"{_STRONG_TAG}{population}"] = best
        model_key = f"{population}/impute_score"
        baseline_key = f"{population}/baseline/{best}/impute_score"
        model_values = [rows.get(fold, {}).get(model_key) for fold in sorted(folds)]
        baseline_values = [fold.get(baseline_key) for fold in ordered]
        if any(value is None for value in (*model_values, *baseline_values)):
            plan.refused = f"no fold values for the gap on {population}"
            return plan
        for values, logged_mean in (
            (model_values, f"cv/test/{model_key}/mean"),
            (baseline_values, f"cv/test/{baseline_key}/mean"),
        ):
            found = [float(value) for value in values if value is not None]
            if logged_mean not in after or abs(float(np.mean(found)) - after[logged_mean]) > fold_tolerance:
                plan.refused = f"{logged_mean}: fold mean {float(np.mean(found)):.6f} against {after.get(logged_mean)}"
                return plan
        pairs = [{model_key: float(m), baseline_key: float(b)} for m, b in zip(model_values, baseline_values) if m is not None and b is not None]
        for key, statistics in best_strong_baseline_gaps(pairs, {population: best}).items():
            plan.parent.update(_summary_metrics(f"cv/test/{key}", statistics))
    return plan


def apply_strong_tier_plan(client: MlflowClient, plan: Plan, index: MirrorIndex | None = None) -> None:
    """Write a plan where a live run would have, mark the tree (the parent last) and re-mirror it."""
    if plan.refused is not None:
        return
    for run_id, metrics in plan.children.items():
        adr0007._write_at_step_zero(client, run_id, metrics)
        client.set_tag(run_id, STRONG_TIER_BACKFILLED_TAG, "true")
    adr0007._write_at_step_zero(client, plan.run_id, plan.parent)
    for key, value in plan.tags.items():
        client.set_tag(plan.run_id, key, value)
    client.set_tag(plan.run_id, STRONG_TIER_BACKFILLED_TAG, "true")
    mirror_run_tree(client, plan.run_id, index=index)


def _scorer(cache_dir: Path | None) -> adr0007.FoldScorerFactory:
    def make(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> adr0007.FoldScorer:
        scorer = BaselineScorer(train, numerical, categorical, cache_dir, families=_FAMILIES)

        def score(test: pd.DataFrame, cells: pd.DataFrame, fills: Mapping[str, float | str]) -> Mapping[str, float]:
            return scorer.score(test, cells, fills).metrics

        return score

    return make


def _pending(run: Run) -> bool:
    """A run this pass has still to write: unmarked, carrying ADR 0007's baselines, lacking an arm."""
    if run.data.tags.get(STRONG_TIER_BACKFILLED_TAG) == "true":
        return False
    populations = {
        match.group(1) for key in run.data.metrics if (match := adr0007._BASELINE_MEAN.fullmatch(key)) is not None
    }
    return bool(populations) and any(
        f"cv/test/{population}/baseline/{arm}/impute_score/mean" not in run.data.metrics
        for population in populations
        for arm in NEW_BASELINES
    )


def _as_planned_from(metrics: Mapping[str, float]) -> dict[str, float]:
    """The run's metrics without what this pass writes, so a tree cut short is planned again."""
    dropped = [f"/baseline/{arm}/" for arm in NEW_BASELINES] + [
        f"/baseline/{BEST_STRONG_BASELINE}/", f"/{GAP_TO_BEST_STRONG_BASELINE}/", f"/{GAP_TO_BEST_STRONG_BASELINE}_pct/"
    ]
    return {key: value for key, value in metrics.items() if not any(part in key for part in dropped)}


def _offline_mismatch(plan: Plan, variant: str, seed: int, offline: Path) -> str | None:
    """How far the planned arms' means are from E41's offline CSV of the same cell, if it has one."""
    path = offline / f"{variant}_s{seed}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    worst = 0.0
    for key, value in plan.parent.items():
        if not key.endswith("/mean") or not any(f"/baseline/{arm}/" in key for arm in NEW_BASELINES):
            continue
        column = key[len("cv/test/"):-len("/mean")]
        if column in frame.columns:
            worst = max(worst, abs(float(frame[column].mean()) - value))
    return f"{worst:.2g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--run", action="append", default=None, help="Restrict to this parent run id (repeatable).")
    parser.add_argument("--dataset", action="append", default=None, help="Restrict to this variant (repeatable).")
    parser.add_argument("--no_cache", action="store_true", help="Neither read nor fill results/baseline_cache/.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="The baseline cache to read and fill.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    parser.add_argument("--check_against", type=Path, default=None, help="E41's per-fold CSVs, to compare the arms with.")
    parser.add_argument("--fold_tolerance", type=float, default=adr0007.TOLERANCE)
    args = parser.parse_args(argv)

    mode = "APPLIED" if args.apply else "DRY RUN"
    cache_dir = None if args.no_cache else (args.cache_dir or Path(__file__).resolve().parents[1] / DEFAULT_BASELINE_CACHE_DIR)
    tracking_uri = setup_mlflow(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = sorted(
        (
            run
            for run in adr0007._imputation_parents(client, args.run)
            if _pending(run) and (not args.dataset or run.data.params.get("dataset_name") in args.dataset)
        ),
        key=adr0007._dataset_bytes,
    )
    print(f"[{mode}] tracking URI: {tracking_uri}; {len(runs)} run(s) to bring to ADR 0021", flush=True)
    written = refused = 0
    index = MirrorIndex(client) if args.apply else None
    for run in runs:
        name = run.info.run_name or run.info.run_id
        started = time.perf_counter()
        naive_refusal = adr0007._naive_refusal(run, name)
        if naive_refusal is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {naive_refusal}", flush=True)
            continue
        arguments = adr0007._arguments(run)
        folds = adr0007.recompute_folds(*arguments, fold_scorer=_scorer(cache_dir))
        plan = plan_strong_tier(
            run.info.run_id, name, _as_planned_from(run.data.metrics), folds,
            adr0007._children(client, run), adr0007._fold_rows(client, run.info.run_id), args.fold_tolerance,
        )
        seconds = time.perf_counter() - started
        if plan.refused is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {plan.refused}", flush=True)
            continue
        offline = _offline_mismatch(plan, arguments[0], arguments[1], args.check_against) if args.check_against else None
        bests = ", ".join(f"{tag[len(_STRONG_TAG):]} {best}" for tag, best in sorted(plan.tags.items()))
        print(
            f"[{mode}] {name}: +{len(plan.parent)} parent metrics on {len(plan.children)} children; strong best {bests}"
            f"{'; against E41 ' + offline if offline is not None else ''} ({seconds:.0f} s)",
            flush=True,
        )
        if args.apply:
            apply_strong_tier_plan(client, plan, index)
            written += 1
    print(f"[{mode}] {written} run tree(s) written and re-mirrored, {refused} refused", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
