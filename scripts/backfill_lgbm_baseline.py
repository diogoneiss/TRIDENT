#!/usr/bin/env python3
"""Backfill ADR 0017 onto runs recorded before it: the ``lgbm`` baseline joins the bar, and
missForest leaves it for a comparison of its own.

Like every baseline, ``lgbm`` depends only on a run's dataset, seed, fold count and
evaluation mask rates, never on its model, so it is recomputed without training, on the
same folds and cells as ``scripts/backfill_baseline_imputers.py`` recomputes, and written
where a live run would have: the seven cross-validation statistics on the parent at step 0,
and each diagnostic child's fold value. With it, per run:

- ``gap_to_missforest`` and ``gap_to_missforest_pct``, the model's paired gap to missForest
  fold by fold (``summary.external_baseline_gaps``), on the parent;
- the best baseline chosen again among the bar's baselines, missForest and the withdrawn
  ``missforest_lgbm`` left out (``summary.OUTSIDE_THE_BAR``). Where that choice differs from
  the run's ``best_baseline/<population>`` tag, every statistic under ``baseline/best`` is
  rewritten as the chosen one's, the tag renamed, and the gap to the bar taken again fold by
  fold. On the runs whose bar ADR 0016's backfill moved to missForest this moves it back.

Every baseline is recomputed through the live cache (all three families), and a run is
written only if that recomputation reproduces, within 1e-9, every mean/mode and missForest
mean it logged: these are its own cells, and the edited missForest module still gives the
numbers ADR 0016 logged. KNN and hgb are recomputed but not checked: on runs recorded on
the Windows PC their last digits differ by machine. Where the bar moves to one of them, its
fold values come from the run's own ``raw_fold_metrics.csv`` when it has them, and the gap
is written only if they average to the logged means. ``missforest_lgbm``'s
numbers stay on the runs that carry them; no API deletes a metric. Rewritten values are
logged again at step 0, MLflow's latest value is the newer one and the history keeps all;
``cv_summary.json`` artifacts are not rewritten. A tree is marked ``lgbm_backfilled=true``,
the parent last, and re-mirrored (ADR 0006) through one shared mirror index.

Dry run by default; runs are taken smallest dataset first; ``--dataset`` (repeatable) splits
a pass so dry runs fill the cache in parallel before a single ``--apply`` writes the store.

    uv run --python 3.11 python scripts/backfill_lgbm_baseline.py [--dataset kc2_20nan]
    uv run --python 3.11 python scripts/backfill_lgbm_baseline.py --apply
"""

from __future__ import annotations

import argparse
import re
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
from src.mlflow_utils import LGBM_BACKFILLED_TAG, setup_mlflow  # noqa: E402
from src.training.baseline_cache import BaselineScorer  # noqa: E402
from src.training.mirroring import MirrorIndex, mirror_run_tree  # noqa: E402
from src.training.summary import (  # noqa: E402
    EXTERNAL_BASELINES,
    GAP_TO_BEST_BASELINE,
    WITHDRAWN_BASELINES,
    external_baseline_gaps,
)
from src.training.tracking import _summary_metrics  # noqa: E402
from src.training.types import DEFAULT_BASELINE_CACHE_DIR  # noqa: E402

NEW_BASELINES = ("lgbm",)
# The baseline whose logged numbers must be reproduced to 1e-9: mean/mode, which says
# these are the run's own cells, truths and naive fills. KNN and hgb are not checked: on
# runs recorded on the Windows PC their last digits differ (KNN breaks distance ties
# differently: 3.4e-4 on vehicle_60nan of 2026-09-25), which is also why the cache keys on
# the machine. missForest is corrected rather than checked (``EXTERNAL``).
_CHECKED = frozenset({"mean_mode"})
# ADR 0016 logged missForest from forests that predicted on several threads, whose sums
# depend on the order the trees finish (a CV mean 1.4e-4 off on kr-vs-kp_60nan, seed 7).
# The single-threaded recomputation replaces the logged numbers wherever they differ.
# The families this ADR 0017 backfill recomputed; ADR 0021's linear family came later and has its
# own backfill.
_ITS_FAMILIES = ("adr0007", "lgbm", "missforest")
_CORRECTED = EXTERNAL_BASELINES
_BEST_TAG = "best_baseline/"
_NAMES = "|".join(EXTERNAL_BASELINES)
_EXTERNAL_STAT = re.compile(rf"cv/test/(impute/.+)/baseline/({_NAMES})/([^/]+)/([^/]+)")
_EXTERNAL_FOLD = re.compile(rf"test/(impute/.+)/baseline/({_NAMES})/([^/]+)")


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
    # Per ``<population>/<metric>`` of an external baseline, how far the logged mean was
    # from the recomputed one, where it was rewritten.
    corrected: dict[str, float] = field(default_factory=dict)
    refused: str | None = None


def plan_lgbm(
    run_id: str,
    name: str,
    logged: Mapping[str, float],
    tags: Mapping[str, str],
    folds: Mapping[int, Mapping[str, float]],
    children: Mapping[str, int],
    rows: Mapping[int, Mapping[str, float]],
    fold_tolerance: float = adr0007.TOLERANCE,
) -> Plan:
    """What ADR 0017 writes to one run: ``lgbm``, the gap to missForest, and the bar again.

    ``logged`` and ``tags`` are the parent's, ``folds`` every baseline's recomputed fold
    values, ``children`` each diagnostic child with the fold it replays, and ``rows`` the
    run's own fold values as ``raw_fold_metrics.csv`` keeps them.
    """
    plan = Plan(run_id, name)
    corrected_parts = [f"/baseline/{baseline}/" for baseline in _CORRECTED]
    without = {key: value for key, value in logged.items() if not any(part in key for part in corrected_parts)}
    base = adr0007.plan_run(run_id, name, without, folds, children, checked=_CHECKED)
    if base.refused is not None:
        plan.refused = base.refused
        return plan
    # An external baseline's planned numbers are kept, for a population and metric, only
    # where some statistic the run logged differs from them, or none was logged at all.
    groups: dict[str, list[tuple[float | None, float]]] = {}
    for key, value in base.parent.items():
        match = _EXTERNAL_STAT.fullmatch(key)
        if match is not None:
            groups.setdefault(f"{match.group(1)}/{match.group(2)}/{match.group(3)}", []).append((logged.get(key), value))
    changed: dict[str, float] = {}
    for group, pairs in groups.items():
        held = [(old, new) for old, new in pairs if old is not None]
        if not held or any(abs(old - new) > adr0007.TOLERANCE for old, new in held):
            population, baseline, metric = group.rsplit("/", 2)
            mean = logged.get(f"cv/test/{population}/baseline/{baseline}/{metric}/mean")
            planned = base.parent[f"cv/test/{population}/baseline/{baseline}/{metric}/mean"]
            changed[group] = abs(planned - mean) if mean is not None else float("inf")
    base.parent = {key: value for key, value in base.parent.items() if _kept(_EXTERNAL_STAT, key, changed)}
    base.children = {
        child: {key: value for key, value in metrics.items() if _kept(_EXTERNAL_FOLD, key, changed)}
        for child, metrics in base.children.items()
    }
    plan.corrected = {
        key.replace(f"/{baseline}/", "/"): distance
        for key, distance in changed.items()
        for baseline in _CORRECTED
        if f"/{baseline}/" in key
    }
    after = {**logged, **base.parent}
    merged = {fold: {**folds.get(fold, {}), **values} for fold, values in rows.items()}

    external, refusal = plan_external_gaps(after, merged, fold_tolerance)
    if refusal is not None:
        plan.refused = refusal
        return plan

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
        gap, refusal = adr0007.plan_gap(
            {**after, **best_metrics},
            {population: new for population, (_, new) in moved.items()},
            merged,
            redo=moved,
            tolerance=fold_tolerance,
        )
        if refusal is not None:
            plan.refused = f"best baseline moved but the gap cannot be taken again: {refusal}"
            return plan

    plan.parent = {**base.parent, **external, **best_metrics, **gap}
    plan.children = base.children
    plan.tags = best_tags
    plan.moved = moved
    return plan


def _kept(pattern: re.Pattern[str], key: str, changed: Mapping[str, float]) -> bool:
    """Whether a planned key is written: anything but an external baseline's, and those
    only for a population and metric whose logged numbers are being corrected."""
    match = pattern.fullmatch(key)
    return match is None or f"{match.group(1)}/{match.group(2)}/{match.group(3)}" in changed


def plan_external_gaps(
    logged: Mapping[str, float], rows: Mapping[int, Mapping[str, float]], tolerance: float = adr0007.TOLERANCE
) -> tuple[dict[str, float], str | None]:
    """The gaps to each external baseline a run lacks, from each fold's own numbers, or why
    not: the model's and the baseline's fold values must average to the run's own means."""
    ordered = [rows[fold] for fold in sorted(rows)]
    metrics: dict[str, float] = {}
    for name in EXTERNAL_BASELINES:
        for population in sorted(adr0007._baseline_populations(logged)):
            if f"cv/test/{population}/baseline/{name}/impute_score/mean" not in logged:
                continue
            if f"cv/test/{population}/gap_to_{name}/impute_score/mean" in logged:
                continue
            checks = {
                f"{population}/impute_score": f"cv/test/{population}/impute_score/mean",
                f"{population}/baseline/{name}/impute_score": f"cv/test/{population}/baseline/{name}/impute_score/mean",
            }
            folds: list[dict[str, float]] = [{} for _ in ordered]
            for key, logged_mean in checks.items():
                values = [fold.get(key) for fold in ordered]
                if not values or any(value is None for value in values):
                    return {}, f"no fold value for {key}"
                found = [float(value) for value in values if value is not None]
                if abs(float(np.mean(found)) - logged[logged_mean]) > tolerance:
                    return {}, f"{key}: fold mean {float(np.mean(found)):.6f}, logged {logged[logged_mean]:.6f}"
                for fold, value in zip(folds, found):
                    fold[key] = value
            for key, statistics in external_baseline_gaps(folds).items():
                metrics.update(_summary_metrics(f"cv/test/{key}", statistics))
    return metrics, None


def apply_lgbm_plan(client: MlflowClient, plan: Plan, index: MirrorIndex | None = None) -> None:
    """Write a plan where a live run would have (step 0, each run's own id), mark every run
    of the tree, rename the moved best baselines, and refresh the tree's mirror.

    The parent's mark is written last of the tree's own writes, so a pass cut short leaves
    an unmarked parent, which the next pass plans again from scratch (``_pending``). One
    ``index`` shared across trees spares a scan of the whole mirror experiment per tree
    (23 s on gorgona8's store of 2026-10-06).
    """
    if plan.refused is not None:
        return
    for run_id, metrics in plan.children.items():
        adr0007._write_at_step_zero(client, run_id, metrics)
        client.set_tag(run_id, LGBM_BACKFILLED_TAG, "true")
    adr0007._write_at_step_zero(client, plan.run_id, plan.parent)
    for key, value in plan.tags.items():
        client.set_tag(plan.run_id, key, value)
    client.set_tag(plan.run_id, LGBM_BACKFILLED_TAG, "true")
    mirror_run_tree(client, plan.run_id, index=index)


def _scorer(cache_dir: Path | None) -> adr0007.FoldScorerFactory:
    """Every family of baselines a live run scores, from the cache a live run reads."""

    def make(train: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> adr0007.FoldScorer:
        scorer = BaselineScorer(train, numerical, categorical, cache_dir, families=_ITS_FAMILIES)

        def score(test: pd.DataFrame, cells: pd.DataFrame, fills: Mapping[str, float | str]) -> Mapping[str, float]:
            return scorer.score(test, cells, fills).metrics

        return score

    return make


# ADR 0017's code reached the checkout at 2026-10-06 22:06 GMT-3. A run started before then never
# logged ``lgbm`` itself, so ``lgbm`` numbers on it without the parent's mark are a pass cut
# short.
_LIVE_SINCE_MS = 1791335174074  # 2026-10-06 22:06 GMT-3


def _pending(run: Run) -> bool:
    """A run this pass has still to write: unmarked, carrying ADR 0007's baselines, and
    either lacking ``lgbm`` or carrying it only because an earlier pass was cut short."""
    if run.data.tags.get(LGBM_BACKFILLED_TAG) == "true" or not _populations(run):
        return False
    lacks = any(
        f"cv/test/{population}/baseline/{name}/impute_score/mean" not in run.data.metrics
        for population in _populations(run)
        for name in NEW_BASELINES
    )
    return lacks or run.info.start_time < _LIVE_SINCE_MS


def _populations(run: Run) -> set[str]:
    return {
        match.group(1)
        for key in run.data.metrics
        if (match := adr0007._BASELINE_MEAN.fullmatch(key)) is not None
    }


def _as_planned_from(metrics: Mapping[str, float]) -> dict[str, float]:
    """The run's metrics without what this pass writes and without the withdrawn baseline's,
    so a run cut short is planned again from scratch and nothing checks the withdrawn one."""
    dropped = [f"/baseline/{name}/" for name in (*NEW_BASELINES, *WITHDRAWN_BASELINES)]
    dropped += [f"/gap_to_{name}/" for name in EXTERNAL_BASELINES]
    dropped += [f"/gap_to_{name}_pct/" for name in EXTERNAL_BASELINES]
    return {key: value for key, value in metrics.items() if not any(part in key for part in dropped)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write. Without it, only report what would be written.")
    parser.add_argument("--run", action="append", default=None, help="Restrict to this parent run id (repeatable).")
    parser.add_argument("--dataset", action="append", default=None, help="Restrict to this variant (repeatable).")
    parser.add_argument("--no_cache", action="store_true", help="Neither read nor fill results/baseline_cache/.")
    parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI.")
    parser.add_argument(
        "--fold_tolerance",
        type=float,
        default=adr0007.TOLERANCE,
        help=(
            "How far fold values may average from the logged means before a gap is refused "
            "(default 1e-9). Raise it, with --run, only for runs recorded on another machine "
            "whose KNN or hgb fold values had to be recomputed here."
        ),
    )
    args = parser.parse_args(argv)

    mode = "APPLIED" if args.apply else "DRY RUN"
    cache_dir = None if args.no_cache else Path(__file__).resolve().parents[1] / DEFAULT_BASELINE_CACHE_DIR
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
    print(f"[{mode}] tracking URI: {tracking_uri}; {len(runs)} run(s) to bring to ADR 0017", flush=True)
    written = refused = moved = 0
    index = MirrorIndex(client)
    for run in runs:
        name = run.info.run_name or run.info.run_id
        started = time.perf_counter()
        naive_refusal = adr0007._naive_refusal(run, name)
        if naive_refusal is not None:
            refused += 1
            print(f"[{mode}] REFUSED {name}: {naive_refusal}", flush=True)
            continue
        folds = adr0007.recompute_folds(*adr0007._arguments(run), fold_scorer=_scorer(cache_dir))
        rows = adr0007._fold_rows(client, run.info.run_id)
        plan = plan_lgbm(
            run.info.run_id,
            name,
            _as_planned_from(run.data.metrics),
            run.data.tags,
            folds,
            adr0007._children(client, run),
            rows,
            args.fold_tolerance,
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
        fixes = (
            f"; missForest corrected on {len(plan.corrected)} metric(s), up to {max(plan.corrected.values()):.2g}"
            if plan.corrected
            else ""
        )
        print(
            f"[{mode}] {name}: +{len(plan.parent)} parent metrics on {len(plan.children)} children"
            f"{'; ' + shifts if shifts else ''}{fixes} ({seconds:.0f} s)",
            flush=True,
        )
        moved += bool(plan.moved)
        if args.apply:
            apply_lgbm_plan(client, plan, index)
            written += 1
    print(
        f"[{mode}] {written} run tree(s) written and re-mirrored, {refused} refused, "
        f"{moved} with a best baseline that moved",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
