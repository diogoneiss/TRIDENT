#!/usr/bin/env python3
"""Command-line entry point for TRIDENT."""

import argparse
import copy
import logging
import time
from typing import Any

from src.training.cli import parse_args, run_from_namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def run_all(args: argparse.Namespace) -> None:
    """Run training sequentially on every discovered dataset."""
    from src.training.discovery import discover_datasets
    from src.training.types import task_spec

    # The batch table reports whatever this task ranks by, not always macro F1.
    task = task_spec(getattr(args, "task", None) or "classification")
    datasets = discover_datasets(nan_level=args.nan_level, limit=args.limit)

    total = len(datasets)
    logger.info("Batch run: %d dataset(s) queued — %s", total, ", ".join(datasets))

    results: list[dict[str, Any]] = []

    for index, dataset_name in enumerate(datasets, start=1):
        logger.info(
            "\n============================== [%d/%d] %s ==============================",
            index,
            total,
            dataset_name,
        )
        run_args = copy.deepcopy(args)
        run_args.dataset_name = dataset_name

        start = time.perf_counter()
        try:
            metrics = run_from_namespace(run_args, return_metrics=True)
            elapsed = time.perf_counter() - start
            score = metrics.get(task.ranking_metric, "N/A") if metrics else "N/A"
            results.append(
                {"dataset": dataset_name, "status": "SUCCESS", "score": score, "time": elapsed}
            )
            logger.info("✓ %s completed in %s", dataset_name, _format_duration(elapsed))
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception("✗ %s failed after %s", dataset_name, _format_duration(elapsed))
            results.append(
                {"dataset": dataset_name, "status": "FAILED", "score": "—", "time": elapsed}
            )

    # Print summary table.
    total_time = sum(r["time"] for r in results)
    successes = sum(1 for r in results if r["status"] == "SUCCESS")
    failures = total - successes

    print("\n" + "=" * 70)
    print("  BATCH RUN SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<25} {'Status':<10} {task.ranking_metric_short_name:<12} {'Time':<10}")
    print("  " + "-" * 57)
    for r in results:
        shown = f"{r['score']:.4f}" if isinstance(r["score"], float) else str(r["score"])
        print(
            f"  {r['dataset']:<25} {r['status']:<10} {shown:<12} {_format_duration(r['time']):<10}"
        )
    print("  " + "-" * 57)
    print(f"  Total: {total} dataset(s) | {successes} succeeded | {failures} failed")
    print(f"  Total time: {_format_duration(total_time)}")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()

    if getattr(args, "all", False):
        run_all(args)
        return

    if args.use_optuna:
        logger.info("Starting hyperparameter optimization for dataset %s", args.dataset_name)
        from opt import run_hyperparameter_optimization

        run_hyperparameter_optimization(args)
        return

    logger.info("Starting training for dataset %s", args.dataset_name)
    run_from_namespace(args)


if __name__ == "__main__":
    main()
