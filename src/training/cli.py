"""Compatibility entry points for command-line and programmatic training."""

import argparse

from .config import build_training_parser, resolve_training_request
from .runner import run_training


def run_from_namespace(args: argparse.Namespace, return_metrics: bool = False) -> dict | None:
    """Run a legacy namespace and retain the flat Optuna return contract."""
    request = resolve_training_request(args)
    result = run_training(request)
    if not return_metrics:
        return None
    if request.cv_folds is not None and len(result.fold_results) > 1:
        return {"dataset": request.dataset.dataset_name, **result.mean_metrics}
    fold = result.fold_results[0]
    return {"fold": fold.fold, "dataset": fold.dataset_name, **fold.metrics}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_training_parser().parse_args(argv)
