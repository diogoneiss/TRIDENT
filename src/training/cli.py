"""Compatibility entry points for command-line and programmatic training."""

import argparse
from typing import Any

from .config import build_training_parser, resolve_training_request, validate_parsed_args
from .runner import run_training

# The flat mapping Optuna consumes: a ``dataset`` key plus that run's metrics, or a
# ``fold``/``dataset`` pair plus one fold's. Values are metric numbers and the two string
# identifiers, so ``Any`` is the width of the contract, not a gap in it.
OptunaMetrics = dict[str, Any]


def run_from_namespace(
    args: argparse.Namespace, return_metrics: bool = False
) -> OptunaMetrics | None:
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
    args = build_training_parser().parse_args(argv)
    validate_parsed_args(args)
    return args
