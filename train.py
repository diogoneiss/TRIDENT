"""Legacy public facade for TRIDENT training."""

from src.training.cli import parse_args, run_from_namespace
from src.training.runner import run_training
from src.training.summary import compute_cv_summary


def main(args, return_metrics: bool = False):
    """Retain the historical ``train.main`` programmatic entry point."""
    return run_from_namespace(args, return_metrics=return_metrics)


if __name__ == "__main__":
    main(parse_args())
