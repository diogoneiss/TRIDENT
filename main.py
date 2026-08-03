#!/usr/bin/env python3
"""Command-line entry point for TRIDENT."""

import logging

from src.training.cli import parse_args, run_from_namespace


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    if args.use_optuna:
        logger.info("Starting hyperparameter optimization for dataset %s", args.dataset_name)
        from opt import run_hyperparameter_optimization

        run_hyperparameter_optimization(args)
        return
    logger.info("Starting training for dataset %s", args.dataset_name)
    run_from_namespace(args)


if __name__ == "__main__":
    main()
