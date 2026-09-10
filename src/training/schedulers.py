"""Selectable learning-rate schedules shared by both training stages.

Every schedule is driven through the same two hooks so the stage loops do not
need to know whether a schedule advances per mini-batch or per epoch:

- ``after_batch()`` runs once after every optimizer step.
- ``after_epoch(validation_loss)`` runs once after each epoch's validation pass.

See ``docs/adr/0003-selectable-learning-rate-schedule.md`` for the rationale
behind each option and for why ``cosine_legacy`` remains the default.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau

from .types import DEFAULT_LR_SCHEDULER, LR_SCHEDULER_NAMES

# Fraction of all optimizer steps spent in linear warmup for ``warmup_cosine``.
WARMUP_FRACTION = 0.1
# ``plateau`` halves the learning rate after this many epochs without a new
# best validation loss. These match PyTorch's documented defaults except the
# factor, which is the value most tabular baselines use.
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 10


def validate_lr_scheduler_name(name: str) -> str:
    if name not in LR_SCHEDULER_NAMES:
        raise ValueError(
            f"Unknown learning-rate scheduler {name!r}; expected one of {', '.join(LR_SCHEDULER_NAMES)}"
        )
    return name


class WarmupCosineMultiplier:
    """Linear warmup to the base rate, then cosine decay to zero, measured in steps.

    The multiplier for step ``k`` (zero-based count of completed optimizer steps)
    is ``(k + 1) / warmup_steps`` during warmup and follows a half cosine from one
    to zero across the remaining steps, so the final step trains at (almost) zero.
    """

    def __init__(self, total_steps: int, warmup_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        self.total_steps = total_steps
        self.warmup_steps = max(1, min(warmup_steps, total_steps))
        self.decay_steps = max(1, total_steps - self.warmup_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = min(1.0, (step + 1 - self.warmup_steps) / self.decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class StageScheduler:
    """One training stage's learning-rate schedule behind a uniform hook interface."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        name: str = DEFAULT_LR_SCHEDULER,
        *,
        epochs: int,
        batches_per_epoch: int,
    ) -> None:
        self.name = validate_lr_scheduler_name(name)
        self._optimizer = optimizer
        self._batch_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._epoch_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._plateau_scheduler: ReduceLROnPlateau | None = None

        if name == "cosine_legacy":
            # The published behaviour: the period is ``epochs`` optimizer steps,
            # so the cosine completes ``batches_per_epoch`` cycles over training.
            self._batch_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif name == "cosine":
            self._epoch_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif name == "warmup_cosine":
            total_steps = max(1, epochs * batches_per_epoch)
            warmup_steps = max(1, round(WARMUP_FRACTION * total_steps))
            self._batch_scheduler = LambdaLR(
                optimizer, WarmupCosineMultiplier(total_steps, warmup_steps)
            )
        elif name == "plateau":
            self._plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
            )
        # ``constant`` deliberately builds nothing: the optimizer keeps its base rate.

    def after_batch(self) -> None:
        if self._batch_scheduler is not None:
            self._batch_scheduler.step()

    def after_epoch(self, validation_loss: float) -> None:
        if self._epoch_scheduler is not None:
            self._epoch_scheduler.step()
        if self._plateau_scheduler is not None:
            self._plateau_scheduler.step(validation_loss)

    @property
    def learning_rate(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])


def batches_per_epoch(row_count: int, batch_size: int) -> int:
    """Number of optimizer steps one epoch performs over ``row_count`` rows."""
    return max(1, math.ceil(row_count / batch_size))
