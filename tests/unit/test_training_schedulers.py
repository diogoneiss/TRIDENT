import math

import pytest
import torch

from src.training.schedulers import (
    PLATEAU_FACTOR,
    PLATEAU_PATIENCE,
    StageScheduler,
    WarmupCosineMultiplier,
    batches_per_epoch,
)
from src.training.types import LR_SCHEDULER_NAMES


BASE_LR = 1e-3


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)


def _run(name: str, epochs: int, batches: int, validation_losses=None) -> list[list[float]]:
    """Learning rate seen by every batch, grouped per epoch."""
    optimizer = _optimizer()
    scheduler = StageScheduler(optimizer, name, epochs=epochs, batches_per_epoch=batches)
    trajectory: list[list[float]] = []
    for epoch in range(epochs):
        epoch_rates = []
        for _ in range(batches):
            epoch_rates.append(scheduler.learning_rate)
            optimizer.step()
            scheduler.after_batch()
        trajectory.append(epoch_rates)
        loss = validation_losses[epoch] if validation_losses is not None else 1.0
        scheduler.after_epoch(loss)
    return trajectory


def test_every_declared_name_builds() -> None:
    for name in LR_SCHEDULER_NAMES:
        StageScheduler(_optimizer(), name, epochs=3, batches_per_epoch=2)


def test_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        StageScheduler(_optimizer(), "linear", epochs=3, batches_per_epoch=2)


def test_cosine_legacy_reproduces_the_per_batch_bug() -> None:
    """The period is ``epochs`` batches, so with 2 batches/epoch it completes a full cycle."""
    epochs, batches = 10, 2
    trajectory = _run("cosine_legacy", epochs, batches)
    flat = [rate for epoch in trajectory for rate in epoch]

    reference = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer(), T_max=epochs)
    expected = []
    for _ in range(epochs * batches):
        expected.append(reference.get_last_lr()[0])
        reference.optimizer.step()
        reference.step()
    assert flat == pytest.approx(expected)
    # Hits zero after ``epochs`` batches (half way through training) and climbs back.
    assert flat[epochs] == pytest.approx(0.0, abs=1e-12)
    assert flat[-1] > 0.5 * BASE_LR


def test_cosine_steps_once_per_epoch_and_ends_annealed() -> None:
    epochs, batches = 10, 4
    trajectory = _run("cosine", epochs, batches)

    assert all(len(set(epoch)) == 1 for epoch in trajectory), "rate must be constant within an epoch"
    per_epoch = [epoch[0] for epoch in trajectory]
    assert per_epoch[0] == pytest.approx(BASE_LR)
    assert per_epoch == sorted(per_epoch, reverse=True)
    assert per_epoch[-1] == pytest.approx(BASE_LR * 0.5 * (1 + math.cos(math.pi * 9 / 10)))


def test_warmup_cosine_warms_up_then_decays_to_zero() -> None:
    epochs, batches = 10, 10
    flat = [rate for epoch in _run("warmup_cosine", epochs, batches) for rate in epoch]
    total = epochs * batches
    warmup = round(0.1 * total)

    assert flat[0] == pytest.approx(BASE_LR / warmup)
    assert flat[:warmup] == sorted(flat[:warmup])
    assert flat[warmup - 1] == pytest.approx(BASE_LR)
    assert max(flat) == pytest.approx(BASE_LR)
    assert flat[warmup:] == sorted(flat[warmup:], reverse=True)
    assert flat[-1] == pytest.approx(0.0, abs=1e-12)


def test_warmup_cosine_multiplier_handles_tiny_horizons() -> None:
    multiplier = WarmupCosineMultiplier(total_steps=1, warmup_steps=5)
    assert multiplier.warmup_steps == 1
    assert multiplier(0) == pytest.approx(1.0)
    assert multiplier(1) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        WarmupCosineMultiplier(total_steps=0, warmup_steps=1)


def test_constant_never_changes_the_rate() -> None:
    flat = [rate for epoch in _run("constant", 5, 3) for rate in epoch]
    assert flat == pytest.approx([BASE_LR] * 15)


def test_plateau_halves_after_patience_epochs_without_improvement() -> None:
    epochs = PLATEAU_PATIENCE + 3
    losses = [1.0] * epochs  # never improves after the first epoch
    trajectory = _run("plateau", epochs, batches=2, validation_losses=losses)
    per_epoch = [epoch[0] for epoch in trajectory]

    # Epoch 0 records the best. PyTorch ignores ``patience`` bad epochs and reduces
    # on the one after, i.e. at the end of epoch ``patience + 1`` (zero-based), so
    # the halved rate is first used by epoch ``patience + 2``.
    assert per_epoch[: PLATEAU_PATIENCE + 2] == pytest.approx([BASE_LR] * (PLATEAU_PATIENCE + 2))
    assert per_epoch[PLATEAU_PATIENCE + 2] == pytest.approx(BASE_LR * PLATEAU_FACTOR)


def test_plateau_keeps_the_rate_while_validation_improves() -> None:
    epochs = PLATEAU_PATIENCE + 3
    losses = [1.0 / (epoch + 1) for epoch in range(epochs)]
    per_epoch = [epoch[0] for epoch in _run("plateau", epochs, 2, validation_losses=losses)]
    assert per_epoch == pytest.approx([BASE_LR] * epochs)


@pytest.mark.parametrize(
    ("rows", "batch", "expected"),
    [(846, 256, 4), (256, 256, 1), (257, 256, 2), (0, 256, 1)],
)
def test_batches_per_epoch_rounds_up(rows: int, batch: int, expected: int) -> None:
    assert batches_per_epoch(rows, batch) == expected
