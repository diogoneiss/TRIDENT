"""ReMasker (Du, Melis and Wang, ICLR 2024) as a baseline imputer, fit on the training fold alone (E157).

Written here from the paper and the reference code at github.com/tydusky/remasker, commit 69b4108 (2023-03-16), whose
settings it keeps. That repository states no licence, so no file of it is copied; the architecture is a standard masked
autoencoder, re-expressed with PyTorch's own transformer layers.

ReMasker is a masked autoencoder over a row's cells. Each column is min-max scaled on the training fold; a cell
becomes a token by one linear map shared by every column (the reference's 1x1 convolution) plus a fixed sine-cosine
position, which is all that names the column. Training re-masks half of each row's tokens at random, the encoder
reads the cells left visible with a class token, and the decoder sees the encoder's states with a learned mask token
everywhere else and predicts every cell through ``tanh / 2 + 0.5``. The loss is the squared error on the re-masked
observed cells plus that on the visible ones, each a mean. To impute, the encoder sees every observed cell.

The reference's settings (``utils.get_args_parser`` and ``remasker_impute.ReMasker``): width 32, encoder depth 6,
decoder depth 4, 4 heads, MLP ratio 4, mask ratio 0.5, 600 epochs, batch 64, AdamW with betas (0.9, 0.95) at
1e-3 (``blr * batch / 64``) under a per-step half-cosine to 1e-5 after 40 warm-up epochs, fp16 autocast with a grad
scaler on the GPU, LayerNorm eps 1e-8. Its parser's ``weight_decay`` 0.05 never reaches the optimizer, which runs at
AdamW's default 0.01; that is kept.

What differs from the reference code, and why:

- **Inductive.** ``fit`` learns the scaling and the model on the training fold, ``impute`` fills any frame through
  them, so the baseline never sees the test rows (ADR 0007).
- **A gap is never a visible cell.** The reference keeps a fixed half of each row's tokens, sorting gaps last; a row
  with fewer observed cells than that keeps some gaps visible as zeros, and its loss on visible cells scores them
  against 0. The paper re-masks observed values only and encodes the observed rest, so here a visible token is an
  observed one (the others are padding in the encoder) and both loss terms read observed cells only. At 40% missing
  and above most rows were affected.
- **Batched imputation.** The reference fills one row at a time; with gaps as padding a batch gives the same fill.
- **Its own random streams.** Training runs inside ``torch.random.fork_rng`` from a fixed seed with cuDNN's flags saved
  and restored, so the global streams every seeded result depends on are untouched (ADR 0007 decision 6).
- **Numerical tables only,** as the reference ("only support numerical values"); a categorical column is refused.

The re-mask count has two rules (``remask``). ``reference``, the default and E157's, keeps the reference code's count
of half a row's tokens, so a row with fewer observed cells than half its columns re-masks none: at 80% missing almost
no observed cell is ever re-masked (E157's Outcome). ``paper``, E158's, counts the ratio over the row's observed cells,
as the paper's §4.4 states.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from src.training.imputation_baselines import BaselineImputer

REMASKER_SEED = 20240507
WIDTH, DEPTH, DECODER_DEPTH, HEADS, MLP_RATIO = 32, 6, 4, 4, 4
MASK_RATIO = 0.5
EPOCHS, BATCH = 600, 64
LEARNING_RATE, MIN_LR, WARMUP_EPOCHS = 1e-3, 1e-5, 40
BETAS, WEIGHT_DECAY = (0.9, 0.95), 0.01
NORM_EPS, SCALE_EPS = 1e-8, 1e-8
FILL_BATCH = 4096


def _positions(width: int, length: int) -> torch.Tensor:
    """The reference's fixed 1-D sine-cosine table, a zero row for the class token first: (length + 1, width)."""
    omega = 1.0 / 10000 ** (np.arange(width // 2, dtype=float) / (width / 2))
    angles = np.outer(np.arange(length, dtype=float), omega)
    table = np.concatenate([np.zeros((1, width)), np.concatenate([np.sin(angles), np.cos(angles)], axis=1)])
    return torch.as_tensor(table, dtype=torch.float32).unsqueeze(0)


def _blocks(depth: int) -> nn.ModuleList:
    return nn.ModuleList(nn.TransformerEncoderLayer(WIDTH, HEADS, WIDTH * MLP_RATIO, dropout=0.0, activation="gelu",
                                                    layer_norm_eps=NORM_EPS, batch_first=True, norm_first=True)
                         for _ in range(depth))


class _Model(nn.Module):
    def __init__(self, length: int) -> None:
        super().__init__()
        self.embed = nn.Linear(1, WIDTH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.register_buffer("positions", _positions(WIDTH, length))
        self.encoder = _blocks(DEPTH)
        self.norm = nn.LayerNorm(WIDTH, eps=NORM_EPS)
        self.decoder_embed = nn.Linear(WIDTH, WIDTH)
        self.decoder = _blocks(DECODER_DEPTH)
        self.decoder_norm = nn.LayerNorm(WIDTH, eps=NORM_EPS)
        self.head = nn.Linear(WIDTH, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """``values`` (B, L) in [0, 1] with gaps at 0, ``visible`` (B, L) the cells the encoder reads: (B, L) fills."""
        positions: torch.Tensor = self.positions  # type: ignore[assignment]
        tokens = self.embed(values.unsqueeze(-1)) + positions[:, 1:]
        x = torch.cat([(self.cls_token + positions[:, :1]).expand(len(values), -1, -1), tokens], dim=1)
        padding = torch.cat([torch.zeros_like(visible[:, :1]), ~visible], dim=1)
        for block in self.encoder:
            x = block(x, src_key_padding_mask=padding)
        x = self.decoder_embed(self.norm(x))
        cells = torch.where(visible.unsqueeze(-1), x[:, 1:], self.mask_token.to(x.dtype))
        x = torch.cat([x[:, :1], cells], dim=1) + positions
        for block in self.decoder:
            x = block(x)
        out: torch.Tensor = torch.tanh(self.head(self.decoder_norm(x))[:, 1:, 0]) / 2 + 0.5
        return out


def _learning_rate(epoch: float) -> float:
    """The reference's ``adjust_learning_rate``: linear warm-up, then a half cosine to MIN_LR, read per step."""
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * epoch / WARMUP_EPOCHS
    return MIN_LR + (LEARNING_RATE - MIN_LR) * 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))


REMASK_RULES = ("reference", "paper")


def _remask(observed: torch.Tensor, rule: str = "reference") -> torch.Tensor:
    """The cells left visible, drawn at random with the gaps sorted last.

    ``reference`` (E157): the reference code's count, int(L * (1 - ratio)) of a row's L tokens, the observed ones among
    them; a row with fewer observed cells than that re-masks none. ``paper`` (E158): the paper's §4.4, the ratio counted
    "after excluding missing values", so a row keeps int(n_obs * (1 - ratio)) of its n_obs observed cells and re-masks
    the rest; on a complete row the two agree.
    """
    if rule not in REMASK_RULES:
        raise ValueError(f"remask rule {rule!r} is not one of {REMASK_RULES}")
    noise = torch.rand(observed.shape, device=observed.device).masked_fill(~observed, 1.0)
    rank = noise.argsort(dim=1).argsort(dim=1)
    if rule == "reference":
        return (rank < int(observed.size(1) * (1 - MASK_RATIO))) & observed
    keep = (observed.sum(dim=1, keepdim=True) * (1 - MASK_RATIO)).floor()
    return (rank < keep) & observed


class RemaskerImputer:
    name = "remasker"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str] = (), epochs: int = EPOCHS,
                 device: torch.device | None = None, remask: str = "reference") -> None:
        if categorical_columns:
            raise ValueError("ReMasker takes numerical columns only, as its reference code")
        if remask not in REMASK_RULES:
            raise ValueError(f"remask rule {remask!r} is not one of {REMASK_RULES}")
        self._remask = remask
        self._numerical = list(numerical_columns)
        self._epochs = epochs
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._low: NDArray[np.float64] = np.zeros(0)
        self._span: NDArray[np.float64] = np.zeros(0)
        self._model: _Model | None = None

    def _scaled(self, frame: pd.DataFrame, missing: NDArray[np.bool_]) -> tuple[torch.Tensor, torch.Tensor]:
        values = (frame[self._numerical].to_numpy(dtype=float) - self._low) / (self._span + SCALE_EPS)
        observed = ~missing & ~np.isnan(values)
        return (torch.as_tensor(np.where(observed, values, 0.0), dtype=torch.float32, device=self._device),
                torch.as_tensor(observed, device=self._device))

    def fit(self, frame: pd.DataFrame) -> None:
        numbers = frame[self._numerical].to_numpy(dtype=float)
        if np.isnan(numbers).all(axis=0).any():
            raise ValueError("a column has no observed value in the training fold")
        self._low = np.nanmin(numbers, axis=0)
        self._span = np.nanmax(numbers, axis=0) - self._low
        values, observed = self._scaled(frame, np.zeros(frame[self._numerical].shape, dtype=bool))
        devices = [self._device] if self._device.type == "cuda" else []
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(REMASKER_SEED)
                self._model = _Model(len(self._numerical)).to(self._device)
                self._train(values, observed)
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark

    def _train(self, values: torch.Tensor, observed: torch.Tensor) -> None:
        assert self._model is not None
        model, cuda = self._model, self._device.type == "cuda"
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)  # type: ignore[attr-defined]
        rows = len(values)
        steps = math.ceil(rows / BATCH)
        model.train()
        for epoch in range(self._epochs):
            order = torch.randperm(rows, device=values.device)
            total = values.new_zeros(())
            for step in range(steps):
                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(epoch + step / steps)
                index = order[step * BATCH:(step + 1) * BATCH]
                x, seen = values[index], observed[index]
                visible = _remask(seen, self._remask)
                remasked = seen & ~visible
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    error = (model(x, visible).float() - x) ** 2
                # Masked means without a host sync: a term with no cell adds 0, as the reference's would NaN.
                loss: torch.Tensor = sum(((error * part).sum() / part.sum().clamp(min=1) for part in (remasked, visible)), x.new_zeros(()))
                total = total + loss.detach()
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.step(optimizer)
                scaler.update()
            if not torch.isfinite(total):  # once an epoch: the reference stops on a non-finite loss
                raise FloatingPointError("ReMasker's loss is not finite")

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        assert self._model is not None, "fit before impute"
        missing = hidden | frame.isna().to_numpy()
        positions = [frame.columns.get_loc(c) for c in self._numerical]
        values, observed = self._scaled(frame, missing[:, positions])
        self._model.eval()
        fills = []
        with torch.no_grad():
            for start in range(0, len(values), FILL_BATCH):
                part = slice(start, start + FILL_BATCH)
                fills.append(self._model(values[part], observed[part]).float().cpu().numpy())
        filled = (np.concatenate(fills) if fills else np.zeros((0, len(positions)))) * (self._span + SCALE_EPS) + self._low
        out = frame.copy()
        numbers = frame[self._numerical].to_numpy(dtype=float)
        gaps = missing[:, positions]
        out[self._numerical] = np.where(gaps, filled, numbers)
        return out


def fit_remasker_imputers(train_frame: pd.DataFrame, numerical_columns: Sequence[str],
                          categorical_columns: Sequence[str], epochs: int = EPOCHS,
                          remask: str = "reference") -> list[BaselineImputer]:
    """ReMasker, fit on this fold's training split. Once per fold."""
    imputer = RemaskerImputer(numerical_columns, categorical_columns, epochs=epochs, remask=remask)
    imputer.fit(train_frame)
    return [imputer]
