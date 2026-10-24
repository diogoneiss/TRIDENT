"""not-MIWAE and MIWAE as inductive baseline imputers, written clean-room (T35; imputation-protocol/25).

From the papers' equations only, no code taken: MIWAE (Mattei and Frellsen, 2019) and not-MIWAE
(Ipsen, Mattei and Frellsen, ICLR 2021). Both share one deep latent variable model of the features:

- **encoder** q(z | x̃): x with its gaps set to 0 (the data are z-scored), an MLP of two tanh
  layers of ``HIDDEN`` units to a Gaussian's mean and log-sd in ``p − 1`` dimensions (the papers'
  UCI choice);
- **decoder** p(x | z): the same MLP shape to a Gaussian's mean and log-sd per feature;
- **prior** p(z): a standard normal.

not-MIWAE adds the missingness model, **self-masking logistic**: P(s_j = 1 | x) =
sigmoid(w_j · (x_j − b_j)), one free (w_j, b_j) per feature, s_j = 1 a gap. Its sign is learnt, not
given. Both train on an importance-weighted bound with ``K`` samples:

    log w_k = log p(x_o | z_k) + log p(z_k) − log q(z_k | x̃)  [+ log p(s | x_o, x_m^(k))],

with x_m^(k) ~ p(x_m | z_k) reparameterised, and the bracket for not-MIWAE only. The fill of a gap
is self-normalised importance sampling with ``DRAWS`` samples: Σ_l w̃_l x̂^(l), where x̂^(l) is the
decoder's mean for MIWAE and the draw x_m^(l) for not-MIWAE (its weights depend on it).

A categorical column, which none of the MNAR tables has, would be filled with the training mode.
Training is seeded, on the CPU, with ``THREADS`` threads.
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

HIDDEN = 128
K = 20
BATCH = 16
STEPS = 100_000
LR = 1e-3
DRAWS = 1000
THREADS = 2
SEED = 0
LOG_2PI = math.log(2 * math.pi)


def _mlp(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, outputs))


def _normal_log(x: torch.Tensor, mean: torch.Tensor, log_sd: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = -0.5 * (LOG_2PI + 2 * log_sd + ((x - mean) / log_sd.exp()) ** 2)
    return out


class _Model(nn.Module):
    def __init__(self, features: int, mnar: bool) -> None:
        super().__init__()
        self.latent = max(features - 1, 1)
        self.encoder = _mlp(features, 2 * self.latent)
        self.decoder = _mlp(self.latent, 2 * features)
        self.mnar = mnar
        self.w = nn.Parameter(torch.zeros(features))
        self.b = nn.Parameter(torch.zeros(features))

    def log_weights(self, x: torch.Tensor, observed: torch.Tensor, samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per row and sample: log w, the decoder's mean and the completed draw; x has 0 in its gaps."""
        q_mean, q_log_sd = self.encoder(x).chunk(2, dim=-1)
        q_log_sd = q_log_sd.clamp(-10, 5)
        z = q_mean + q_log_sd.exp() * torch.randn(samples, *q_mean.shape)  # (S, n, d)
        log_q = _normal_log(z, q_mean, q_log_sd).sum(-1)
        log_prior = (-0.5 * (LOG_2PI + z**2)).sum(-1)
        mean, log_sd = self.decoder(z).chunk(2, dim=-1)
        log_sd = log_sd.clamp(-10, 5)
        log_px = (_normal_log(x, mean, log_sd) * observed).sum(-1)
        log_w = log_px + log_prior - log_q
        draw = mean + log_sd.exp() * torch.randn_like(mean)
        completed = x * observed + draw * (1 - observed)
        if self.mnar:
            logits = self.w * (completed - self.b)
            missing = 1 - observed
            # log Bernoulli(s | sigmoid(logits)), s = 1 for a gap.
            log_ps = -(nn.functional.binary_cross_entropy_with_logits(logits, missing.expand_as(logits), reduction="none")).sum(-1)
            log_w = log_w + log_ps
        return log_w, mean, completed


class VaeImputer:
    """MIWAE (``mnar=False``) or not-MIWAE (``mnar=True``), fitted on a training fold."""

    def __init__(self, numerical: Sequence[str], categorical: Sequence[str], mnar: bool) -> None:
        self.name = "notmiwae" if mnar else "miwae"
        self.numerical, self.categorical, self.mnar = list(numerical), list(categorical), mnar
        self.modes: dict[str, str] = {}
        self.model: _Model | None = None
        # E72's fill: not-MIWAE averages its draws, MIWAE the decoder's means, over DRAWS samples.
        # E73 adds "mean" for not-MIWAE: the decoder's means under not-MIWAE's weights.
        self.fill = "draws" if mnar else "mean"
        self.draws = DRAWS

    def fit(self, frame: pd.DataFrame) -> None:
        torch.set_num_threads(THREADS)
        generator = torch.Generator().manual_seed(SEED)
        torch.manual_seed(SEED)
        self.modes = {c: str(frame[c].mode(dropna=True).iloc[0]) for c in self.categorical}
        values = frame[self.numerical].to_numpy(dtype=np.float64)
        observed = ~np.isnan(values)
        x = torch.tensor(np.where(observed, values, 0.0), dtype=torch.float32)
        mask = torch.tensor(observed, dtype=torch.float32)
        model = _Model(len(self.numerical), self.mnar)
        optimiser = torch.optim.Adam(model.parameters(), lr=LR)
        rows = x.shape[0]
        for _ in range(STEPS):
            batch = torch.randint(rows, (min(BATCH, rows),), generator=generator)
            log_w, _, _ = model.log_weights(x[batch], mask[batch], K)
            loss = -(torch.logsumexp(log_w, dim=0) - math.log(K)).mean()
            optimiser.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
        self.model = model.eval()

    @torch.no_grad()
    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit first")
        torch.manual_seed(SEED)
        out = frame.copy()
        columns = frame.columns.get_indexer(pd.Index(self.numerical))
        values = frame[self.numerical].to_numpy(dtype=np.float64)
        observed = ~(np.isnan(values) | hidden[:, columns])
        x = torch.tensor(np.where(observed, values, 0.0), dtype=torch.float32)
        mask = torch.tensor(observed, dtype=torch.float32)
        fills = []
        chunk = max(1, 64 * DRAWS // self.draws)
        for start in range(0, x.shape[0], chunk):
            log_w, mean, completed = self.model.log_weights(x[start:start + chunk], mask[start:start + chunk], self.draws)
            weights = torch.softmax(log_w, dim=0).unsqueeze(-1)
            guess = completed if self.fill == "draws" else mean
            fills.append((weights * guess).sum(0))
        filled = torch.cat(fills).numpy().astype(np.float64)
        out[self.numerical] = np.where(observed, values, filled)
        for c in self.categorical:
            gaps = frame[c].isna().to_numpy() | hidden[:, int(frame.columns.get_indexer(pd.Index([c]))[0])]
            out[c] = np.where(gaps, self.modes[c], out[c].to_numpy(dtype=object))
        return out


def fit_vae_imputers(frame: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> list[BaselineImputer]:
    """MIWAE and not-MIWAE on one training fold, in that order."""
    imputers: list[BaselineImputer] = [VaeImputer(numerical, categorical, mnar=False), VaeImputer(numerical, categorical, mnar=True)]
    for imputer in imputers:
        imputer.fit(frame)
    return imputers


class _FillView:
    """One fitted not-MIWAE read with another fill (E73): shares the model, renames the method."""

    def __init__(self, fitted: VaeImputer, name: str, fill: str, draws: int) -> None:
        self.name, self.fitted, self.fill, self.draws = name, fitted, fill, draws

    def fit(self, frame: pd.DataFrame) -> None:
        raise RuntimeError("a fill view reads an already fitted model")

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        fill, draws = self.fitted.fill, self.fitted.draws
        self.fitted.fill, self.fitted.draws = self.fill, self.draws
        try:
            return self.fitted.impute(frame, hidden)
        finally:
            self.fitted.fill, self.fitted.draws = fill, draws


E73_DRAWS = 10_000


def fit_notmiwae_fills(frame: pd.DataFrame, numerical: Sequence[str], categorical: Sequence[str]) -> list[BaselineImputer]:
    """E73: one not-MIWAE fit (E72's settings) read two ways with 10,000 draws: ``notmiwae_draws``
    (E72's fill, more draws) and ``notmiwae_mean`` (the decoder's means under its weights)."""
    fitted = VaeImputer(numerical, categorical, mnar=True)
    fitted.fit(frame)
    return [_FillView(fitted, "notmiwae_draws", "draws", E73_DRAWS), _FillView(fitted, "notmiwae_mean", "mean", E73_DRAWS)]

