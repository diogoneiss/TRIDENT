"""MaCoDE (An et al., AAAI 2025) as a baseline imputer, fit on the training fold alone.

Vendored from github.com/an-seunghwan/MaCoDE at commit 8867879 (2025-09-04): ``macode/model.py``
and the parts of ``dataset.py`` and ``macode.py`` an imputer needs, under the MIT licence below.

MaCoDE is a masked transformer over discretised rows. Each continuous column is mapped through
its empirical CDF to [0, 1] and cut into ``bins`` equal bins; a categorical column keeps its
category. Every column has its own embedding table, whose row 0 is the word for a masked or
missing input, and its own output head. Training masks each row's observed cells with a rate
drawn per row and predicts the masked cells' bins or categories (cross-entropy). There is no
pre-training stage: the masked-prediction model is the imputer.

What changed from the released code, and why:

- **Inductive.** The released ``impute`` only fills the frame the model was trained on. Here
  ``fit`` learns the CDFs, the bins and the vocabularies on the training fold, and ``impute``
  encodes any frame through them, so the baseline never sees the test rows (ADR 0007).
- **The bins reach the data.** The released ``MaCoDE`` built its dataset with the default 10
  bins while the model was built for ``bins`` (the review's blocker, 2026-10-16).
- **A table of one column type works.** The released embedding stacked an empty list on a table
  without both kinds of column.
- **The paper's settings, not the package's:** 50 bins, width 128, 4 heads, 2 layers, no dropout,
  500 epochs, batch 1,024, AdamW with learning rate 1e-3 and weight decay 1e-3 (appendix).
- **Its own random streams.** Training runs inside ``torch.random.fork_rng`` from a fixed seed,
  with cuDNN's flags saved and restored, so the global streams every seeded result depends on
  are untouched (ADR 0007 decision 6). GPU training is still not bit-reproducible across
  machines.
- **The fill is a point estimate.** A number is the posterior mean over the bins, each bin
  standing for the training fold's mean of its observed values (the inverse CDF of the bin's
  midpoint where the fold has none); a category is the most probable one. The released code
  samples a bin and a uniform quantile inside it, which only adds variance to a squared-error
  score.
- A category the training fold never shows enters as the missing word, as for ``hgb``.

MIT License

Copyright (c) 2024 an-seunghwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.preprocessing import QuantileTransformer

from src.embedder import as_category_strings

from .imputation_baselines import _NOT_A_CLASS, BaselineImputer, _position

# The paper's settings (appendix), the same for every table, untuned there and here.
BINS = 50
WIDTH = 128
HEADS = 4
LAYERS = 2
EPOCHS = 500
BATCH = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
# The baseline's own seed: the same training fold gives the same fill on one machine.
MACODE_SEED = 0
# Rows per forward pass when filling.
FILL_BATCH = 4096


class _Model(nn.Module):
    """One embedding table and one output head per column, a transformer encoder between."""

    def __init__(self, sizes: Sequence[int], width: int, heads: int, layers: int) -> None:
        super().__init__()
        # Row 0 of every table is the masked-or-missing word; a column with ``size`` outputs
        # reads ``size + 1`` input words.
        self.embeddings = nn.ModuleList(nn.Embedding(size + 1, width) for size in sizes)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in sizes)
        for layer in [*self.embeddings, *self.heads]:
            nn.init.kaiming_uniform_(layer.weight)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dropout=0.0, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, codes: torch.Tensor) -> list[torch.Tensor]:
        tokens = torch.stack([table(codes[:, j]) for j, table in enumerate(self.embeddings)], dim=1)
        hidden = self.encoder(tokens)
        return [head(hidden[:, j, :]) for j, head in enumerate(self.heads)]


@dataclass
class _Numerical:
    """A continuous column's CDF and the value each bin stands for, in the frame's space."""

    transformer: QuantileTransformer | None
    bin_values: NDArray[np.float64]
    constant: float


class MacodeImputer:
    """MaCoDE fit on one training fold: learns the bins and vocabularies, then fills any frame."""

    name = "macode"

    def __init__(
        self,
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        epochs: int = EPOCHS,
        bins: int = BINS,
        device: torch.device | None = None,
    ) -> None:
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._epochs = epochs
        self._bins = bins
        self._edges = np.linspace(0.0, 1.0, bins + 1)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._columns: dict[str, _Numerical] = {}
        self._classes: dict[str, list[str]] = {}
        self._model: _Model | None = None

    def _bin(self, transformer: QuantileTransformer, values: NDArray[np.float64]) -> NDArray[np.int64]:
        quantiles = transformer.transform(values.reshape(-1, 1)).ravel()
        quantiles = np.where(quantiles >= 1.0, 1.0 - 1e-6, quantiles)
        return np.digitize(quantiles, self._edges).astype(np.int64)  # 1..bins

    def _fit_numerical(self, values: NDArray[np.float64]) -> _Numerical:
        observed = values[~np.isnan(values)]
        if len(np.unique(observed)) < 2:
            constant = float(observed[0]) if len(observed) else 0.0
            return _Numerical(None, np.full(self._bins, constant), constant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # n_quantiles above the row count is capped, as intended
            transformer = QuantileTransformer(output_distribution="uniform", subsample=None).fit(observed.reshape(-1, 1))
        codes = self._bin(transformer, observed)
        midpoints = (self._edges[:-1] + self._edges[1:]) / 2
        fallback = transformer.inverse_transform(midpoints.reshape(-1, 1)).ravel()
        means = np.array([observed[codes == b].mean() if (codes == b).any() else fallback[b - 1] for b in range(1, self._bins + 1)])
        return _Numerical(transformer, means.astype(np.float64), float(observed.mean()))

    def _codes(self, frame: pd.DataFrame, missing: NDArray[np.bool_]) -> NDArray[np.int64]:
        """Bins 1..L and categories 1..K, with 0 wherever the cell is hidden, missing or unseen."""
        blocks: list[NDArray[np.int64]] = []
        for column in self._numerical:
            values = frame[column].to_numpy(dtype=float)
            gap = missing[:, _position(frame, column)] | np.isnan(values)
            fitted = self._columns[column]
            codes = np.zeros(len(frame), dtype=np.int64)
            if fitted.transformer is not None and (~gap).any():
                codes[~gap] = self._bin(fitted.transformer, values[~gap])
            blocks.append(codes)
        for column in self._categorical:
            lookup = {category: code + 1 for code, category in enumerate(self._classes[column])}
            strings = as_category_strings(frame[column]).tolist()
            codes = np.array([lookup.get(value, 0) for value in strings], dtype=np.int64)
            codes[missing[:, _position(frame, column)]] = 0
            blocks.append(codes)
        return np.column_stack(blocks)

    def fit(self, frame: pd.DataFrame) -> None:
        self._columns = {column: self._fit_numerical(frame[column].to_numpy(dtype=float)) for column in self._numerical}
        self._classes = {}
        for column in self._categorical:
            classes = sorted(set(as_category_strings(frame[column]).tolist()) - _NOT_A_CLASS)
            if not classes:
                raise ValueError(f"column '{column}' has no observed category in the training fold")
            self._classes[column] = classes
        codes = self._codes(frame, np.zeros(frame.shape, dtype=bool))
        sizes = [self._bins] * len(self._numerical) + [len(self._classes[c]) for c in self._categorical]
        devices = [self._device] if self._device.type == "cuda" else []
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(MACODE_SEED)
                self._model = _Model(sizes, WIDTH, HEADS, LAYERS).to(self._device)
                self._train(torch.as_tensor(codes, device=self._device))
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark

    def _train(self, data: torch.Tensor) -> None:
        assert self._model is not None
        model = self._model
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        rows, width = data.shape
        model.train()
        for _ in range(self._epochs):
            order = torch.randperm(rows, device=data.device)
            for start in range(0, rows, BATCH):
                batch = data[order[start:start + BATCH]]
                missing = batch == 0
                # The released code's low-discrepancy sampler: one masking rate a row,
                # stratified across the batch, each observed cell masked with that rate.
                jitter = torch.rand(batch.size(0), device=batch.device)
                rate = ((torch.arange(batch.size(0), device=batch.device) % width) + jitter) / width
                masked = torch.rand(batch.shape, device=batch.device) < rate.unsqueeze(1)
                target = masked & ~missing
                inputs = batch.masked_fill(masked | missing, 0)
                predictions = model(inputs)
                loss = batch.new_zeros((), dtype=torch.float32)
                for j, logits in enumerate(predictions):
                    selected = target[:, j]
                    if selected.any():
                        loss = loss + F.cross_entropy(logits[selected], batch[selected, j] - 1, reduction="sum") / batch.size(0)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        assert self._model is not None, "fit before impute"
        missing = hidden | frame.isna().to_numpy()
        codes = self._codes(frame, missing)
        probabilities: list[list[NDArray[np.float32]]] = [[] for _ in range(codes.shape[1])]
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(codes), FILL_BATCH):
                batch = torch.as_tensor(codes[start:start + FILL_BATCH], device=self._device)
                for j, logits in enumerate(self._model(batch)):
                    probabilities[j].append(torch.softmax(logits, dim=1).cpu().numpy())
        stacked = [np.concatenate(p) if p else np.empty((0, 0), dtype=np.float32) for p in probabilities]
        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        for j, column in enumerate(self._numerical):
            gaps = missing[:, _position(frame, column)]
            numbers = frame[column].to_numpy(dtype=float).copy()
            fitted = self._columns[column]
            if gaps.any():
                numbers[gaps] = fitted.constant if fitted.transformer is None else stacked[j][gaps] @ fitted.bin_values
            columns[column] = numbers
        for k, column in enumerate(self._categorical):
            j = len(self._numerical) + k
            gaps = missing[:, _position(frame, column)]
            strings = as_category_strings(frame[column]).astype(object)
            if gaps.any():
                strings[gaps] = np.array(self._classes[column], dtype=object)[stacked[j][gaps].argmax(axis=1)]
            columns[column] = strings
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)


def fit_macode_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    epochs: int = EPOCHS,
) -> list[BaselineImputer]:
    """MaCoDE, fit on this fold's training split. Once per fold."""
    imputer = MacodeImputer(numerical_columns, categorical_columns, epochs=epochs)
    imputer.fit(train_frame)
    return [imputer]
