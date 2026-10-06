"""The baseline imputers' scores, cached across runs that share a fold (ADR 0007).

The baseline imputers are fit on a fold's training rows and scored on exactly the cells the
model is scored on. Neither step depends on the model, so every arm of a study and every
trial of a search recomputes the same numbers. ``BaselineScorer`` answers from a file when
the same computation was done before, and fits the imputers only when some population is
not cached, once per fold at most.

The key is a hash of everything that enters the computation: the machine (hgb's last digits
differ between the two machines), the library versions, the source of the modules that
compute the numbers, the training and test frames, the scored cells' positions, kinds and
truths (never the model's guesses, which differ per arm), the naive baselines and the column
lists. A change to any of them misses, so a stale entry cannot be read.

Each family of baselines has entries of its own: ADR 0007's four, keyed exactly as before
the missForests of ADR 0016 joined, so not one earlier entry was invalidated, and the two
missForests, keyed also on their own module's source. Editing missForest misses only its
entries; editing the ADR 0007 module, which missForest reads too, misses both.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from . import imputation_baselines, imputation_metrics, missforest
from .imputation_baselines import BaselineImputer, fit_baseline_imputers, score_baselines
from .imputation_metrics import ImputationScores
from .missforest import fit_missforest_imputers

_PER_COLUMN = ["column", "metric", "value"]


def _frame_digest(frame: pd.DataFrame) -> bytes:
    columns = json.dumps([[str(name), str(dtype)] for name, dtype in frame.dtypes.items()])
    rows = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return columns.encode() + np.ascontiguousarray(rows).tobytes()


@dataclass(frozen=True)
class _Family:
    """Baselines cached together: the modules their numbers come from, the libraries
    beyond scikit-learn, pandas and numpy that compute them, and how to fit them."""

    name: str
    sources: tuple[ModuleType, ...]
    libraries: tuple[str, ...]
    fit: Callable[[pd.DataFrame, Sequence[str], Sequence[str]], list[BaselineImputer]]


# In the order their metrics and per-column rows are reported. The name is not part of
# the key.
_FAMILIES = (
    _Family("adr0007", (imputation_baselines, imputation_metrics), (), fit_baseline_imputers),
    _Family(
        "missforest",
        (imputation_baselines, imputation_metrics, missforest),
        ("lightgbm",),
        fit_missforest_imputers,
    ),
)
FAMILIES = tuple(family.name for family in _FAMILIES)


def _environment_digest(family: _Family) -> bytes:
    sources = b""
    for module in family.sources:
        assert module.__file__ is not None  # all are plain source modules
        sources += Path(module.__file__).read_bytes()
    libraries = [version(name) for name in ("scikit-learn", "pandas", "numpy", *family.libraries)]
    return json.dumps([socket.gethostname(), *libraries]).encode() + sources


class BaselineScorer:
    """Scores one fold's baseline imputers on any population of its cells, from the cache
    when it can. ``hits`` counts the lookups the cache answered, one per family."""

    def __init__(
        self,
        train_frame: pd.DataFrame,
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        cache_dir: Path | None,
        families: Collection[str] = FAMILIES,
    ) -> None:
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"no family of baselines is called {sorted(unknown)}; there are {list(FAMILIES)}")
        # Every family a live run scores; a backfill that recomputes only some names them.
        self._families = [index for index, family in enumerate(_FAMILIES) if family.name in families]
        self._train_frame = train_frame
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._cache_dir = cache_dir
        self._imputers: dict[int, list[BaselineImputer]] = {}
        self.hits = 0

    def score(
        self, test_frame: pd.DataFrame, cells: pd.DataFrame, baselines: Mapping[str, float | str]
    ) -> ImputationScores:
        parts = [self._score(family, test_frame, cells, baselines) for family in self._families]
        return ImputationScores(
            metrics={name: value for part in parts for name, value in part.metrics.items()},
            per_column=pd.concat([part.per_column for part in parts], ignore_index=True),
        )

    def _score(
        self, family: int, test_frame: pd.DataFrame, cells: pd.DataFrame, baselines: Mapping[str, float | str]
    ) -> ImputationScores:
        if self._cache_dir is None or cells.empty:
            return score_baselines(self._fitted(family), test_frame, cells, baselines)
        path = self._cache_dir / f"{self._key(_FAMILIES[family], test_frame, cells, baselines)}.json"
        if path.exists():
            self.hits += 1
            # A hit refreshes the entry's modification time, so ``scripts/prune_cache.py``
            # drops the entries used longest ago first.
            os.utime(path)
            return _load(path)
        scores = score_baselines(self._fitted(family), test_frame, cells, baselines)
        _store(path, scores)
        return scores

    def _fitted(self, family: int) -> list[BaselineImputer]:
        if family not in self._imputers:
            self._imputers[family] = _FAMILIES[family].fit(
                self._train_frame, self._numerical, self._categorical
            )
        return self._imputers[family]

    def _key(
        self,
        family: _Family,
        test_frame: pd.DataFrame,
        cells: pd.DataFrame,
        baselines: Mapping[str, float | str],
    ) -> str:
        scored = cells[["row", "column", "kind", "actual"]].sort_values(["row", "column"])
        digest = hashlib.sha256()
        digest.update(_environment_digest(family))
        digest.update(json.dumps([self._numerical, self._categorical]).encode())
        digest.update(json.dumps(sorted((str(k), repr(v)) for k, v in baselines.items())).encode())
        for frame in (self._train_frame, test_frame, scored.astype(str).reset_index(drop=True)):
            digest.update(_frame_digest(frame))
        return digest.hexdigest()


def _store(path: Path, scores: ImputationScores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": scores.metrics,
        "per_column": {name: scores.per_column[name].tolist() for name in _PER_COLUMN},
    }
    # Written aside and renamed, so a concurrent reader never sees half a file.
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(payload))
    os.replace(partial, path)


def _load(path: Path) -> ImputationScores:
    payload = json.loads(path.read_text())
    per_column = pd.DataFrame(payload["per_column"], columns=_PER_COLUMN)
    return ImputationScores(metrics=dict(payload["metrics"]), per_column=per_column)
