"""One reading of imputation runs for every report (T21; plan 2026-10-17, §2a).

Read-only on the store. Every report that compares the model with baselines reads through here,
so the same words mean the same numbers everywhere:

- **The model's readout.** K is the checkpoint chosen on the validation gaps (ADR 0015), H the
  loss's, M the validation mask's (E43). Which column holds each depends on the run's
  ``decode_checkpoint`` tag: a ``loss`` run's headline is H and its K is ``induced_checkpoint``;
  an ``induced`` run's headline is K and its H is ``loss_checkpoint``. ``readout_key`` picks it,
  and refuses rather than guess. The store's ``gap_to_*`` keys on E34/E35 are H-based and are
  never read as K. KC is K after the calibration on the validation gaps: the headline of a run
  ADR 0027 calibrated (its K then sits in ``uncalibrated``), the ``calibrated`` diagnostic of an
  older one.
- **The bar** (the weak tier) is the best of the logged baselines outside
  ``summary.OUTSIDE_THE_BAR`` and outside ``STRONG``, chosen once per run on its CV mean (ADR 0007
  decision 8); never a name tuple. The **strong tier**
  (plan §2b) is the best of ``STRONG``; **the best of all** the best of every baseline but the
  withdrawn one; **the per-type bar** the best baseline per kind of cell, recombined with
  ``impute_score``'s cell weights (item 7b, D1).
- **The verdict rule**: the 95% t-interval of the pooled fold differences excludes zero and every
  seed's mean difference agrees in sign. With a SESOI, a call without a verdict whose interval lies
  inside ±SESOI of the opponent's mean is "equivalent" (D3: 1%).
- **Across tables**: the per-table mean relative gap, a two-sided Wilcoxon signed-rank test over
  tables, Holm across opponents (D2). Every count is printed with the test named beside it.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import t, wilcoxon

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.summary import OUTSIDE_THE_BAR, STRONG_BASELINES, WITHDRAWN_BASELINES  # noqa: E402

# The strong tier of baselines (ADR 0021): missForest and E41's arms. The weak tier is the bar.
STRONG = STRONG_BASELINES
# Every other baseline a frame can carry, classified so that none slips into the bar unread:
# the live bar's members (checked only as known names; the bar itself is "not outside"), E41's
# diagnostic arm, and E42's tuned families with their bar.
WEAK = ("mean_mode", "knn5", "knn10", "hgb", "lgbm")
DIAGNOSTIC = ("missforest_seq",)
# Deep baselines read from offline passes (E48's MaCoDE): outside the bar and the strong tier,
# compared on their own and in the best of all.
DEEP = ("macode",)
_TUNED = re.compile(r"_tuned(_masked)?$|^tuned_bar$")
RULE = "fold-pair t, seed agreement"
_SOURCE = """r.status = 'FINISHED' and r.lifecycle_stage = 'active'
            and r.run_uuid not in (select run_uuid from tags where key = 'is_mirror' and value = 'true')"""
_STORE: sqlite3.Connection | None = None


@dataclass(frozen=True)
class Run:
    """One finished source parent: where its fold table lives and which checkpoint it kept."""

    variant: str
    seed: int
    uri: str
    decode_checkpoint: str


def _store() -> sqlite3.Connection:
    global _STORE
    if _STORE is None:
        _STORE = sqlite3.connect("file:mlflow.db?mode=ro", uri=True, timeout=30)
    return _STORE


def parents(experiment: str, arm: str | None = None) -> dict[tuple[str, int], list[Run]]:
    """Finished source parents of an ``experiment`` tag, by (variant, seed). ``arm=None`` takes
    runs with any arm tag or none; a named arm takes only runs tagged with it."""
    arm_clause = "" if arm is None else "join tags a on a.run_uuid = r.run_uuid and a.key = 'arm' and a.value = ?"
    rows = _store().execute(
        f"""
        select d.value, s.value, r.artifact_uri, coalesce(c.value, 'loss') from runs r
        join tags x on x.run_uuid = r.run_uuid and x.key = 'experiment' and x.value = ?
        join tags d on d.run_uuid = r.run_uuid and d.key = 'dataset'
        join tags s on s.run_uuid = r.run_uuid and s.key = 'seed'
        join tags p on p.run_uuid = r.run_uuid and p.key = 'run_role' and p.value = 'parent'
        {arm_clause}
        left join tags c on c.run_uuid = r.run_uuid and c.key = 'decode_checkpoint'
        where {_SOURCE}""",
        (experiment,) if arm is None else (experiment, arm),
    )
    found: dict[tuple[str, int], list[Run]] = {}
    for variant, seed, uri, checkpoint in rows:
        found.setdefault((str(variant), int(seed)), []).append(Run(str(variant), int(seed), str(uri), str(checkpoint)))
    return found


def artifact_path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    return Path(path[1:] if re.match(r"^/[A-Za-z]:", path) else path)


def fold_table(run: Run) -> pd.DataFrame:
    frame = pd.read_csv(artifact_path(run.uri) / "metrics" / "raw_fold_metrics.csv")
    return frame.sort_values("fold").reset_index(drop=True)


def readout_key(columns: Iterable[str], decode_checkpoint: str, readout: str, population: str, metric: str) -> str:
    """The column holding the model's ``readout`` (K, H, M or KC) of ``metric`` on ``population``.

    K, H and M are always uncalibrated: on a run ADR 0027 calibrated, the headline is KC and the
    run's own checkpoint, uncalibrated, sits in ``uncalibrated``.
    """
    available = set(columns)
    own = {"induced": "induced", "loss": "loss", "masked": "masked"}[decode_checkpoint]
    headline = f"impute/{population}/{metric}"
    calibrated_run = f"impute/{population}/uncalibrated/{metric}" in available
    if readout == "KC":
        if own == "induced":
            key = headline if calibrated_run else f"impute/{population}/calibrated/{metric}"
        else:
            key = f"impute/{population}/induced_checkpoint_calibrated/{metric}"
    else:
        family = {"K": "induced", "H": "loss", "M": "masked"}[readout]
        if family == own:
            key = f"impute/{population}/uncalibrated/{metric}" if calibrated_run else headline
        else:
            key = f"impute/{population}/{family}_checkpoint/{metric}"
    if key not in available:
        raise KeyError(f"a {decode_checkpoint!r} run carries no {readout} readout ({key})")
    return key


def model(frame: pd.DataFrame, run: Run, readout: str = "K", population: str = "induced", metric: str = "impute_score") -> NDArray[np.float64]:
    return np.asarray(frame[readout_key(frame.columns, run.decode_checkpoint, readout, population, metric)], dtype=float)


def baselines_in(columns: Iterable[str], population: str) -> list[str]:
    """The baselines a frame scores on ``population``; refuses one no class of this module names,
    so a new arm can never join the bar or a tier without being read."""
    pattern = re.compile(rf"^impute/{re.escape(population)}/baseline/([^/]+)/impute_score$")
    names = sorted({m.group(1) for column in columns if (m := pattern.match(column))})
    unknown = [n for n in names if n not in (*WEAK, *STRONG, *DIAGNOSTIC, *DEEP, *OUTSIDE_THE_BAR) and not _TUNED.search(n)]
    if unknown:
        raise KeyError(f"baselines no class names: {unknown}; add them to WEAK, STRONG or DIAGNOSTIC")
    return names


def _excluded(name: str) -> bool:
    return name in OUTSIDE_THE_BAR or name in STRONG or name in DIAGNOSTIC or name in DEEP or bool(_TUNED.search(name))


def _best(frame: pd.DataFrame, names: Sequence[str], population: str) -> tuple[str, NDArray[np.float64]]:
    if not names:
        raise KeyError(f"no baseline to choose from on {population}")
    best = min(names, key=lambda name: (float(frame[f"impute/{population}/baseline/{name}/impute_score"].mean()), name))
    return best, np.asarray(frame[f"impute/{population}/baseline/{best}/impute_score"], dtype=float)


def opponent(frame: pd.DataFrame, which: str, population: str = "induced") -> tuple[str, NDArray[np.float64]]:
    """An opponent's name and its fold values on this run: ``bar``, ``strong``, ``best_of_all``,
    ``per_type``, or a baseline's own name."""
    names = baselines_in(frame.columns, population)
    if which == "bar":
        return _best(frame, [n for n in names if not _excluded(n)], population)
    if which == "strong":
        return _best(frame, [n for n in names if n in STRONG], population)
    if which == "best_of_all":
        return _best(
            frame,
            [n for n in names if n not in WITHDRAWN_BASELINES and n not in ("best", "best_strong") and n not in DIAGNOSTIC and not _TUNED.search(n)],
            population,
        )
    if which == "per_type":
        return "per_type", per_type_bar(frame, population)
    if which in names:
        return which, np.asarray(frame[f"impute/{population}/baseline/{which}/impute_score"], dtype=float)
    raise KeyError(f"no opponent {which!r} on {population}")


def kind_ratios(frame: pd.DataFrame, prefix: str, population: str) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Per fold, the numerical RMSE and the categorical error of ``prefix`` (a key stem such as
    ``impute/induced`` or ``impute/induced/baseline/hgb``) over the mean/mode fill's, as
    ``impute_score`` weighs them; None for a kind the population has no cell of."""
    naive = f"impute/{population}/baseline/mean_mode"
    numbers = categories = None
    if f"{prefix}/rmse_num_z" in frame.columns and f"{naive}/rmse_num_z" in frame.columns:
        numbers = np.asarray(frame[f"{prefix}/rmse_num_z"] / frame[f"{naive}/rmse_num_z"], dtype=float)
    if f"{prefix}/acc_cat" in frame.columns and f"{naive}/acc_cat" in frame.columns:
        categories = np.asarray((1 - frame[f"{prefix}/acc_cat"]) / (1 - frame[f"{naive}/acc_cat"]), dtype=float)
    return numbers, categories


def per_type_bar(frame: pd.DataFrame, population: str = "induced") -> NDArray[np.float64]:
    """Item 7b: the bar's best baseline chosen per kind of cell, on that kind's CV-mean ratio, and
    the two recombined with the model's cell weights. Equal to the bar on a table of one kind."""
    names = [n for n in baselines_in(frame.columns, population) if not _excluded(n)]
    numbers = np.asarray(frame.get(f"impute/{population}/n_num_cells", 0), dtype=float) * np.ones(len(frame))
    categories = np.asarray(frame.get(f"impute/{population}/n_cat_cells", 0), dtype=float) * np.ones(len(frame))
    total = numbers + categories
    score = np.zeros(len(frame))
    for kind, weight in ((0, numbers), (1, categories)):
        if not weight.any():
            continue
        candidates = {}
        for name in names:
            ratio = kind_ratios(frame, f"impute/{population}/baseline/{name}", population)[kind]
            if ratio is not None:
                candidates[name] = ratio
        best = min(candidates, key=lambda name: (float(np.mean(candidates[name])), name))
        score += np.where(total > 0, weight / np.where(total > 0, total, 1), 0.0) * candidates[best]
    return score


@dataclass(frozen=True)
class Verdict:
    call: str  # "better", "worse", "equivalent" or "none", for the first of the two compared
    mean: float
    low: float
    high: float
    seeds: tuple[float, ...]


def verdict(per_seed: Sequence[NDArray[np.float64]], sesoi: float | None = None) -> Verdict:
    """The project's rule on (A - B) fold differences, one array per seed; lower is better, so
    "better" means A's interval lies below zero with every seed agreeing. ``sesoi`` is the
    equivalence margin in the differences' own units."""
    pooled = np.concatenate(per_seed)
    mean = float(pooled.mean())
    margin = float(t.ppf(0.975, len(pooled) - 1) * pooled.std(ddof=1) / np.sqrt(len(pooled)))
    low, high = mean - margin, mean + margin
    seeds = tuple(float(np.mean(d)) for d in per_seed)
    if high < 0 and all(s < 0 for s in seeds):
        call = "better"
    elif low > 0 and all(s > 0 for s in seeds):
        call = "worse"
    elif sesoi is not None and -sesoi < low and high < sesoi:
        call = "equivalent"
    else:
        call = "none"
    return Verdict(call, mean, low, high, seeds)


def tally(calls: Iterable[str]) -> str:
    counts = {name: 0 for name in ("better", "none", "equivalent", "worse")}
    for call in calls:
        counts[call] += 1
    shown = [f"better {counts['better']}", f"no difference {counts['none']}"]
    if counts["equivalent"]:
        shown.append(f"equivalent {counts['equivalent']}")
    shown.append(f"worse {counts['worse']}")
    return " / ".join(shown) + f" ({RULE})"


def table_of(variant: str) -> str:
    return variant.split("_")[0]


def across_tables(relative_gaps: Mapping[str, float]) -> tuple[int, int, float]:
    """Per table, the mean relative gap over its variants; a two-sided Wilcoxon signed-rank test
    of those means against zero. Returns (tables, tables below zero, p)."""
    by_table: dict[str, list[float]] = {}
    for variant, gap in relative_gaps.items():
        by_table.setdefault(table_of(variant), []).append(gap)
    means = np.array([np.mean(gaps) for gaps in by_table.values()])
    p = float(wilcoxon(means).pvalue) if len(means) and np.any(means != 0) else 1.0
    return len(means), int((means < 0).sum()), p


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm's step-down adjustment."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def seed_effects(per_seed: Sequence[NDArray[np.float64]]) -> tuple[float, float, float]:
    """The sensitivity reading: the seed means as the unit, a t-interval on seeds - 1 degrees of
    freedom (a one-way random-effects reading of the seed)."""
    means = np.array([float(np.mean(d)) for d in per_seed])
    centre = float(means.mean())
    margin = float(t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / np.sqrt(len(means)))
    return centre, centre - margin, centre + margin


# The development set's cells on today's default (plan §2a; ticket imputation-protocol/01):
# per seed set, which experiment (and arm) carries each variant's K. ``labelled`` variants ran a
# configuration that is no longer the default and are never pooled with other seed sets.
FORTY_WITH_REMOVED_FILES = ("credit-g_40nan", "kr-vs-kp_40nan", "spambase_40nan")
BIG_ENCODER = ("pendigits_20nan", "letter_20nan")


@dataclass(frozen=True)
class Source:
    experiment: str
    arm: str | None
    labelled: bool = False


def development_source(variant: str, seeds: str) -> Source | None:
    """Where today's default K for ``variant`` on a seed set (``"101/202/303"``, ``"42/7/13"`` or
    ``"404/505/606"``) lives, or None where that set has no run of it."""
    if seeds == "101/202/303":
        if variant in BIG_ENCODER:
            return Source("architecture-large-2026-10-06", "ALL")
        if variant in FORTY_WITH_REMOVED_FILES:
            return Source("retune-2026-10-15", "D")
        return Source("selection-2026-10-03", None)
    if seeds == "42/7/13":
        return Source("selection-replication-2026-10-04", None, labelled=variant in (*FORTY_WITH_REMOVED_FILES, *BIG_ENCODER))
    if seeds == "404/505/606":
        return Source("calibration-2026-10-16", None)
    raise KeyError(seeds)
