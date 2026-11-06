"""E151 (imputation-protocol/54, T43): 90% intervals calibrated per column and per tertile of the row's gap
count, on E75's ledgers (primary), E76's (development) and E84's (descriptive). Offline, nothing written.

    uv run --python 3.11 python -B scripts/experiments/conformal_row_gaps.py [--check]

Four arms per numerical cell, cross-fitted on the same seed's other four folds at the split-conformal level:
C and S per column (E77's constant and spread-scaled), MC and MS per column and bin. Q1 reads u(MC) − u(C),
u the worst bin's under-coverage of 0.90; Q2 reads (width MS − width MC) / width MC. ``--check`` runs the
reader's self-check on synthetic cells only.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import remask_ensemble as e75  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments import spread_mechanisms as e84  # noqa: E402

TARGET = 0.9
MIN_FIT = 30
SESOI = 0.01
ARMS = ("C", "S", "MC", "MS")
READOUTS = {"K": ("induced_remask", "induced"), "H": ("induced_remask_loss_checkpoint", "induced_remask_loss_checkpoint")}
Ledger = Callable[[str, int, int], pd.DataFrame]


def _quantile(scores: np.ndarray) -> float:
    """The split-conformal quantile: the ceil((n + 1) * 0.9)-th smallest score, or the largest."""
    ranked = np.sort(scores)
    return float(ranked[min(len(ranked), math.ceil((len(ranked) + 1) * TARGET)) - 1])


def _floor(spread: pd.Series) -> float:
    """E77's amendment: the smallest positive spread, or 1 where there is none."""
    positive = spread[spread > 0]
    return float(positive.min()) if len(positive) else 1.0


def _halves(fit: pd.DataFrame, spread: pd.Series) -> tuple[float, pd.Series]:
    """The constant half-width and the spread-scaled half-widths for cells of ``spread``, from ``fit``."""
    floor = _floor(fit["spread"])
    q = _quantile((fit["error"] / fit["spread"].clip(lower=floor)).to_numpy())
    return _quantile(fit["error"].to_numpy()), q * spread.clip(lower=floor)


def _cells(frame: pd.DataFrame, population: str, centre: str) -> pd.DataFrame:
    """A fold's numerical cells with error, spread and b, the gaps in the cell's row."""
    cells = e75._numerical(frame, population, centre)
    gaps = frame[frame["population"] == "induced"].groupby("row").size()
    return cells.assign(b=cells["row"].map(gaps).astype(float))


def fold_arms(cells: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    """Per held-out cell: its bin, whether its group fell back, and each arm's half-width."""
    edges = np.unique(np.quantile(fit["b"], [1 / 3, 2 / 3]))
    cells = cells.assign(bin=np.searchsorted(edges, cells["b"], side="left"))
    fit = fit.assign(bin=np.searchsorted(edges, fit["b"], side="left"))
    parts = []
    for column, group in cells.groupby("column"):
        fit_column = fit[fit["column"] == column]
        if fit_column.empty:
            continue
        w, scaled = _halves(fit_column, group["spread"])
        out = group.assign(C=w, S=scaled, MC=w, MS=scaled, fallback=True)
        for level, cell_bin in group.groupby("bin"):
            fit_bin = fit_column[fit_column["bin"] == level]
            if len(fit_bin) >= MIN_FIT:
                w_bin, scaled_bin = _halves(fit_bin, cell_bin["spread"])
                out.loc[cell_bin.index, ["MC", "fallback"]] = [w_bin, False]
                out.loc[cell_bin.index, "MS"] = scaled_bin
        parts.append(out)
    return pd.concat(parts) if parts else pd.DataFrame()


def fold_measures(arms: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"fallback": float(arms["fallback"].mean())}
    top = arms["bin"].max()
    for arm in ARMS:
        covered = arms["error"] <= arms[arm]
        by_bin = covered.groupby(arms["bin"]).mean()
        out[f"u_{arm}"] = max(0.0, float((TARGET - by_bin).max()))
        out[f"cover_{arm}"] = float(covered.mean())
        out[f"top_{arm}"] = float(by_bin.loc[top])
        out[f"low_{arm}"] = float(by_bin.iloc[0])
        out[f"width_{arm}"] = float(2 * arms[arm].mean())
    return out


def variant_folds(variant: str, seeds: tuple[int, ...], ledger: Ledger, population: str, centre: str) -> list[list[dict[str, float]]]:
    """Per seed, the five held-out folds' measures (empty where the readout is absent)."""
    per_seed = []
    for seed in seeds:
        folds = {k: _cells(ledger(variant, seed, k), population, centre) for k in range(1, 6)}
        if any(f.empty for f in folds.values()):
            return []
        per_seed.append([fold_measures(fold_arms(folds[k], pd.concat([f for j, f in folds.items() if j != k], ignore_index=True)))
                         for k in range(1, 6)])
    return per_seed


def read_set(name: str, variants: list[str], seeds: tuple[int, ...], ledger: Ledger, ruled: bool) -> None:
    print(f"\n## {name}\n")
    calls: dict[str, dict[str, dict[str, str]]] = {r: {"Q1": {}, "Q2": {}} for r in READOUTS}
    for variant in variants:
        lines = []
        for readout, (population, centre) in READOUTS.items():
            per_seed = variant_folds(variant, seeds, ledger, population, centre)
            if not per_seed:
                continue
            arrays = {key: [np.array([f[key] for f in folds]) for folds in per_seed] for key in per_seed[0][0]}
            q1 = common.verdict([mc - c for mc, c in zip(arrays["u_MC"], arrays["u_C"])], sesoi=SESOI)
            q2 = common.verdict([(ms - mc) / mc for ms, mc in zip(arrays["width_MS"], arrays["width_MC"])], sesoi=SESOI)
            ms_c = common.verdict([(ms - c) / c for ms, c in zip(arrays["width_MS"], arrays["width_C"])], sesoi=SESOI)
            calls[readout]["Q1"][variant], calls[readout]["Q2"][variant] = q1.call, q2.call
            mean = {key: float(np.concatenate(v).mean()) for key, v in arrays.items()}
            coverage = ", ".join(f"{arm} {mean[f'cover_{arm}']:.3f} (low bin {mean[f'low_{arm}']:.3f}, top {mean[f'top_{arm}']:.3f})" for arm in ARMS)
            lines.append(f"{readout}: Q1 u(MC) − u(C) {q1.mean:+.4f} [{q1.low:+.4f}, {q1.high:+.4f}] {q1.call}, u(C) {mean['u_C']:.3f}, u(MC) {mean['u_MC']:.3f}; "
                         f"Q2 (MS − MC) / MC {q2.mean:+.3f} [{q2.low:+.3f}, {q2.high:+.3f}] {q2.call}; (MS − C) / C {ms_c.mean:+.3f} {ms_c.call}; "
                         f"coverage {coverage}; fallback {mean['fallback']:.2f}")
        print(f"- `{variant}`: " + ("\n  ".join(lines) if lines else "no numerical cells"))
    print()
    for readout, by_question in calls.items():
        for question, by_variant in by_question.items():
            if by_variant:
                print(f"- {question}, {readout}: {common.tally(by_variant.values())}")
    if not ruled:
        return
    for question, by_variant in calls["K"].items():
        better = [v for v, c in by_variant.items() if c == "better"]
        worse = [v for v, c in by_variant.items() if c == "worse"]
        reading = "supported" if len(better) >= 10 and len(worse) <= 1 else "refuted" if len(better) <= 5 or len(worse) >= 4 else "mixed"
        print(f"{name} {question} (K): better on {len(better)} of {len(by_variant)}, worse on {worse or 'none'} -> {reading} ({common.RULE})")


def check() -> None:
    """Synthetic cells whose error scale grows with b: C under-covers the top bin, MC does not; an
    oracle spread (the error's own scale) makes MS narrower than MC."""
    rng = np.random.default_rng(151)
    frames = []
    for fold in range(1, 6):
        b = rng.integers(1, 10, 3000).astype(float)
        scale = np.where(b > 6, 3.0, 1.0) * rng.uniform(0.2, 2.0, 3000)
        frames.append(pd.DataFrame({"column": "x", "b": b, "spread": scale, "error": np.abs(rng.normal(0, scale))}))
    measures = [fold_measures(fold_arms(frames[k], pd.concat(frames[:k] + frames[k + 1:], ignore_index=True))) for k in range(5)]
    mean = {key: float(np.mean([m[key] for m in measures])) for key in measures[0]}
    assert mean["top_C"] < 0.85 and abs(mean["top_MC"] - TARGET) < 0.02, mean
    assert mean["width_MS"] < 0.9 * mean["width_MC"], mean
    assert abs(_quantile(np.arange(1.0, 11.0)) - 10.0) < 1e-12 and abs(_quantile(np.arange(1.0, 101.0)) - 91.0) < 1e-12
    print("check passed:", {k: round(v, 3) for k, v in mean.items() if k.startswith(("top_", "width_", "u_"))})


def main() -> None:
    check()
    read_set("E75, primary (MCAR, confirmation)", list(e75.VARIANTS), e75.SEEDS, e75._ledger, ruled=True)
    e75._use_development()
    development = [v for v in e75.VARIANTS if not v.startswith("kr-vs-kp")]
    read_set("E76, development (MCAR)", development, e75.SEEDS, e75._ledger, ruled=True)
    read_set("E84, descriptive (MNAR and MAR, 60nan)", list(e84.SHIFTED), e84.SEEDS, e84._ledger, ruled=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    elif not sys.argv[1:]:
        main()
    else:
        sys.exit(__doc__)
