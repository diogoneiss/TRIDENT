"""E98's reading (classification-step-zero/07): H1c, tuned TRIDENT against the tuned GBDT bar.

    uv run --python 3.11 python -B scripts/experiments/stage4_tally.py

Run on gorgona8 after medusa1's outputs are copied under ``STATE/medusa`` (its ``optuna/*/*/optuna_*/
best_model`` and ``eval``) and the references under ``STATE/refs`` (E87's ``results`` as ``refs/e87``,
E92's ``csv`` as ``refs/e92``). A variant's TRIDENT folds come from medusa1 when it has them, else
from gorgona8 (the ticket's amendments: kc2_00nan is read from medusa1). Per level, every difference is
test macro-F1 paired by fold and seed, pooled over the tables with one array per seed, read with the
project's rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.cls_stage2_run import BASES, LEVELS, fold_metrics  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-30-step-zero")
SEEDS = (42, 7, 13)
COLLAPSED = 5


def trident(variant: str, seed: int) -> pd.DataFrame | None:
    for root in (STATE / "medusa", STATE):
        if seed == 42:
            found = [f for d in sorted(root.glob(f"optuna/{variant}/{variant}/optuna_*/best_model")) if (f := fold_metrics(d)) is not None]
            if found:
                return found[-1]
        else:
            frame = fold_metrics(root / "eval" / f"{variant}_s{seed}")
            if frame is not None:
                return frame
    return None


def gbdt(variant: str) -> tuple[pd.DataFrame, str] | None:
    csv, js = STATE / "gbdt" / "csv" / f"{variant}.csv", STATE / "gbdt" / "json" / f"{variant}.json"
    if not csv.exists() or not js.exists():
        return None
    return pd.read_csv(csv), str(pd.read_json(js, typ="series")["bar"])


def fork(variant: str, seed: int) -> pd.DataFrame | None:
    return fold_metrics(STATE / "refs" / "e87" / f"{variant}_s{seed}_A")


def tabpfn(variant: str, seed: int) -> pd.DataFrame | None:
    path = STATE / "refs" / "e92" / f"{variant}_s{seed}.csv"
    return pd.read_csv(path).sort_values("fold").reset_index(drop=True) if path.exists() else None


def _f1(frame: pd.DataFrame, column: str = "f1_macro") -> np.ndarray:
    return frame.sort_values("fold")[column].to_numpy(dtype=float)


def main() -> None:
    missing: list[str] = []
    calls: dict[str, str] = {}
    kept: list[float] = []
    for level in LEVELS:
        diffs: dict[str, dict[int, list[np.ndarray]]] = {k: {s: [] for s in SEEDS} for k in ("bar", "fork", "tabpfn", "tabpfn_balanced", "missforest+ind/hgb")}
        bars: dict[str, str] = {}
        means: dict[str, list[float]] = {"TRIDENT": [], "bar": []}
        for base in BASES:
            variant = f"{base}_{level}nan"
            g = gbdt(variant)
            if g is None:
                missing.append(f"{variant} gbdt")
                continue
            table, bar = g
            bars[base] = bar.split("/")[1]
            for seed in SEEDS:
                t = trident(variant, seed)
                if t is None:
                    missing.append(f"{variant} s{seed} trident")
                    continue
                ours = _f1(t)
                kept.extend(t["kept_epoch"].to_numpy(dtype=float).tolist())
                rows = table[table.seed == seed]
                for arm in ("bar", "missforest+ind/hgb"):
                    theirs = _f1(rows[rows.arm == (bar if arm == "bar" else arm)])
                    diffs[arm][seed].append(ours - theirs)
                means["TRIDENT"].append(float(ours.mean()))
                means["bar"].append(float(_f1(rows[rows.arm == bar]).mean()))
                f = fork(variant, seed)
                if f is not None:
                    diffs["fork"][seed].append(ours - _f1(f))
                p = tabpfn(variant, seed)
                if p is not None:
                    diffs["tabpfn"][seed].append(ours - _f1(p))
                    diffs["tabpfn_balanced"][seed].append(ours - _f1(p, "f1_macro_balanced"))
        print(f"## {level}nan · TRIDENT mean {np.mean(means['TRIDENT']):.4f}, bar mean {np.mean(means['bar']):.4f} · bar per table: "
              + ", ".join(f"{b} {a}" for b, a in bars.items()))
        for name, label in (("bar", "TRIDENT − tuned GBDT bar"), ("missforest+ind/hgb", "TRIDENT − missForest+ind+HGB"),
                            ("fork", "TRIDENT − fork budget (E87 A)"), ("tabpfn", "TRIDENT − TabPFN"), ("tabpfn_balanced", "TRIDENT − TabPFN balanced")):
            per_seed = [np.concatenate(diffs[name][s]) for s in SEEDS if diffs[name][s]]
            if len(per_seed) < len(SEEDS):
                print(f"- {label}: incomplete")
                continue
            v = common.verdict(per_seed)
            # The rule reads lower as better; here a positive difference favours TRIDENT.
            reading = {"better": "the other higher", "worse": "TRIDENT higher", "none": "no verdict"}.get(v.call, v.call)
            if name == "bar":
                calls[level] = reading
            print(f"- {label}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] seeds {', '.join(f'{s:+.4f}' for s in v.seeds)}, "
                  f"{sum(len(d) for d in per_seed)} fold pairs -> {reading}")
        print()
    print(f"- kept epochs: {len(kept)} folds, median {np.median(kept) if kept else float('nan'):.0f}, collapsed (below {COLLAPSED}): {sum(1 for k in kept if k < COLLAPSED)}")
    if missing:
        print(f"- missing: {len(missing)}: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
        print("\nPrimary (H1c): incomplete")
        return
    h1c = [calls[lv] for lv in ("40", "60", "80")]
    if any(c == "the other higher" for c in h1c):
        reading = "refuted: the bar is higher at " + ", ".join(f"{lv}nan" for lv, c in zip(("40", "60", "80"), h1c) if c == "the other higher")
    elif all(c == "TRIDENT higher" for c in h1c):
        reading = "supported: TRIDENT higher at 40, 60 and 80nan"
    else:
        reading = "unconfirmed, not refuted"
    print(f"\nPrimary (H1c, 40/60/80nan): {', '.join(f'{lv}nan {c}' for lv, c in zip(('40', '60', '80'), h1c))} -> {reading} ({common.RULE})")


if __name__ == "__main__":
    main()
