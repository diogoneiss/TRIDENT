"""The thesis's imputation table on the development set, from existing runs (T21; plan 2026-10-17, §2a).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/thesis_table.py [--seeds 101/202/303|42/7/13|404/505/606] [--linear DIR] [--tuned DIR] [--tabpfn DIR|none]

Per seed set, the model's K on today's default (``report_common.development_source``: E34 K,
E38 ALL and E39 D on 101/202/303; E35 K on 42/7/13, five variants labelled; E40 on 404/505/606)
against every opponent on the same folds:

- the bar (the weak tier);
- the tuned bar (E42's validation-only budget, the tuned weak tier's best);
- missForest;
- the strong tier (missForest and E41's arms);
- the tuned strong tier (E42's tuned missForest and E41's arms, the best by CV mean);
- the best of all;
- the per-type bar (item 7b).

The baselines a run's fold CSV lacks (lgbm and missForest on E34/E35, E41's arms everywhere)
come from E41's offline per-fold CSVs (``--linear``), which reproduce the logged baselines
exactly; the tuned families come from E42's per-fold CSVs (``--tuned``). TabPFN v2, the strong
tier's offline member since ADR 0028, comes from E55's per-fold CSVs (``--tabpfn``; ``none``
reads the strong tier as before the ADR); it joins the strong and tuned strong tiers and the best
of all wherever a seed set has it. A development win under
D4 needs K to beat both the bar and the tuned bar. Every count is printed with its test. Labelled
variants are never pooled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402

DEVELOPMENT = [
    *(f"{table}_{level}nan" for table in ("kr-vs-kp", "spambase", "credit-g") for level in (20, 40, 60, 80)),
    "vehicle_20nan", "vehicle_60nan", "biodeg_20nan", "biodeg_60nan", "kc2_20nan", "kc2_60nan",
    "pendigits_20nan", "letter_20nan", "electricity_20nan",
]
SEED_SETS = {"101/202/303": [101, 202, 303], "42/7/13": [42, 7, 13], "404/505/606": [404, 505, 606]}
E40_VARIANTS = {f"{t}_{level}nan" for t in ("kr-vs-kp", "spambase", "credit-g") for level in (40, 60, 80)} | {
    "vehicle_60nan", "biodeg_60nan", "kc2_60nan"}
OPPONENTS = ("bar", "tuned", "missforest", "strong", "tuned_strong", "best_of_all", "per_type")
SESOI = 0.01
DEFAULT_LINEAR = Path("/scratch2/diogoneiss/trident-2026-10-17-linear-baselines/csv")
DEFAULT_TUNED = Path("/scratch2/diogoneiss/trident-2026-10-17-baseline-budget")
TUNED_BAR = "impute/induced/baseline/tuned_bar/impute_score"
DEFAULT_TABPFN = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn/csv")
LINEAR_ARMS = ("condexp", "softimpute", "ice")


def _cells(variant: str, seeds: str, linear: Path, tuned: Path | None = None, tabpfn: Path | None = None) -> list[tuple[common.Run, pd.DataFrame]] | None:
    source = common.development_source(variant, seeds)
    if source is None or (seeds == "404/505/606" and variant not in E40_VARIANTS):
        return None
    found = common.parents(source.experiment, source.arm)
    cells = []
    for seed in SEED_SETS[seeds]:
        runs = found.get((variant, seed), [])
        if len(runs) != 1:
            return None
        frame = common.fold_table(runs[0])
        extra_path = linear / f"{variant}_s{seed}.csv"
        if extra_path.exists():
            extra = pd.read_csv(extra_path).sort_values("fold").reset_index(drop=True)
            shared = [c for c in extra.columns if c.startswith("impute/") and c in frame.columns and "/baseline/" in c]
            worst = max((float((extra[c] - frame[c]).abs().max()) for c in shared), default=0.0)
            if worst > 1e-9:
                sys.exit(f"{variant} s{seed}: E41's recomputed baselines differ from the run's by {worst:g}")
            new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
            frame = pd.concat([frame, extra[new]], axis=1)
        if tuned is not None:
            # E42's two passes: the weak tier's (with its tuned bar) and missForest's, whose own
            # ``tuned_bar`` column (mean/mode against the tuned missForest) is left out.
            for directory, only_missforest in ((tuned / "csv", False), (tuned / "csv_missforest", True)):
                path = directory / f"{variant}_s{seed}.csv"
                if path.exists():
                    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
                    new = [
                        c for c in extra.columns
                        if c.startswith("impute/") and c not in frame.columns and (not only_missforest or "/missforest_tuned" in c)
                    ]
                    frame = pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)
        if tabpfn is not None and (tabpfn / f"{variant}_s{seed}.csv").exists():
            extra = pd.read_csv(tabpfn / f"{variant}_s{seed}.csv").sort_values("fold").reset_index(drop=True)
            new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
            frame = pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)
        cells.append((runs[0], frame))
    return cells


def _opponent(frame: pd.DataFrame, which: str) -> tuple[str, np.ndarray]:
    if which == "tuned":
        if TUNED_BAR not in frame.columns:
            raise KeyError(which)
        return "tuned_bar", np.asarray(frame[TUNED_BAR], dtype=float)
    if which == "tuned_strong":
        names = ["missforest_tuned", *LINEAR_ARMS, *(n for n in common.OFFLINE_STRONG if f"impute/induced/baseline/{n}/impute_score" in frame.columns)]
        keys = [f"impute/induced/baseline/{name}/impute_score" for name in names]
        if any(key not in frame.columns for key in keys):
            raise KeyError(which)
        best = min(range(len(keys)), key=lambda i: float(frame[keys[i]].mean()))
        return names[best], np.asarray(frame[keys[best]], dtype=float)
    return common.opponent(frame, which)


def _contrast(cells: list[tuple[common.Run, pd.DataFrame]], which: str) -> tuple[str, common.Verdict] | None:
    per_seed, scale, names = [], [], []
    for run, frame in cells:
        if which in ("strong", "best_of_all") and "condexp" not in common.baselines_in(frame.columns, "induced"):
            return None
        try:
            name, theirs = _opponent(frame, which)
        except KeyError:
            return None
        per_seed.append(common.model(frame, run, "K") - theirs)
        scale.append(float(theirs.mean()))
        names.append(name)
    label = names[0] if len(set(names)) == 1 else "/".join(sorted(set(names)))
    return label, common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="101/202/303", choices=list(SEED_SETS))
    parser.add_argument("--linear", type=Path, default=DEFAULT_LINEAR)
    parser.add_argument("--tuned", type=Path, default=DEFAULT_TUNED)
    parser.add_argument("--tabpfn", default=str(DEFAULT_TABPFN))
    options = parser.parse_args()
    print(f"## Development set, seeds {options.seeds}, induced impute_score, K readout")
    print()
    print("| variant | K | " + " | ".join(f"K − {o}" for o in OPPONENTS) + " |")
    print("|---|---|" + "---|" * len(OPPONENTS))
    calls: dict[str, dict[str, str]] = {o: {} for o in OPPONENTS}
    gaps: dict[str, dict[str, float]] = {o: {} for o in OPPONENTS}
    for variant in DEVELOPMENT:
        cells = _cells(variant, options.seeds, options.linear, options.tuned, None if options.tabpfn == "none" else Path(options.tabpfn))
        if cells is None:
            continue
        source = common.development_source(variant, options.seeds)
        flag = " †" if source is not None and source.labelled else ""
        k = float(np.concatenate([common.model(frame, run, "K") for run, frame in cells]).mean())
        row = [f"`{variant}`{flag}", f"{k:.4f}"]
        for which in OPPONENTS:
            result = _contrast(cells, which)
            if result is None:
                row.append("—")
                continue
            name, v = result
            row.append(f"{name}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            if not flag:
                calls[which][variant] = v.call
                gaps[which][variant] = v.mean / max(k - v.mean, 1e-12)
        print("| " + " | ".join(row) + " |")
    print()
    print("† ran a configuration that is no longer the default (removed file or old encoder); left out of the counts.")
    print()
    pvalues = {}
    for which in OPPONENTS:
        if not calls[which]:
            print(f"- K vs {which}: not available")
            continue
        tables, below, p = common.across_tables(gaps[which])
        pvalues[which] = p
        print(f"- K vs {which} on {len(calls[which])} variants: {common.tally(calls[which].values())}; "
              f"tables below zero {below} of {tables} (two-sided Wilcoxon on per-table mean relative gaps, p = {p:.4f})")
        for level in (20, 40, 60, 80):
            at = [c for v, c in calls[which].items() if v.endswith(f"_{level}nan")]
            if at:
                print(f"  - {level}nan: {common.tally(at)}")
    if pvalues:
        adjusted = common.holm(pvalues)
        print("- Holm across opponents: " + ", ".join(f"{k} {v:.4f}" for k, v in adjusted.items()))
    both = [v for v in calls["bar"] if calls["bar"][v] == "better" and calls["tuned"].get(v) == "better"]
    lost = [v for v in calls["bar"] if "worse" in (calls["bar"][v], calls["tuned"].get(v))]
    if calls["tuned"]:
        print(f"- D4, a win needs K better than both the bar and the tuned bar: {len(both)} wins of {len(calls['bar'])}; "
              f"worse than either on {len(lost)} ({', '.join(lost) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
