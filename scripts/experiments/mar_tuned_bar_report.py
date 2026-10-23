"""Pre-registered analysis of E68: the tuned bar on the MAR folds (imputation-protocol/23, T34).
Read-only.

    uv run --python 3.11 python scripts/experiments/mar_tuned_bar_report.py [--tally]

ADR 0022 decision 5: every study that claims against the bar runs E42's driver
(``baseline_budget_run.py``) on its own folds, and D4 counts a win only against both bars. Here the
driver's CSVs (``<state>/csv``) are merged into the fold tables of:

- E64's family (arm G, seeds 404/505/606): 6 tables × 20 and 60nan, MAR and control;
- E63's smoke (arm G, seeds 42/7/13): 4 tables at 60nan, MAR and control.

On the induced ``impute_score`` over 15 fold pairs, with the verdict rule:

- per mechanism, with the 1% SESOI: H − bar, H − tuned bar, K − bar and K − tuned bar, and D4
  (better than both bars) on H and on K;
- the mechanism effect, fold-paired: (H − tuned bar) and (K − tuned bar), MAR minus control;
- **primary,** over E64's 12 table-rates, (H − tuned bar) MAR minus control: ``supported`` (MAR costs
  the model ground against the tuned bar) if worse on 4 or more and better on at most 1;
  ``refuted`` if worse on at most 1; ``mixed`` otherwise. E63's four are secondary.

Under MAR the mean/mode denominator comes from observed values the mechanism biases, so scores are
compared across mechanisms only as differences of within-variant gaps.
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

STATE = Path("/scratch2/diogoneiss/trident-2026-10-23-mar-tuned-bar")
TUNED = "impute/induced/baseline/tuned_bar/impute_score"
SESOI = 0.01
# (label, experiment, seeds, [(table-rate label, MAR variant, control variant)])
FAMILY = ("E64", "mar-family-2026-10-22", [404, 505, 606],
          [(f"{b}_{lv}", f"{b}_marf_{lv}", f"{b}_marfctl_{lv}")
           for b in ("biodeg", "kc2", "vehicle", "kr-vs-kp", "spambase", "credit-g") for lv in ("20nan", "60nan")])
SMOKE = ("E63", "mar-smoke-2026-10-22", [42, 7, 13],
         [(f"{b}_60nan", f"{b}_mar_60nan", f"{b}_marctl_60nan") for b in ("biodeg", "kc2", "vehicle", "kr-vs-kp")])
Cells = list[tuple[common.Run, pd.DataFrame]]


def pairs() -> list[tuple[str, int]]:
    """Every (variant, seed) the driver must cover."""
    return [(v, s) for _, _, seeds, rows in (FAMILY, SMOKE) for _, mar, ctl in rows for v in (mar, ctl) for s in seeds]


def _merge(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
    return pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)


def cells(found: dict[tuple[str, int], list[common.Run]], variant: str, seeds: list[int]) -> Cells | None:
    out = []
    for seed in seeds:
        runs = found.get((variant, seed), [])
        path = STATE / "csv" / f"{variant}_s{seed}.csv"
        if len(runs) != 1 or not path.exists():
            return None
        out.append((runs[0], _merge(common.fold_table(runs[0]), path)))
    return out


def gaps(found: Cells, readout: str, which: str) -> tuple[list[np.ndarray], float]:
    per_seed, scale = [], []
    for run, frame in found:
        theirs = np.asarray(frame[TUNED], dtype=float) if which == "tuned" else common.opponent(frame, which)[1]
        per_seed.append(common.model(frame, run, readout) - theirs)
        scale.append(float(theirs.mean()))
    return per_seed, float(np.mean(scale))


def study(label: str, experiment: str, seeds: list[int], rows: list[tuple[str, str, str]]) -> dict[str, str]:
    found = common.parents(experiment, "G")
    primary: dict[str, str] = {}
    counts: dict[str, list[str]] = {}
    print(f"# {label} ({experiment}, seeds {'/'.join(map(str, seeds))})\n")
    for name, mar_variant, ctl_variant in rows:
        mar, ctl = cells(found, mar_variant, seeds), cells(found, ctl_variant, seeds)
        if mar is None or ctl is None:
            print(f"## `{name}`: incomplete\n")
            continue
        print(f"## `{name}`")
        for tag, found_m in (("MAR", mar), ("control", ctl)):
            parts, d4 = [], []
            for readout in ("H", "K"):
                calls = {}
                for which in ("bar", "tuned"):
                    per_seed, scale = gaps(found_m, readout, which)
                    v = common.verdict(per_seed, sesoi=SESOI * scale)
                    calls[which] = v.call
                    parts.append(f"{readout} − {'tuned bar' if which == 'tuned' else 'bar'} {v.mean:+.4f} {v.call}")
                win = calls["bar"] == "better" and calls["tuned"] == "better"
                d4.append(f"D4 on {readout}: {'win' if win else 'no win'}")
                counts.setdefault(f"{tag} D4 on {readout}", []).append("better" if win else "none")
            print(f"- {tag}: " + "; ".join(parts) + "; " + "; ".join(d4))
        for readout in ("H", "K"):
            a, _ = gaps(mar, readout, "tuned")
            b, _ = gaps(ctl, readout, "tuned")
            v = common.verdict([x - y for x, y in zip(a, b)])
            key = f"({readout} − tuned bar) MAR minus control"
            counts.setdefault(key, []).append(v.call)
            if readout == "H":
                primary[name] = v.call
            print(f"  - {key}: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print()
    for key, tallied in counts.items():
        if key.startswith("("):
            print(f"- {key}: {common.tally(tallied)}")
        else:
            print(f"- {key}: {sum(c == 'better' for c in tallied)} wins of {len(tallied)}")
    print()
    return primary


def tally() -> None:
    primary = study(*FAMILY)
    study(*SMOKE)
    expected = len(FAMILY[3])
    if len(primary) < expected:
        print(f"Primary: incomplete ({len(primary)} of {expected})")
        return
    worse = [k for k, c in primary.items() if c == "worse"]
    better = [k for k, c in primary.items() if c == "better"]
    reading = "supported" if len(worse) >= 4 and len(better) <= 1 else "refuted" if len(worse) <= 1 else "mixed"
    print(f"Primary, E64's (H − tuned bar) MAR minus control: worse on {worse or 'none'}, better on {better or 'none'} "
          f"-> {reading} ({common.RULE})")


if __name__ == "__main__":
    if sys.argv[1:] in ([], ["--tally"]):
        tally()
    elif sys.argv[1:] == ["--pairs"]:
        for variant, seed in pairs():
            print(variant, seed)
    else:
        sys.exit(__doc__)
