"""Pre-registered analysis of E59: ADR 0018's bigger encoder on the large tables at 40-80nan
(imputation-protocol/14). Read-only.

    uv run --python 3.11 python scripts/experiments/large_encoder_report.py [--tally | --todo 1|2|3|4]

ALL (tag experiment=large-encoder-2026-10-21) against G, E44's runs of today's default on the same
seeds (707/808/909), on pendigits and letter at 40, 60 and 80nan. On the induced ``impute_score``
over 15 fold pairs, the verdict rule and the 1% SESOI:

- **primary:** ALL − G on K (uncalibrated);
- **secondary:** the same on KC (the default output at these rates, ADR 0027) and on H; and each
  arm's KC − TabPFN (E55's CSVs on the same folds), to see whether the bigger encoder closes the
  80nan gap.
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

EXPERIMENT = "large-encoder-2026-10-21"
REFERENCE = "confirmation-2026-10-17"
VARIANTS = tuple(f"{t}_{level}nan" for t in ("pendigits", "letter") for level in (40, 60, 80))
SEEDS = [707, 808, 909]
SESOI = 0.01
QUEUES = 4
TABPFN_DIR = Path("/scratch2/diogoneiss/trident-2026-10-21-tabpfn/csv")
TABPFN = "impute/induced/baseline/tabpfn/impute_score"
CELL_MINUTES = {"pendigits": 67.0, "letter": 110.0}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[tuple[str, str | None], dict[tuple[str, int], list[common.Run]]] = {}


def cells(variant: str, experiment: str, arm: str | None) -> Cells | None:
    if (experiment, arm) not in _FOUND:
        _FOUND[experiment, arm] = common.parents(experiment, arm)
    found = []
    for seed in SEEDS:
        runs = _FOUND[experiment, arm].get((variant, seed), [])
        if len(runs) != 1:
            return None
        found.append((runs[0], common.fold_table(runs[0])))
    return found


def readout(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "KC":
        key = "impute/induced/calibrated/impute_score"
        if key in frame.columns:
            return np.asarray(frame[key], dtype=float)
        return np.asarray(frame["impute/induced/impute_score"], dtype=float)
    return common.model(frame, run, name)


def contrast(a: Cells, b: Cells, name: str) -> common.Verdict:
    per_seed, scale = [], []
    for (ra, fa), (rb, fb) in zip(a, b):
        x, y = readout(ra, fa, name), readout(rb, fb, name)
        per_seed.append(x - y)
        scale.append(float(y.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def against_tabpfn(found: Cells, variant: str) -> common.Verdict | None:
    per_seed, scale = [], []
    for run, frame in found:
        path = TABPFN_DIR / f"{variant}_s{run.seed}.csv"
        if not path.exists():
            return None
        theirs = pd.read_csv(path).sort_values("fold")[TABPFN].to_numpy(dtype=float)
        per_seed.append(readout(run, frame, "KC") - theirs)
        scale.append(float(theirs.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def tally() -> None:
    calls: dict[str, str] = {}
    for variant in VARIANTS:
        g, all_ = cells(variant, REFERENCE, None), cells(variant, EXPERIMENT, "ALL")
        if g is None or all_ is None:
            print(f"- `{variant}`: incomplete")
            continue
        k = float(np.concatenate([common.model(f, r, "K") for r, f in g]).mean())
        parts = []
        for name in ("K", "KC", "H"):
            v = contrast(all_, g, name)
            parts.append(f"ALL − G, {name} {v.mean:+.4f} ({100 * v.mean / k:+.1f}% of G's K) [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            if name == "K":
                calls[variant] = v.call
        for arm, found in (("G", g), ("ALL", all_)):
            t = against_tabpfn(found, variant)
            parts.append(f"{arm} KC − tabpfn " + ("not available" if t is None else f"{t.mean:+.4f} {t.call}"))
        print(f"- `{variant}`: " + "; ".join(parts))
    if len(calls) < len(VARIANTS):
        print("\nPrimary: incomplete")
        return
    better = [v for v, c in calls.items() if c == "better"]
    worse = [v for v, c in calls.items() if c == "worse"]
    reading = "supported" if len(better) >= 4 and not worse else "refuted" if len(better) <= 1 or len(worse) >= 2 else "mixed"
    print(f"\nPrimary, ALL − G on K: better on {better or 'none'}, worse on {worse or 'none'} -> {reading} ({common.RULE})")


def queue(name: str) -> list[tuple[str, int, str]]:
    work = sorted(((CELL_MINUTES[v.split("_")[0]], v, s) for v in VARIANTS for s in SEEDS), key=lambda i: (-i[0], i[1], i[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int, str]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed, "ALL"))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        done = common.parents(EXPERIMENT, "ALL")
        for variant, seed, arm in queue(argv[1]):
            if (variant, seed) not in done:
                print(variant, seed, arm)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
