"""Pre-registered analysis of E62: today's default on fresh MCAR draws (imputation-protocol/17, T31).
Read-only.

    uv run --python 3.11 python scripts/experiments/mcar_draws_report.py [--tally | --todo 1|2|3|4 | --gate <raw_fold_metrics.csv>]

Four shipped variants (pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan, biodeg_80nan), each in
three MCAR draws: draw 1 is the shipped file, draws 2 and 3 come from
``datasets/generate_mcar_draws.py``. Every draw runs today's default at seeds 42/7/13 (tag
experiment=mcar-draws-2026-10-22), on the same folds, with TabPFN's offline CSVs merged (ADR 0028).
On the induced ``impute_score`` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- per draw: K − bar, K − strong (with TabPFN), K − TabPFN, and KC − strong (KC: the default output
  at 30% missing and above, ADR 0027);
- across a variant's three draws, per contrast: **same** (one call), a **reversal** (better in one
  draw and worse in another) or a **drift** (any other change);
- **primary:** K − bar and K − strong, on all four variants; ``holds`` with no reversal, ``tied to
  the draw`` with any, named.

``--gate`` compares a ``--no-mlflow`` run of credit-g_60nan at seed 42, the shipped draw, with E35's
run fold by fold on K.
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

EXPERIMENT = "mcar-draws-2026-10-22"
ORIGINALS = ("pendigits_20nan", "kr-vs-kp_40nan", "credit-g_60nan", "biodeg_80nan")
DRAWS = (1, 2, 3)
SEEDS = [42, 7, 13]
SESOI = 0.01
QUEUES = 4
STATE = Path("/scratch2/diogoneiss/trident-2026-10-22-mcar-draws")
TABPFN_DIR = STATE / "tabpfn"
CELL_MINUTES = {"pendigits": 65.0, "kr-vs-kp": 28.0, "credit-g": 7.0, "biodeg": 6.0}
CONTRASTS = (("K", "bar"), ("K", "strong"), ("K", "tabpfn"), ("KC", "strong"))
PRIMARY = (("K", "bar"), ("K", "strong"))
Cells = list[tuple[common.Run, pd.DataFrame]]


def draw_name(original: str, draw: int) -> str:
    if draw == 1:
        return original
    base, tag = original.split("_", 1)
    return f"{base}_d{draw}_{tag}"


def variants() -> list[str]:
    return [draw_name(o, d) for o in ORIGINALS for d in DRAWS]


def _merge(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    extra = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    new = [c for c in extra.columns if c.startswith("impute/") and c not in frame.columns]
    return pd.concat([frame.reset_index(drop=True), extra[new]], axis=1)


def cells(found: dict[tuple[str, int], list[common.Run]], variant: str) -> Cells | None:
    out = []
    for seed in SEEDS:
        runs = found.get((variant, seed), [])
        path = TABPFN_DIR / f"{variant}_s{seed}.csv"
        if len(runs) != 1 or not path.exists():
            return None
        out.append((runs[0], _merge(common.fold_table(runs[0]), path)))
    return out


def readout(run: common.Run, frame: pd.DataFrame, name: str) -> np.ndarray:
    """K, uncalibrated; KC, the calibrated output (the headline where ADR 0027 applied it)."""
    if name == "KC":
        key = "impute/induced/calibrated/impute_score"
        if key in frame.columns:
            return np.asarray(frame[key], dtype=float)
        return np.asarray(frame["impute/induced/impute_score"], dtype=float)
    return common.model(frame, run, name)


def contrast(found: Cells, name: str, which: str) -> tuple[common.Verdict, str]:
    per_seed, scale, names = [], [], set()
    for run, frame in found:
        opponent, theirs = common.opponent(frame, which)
        per_seed.append(readout(run, frame, name) - theirs)
        scale.append(float(theirs.mean()))
        names.add(opponent)
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale))), "/".join(sorted(names))


def pattern(calls: list[str]) -> str:
    if len(set(calls)) == 1:
        return "same"
    if "better" in calls and "worse" in calls:
        return "reversal"
    return "drift"


def tally() -> None:
    found = common.parents(EXPERIMENT)
    reversals: list[str] = []
    complete = 0
    for original in ORIGINALS:
        print(f"## `{original}`")
        draws = {d: cells(found, draw_name(original, d)) for d in DRAWS}
        if any(c is None for c in draws.values()):
            missing = [draw_name(original, d) for d, c in draws.items() if c is None]
            print(f"- incomplete: {missing}\n")
            continue
        complete += 1
        readings: dict[tuple[str, str], list[common.Verdict]] = {c: [] for c in CONTRASTS}
        for d, found_d in draws.items():
            assert found_d is not None
            parts = []
            for name, which in CONTRASTS:
                v, opponent = contrast(found_d, name, which)
                readings[name, which].append(v)
                parts.append(f"{name} − {which} ({opponent}) {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            label = "shipped" if d == 1 else f"seed {42 + int(original.split('_')[1][:2]) + 1000 * (d - 1)}"
            print(f"- draw {d} ({label}): " + "; ".join(parts))
        for (name, which), vs in readings.items():
            calls = [v.call for v in vs]
            means = [v.mean for v in vs]
            half = float(np.mean([(v.high - v.low) / 2 for v in vs]))
            kind = pattern(calls)
            flag = " (primary)" if (name, which) in PRIMARY else ""
            print(f"  - {name} − {which}{flag}: {' / '.join(calls)} -> {kind}; the draws' means span "
                  f"{max(means) - min(means):.4f} against a mean interval half-width of {half:.4f}")
            if (name, which) in PRIMARY and kind == "reversal":
                reversals.append(f"{original} {name} − {which}")
        print()
    if complete < len(ORIGINALS):
        print(f"Primary: incomplete ({complete} of {len(ORIGINALS)})")
        return
    reading = "holds" if not reversals else "tied to the draw"
    print(f"Primary: reversals on K − bar and K − strong: {reversals or 'none'} -> {reading} ({common.RULE})")


def gate(path: Path) -> None:
    """The shipped draw of credit-g_60nan at seed 42 against E35's run, fold by fold on K."""
    mine = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    e35 = common.parents("selection-replication-2026-10-04")[("credit-g_60nan", 42)]
    if len(e35) != 1:
        sys.exit(f"E35 has {len(e35)} runs of credit-g_60nan at seed 42")
    theirs = common.model(common.fold_table(e35[0]), e35[0], "K")
    key = "impute/induced/uncalibrated/impute_score"
    ours = np.asarray(mine[key if key in mine.columns else "impute/induced/impute_score"], dtype=float)
    gap = np.abs(ours - theirs)
    print(f"gate: K per fold {np.round(ours, 6).tolist()} against E35's {np.round(theirs, 6).tolist()}; largest difference {gap.max():.2e}")


def queue(name: str) -> list[tuple[str, int]]:
    work = sorted(((CELL_MINUTES[v.split("_")[0]], v, s) for v in variants() for s in SEEDS), key=lambda i: (-i[0], i[1], i[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        done = common.parents(EXPERIMENT)
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in done:
                print(variant, seed)
    elif argv[:1] == ["--gate"] and len(argv) == 2:
        gate(Path(argv[1]))
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
