"""Pre-registered analysis of E67: the gaps as ``[NULL]`` in the decode stage, alone
(imputation-protocol/22, T33). Read-only.

    uv run --python 3.11 python scripts/experiments/null_gaps_report.py [--tally | --todo 1|2|3|4 | --gate <raw_fold_metrics.csv>]

N (tag experiment=null-gaps-2026-10-23, arm N) is today's default with ``DECODE_GAP_TOKEN null``.
G is E45's or E46's run of the same variant and seed (arm G), on the same folds. Eight MNAR pairs:
biodeg's tilt dial at 60nan (slopes 0.25, 0.5, 1.0, 2.0) against biodeg_60nan, and E46's four
new pairs. On the induced ``impute_score`` over 15 fold pairs, with the verdict rule:

- per pair and mechanism, with the 1% SESOI of G's value: N − G on H and on K;
- the interaction, fold-paired: (N − G), MNAR minus MCAR, on H and on K;
- **primary,** the interaction on H over the 8 pairs: ``supported`` (the ``[NULL]`` token helps
  more under value-dependent missingness) if better on 3 or more and worse on none; ``refuted`` if
  better on at most 1; ``mixed`` otherwise;
- **the dial,** descriptive: the interaction on H by slope.

``--gate`` compares a ``--no-mlflow`` G cell of biodeg_probit_60nan at seed 42 with E45's run,
fold by fold on K and H.
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

EXPERIMENT = "null-gaps-2026-10-23"
SMOKE, FAMILY = "mnar-smoke-2026-10-17", "mnar-family-2026-10-17"
SEEDS = [42, 7, 13]
SESOI = 0.01
QUEUES = 4
DIAL = ("0p25", "", "1p0", "2p0")
# (MNAR variant, its G's study, MCAR control, the control's G's study); the dial's control is
# E45's, as ``mnar_family_report.dial`` and E58 read it.
PAIRS = [
    *((f"biodeg_probit{s}_60nan", SMOKE if s == "" else FAMILY, "biodeg_60nan", SMOKE) for s in DIAL),
    ("spambase_probit_60nan", FAMILY, "spambase_60nan", FAMILY),
    *((f"{t}_probit_20nan", FAMILY, f"{t}_20nan", FAMILY) for t in ("biodeg", "kc2", "vehicle")),
]
CELL_MINUTES = {"spambase": 30.0, "biodeg": 6.0, "kc2": 5.0, "vehicle": 5.0}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[tuple[str, str], dict[tuple[str, int], list[common.Run]]] = {}


def variants() -> list[str]:
    return sorted({v for mnar, _, mcar, _ in PAIRS for v in (mnar, mcar)})


def cells(variant: str, experiment: str, arm: str) -> Cells | None:
    if (experiment, arm) not in _FOUND:
        _FOUND[experiment, arm] = common.parents(experiment, arm)
    out = []
    for seed in SEEDS:
        runs = _FOUND[experiment, arm].get((variant, seed), [])
        if len(runs) != 1:
            return None
        out.append((runs[0], common.fold_table(runs[0])))
    return out


def arm_gap(n: Cells, g: Cells, readout: str) -> tuple[list[np.ndarray], float]:
    per_seed, scale = [], []
    for (rn, fn), (rg, fg) in zip(n, g):
        assert rn.seed == rg.seed
        theirs = common.model(fg, rg, readout)
        per_seed.append(common.model(fn, rn, readout) - theirs)
        scale.append(float(theirs.mean()))
    return per_seed, float(np.mean(scale))


def tally() -> None:
    interactions: dict[str, dict[str, str]] = {"H": {}, "K": {}}
    for mnar, mnar_from, mcar, mcar_from in PAIRS:
        found = {
            ("N", "mnar"): cells(mnar, EXPERIMENT, "N"), ("G", "mnar"): cells(mnar, mnar_from, "G"),
            ("N", "mcar"): cells(mcar, EXPERIMENT, "N"), ("G", "mcar"): cells(mcar, mcar_from, "G"),
        }
        if any(v is None for v in found.values()):
            print(f"## `{mnar}` against `{mcar}`: incomplete\n")
            continue
        print(f"## `{mnar}` against `{mcar}`")
        for readout in ("H", "K"):
            gaps = {}
            parts = []
            for mechanism in ("mnar", "mcar"):
                n, g = found["N", mechanism], found["G", mechanism]
                assert n is not None and g is not None
                per_seed, scale = arm_gap(n, g, readout)
                gaps[mechanism] = per_seed
                v = common.verdict(per_seed, sesoi=SESOI * scale)
                parts.append(f"{mechanism.upper()} N − G {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            inter = common.verdict([a - b for a, b in zip(gaps["mnar"], gaps["mcar"])])
            interactions[readout][mnar] = inter.call
            print(f"- {readout}: " + "; ".join(parts) + f"; (N − G), MNAR minus MCAR {inter.mean:+.4f} [{inter.low:+.4f}, {inter.high:+.4f}] {inter.call}")
        print()
    print("## The dial (biodeg at 60nan), (N − G on H), MNAR minus MCAR by slope")
    for slope in DIAL:
        name = f"biodeg_probit{slope}_60nan"
        print(f"- slope {slope or '0p5'}: {interactions['H'].get(name, 'incomplete')}")
    print()
    for readout in ("H", "K"):
        print(f"- (N − G on {readout}), MNAR minus MCAR: {common.tally(interactions[readout].values())}")
    if len(interactions["H"]) < len(PAIRS):
        print(f"\nPrimary: incomplete ({len(interactions['H'])} of {len(PAIRS)})")
        return
    better = [k for k, c in interactions["H"].items() if c == "better"]
    worse = [k for k, c in interactions["H"].items() if c == "worse"]
    reading = "supported" if len(better) >= 3 and not worse else "refuted" if len(better) <= 1 else "mixed"
    print(f"\nPrimary, (N − G on H), MNAR minus MCAR: better on {better or 'none'}, worse on {worse or 'none'} -> {reading} ({common.RULE})")


def gate(path: Path) -> None:
    mine = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
    e45 = common.parents(SMOKE, "G")[("biodeg_probit_60nan", 42)]
    if len(e45) != 1:
        sys.exit(f"E45 has {len(e45)} G runs of biodeg_probit_60nan at seed 42")
    frame = common.fold_table(e45[0])
    for readout, key in (("K", "impute/induced/uncalibrated/impute_score"), ("H", "impute/induced/loss_checkpoint/impute_score")):
        theirs = common.model(frame, e45[0], readout)
        column = key if key in mine.columns else "impute/induced/impute_score"
        ours = np.asarray(mine[column], dtype=float)
        print(f"gate {readout}: {np.round(ours, 6).tolist()} against E45's {np.round(theirs, 6).tolist()}; largest difference {np.abs(ours - theirs).max():.2e}")


def queue(which: str) -> list[tuple[str, int]]:
    work = sorted(((CELL_MINUTES[v.split("_")[0]], v, s) for v in variants() for s in SEEDS), key=lambda i: (-i[0], i[1], i[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(which) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        done = common.parents(EXPERIMENT, "N")
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in done:
                print(variant, seed)
    elif argv[:1] == ["--gate"] and len(argv) == 2:
        gate(Path(argv[1]))
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
