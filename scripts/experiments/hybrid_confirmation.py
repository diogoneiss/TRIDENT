"""E90, E91 and E93: the deployable hybrid's `table` rule on fresh seeds (imputation-protocol/34 and 35, T39).

    uv run --python 3.11 python -B scripts/experiments/hybrid_confirmation.py --study <e90|e91|e93> train <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_confirmation.py --todo <n>
    uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_confirmation.py --study <e90|e91> cells <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/hybrid_confirmation.py --study <e90|e91> --tally

E82's shape on seeds 4545/5656/6767: ``train`` is ``deployable_hybrid.train`` (H, with the hidden observed
validation cells read in place), ``cells`` is ``hybrid_selection_rules.cells`` (TabPFN refit, both cell
tables kept), each pointed at this study's experiment tag and state. E90 runs E82's 10 hard variants;
E91 seven variants where the model already leads or ties; E93 E84's 12 MNAR and MAR variants at 60nan. ``--todo`` deals both studies' cells,
over ``QUEUES`` training queues, the longest first. ``--tally`` scores H, TabPFN, the equal blend and
the four rules of E89, and applies this study's rule to ``table``.
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

from scripts.experiments import blend_weights as weights  # noqa: E402
from scripts.experiments import deployable_hybrid as e82  # noqa: E402
from scripts.experiments import hybrid_selection_rules as rules  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.spread_mechanisms import SHIFTED  # noqa: E402

SEEDS = (4545, 5656, 6767)
STUDIES = {
    "e90": ("hybrid-confirmation-2026-10-28", Path("/scratch2/diogoneiss/trident-2026-10-28-hybrid-confirmation"), weights.HARD),
    "e91": ("hybrid-generality-2026-10-28", Path("/scratch2/diogoneiss/trident-2026-10-28-hybrid-generality"),
            ("biodeg_40nan", "kc2_40nan", "kc2_80nan", "pendigits_40nan", "pendigits_60nan", "pendigits_80nan", "vehicle_40nan")),
    "e93": ("hybrid-mechanisms-2026-10-29", Path("/scratch2/diogoneiss/trident-2026-10-29-hybrid-mechanisms"), tuple(SHIFTED)),
}
# Minutes a cell, from E82's and E44's runs of the same tables; they only deal the cells over the queues.
MINUTES = {"electricity": 97, "pendigits": 37, "spambase": 37, "kr-vs-kp": 12, "biodeg": 5, "credit-g": 3.5, "vehicle": 3, "kc2": 2.7}
QUEUES = 4


def use(study: str) -> tuple[str, ...]:
    experiment, state, variants = STUDIES[study]
    e82.EXPERIMENT, e82.STATE, e82.SEEDS, e82.VARIANTS = experiment, state, SEEDS, variants
    rules.STATE = state
    return variants


def todo(which: int) -> None:
    done = {study: common.parents(experiment) for study, (experiment, _, _) in STUDIES.items()}
    work = sorted(((MINUTES[v.split("_")[0]], study, v, s) for study, (_, _, variants) in STUDIES.items() for v in variants for s in SEEDS),
                  key=lambda w: (-w[0], w[1], w[2], w[3]))
    loads = [0.0] * QUEUES
    for cost, study, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += cost
        # The store copy can be ten minutes behind; a finished cell's last selection file is not.
        written = (STUDIES[study][1] / "selection" / f"{variant}_s{seed}" / "selection_fold_5.csv").exists()
        if target == which - 1 and (variant, seed) not in done[study] and not written:
            print(study, variant, seed)


def tally(study: str) -> None:
    variants = use(study)
    done = [(v, s) for v in variants for s in SEEDS if (rules.STATE / "cells" / f"{v}_s{s}" / "scale_fold_5.json").exists()]
    if len(done) < len(variants) * len(SEEDS):
        print(f"Primary: incomplete ({len(done)} of {len(variants) * len(SEEDS)} cells)")
        return
    table = pd.concat([rules.scores(v, s) for v, s in done], ignore_index=True)
    calls: dict[str, dict[str, dict[str, str]]] = {rule: {} for rule in rules.RULES}
    for variant in variants:
        values = {m: weights._values(table, variant, list(SEEDS), m) for m in ("model", "tabpfn", "equal", *rules.RULES)}
        parts = []
        for rule in rules.RULES:
            calls[rule][variant] = {}
            for other in ("model", "tabpfn", "equal"):
                v = common.verdict([a - b for a, b in zip(values[rule], values[other])], sesoi=rules.SESOI * float(np.mean([o.mean() for o in values[other]])))
                calls[rule][variant][other] = v.call
                if rule == "table" or other == "tabpfn":
                    parts.append(f"{rule} − {other} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        print(f"- `{variant}` (H {np.concatenate(values['model']).mean():.4f}, TabPFN {np.concatenate(values['tabpfn']).mean():.4f}): " + "; ".join(parts))
    print()
    for rule in rules.RULES:
        over = [v for v, c in calls[rule].items() if c["model"] == "better"]
        under = [v for v, c in calls[rule].items() if c["model"] == "worse"]
        behind = [v for v, c in calls[rule].items() if c["tabpfn"] == "worse"]
        if study in ("e90", "e93"):
            reading = ("supported" if len(over) >= 8 and not under and len(behind) <= 2
                       else "refuted" if len(over) <= 5 or len(under) >= 2 or len(behind) >= 5 else "mixed")
        else:
            reading = ("supported" if not under and len(behind) <= 1
                       else "refuted" if len(under) >= 2 or len(behind) >= 3 else "mixed")
        label = "Primary" if rule == "table" else "Secondary"
        print(f"{label} `{rule}`: better than H on {len(over)} of {len(variants)}, worse on {under or 'none'}, "
              f"worse than TabPFN on {len(behind)} ({behind or 'none'}) -> {reading}")
    print(f"\n({common.RULE}; SESOI {rules.SESOI:.0%} of the opponent's mean)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--todo"] and len(argv) == 2:
        todo(int(argv[1]))
    elif argv[:1] == ["--study"] and len(argv) >= 3 and argv[1] in STUDIES:
        use(argv[1])
        rest = argv[2:]
        if rest[:1] == ["train"] and len(rest) == 3:
            e82.train(rest[1], int(rest[2]), False)
        elif rest[:1] == ["cells"] and len(rest) == 3:
            rules.cells(rest[1], int(rest[2]))
        elif rest == ["--tally"]:
            tally(argv[1])
        else:
            sys.exit(__doc__)
    else:
        sys.exit(__doc__)
