"""E164 and E165: the deployable hybrid with missForest as the other side, and a rule that picks the side, on every
variant with T39's cell tables (imputation-protocol/60 and /61).

    uv run --python 3.11 python -B scripts/experiments/missforest_generality.py cells <study> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/missforest_generality.py --lanes <n>
    uv run --python 3.11 python -B scripts/experiments/missforest_generality.py --tally <e164|e165|e168>

``cells`` adds missForest's fills (refit per training fold, asserted equal to the run's logged missForest score fold by
fold) beside H's and TabPFN's in E90's, E91's, E93's and E95's cell tables (seeds 4545/5656/6767). E90/E91/E93's tables
are verified copies from gorgona8 under ``SOURCE``; E95's are medusa1's own. Everything runs on medusa1, CPU only,
nothing in the store. ``--lanes`` starts detached lanes over the missing pairs, largest tables first; progress goes to
``STATE/progress.log``. ``--tally e164`` reads `table_missforest` against H and missForest; ``--tally e165`` reads the
pick (missforest_hybrid.both_sides) against H and each fixed side; ``--tally e168`` the cross-fit pick (imputation-protocol/63),
on these development tables, against H, `table_tabpfn` and the pick. Run from the checkout root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

STATE = Path("/scratch/diogoneiss/trident-2026-11-08-missforest-generality")
SOURCE = STATE / "source"
SEEDS = (4545, 5656, 6767)
from scripts.experiments import hybrid_confirmation as confirmation  # noqa: E402

# study: (experiment, cell tables, variants); the variants are hybrid_confirmation.STUDIES's
_TABLES = {"e90": SOURCE / "confirmation", "e91": SOURCE / "generality", "e93": SOURCE / "mechanisms",
           "e95": Path("/scratch/diogoneiss/trident-2026-10-29-hybrid-20nan/cells")}
STUDIES: dict[str, tuple[str, Path, tuple[str, ...]]] = {
    k: (confirmation.STUDIES[k][0], tables, tuple(confirmation.STUDIES[k][2])) for k, tables in _TABLES.items()}
# Where the runs' own H trails their logged missForest by more than 1% of missForest's score (complete-table scores,
# named before the cells, from the four studies' parent runs on 2026-11-08): E164's secondary scope.
TRAILS = ("biodeg_80nan", "vehicle_80nan", "credit-g_40nan", "biodeg_probit0p25_60nan", "biodeg_probit_60nan",
          "vehicle_probit_60nan", "electricity_20nan")
ROWS = {"electricity": 45312, "letter": 20000, "pendigits": 10992, "spambase": 4601, "kr-vs-kp": 3196, "biodeg": 1055,
        "credit-g": 1000, "vehicle": 846, "kc2": 522}


def _pairs() -> list[tuple[str, str, int]]:
    return [(study, v, s) for study, (_, _, variants) in STUDIES.items() for v in variants for s in SEEDS]


def _done(study: str, variant: str, seed: int) -> bool:
    return (STATE / "cells" / study / f"{variant}_s{seed}" / "selection_fold_5.csv").exists()


def cells(study: str, variant: str, seed: int) -> None:
    from scripts.experiments import missforest_hybrid as mh

    experiment, source, _ = STUDIES[study]
    mh.add_missforest(experiment, source, STATE / "cells" / study, variant, seed)


def lanes(n: int) -> None:
    todo = sorted((p for p in _pairs() if not _done(*p)), key=lambda p: -ROWS[p[1].split("_")[0]])
    loads, assigned = [0.0] * n, [[] for _ in range(n)]  # type: list[float], list[list[tuple[str, str, int]]]
    for pair in todo:
        k = loads.index(min(loads))
        loads[k] += ROWS[pair[1].split("_")[0]]
        assigned[k].append(pair)
    STATE.mkdir(parents=True, exist_ok=True)
    log = STATE / "progress.log"
    env = "export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
    for k, group in enumerate(assigned, start=1):
        steps = [f"echo \"$(date '+%F %T %z') start {st} {v} s{s}\" >> {log}; "
                 f"uv run --python 3.11 python -B scripts/experiments/missforest_generality.py cells {st} {v} {s} >> {STATE}/lane{k}.log 2>&1 "
                 f"&& echo \"$(date '+%F %T %z') done {st} {v} s{s}\" >> {log} || echo \"$(date '+%F %T %z') FAILED {st} {v} s{s}\" >> {log}"
                 for st, v, s in group]
        script = f"{env}; " + "; ".join(steps) + f"; echo \"$(date '+%F %T %z') lane {k} complete\" >> {log}"
        subprocess.Popen(["setsid", "nohup", "bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"lane {k}: {len(group)} pairs")


def tally(which: str) -> None:
    import numpy as np

    from scripts.experiments import missforest_hybrid as mh
    from scripts.experiments import report_common as common

    missing = [p for p in _pairs() if not _done(*p)]
    if missing:
        print(f"incomplete: {len(missing)} of {len(_pairs())} pairs")
        return
    pairs = {"e164": (("table_missforest", "H"), ("table_missforest", "missforest")),
             "e165": (("pick", "H"), ("pick", "table_tabpfn"), ("pick", "table_missforest")),
             "e168": (("crossfit", "H"), ("crossfit", "table_tabpfn"), ("crossfit", "pick"))}[which]
    calls: dict[str, dict[str, str]] = {}
    shares: list[float] = []
    for study, (_, _, variants) in STUDIES.items():
        print(f"## {study}")
        for variant in variants:
            per = [mh.both_sides(STATE / "cells" / study / f"{variant}_s{s}") for s in SEEDS]
            values = {m: [np.asarray(p[m]) for p in per] for m in per[0]}
            parts = []
            for first, second in pairs:
                scale = float(np.mean([v.mean() for v in values[second]]))
                v = common.verdict([a - b for a, b in zip(values[first], values[second])], sesoi=mh.SESOI * scale)
                calls.setdefault(variant, {})[f"{first}-{second}"] = v.call
                parts.append(f"{first} − {second} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
            share = float(np.concatenate(values["crossfit_missforest" if which == "e168" else "picked_missforest"]).mean())
            shares.append(share)
            levels = ", ".join(f"{m} {np.concatenate(values[m]).mean():.4f}" for m in ("H", "tabpfn", "missforest", "table_tabpfn", "table_missforest", "pick")
                           + (("crossfit",) if which == "e168" else ()))
            print(f"- `{variant}`{' (trails)' if variant in TRAILS else ''}: {levels}; {'cross-fit ' if which == 'e168' else ''}pick took missForest on {share:.0%} of folds\n  - "
                  + "\n  - ".join(parts))
    n = len(calls)

    def count(key: str, call: str, scope: tuple[str, ...] | None = None) -> list[str]:
        return [v for v, c in calls.items() if c[key] == call and (scope is None or v in scope)]

    print()
    if which == "e164":
        under, behind = count("table_missforest-H", "worse"), count("table_missforest-missforest", "worse")
        reading = ("supported" if not under and len(behind) <= 2 else "refuted" if len(under) >= 2 or len(behind) >= 5 else "mixed")
        print(f"PRIMARY `table_missforest` on {n} variants: worse than H on {under or 'none'}, worse than missForest on "
              f"{behind or 'none'} -> {reading}")
        print(f"SECONDARY better than H on {len(count('table_missforest-H', 'better'))} of {n}; on the {len(TRAILS)} where H trails "
              f"missForest: better than H on {count('table_missforest-H', 'better', TRAILS) or 'none'}")
    else:
        rule = "crossfit" if which == "e168" else "pick"
        under = count(f"{rule}-H", "worse")
        over, behind = count(f"{rule}-table_tabpfn", "better"), count(f"{rule}-table_tabpfn", "worse")
        reading = ("supported" if not under and len(behind) <= 1 and len(over) >= 2
                   else "refuted" if len(behind) >= 3 or not over else "mixed")
        print(f"{'DEVELOPMENT' if which == 'e168' else 'PRIMARY'} {rule} on {n} variants: worse than H on {under or 'none'}; against "
              f"`table_tabpfn` better on {over or 'none'}, worse on {behind or 'none'} -> {reading}")
        if which == "e168":
            print(f"`crossfit` − `pick`: better on {count('crossfit-pick', 'better') or 'none'}, worse on {count('crossfit-pick', 'worse') or 'none'}; "
                  f"the cross-fit pick took missForest on {np.mean(shares):.0%} of folds overall")
        else:
            print(f"SECONDARY against `table_missforest`: better on {len(count('pick-table_missforest', 'better'))}, worse on "
                  f"{count('pick-table_missforest', 'worse') or 'none'}; the pick took missForest on {np.mean(shares):.0%} of folds overall")
    print(f"({common.RULE}; SESOI {mh.SESOI:.0%} of the opponent's mean)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["cells"] and len(argv) == 4:
        cells(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--lanes"] and len(argv) == 2:
        lanes(int(argv[1]))
    elif argv[:1] == ["--tally"] and len(argv) == 2 and argv[1] in ("e164", "e165", "e168"):
        tally(argv[1])
    else:
        sys.exit(__doc__)
