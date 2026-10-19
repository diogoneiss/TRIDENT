"""Pre-registered analysis of the deployable calibration study (imputation-calibration/02, E47).

Read-only. Run from the checkout root:

    uv run --python 3.11 python scripts/experiments/deployable_calibration_report.py \
        [--tally | --integrity | --todo 1|2|3|4]

Tag experiment=deployable-calibration-2026-10-18, seeds 404, 505 and 606, on E40's 12 variants at
40% missing and above. Each run keeps H (the loss's checkpoint) as its headline and also logs, on
the same trajectory:

- HC_m, H after a calibration fitted on the fixed validation mask's cells (observed cells, so a
  deployed model can fit it): ``impute/<population>/masked_calibrated/*``;
- HC_g, H after the calibration fitted on the validation gaps (an oracle):
  ``impute/induced/calibrated/*``;
- K and KC, which must equal E40's cells fold by fold (``--integrity``).

Every reading is on ``impute_score`` over 15 fold pairs with ``report_common``'s verdict rule and the
1% SESOI (ADR 0020).
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

EXPERIMENT = "deployable-calibration-2026-10-18"
REFERENCE = "calibration-2026-10-16"  # E40: the same cells, K headline and KC diagnostic
VARIANTS = [
    *(f"{base}_{level}nan" for base in ("kr-vs-kp", "spambase", "credit-g") for level in (40, 60, 80)),
    "vehicle_60nan", "biodeg_60nan", "kc2_60nan",
]
NAMED = ["kr-vs-kp_80nan", "spambase_60nan", "spambase_80nan", "credit-g_60nan", "credit-g_80nan", "vehicle_60nan"]
SEEDS = [404, 505, 606]
SESOI = 0.01
QUEUES = 4
CELL_MINUTES = {
    "kr-vs-kp_40nan": 19.3, "kr-vs-kp_60nan": 19.0, "kr-vs-kp_80nan": 17.3,
    "spambase_40nan": 36.0, "spambase_60nan": 33.3, "spambase_80nan": 37.0,
    "credit-g_40nan": 4.7, "credit-g_60nan": 4.3, "credit-g_80nan": 5.0,
    "vehicle_60nan": 4.5, "biodeg_60nan": 9.9, "kc2_60nan": 3.6,
}
Cells = list[tuple[common.Run, pd.DataFrame]]
_FOUND: dict[str, dict[tuple[str, int], list[common.Run]]] = {}


def _runs(experiment: str) -> dict[tuple[str, int], list[common.Run]]:
    if experiment not in _FOUND:
        _FOUND[experiment] = common.parents(experiment)
    return _FOUND[experiment]


def cells(variant: str, experiment: str = EXPERIMENT) -> Cells | None:
    found = []
    for seed in SEEDS:
        runs = _runs(experiment).get((variant, seed), [])
        if len(runs) > 1:
            sys.exit(f"{variant} s{seed}: {len(runs)} finished runs; the design allows one")
        if not runs:
            return None
        found.append((runs[0], common.fold_table(runs[0])))
    return found


def values(run: common.Run, frame: pd.DataFrame, name: str, population: str) -> np.ndarray:
    if name in ("H", "K", "M", "KC"):
        return common.model(frame, run, name, population)
    if name == "HC_m":
        return np.asarray(frame[f"impute/{population}/masked_calibrated/impute_score"], dtype=float)
    if name == "HC_g":
        return np.asarray(frame[f"impute/{population}/calibrated/impute_score"], dtype=float)
    return common.opponent(frame, name, population)[1]


def contrast(found: Cells, first: str, second: str, population: str = "induced") -> common.Verdict:
    per_seed, scale = [], []
    for run, frame in found:
        a, b = values(run, frame, first, population), values(run, frame, second, population)
        per_seed.append(a - b)
        scale.append(float(b.mean()))
    return common.verdict(per_seed, sesoi=SESOI * float(np.mean(scale)))


def line(label: str, v: common.Verdict) -> str:
    return f"{label} {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}"


def tally() -> None:
    primary: dict[str, str] = {}
    for variant in VARIANTS:
        found = cells(variant)
        if found is None:
            print(f"- `{variant}`: incomplete")
            continue
        named = " (named)" if variant in NAMED else ""
        readings = {
            "HC_m − H": contrast(found, "HC_m", "H"),
            "HC_g − H": contrast(found, "HC_g", "H"),
            "masked: HC_m − H": contrast(found, "HC_m", "H", "masked"),
            "KC − K": contrast(found, "KC", "K"),
            "HC_m − K": contrast(found, "HC_m", "K"),
            "H − bar": contrast(found, "H", "bar"),
            "HC_m − bar": contrast(found, "HC_m", "bar"),
            "HC_m − strong": contrast(found, "HC_m", "strong"),
        }
        primary[variant] = readings["HC_m − H"].call
        gain_m, gain_g = -readings["HC_m − H"].mean, -readings["HC_g − H"].mean
        kept = f"{100 * gain_m / gain_g:.0f}%" if gain_g > 0 else "n/a"
        alphas = {
            key: float(np.mean([frame[f"impute/induced/{key}/mean_alpha"].mean() for _, frame in found]))
            for key in ("masked_calibrated", "calibrated")
            if all(f"impute/induced/{key}/mean_alpha" in frame.columns for _, frame in found)
        }
        fit_cells = float(np.mean([frame["impute/induced/masked_calibrated/n_fit_cells"].mean() for _, frame in found]))
        print(f"- `{variant}`{named}: " + "; ".join(line(k, v) for k, v in readings.items())
              + f"; HC_m keeps {kept} of HC_g's gain; mean alpha " + ", ".join(f"{k} {a:.3f}" for k, a in alphas.items())
              + f"; validation mask cells per fold {fit_cells:.0f}")
    if len(primary) < len(VARIANTS):
        print("\nPrimary: incomplete")
        return
    better_named = sum(primary[v] == "better" for v in NAMED)
    worse = [v for v in VARIANTS if primary[v] == "worse"]
    if better_named >= 4 and not worse:
        reading = "supported"
    elif len(worse) >= 2 or better_named <= 1:
        reading = "refuted"
    else:
        reading = "mixed"
    print()
    print(f"Primary, HC_m − H on the induced impute_score: better on {better_named} of the {len(NAMED)} named; "
          f"worse on {len(worse)} of {len(VARIANTS)} ({', '.join(worse) or 'none'}) -> {reading} ({common.RULE})")
    print("All calls: " + common.tally(primary.values()))


def integrity() -> None:
    """K and KC must equal E40's headline and calibrated readout, fold by fold."""
    for variant in VARIANTS:
        ours, theirs = cells(variant), cells(variant, REFERENCE)
        if ours is None or theirs is None:
            print(f"  {variant}: missing")
            continue
        worst = 0.0
        for (run_a, a), (run_b, b) in zip(ours, theirs):
            for readout in ("K", "KC"):
                # An all-categorical table (kr-vs-kp) has no numerical metric.
                for metric in ("impute_score", "rmse_num_z", "err_cat"):
                    try:
                        x = common.model(a, run_a, readout, "induced", metric)
                        y = common.model(b, run_b, readout, "induced", metric)
                    except KeyError:
                        continue
                    worst = max(worst, float(np.abs(x - y).max()))
        print(f"  {variant}: largest difference from E40 {worst:g}")


def queue(name: str) -> list[tuple[str, int]]:
    work = sorted(((CELL_MINUTES[v], v, s) for v in VARIANTS for s in SEEDS), key=lambda item: (-item[0], item[1], item[2]))
    loads = [0.0] * QUEUES
    assigned: list[list[tuple[str, int]]] = [[] for _ in range(QUEUES)]
    for minutes, variant, seed in work:
        target = loads.index(min(loads))
        loads[target] += minutes
        assigned[target].append((variant, seed))
    return assigned[int(name) - 1]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--integrity"]:
        integrity()
    elif argv[:1] == ["--todo"] and len(argv) == 2:
        for variant, seed in queue(argv[1]):
            if (variant, seed) not in _runs(EXPERIMENT):
                print(variant, seed)
    elif argv in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
