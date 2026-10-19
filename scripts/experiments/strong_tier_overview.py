"""The TRIDENT-against-baselines page, again with the strong tier (ADR 0021), read-only.

    uv run --python 3.11 python scripts/experiments/strong_tier_overview.py <old_page.html> <template.html> <out.html>

The first version of the page (artifact "TRIDENT contra os baselines de imputação", 2026-10-15)
inlines its data as ``const DATA = {...};``. This script rebuilds every number of that data from the
same runs (E35, tag experiment=selection-replication-2026-10-04, seeds 42/7/13, the model's K
readout ``induced_checkpoint``), refuses unless it reproduces the old numbers to 1e-9, and adds the
strong tier: condexp, SoftImpute and ICE from E41's per-fold CSVs (which the backfill dry run matched
to the store within 4.4e-16) beside missForest, and per population the strong tier's best (the
lowest mean over the 15 folds, per seed as each run's ``best_strong_baseline`` tag would choose) and
the model's paired difference to it. The section on why some tables impute better keeps the old
page's data. The template's ``__DATA__`` is replaced by the data and written to ``out``.

The verdict is the old page's: the 95% t-interval over the 15 fold differences excludes zero and the
three seeds' mean differences agree in sign.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import report_common as common  # noqa: E402
from scripts.experiments.thesis_table import DEFAULT_LINEAR, _cells  # noqa: E402

SEEDS = "42/7/13"
POPULATIONS = ("induced", "masked")
WEAK = ("mean_mode", "knn5", "knn10", "hgb", "lgbm")
STRONG = ("missforest", "condexp", "softimpute", "ice")
NEW = ("condexp", "softimpute", "ice")
CALLS = {"better": "model better", "worse": "baseline better", "none": "no detectable difference"}
TOLERANCE = 1e-9


def _interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def _stats(per_seed: list[np.ndarray]) -> dict[str, object]:
    mean, lo, hi = _interval(np.concatenate(per_seed))
    return {"mean": mean, "lo": lo, "hi": hi, "seeds": [float(s.mean()) for s in per_seed]}


def _call(per_seed: list[np.ndarray]) -> str:
    _, lo, hi = _interval(np.concatenate(per_seed))
    signs = {np.sign(s.mean()) for s in per_seed}
    if hi < 0 and signs == {-1.0}:
        return "model better"
    if lo > 0 and signs == {1.0}:
        return "baseline better"
    return "no detectable difference"


def _against(model: list[np.ndarray], other: list[np.ndarray], scale: float) -> dict[str, object]:
    diffs = [a - b for a, b in zip(model, other)]
    pooled = float(np.concatenate(diffs).mean())
    return {"difference": {**_stats(diffs), "pct": 100 * pooled / scale}, "call": _call(diffs)}


def _close(a: object, b: object, where: str) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict)
        for key in a:
            _close(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), where
        if abs(float(a) - float(b)) > TOLERANCE:
            sys.exit(f"{where}: the old page has {a}, the store gives {b}")
    elif a != b:
        sys.exit(f"{where}: the old page has {a!r}, the store gives {b!r}")


def population(cells: list[tuple[common.Run, pd.DataFrame]], population: str) -> dict[str, object]:
    def values(name: str) -> list[np.ndarray]:
        out = []
        for run, frame in cells:
            if name == "model":
                out.append(common.model(frame, run, "K", population))
            else:
                out.append(frame[f"impute/{population}/baseline/{name}/impute_score"].to_numpy(dtype=float))
        return out

    folds = {name: values(name) for name in ("model", *WEAK, *STRONG)}
    methods = {name: _stats(per_seed) for name, per_seed in folds.items()}
    means = {name: float(np.concatenate(per_seed).mean()) for name, per_seed in folds.items()}

    def best(names: tuple[str, ...]) -> tuple[str, list[str]]:
        overall = min(names, key=lambda n: means[n])
        by_seed = [min(names, key=lambda n: float(folds[n][k].mean())) for k in range(len(cells))]
        return overall, by_seed

    weak, weak_by_seed = best(WEAK)
    strong, strong_by_seed = best(STRONG)
    to_weak = _against(folds["model"], folds[weak], means[weak])
    return {
        "methods": methods,
        "best": weak,
        "best_by_seed": weak_by_seed,
        "difference": to_weak["difference"],
        "call": to_weak["call"],
        "external": {"missforest": _against(folds["model"], folds["missforest"], means["missforest"])},
        "strong": {
            "best": strong,
            "best_by_seed": strong_by_seed,
            **_against(folds["model"], folds[strong], means[strong]),
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.exit(__doc__)
    old_page, template, out = (Path(a) for a in argv)
    line = next(l for l in old_page.read_text().splitlines() if l.startswith("const DATA = "))
    old = json.loads(line[len("const DATA = "):].rstrip().rstrip(";"))
    variants = []
    for before in old["variants"]:
        cells = _cells(before["variant"], SEEDS, DEFAULT_LINEAR)
        if cells is None:
            sys.exit(f"{before['variant']}: no complete cells on {SEEDS}")
        after = {key: value for key, value in before.items() if key != "populations"}
        after["populations"] = {}
        for pop in POPULATIONS:
            rebuilt = population(cells, pop)
            was = before["populations"][pop]
            for key in ("best", "best_by_seed", "difference", "call", "external"):
                _close(was[key], rebuilt[key], f"{before['variant']}.{pop}.{key}")
            methods = rebuilt["methods"]
            assert isinstance(methods, dict)
            for name, stats in was["methods"].items():
                _close(stats, methods[name], f"{before['variant']}.{pop}.methods.{name}")
            after["populations"][pop] = {**rebuilt, "cells_per_fold": was["cells_per_fold"]}
        source = common.development_source(before["variant"], SEEDS)
        after["labelled"] = bool(source is not None and source.labelled)
        variants.append(after)
    meta = {
        **old["meta"],
        "baselines": [*WEAK, *STRONG],
        "strong": list(STRONG),
        "linear_source": str(DEFAULT_LINEAR),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M") + " GMT-3",
        "first_version_generated": old["meta"]["generated"],
    }
    data = {"meta": meta, "variants": variants, "why": old["why"]}
    page = template.read_text()
    assert page.count("__DATA__") == 1
    out.write_text(page.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out}: {len(variants)} variants; the old page's numbers reproduced within {TOLERANCE:g}")
    for pop in POPULATIONS:
        calls = [v["populations"][pop]["strong"]["call"] for v in variants]
        bests = [v["populations"][pop]["strong"]["best"] for v in variants]
        print(pop, "vs strong:", {c: calls.count(c) for c in set(calls)}, "best:", {b: bests.count(b) for b in set(bests)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
