"""Pre-registered analysis of E72: does an imputer that models the mechanism shrink the MNAR bias?
(imputation-protocol/25, T35). Read-only.

    uv run --python 3.11 python scripts/experiments/vae_mnar_report.py [--tally]

Reads ``metric_panel.py --baselines --fitters vae --cell_suffix _G`` CSVs (method rows ``model``,
``miwae``, ``notmiwae``) in ``<STATE>/e45`` and ``<STATE>/e46``, and E45/E46's panel
(``mnar_bias_report.DEFAULT_PANEL``) for the other nine methods. On E45/E46's ten MNAR pairs, over
15 fold pairs, with the verdict rule:

- **per pair and mechanism:** notmiwae − miwae on ``abs_bias`` (the mean over numerical columns of
  each column's |mean(fill − truth)|), on ``signed_bias`` and on ``impute_score``; notmiwae − model
  (K, the ledger's fills) on ``abs_bias`` and ``impute_score``;
- **the interaction, fold-paired:** (notmiwae − miwae) on ``abs_bias``, MNAR minus MCAR;
- **primary,** that interaction over the 10 pairs: ``supported`` (modelling the mechanism shrinks
  the bias MNAR adds) if better on 4 or more and worse on at most 1; ``refuted`` if better on at
  most 1; ``mixed`` otherwise;
- **descriptive:** the 12 methods ranked by the MNAR − MCAR shift of ``signed_bias``, as E45/E46's
  measure 5 ranked ten.
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

from scripts.experiments import mnar_bias_report as bias  # noqa: E402
from scripts.experiments import report_common as common  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-24-vae-mnar")
SESOI = 0.01
VAE = ("miwae", "notmiwae")


def _load(tag: str, variant: str) -> pd.DataFrame | None:
    ours = bias._load(STATE, tag, variant)
    theirs = bias._load(bias.DEFAULT_PANEL, tag, variant)
    if ours is None or theirs is None:
        return None
    return pd.concat([ours[ours["method"].isin(VAE)], theirs], ignore_index=True)


def _diff(frame: pd.DataFrame, a: str, b: str, metric: str) -> list[np.ndarray]:
    return [x - y for x, y in zip(bias._per_seed(frame, a, metric), bias._per_seed(frame, b, metric))]


def tally() -> None:
    primary: dict[str, str] = {}
    counts: dict[str, list[str]] = {}
    for mnar, mnar_tag, mcar, mcar_tag in bias.PAIRS:
        found = {"MNAR": _load(mnar_tag, mnar), "MCAR": _load(mcar_tag, mcar)}
        if any(f is None for f in found.values()):
            print(f"## `{mnar}` against `{mcar}`: incomplete\n")
            continue
        print(f"## `{mnar}` against `{mcar}`")
        for tag, frame in found.items():
            assert frame is not None
            parts = []
            for a, b, metric in (("notmiwae", "miwae", "abs_bias"), ("notmiwae", "miwae", "signed_bias"), ("notmiwae", "miwae", "impute_score"),
                                 ("notmiwae", "model", "abs_bias"), ("notmiwae", "model", "impute_score")):
                d = _diff(frame, a, b, metric)
                scale = float(np.concatenate(bias._per_seed(frame, b, metric)).mean())
                v = common.verdict(d, sesoi=SESOI * abs(scale)) if metric == "impute_score" else common.verdict(d)
                call = v.call if metric != "signed_bias" else {"better": "lower", "worse": "higher"}.get(v.call, v.call)
                parts.append(f"{a} − {b} {metric} {v.mean:+.4f} {call}")
                if tag == "MNAR":
                    counts.setdefault(f"MNAR: {a} − {b} on {metric}", []).append(call)
            levels = ", ".join(f"{m} {np.concatenate(bias._per_seed(frame, m, 'signed_bias')).mean():+.3f}" for m in ("model", "miwae", "notmiwae"))
            print(f"- {tag}: " + "; ".join(parts) + f" (signed bias: {levels})")
        mnar_frame, mcar_frame = found["MNAR"], found["MCAR"]
        assert mnar_frame is not None and mcar_frame is not None
        under_mnar = _diff(mnar_frame, "notmiwae", "miwae", "abs_bias")
        under_mcar = _diff(mcar_frame, "notmiwae", "miwae", "abs_bias")
        v = common.verdict([x - y for x, y in zip(under_mnar, under_mcar)])
        primary[mnar] = v.call
        print(f"  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        shifts = {}
        for method in (*bias.METHODS, *VAE):
            x, y = bias._per_seed(mnar_frame, method, "signed_bias"), bias._per_seed(mcar_frame, method, "signed_bias")
            shifts[method] = float(np.mean([p.mean() - q.mean() for p, q in zip(x, y)]))
        order = sorted(shifts, key=lambda m: abs(shifts[m]))
        print("  - by |MNAR − MCAR shift of signed bias| (1 moves least): " + ", ".join(f"{i + 1}. {m} {shifts[m]:+.3f}" for i, m in enumerate(order)))
        print()
    for key, calls in counts.items():
        print(f"- {key}: {common.tally(calls)}")
    if len(primary) < len(bias.PAIRS):
        print(f"\nPrimary: incomplete ({len(primary)} of {len(bias.PAIRS)})")
        return
    better = [k for k, c in primary.items() if c == "better"]
    worse = [k for k, c in primary.items() if c == "worse"]
    reading = "supported" if len(better) >= 4 and len(worse) <= 1 else "refuted" if len(better) <= 1 else "mixed"
    print(f"\nPrimary, (notmiwae − miwae on abs_bias), MNAR minus MCAR: better on {better or 'none'}, worse on {worse or 'none'} "
          f"-> {reading} ({common.RULE})")


if __name__ == "__main__":
    if sys.argv[1:] in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
