"""Pre-registered analysis of E73: not-MIWAE with a mean-based fill on the pairs where E72's fill
failed (imputation-protocol/26, T35). Read-only.

    uv run --python 3.11 python scripts/experiments/notmiwae_fill_report.py [--tally]

Reads ``metric_panel.py --fitters notmiwae_fills`` CSVs (``notmiwae_mean``, ``notmiwae_draws``) in
``<STATE>/e45`` and ``<STATE>/e46``, MIWAE from E72's (same folds, same settings, same seed), and the
model from E45/E46's panel. On the six pairs of biodeg and spambase, over 15 fold pairs:

- per pair and mechanism: notmiwae_mean − miwae on ``abs_bias``, ``signed_bias`` and ``impute_score``
  (1% SESOI), notmiwae_mean − model on ``impute_score``, and each method's mean ``impute_score``
  (a pair whose notmiwae_mean is not below the mean fill, 1.0, under either mechanism is flagged);
- the interaction, fold-paired: (notmiwae_mean − miwae on ``abs_bias``), MNAR minus MCAR;
- **primary,** that interaction over the 6 pairs: ``supported`` if better on 3 or more and worse on
  at most 1; ``refuted`` if better on at most 1; ``mixed`` otherwise;
- descriptive: the 13 methods ranked by the MNAR − MCAR shift of ``signed_bias``.
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
from scripts.experiments import vae_mnar_report as e72  # noqa: E402

STATE = Path("/scratch2/diogoneiss/trident-2026-10-24-notmiwae-fill")
SESOI = 0.01
FILLS = ("notmiwae_mean", "notmiwae_draws")
PAIRS = [p for p in bias.PAIRS if p[0].startswith(("biodeg", "spambase"))]


def _load(tag: str, variant: str) -> pd.DataFrame | None:
    ours = bias._load(STATE, tag, variant)
    earlier = e72._load(tag, variant)
    if ours is None or earlier is None:
        return None
    return pd.concat([ours[ours["method"].isin(FILLS)], earlier], ignore_index=True)


def tally() -> None:
    primary: dict[str, str] = {}
    flagged: list[str] = []
    for mnar, mnar_tag, mcar, mcar_tag in PAIRS:
        found = {"MNAR": _load(mnar_tag, mnar), "MCAR": _load(mcar_tag, mcar)}
        if any(f is None for f in found.values()):
            print(f"## `{mnar}` against `{mcar}`: incomplete\n")
            continue
        print(f"## `{mnar}` against `{mcar}`")
        for tag, frame in found.items():
            assert frame is not None
            parts = []
            for a, b, metric in (("notmiwae_mean", "miwae", "abs_bias"), ("notmiwae_mean", "miwae", "signed_bias"),
                                 ("notmiwae_mean", "miwae", "impute_score"), ("notmiwae_mean", "model", "impute_score")):
                d = e72._diff(frame, a, b, metric)
                scale = float(np.concatenate(bias._per_seed(frame, b, metric)).mean())
                v = common.verdict(d, sesoi=SESOI * abs(scale)) if metric == "impute_score" else common.verdict(d)
                call = v.call if metric != "signed_bias" else {"better": "lower", "worse": "higher"}.get(v.call, v.call)
                parts.append(f"{a} − {b} {metric} {v.mean:+.4f} {call}")
            levels = {m: float(np.concatenate(bias._per_seed(frame, m, "impute_score")).mean()) for m in ("model", "miwae", "notmiwae", *FILLS)}
            if levels["notmiwae_mean"] >= 1.0 and mnar not in flagged:
                flagged.append(mnar)
            signed = ", ".join(f"{m} {np.concatenate(bias._per_seed(frame, m, 'signed_bias')).mean():+.3f}" for m in ("model", "miwae", "notmiwae_mean"))
            print(f"- {tag}: " + "; ".join(parts))
            print(f"  - impute_score: " + ", ".join(f"{m} {s:.3f}" for m, s in levels.items()) + f"; signed bias: {signed}")
        mnar_frame, mcar_frame = found["MNAR"], found["MCAR"]
        assert mnar_frame is not None and mcar_frame is not None
        under_mnar = e72._diff(mnar_frame, "notmiwae_mean", "miwae", "abs_bias")
        under_mcar = e72._diff(mcar_frame, "notmiwae_mean", "miwae", "abs_bias")
        v = common.verdict([x - y for x, y in zip(under_mnar, under_mcar)])
        primary[mnar] = v.call
        print(f"  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: {v.mean:+.4f} [{v.low:+.4f}, {v.high:+.4f}] {v.call}")
        shifts = {}
        for method in (*bias.METHODS, "miwae", *FILLS):
            x, y = bias._per_seed(mnar_frame, method, "signed_bias"), bias._per_seed(mcar_frame, method, "signed_bias")
            shifts[method] = float(np.mean([p.mean() - q.mean() for p, q in zip(x, y)]))
        order = sorted(shifts, key=lambda m: abs(shifts[m]))
        print("  - by |MNAR − MCAR shift of signed bias| (1 moves least): " + ", ".join(f"{i + 1}. {m} {shifts[m]:+.3f}" for i, m in enumerate(order)))
        print()
    print(f"- pairs where notmiwae_mean is not below the mean fill under a mechanism: {flagged or 'none'}")
    if len(primary) < len(PAIRS):
        print(f"\nPrimary: incomplete ({len(primary)} of {len(PAIRS)})")
        return
    better = [k for k, c in primary.items() if c == "better"]
    worse = [k for k, c in primary.items() if c == "worse"]
    reading = "supported" if len(better) >= 3 and len(worse) <= 1 else "refuted" if len(better) <= 1 else "mixed"
    print(f"\nPrimary, (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: better on {better or 'none'}, worse on {worse or 'none'} "
          f"-> {reading} ({common.RULE})")


if __name__ == "__main__":
    if sys.argv[1:] in ([], ["--tally"]):
        tally()
    else:
        sys.exit(__doc__)
