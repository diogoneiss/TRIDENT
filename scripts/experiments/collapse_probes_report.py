"""Pre-registered analysis of E53, T28's D2: collapse and linear probes on the pre-training stage,
and the encoder's drift (imputation-pretraining/11).

Read-only. Reads the CSVs ``pretrain_probe.py`` wrote:

    uv run --python 3.11 python scripts/experiments/collapse_probes_report.py [csv_dir]

Every call uses setting N, kind ``all``, seeds 101/202/303, and the rule fixed in the ticket:

- collapse on a table: on every seed, at the last epoch the within-column share is below the
  start's, |centred cosine| < 2 standard errors and NMSE >= 1; N collapsing on 2 or more of 3;
- value information on a table: the final probe score below the start's on every seed and the
  mean gain over 1% of the start's mean; present under N on 2 or more of 3;
- H5 on credit-g: N_lr4 or N_b64 changes the collapse call, or moves the final probe score by
  more than 1% of N's with every seed in one direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = Path("/scratch2/diogoneiss/trident-2026-10-20-collapse-probes/csv")
TABLES = ("credit-g_20nan", "kr-vs-kp_40nan", "pendigits_20nan")
SEEDS = (101, 202, 303)
MARGIN = 0.01


def _read(directory: Path, suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(directory.glob(f"*_{suffix}.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collapse_call(collapse: pd.DataFrame, table: str, setting: str) -> tuple[str, list[str]]:
    lines: list[str] = []
    hits = 0
    for seed in SEEDS:
        rows = collapse[(collapse.variant == table) & (collapse.seed == seed) & (collapse.setting == setting) & (collapse.kind == "all")]
        if rows.empty:
            return "incomplete", lines
        start = rows[rows.stage == "start"].iloc[0]
        last = rows[rows.stage == "train"].sort_values("epoch").iloc[-1]
        falls = last.within_share < start.within_share
        flat = abs(last.centred_cosine) < 2 * last.cosine_se
        poor = last.nmse >= 1
        hits += bool(falls and flat and poor)
        lines.append(
            f"s{seed}: within share {start.within_share:.4f} -> {last.within_share:.4f}, raw {start.raw_within_share:.4f} -> "
            f"{last.raw_within_share:.4f}; centred cosine {last.centred_cosine:+.4f} ({last.centred_cosine / last.cosine_se:+.1f} SE); "
            f"NMSE {start.nmse:.3f} -> {last.nmse:.3f}; erank target {last.erank_target:.2f}, prediction {last.erank_prediction:.2f}"
        )
    return ("collapse" if hits == len(SEEDS) else "no collapse"), lines


def _probe(probes: pd.DataFrame, table: str, setting: str, stage: str) -> np.ndarray | None:
    rows = probes[(probes.variant == table) & (probes.setting == setting) & (probes.stage == stage)].set_index("seed")
    if any(seed not in rows.index for seed in SEEDS):
        return None
    return np.asarray([rows.loc[seed, "probe_score"] for seed in SEEDS], dtype=float)


def _value_call(probes: pd.DataFrame, table: str, setting: str) -> tuple[str, str]:
    start, final = _probe(probes, table, setting, "start"), _probe(probes, table, setting, "final")
    if start is None or final is None:
        return "incomplete", ""
    gain = start - final
    present = bool((gain > 0).all()) and float(gain.mean()) > MARGIN * float(start.mean())
    by_kind = probes[(probes.variant == table) & (probes.setting == setting)].groupby("stage")[["probe_num_ratio", "probe_cat_ratio"]].mean()
    kinds = "; ".join(f"{stage}: num {row.probe_num_ratio:.4f}, cat {row.probe_cat_ratio:.4f}" for stage, row in by_kind.iterrows())
    detail = (f"probe start {np.round(start, 4).tolist()} -> final {np.round(final, 4).tolist()}, mean gain "
              f"{float(gain.mean()):+.4f} ({100 * float(gain.mean()) / float(start.mean()):+.1f}% of the start); by kind ({kinds})")
    return ("present" if present else "absent"), detail


def main(argv: list[str]) -> int:
    directory = Path(argv[0]) if argv else DEFAULT
    collapse, probes, drift = _read(directory, "collapse"), _read(directory, "probe"), _read(directory, "drift")
    calls: dict[str, dict[str, str]] = {"collapse": {}, "value": {}}
    for setting in ("N", "A"):
        print(f"## Setting {setting}")
        for table in TABLES:
            collapsed, lines = _collapse_call(collapse, table, setting)
            value, detail = _value_call(probes, table, setting)
            print(f"- {table}: {collapsed}; value information {value}")
            for line in lines:
                print(f"  - {line}")
            if detail:
                print(f"  - {detail}")
            if setting == "N":
                calls["collapse"][table], calls["value"][table] = collapsed, value
        print()
    if any(call == "incomplete" for kind in calls.values() for call in kind.values()):
        print("Primary: incomplete")
    else:
        collapsing = sum(call == "collapse" for call in calls["collapse"].values())
        informative = sum(call == "present" for call in calls["value"].values())
        print(f"Primary: N collapsing: {'yes' if collapsing >= 2 else 'no'} ({collapsing} of 3); "
              f"value information under N: {'present' if informative >= 2 else 'no value information'} ({informative} of 3)")
    reference = _probe(probes, "credit-g_20nan", "N", "final")
    base_call = calls["collapse"].get("credit-g_20nan")
    for setting in ("N_lr4", "N_b64"):
        final = _probe(probes, "credit-g_20nan", setting, "final")
        collapsed, _ = _collapse_call(collapse, "credit-g_20nan", setting)
        if reference is None or final is None or collapsed == "incomplete":
            print(f"H5 {setting}: incomplete")
            continue
        shift = final - reference
        moves = collapsed != base_call or (
            abs(float(shift.mean())) > MARGIN * float(reference.mean()) and (bool((shift > 0).all()) or bool((shift < 0).all())))
        print(f"H5 {setting}: collapse {collapsed} (N: {base_call}); final probe {np.round(final, 4).tolist()} against N's "
              f"{np.round(reference, 4).tolist()} (mean {float(shift.mean()):+.4f}) -> {'moves' if moves else 'does not move'}")
    if not drift.empty:
        print("\n## Drift across today's decode stage (seed 101, fold 1)")
        columns = ["variant", "start", "linear_cka", "induced_impute_score", *[c for c in drift.columns if c.startswith("rel_change/")]]
        print(drift[columns].sort_values(["variant", "start"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
