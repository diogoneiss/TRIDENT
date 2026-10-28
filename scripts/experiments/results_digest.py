"""The summary lines of every reader behind the consolidated results report, in one pass. Read-only.

    uv run --python 3.11 python scripts/experiments/results_digest.py

Runs each study's pre-registered reader as it is (none is changed) and keeps only its summary
lines: section headers, tallies and the "Primary" verdict. The per-variant lines stay in each
ticket's Outcome, which this digest points to. Run from the checkout root.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")

E = "scripts/experiments"
SUMMARY = re.compile(r"^(##|Primary|- \(|- P − G|- \d+nan: |  - calls that change|  - .*(better|worse|none|reversal|drift|same))")
# (label, ticket, command)
READERS: list[tuple[str, str, list[str]]] = [
    ("E55/E57: confirmation set against TabPFN and the strong tier", "imputation-protocol/10, 12",
     [f"{E}/tabpfn_report.py"]),
    ("E58: the other seed sets and MNAR, with TabPFN", "imputation-protocol/13",
     [f"{E}/tabpfn_seed_sets_report.py"]),
    ("E51: fresh seeds at 20nan", "imputation-protocol/08", [f"{E}/default_refresh_report.py", "--tally"]),
    ("E62: fresh MCAR draws", "imputation-protocol/17", [f"{E}/mcar_draws_report.py", "--tally"]),
    ("E63: MAR against matched MCAR controls (smoke)", "imputation-protocol/18", [f"{E}/mar_smoke_report.py", "--tally"]),
    ("E64: the MAR family", "imputation-protocol/19", [f"{E}/mar_family_report.py", "--tally"]),
    ("E65: propensity-weighted decode masks", "imputation-protocol/20", [f"{E}/mar_propensity_report.py", "--tally"]),
    ("E60: a TRIDENT-TabPFN blend", "imputation-protocol/15",
     [f"{E}/tabpfn_blend.py", "--tally", "/scratch2/diogoneiss/trident-2026-10-21-tabpfn-blend/csv"]),
    ("E66: the blend on E40's tables and seeds", "imputation-protocol/21",
     [f"{E}/tabpfn_blend.py", "--tally", "--set", "e40", "/scratch2/diogoneiss/trident-2026-10-23-tabpfn-blend-e40/csv"]),
    ("E61: where K loses to TabPFN, per column", "imputation-protocol/16",
     [f"{E}/categorical_deficit.py", "--tally", "/scratch2/diogoneiss/trident-2026-10-22-categorical-deficit/csv"]),
    ("E59: the bigger encoder at 40-80nan", "imputation-protocol/14", [f"{E}/large_encoder_report.py", "--tally"]),
    ("E52: a frozen pre-trained encoder against a random one", "imputation-pretraining/10",
     [f"{E}/frozen_encoder_report.py", "--tally"]),
    ("E53: collapse and probes", "imputation-pretraining/11", [f"{E}/collapse_probes_report.py"]),
    # Added 2026-10-28 for the report's E67-E83 update.
    ("E67: the gaps as [NULL] in the decode stage, alone", "imputation-protocol/22", [f"{E}/null_gaps_report.py", "--tally"]),
    ("E68: the tuned bar on the MAR folds", "imputation-protocol/23", [f"{E}/mar_tuned_bar_report.py", "--tally"]),
    ("E69: per-column weights for the K + TabPFN blend", "imputation-protocol/24",
     [f"{E}/blend_weights.py", "--tally", "/scratch2/diogoneiss/trident-2026-10-23-blend-weights"]),
    ("E71: classification, what pre-training buys", "classification-step-zero/02",
     [f"{E}/cls_stage2_run.py", "--tally", "/scratch2/diogoneiss/trident-2026-10-24-cls-stage2"]),
    ("E72: not-MIWAE under MNAR", "imputation-protocol/25", [f"{E}/vae_mnar_report.py", "--tally"]),
    ("E74: the KC + TabPFN hybrid as an output", "imputation-protocol/27", [f"{E}/hybrid_blend.py", "--tally"]),
    ("E75: a test-time re-masking ensemble", "imputation-protocol/28", [f"{E}/remask_ensemble.py", "--tally"]),
    ("E76: the spread on the development variants", "imputation-protocol/29", [f"{E}/remask_ensemble.py", "--study", "e76", "--tally"]),
    ("E77: intervals from the spread", "imputation-protocol/30", [f"{E}/remask_ensemble.py", "--conformal"]),
    ("E78: the occlusion map at 40nan", "imputation-interpretability/01", [f"{E}/occlusion_map.py", "--tally"]),
    ("E79: the occlusion map at 60nan", "imputation-interpretability/02", [f"{E}/occlusion_map.py", "--study", "e79", "--tally"]),
    ("E80: the spread and missing top sources", "imputation-interpretability/03", [f"{E}/occlusion_spread.py"]),
    ("E81: gap-pattern reliance under MNAR", "imputation-interpretability/04", [f"{E}/gap_reliance.py", "--tally"]),
    ("E82: the deployable hybrid", "imputation-protocol/31", [f"{E}/deployable_hybrid.py", "--tally"]),
    ("E83: TabPFN v2's classifier", "classification-step-zero/03", [f"{E}/tabpfn_classification.py", "--tally"]),
    ("E84: the spread under MNAR and MAR", "imputation-protocol/32", [f"{E}/spread_mechanisms.py", "--tally"]),
    ("E85: the occlusion map at 20nan and 80nan", "imputation-interpretability/05", [f"{E}/occlusion_map.py", "--study", "e85", "--tally"]),
]


def main() -> int:
    for label, ticket, command in READERS:
        done = subprocess.run(["uv", "run", "--python", "3.11", "python", "-B", *command], capture_output=True, text=True)
        kept = [line for line in done.stdout.splitlines() if SUMMARY.match(line)]
        print(f"### {label} ({ticket})\n\n```text")
        print("\n".join(kept) if kept else f"(no summary lines; exit {done.returncode})")
        print("```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
