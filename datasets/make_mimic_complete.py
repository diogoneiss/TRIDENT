"""A complete numerical table cut from a natural MIMIC-IV table, to inject known mechanisms into (E156,
docs/tickets/c15-mnar-prediction/12-mechanism-split.md).

    uv run --python 3.11 python datasets/make_mimic_complete.py <source root> <study root> <base>

Reads ``<source root>/<base>/<base>_natural.csv`` and writes ``<study root>/<base>-semi/<base>-semi_00nan.csv``: the
numerical columns (``gender`` and ``admission_type`` left out) with at most 5% natural gaps whose most frequent value
covers at most half the rows (the vasopressor rates, mostly 0, and ``gcs_min`` fail it: self-masking on a column that
is one value would be MCAR), and the rows complete on them, in their order, with ``class``. An existing file is
rewritten only if identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MAX_GAPS, MAX_MODE = 0.05, 0.5


def complete(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=["class", "gender", "admission_type"]).select_dtypes("number")
    keep = [c for c in x.columns
            if x[c].isna().mean() <= MAX_GAPS and x[c].value_counts(normalize=True).iloc[0] <= MAX_MODE]
    out = frame[keep + ["class"]].dropna().reset_index(drop=True)
    assert not out.isna().any().any() and out["class"].nunique() == 2
    return out


def main(source: Path, study: Path, base: str) -> None:
    out = complete(pd.read_csv(source / base / f"{base}_natural.csv"))
    target = study / f"{base}-semi" / f"{base}-semi_00nan.csv"
    text = out.to_csv(index=False)
    if target.exists() and target.read_text() != text:
        sys.exit(f"{target} exists and differs; not rewritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"{target}: {len(out)} rows, columns {list(out.columns[:-1])}, deaths {out['class'].mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
