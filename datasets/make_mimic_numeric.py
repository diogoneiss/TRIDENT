"""E152's table: ``mimic-sofa_natural`` with its categorical columns made numerical, since C15's arms take
numerical tables only (docs/tickets/mimic/06-sofa-positive-control.md).

    uv run --python 3.11 python datasets/make_mimic_numeric.py <source root> <study root>

Reads ``<source root>/mimic-sofa/mimic-sofa_natural.csv`` and writes
``<study root>/mimic-sofa-num/mimic-sofa-num_natural.csv``: ``gender`` 0/1 (F/M), ``pf_on_vent`` 0/1 (no/yes) with its
gaps kept, ``admission_type`` one-hot over its values (``adm_<value>``). Every other cell is the source's own value and the
rows keep their order. An existing file is rewritten only if identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    assert set(out["gender"].dropna()) <= {"F", "M"} and out["gender"].notna().all()
    out["gender"] = (out["gender"] == "M").astype(float)
    vent = out["pf_on_vent"]
    assert set(vent.dropna()) <= {"yes", "no"}, set(vent.dropna())
    out["pf_on_vent"] = vent.map({"no": 0.0, "yes": 1.0})
    assert out["admission_type"].notna().all()
    dummies = pd.get_dummies(out.pop("admission_type"), prefix="adm", dtype=float)
    dummies.columns = [c.replace(" ", "_").replace(".", "") for c in dummies.columns]
    label = out.pop("class")
    out = pd.concat([out, dummies, label], axis=1)
    assert out.drop(columns=list(dummies.columns)).isna().sum().equals(frame.drop(columns=["admission_type"]).isna().sum())
    return out


def main(source: Path, study: Path) -> None:
    frame = pd.read_csv(source / "mimic-sofa" / "mimic-sofa_natural.csv")
    out = numeric(frame)
    target = study / "mimic-sofa-num" / "mimic-sofa-num_natural.csv"
    text = out.to_csv(index=False)
    if target.exists() and target.read_text() != text:
        sys.exit(f"{target} exists and differs; not rewritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"{target}: {len(out)} rows, {out.shape[1] - 1} features, gaps per column {out.isna().sum()[lambda s: s > 0].to_dict()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
