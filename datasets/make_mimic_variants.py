"""Cut E149's variants: real gap patterns transplanted onto complete rows
(docs/tickets/mimic/04-pattern-transplant.md).

    TRIDENT_DATASETS_ROOT=<root> uv run --python 3.11 python datasets/make_mimic_variants.py

For ``mimic-cardio`` and ``mimic-sofa`` it reads ``<base>_natural.csv`` and the table of its complete rows,
``<base>-donor_00nan.csv`` (both written by ``build_mimic.py``), and writes three variants of the donor table beside
it, with the same count of gaps in every column:

  tmatch  each donor row takes the whole gap pattern of its nearest incomplete stay
  trand   tmatch's patterns, permuted among the donor rows
  mcar    in each column, tmatch's count of gaps at uniform positions

The rules the ticket leaves open are fixed here:

- The pool is every row of the natural table with a gap in any feature; the donors are the other rows.
- Distance is Euclidean over the numeric columns with under 1% of gaps in the natural table, each standardised by
  its mean and standard deviation over the whole cohort. A gap in one of them counts as the column's mean.
- Of equally near stays, the first in the table's order.
- One seed, ``SEED``, draws trand's permutation and mcar's positions. The training seeds are another matter.
- A variant is named ``<base>-donor_<variant>_<NN>nan``, NN being the nearest of 20/40/60/80 to the share of
  feature cells that are gaps. It is a label.

Every cell that is not a gap is the donor file's own text, the label is never touched and the rows keep their order,
so the training code finds each gap's truth in ``<base>-donor_00nan``. ``<base>-donor_sources.csv`` holds, row by
row, the stay each donor row took its pattern from under tmatch and under trand. An existing file is rewritten only
if identical: the data directory is outside the repository, so this script is the record.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "class"
SEED = 42
FLOOR = 1000  # the ticket's: a donor table under 1,000 rows is not run
LEVELS = (20, 40, 60, 80)
MATCHING_GAP_LIMIT = 0.01
CHUNK = 100


def nearest_incomplete(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The positions of the complete rows, and for each the position of its nearest incomplete row."""
    gaps = features.isna()
    matching = [
        name for name in features.columns
        if pd.api.types.is_numeric_dtype(features[name]) and gaps[name].mean() < MATCHING_GAP_LIMIT
    ]
    spread = features[matching].std()
    assert (spread > 0).all(), f"a matching column is constant: {list(spread[spread == 0].index)}"
    standardised = ((features[matching] - features[matching].mean()) / spread).fillna(0.0).to_numpy()
    incomplete = gaps.any(axis=1).to_numpy()
    donors, pool = np.flatnonzero(~incomplete), np.flatnonzero(incomplete)
    nearest = np.concatenate([
        # argmin returns the first of equal distances, and the pool is in the table's order.
        ((standardised[chunk][:, None, :] - standardised[pool][None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for chunk in np.array_split(donors, max(1, len(donors) // CHUNK))
    ])
    return donors, pool[nearest]


def cut(tmatch: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The three gap masks from tmatch's, and the permutation trand applied to its rows."""
    stream = np.random.default_rng(seed)
    order = stream.permutation(len(tmatch))
    mcar = np.zeros_like(tmatch)
    for column, count in enumerate(tmatch.sum(axis=0)):
        mcar[stream.choice(len(tmatch), int(count), replace=False), column] = True
    return {"tmatch": tmatch, "trand": tmatch[order], "mcar": mcar}, order


def write_if_new(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        sys.exit(f"{path} exists and differs; refusing to overwrite")
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("TRIDENT_DATASETS_ROOT"))
    parser.add_argument("--bases", nargs="+", default=["mimic-cardio", "mimic-sofa"])
    args = parser.parse_args()
    if args.root is None:
        parser.error("set TRIDENT_DATASETS_ROOT or pass --root")

    for base in args.bases:
        donor_base = f"{base}-donor"
        natural = pd.read_csv(args.root / base / f"{base}_natural.csv")
        index = pd.read_csv(args.root / base / f"{base}_index.csv")
        features = natural.drop(columns=[LABEL])
        donors, sources = nearest_incomplete(features)

        directory = args.root / donor_base
        # As text, so that every cell left alone is written back as it was read.
        complete = pd.read_csv(directory / f"{donor_base}_00nan.csv", dtype=str, keep_default_na=False)
        donor_index = pd.read_csv(directory / f"{donor_base}_index.csv")
        assert len(complete) >= FLOOR, f"{donor_base} has {len(complete)} rows, under the floor of {FLOOR}"
        assert list(complete.columns) == list(natural.columns)
        assert donor_index["stay_id"].tolist() == index["stay_id"].iloc[donors].tolist(), "the donor table is stale"
        assert not set(index["stay_id"].iloc[sources]) & set(donor_index["stay_id"]), "a pattern came from a donor"

        masks, order = cut(features.isna().to_numpy()[sources], SEED)
        counts = masks["tmatch"].sum(axis=0)
        level = min(LEVELS, key=lambda candidate: abs(candidate - 100 * masks["tmatch"].mean()))
        print(f"\n{donor_base}: {len(complete)} rows, patterns from {len(set(sources))} of "
              f"{int(features.isna().any(axis=1).sum())} incomplete stays, "
              f"{masks['tmatch'].mean():.1%} of feature cells are gaps, label {level}nan")
        for name, mask in masks.items():
            assert (mask.sum(axis=0) == counts).all(), "the variants differ in a column's count of gaps"
            variant = complete.copy()
            variant[features.columns] = complete[features.columns].mask(mask, "")
            path = directory / f"{donor_base}_{name}_{level}nan.csv"
            write_if_new(path, variant)
            written = pd.read_csv(path, dtype=str, keep_default_na=False)
            assert written[LABEL].equals(complete[LABEL])
            assert ((written[features.columns] == "") == mask).all().all()
            assert written[features.columns].where(~mask, complete[features.columns]).equals(complete[features.columns])
            distinct = len(np.unique(mask, axis=0))
            print(f"  {path.name}: {int(mask.sum())} gaps, {distinct} distinct row patterns, "
                  f"{int((~mask.any(axis=1)).sum())} rows left complete")
        gapped = [(name, int(count)) for name, count in zip(features.columns, counts) if count]
        print("  gaps by column: " + ", ".join(f"{name} {count / len(complete):.1%}" for name, count in gapped))
        write_if_new(directory / f"{donor_base}_sources.csv", pd.DataFrame({
            "stay_id": donor_index["stay_id"],
            "tmatch_source_stay_id": index["stay_id"].to_numpy()[sources],
            "trand_source_stay_id": index["stay_id"].to_numpy()[sources[order]],
        }))


if __name__ == "__main__":
    main()
