"""Fresh MCAR draws of a variant, from its complete table, with the shipped mechanism (T31, E62).

    uv run --python 3.11 python datasets/generate_mcar_draws.py [--variants pendigits_20nan ...] [--draws 2 3]

Each shipped ``<base>_<pp>nan`` variant is one draw: ``generate_splits.inject_nans`` on
``<base>_00nan`` with seed ``42 + pp``. This script first regenerates that draw and refuses to go on
unless it equals the shipped file cell for cell, so the mechanism is the one the variants were
made with. It then writes draw ``d`` with seed ``42 + pp + 1000 * (d - 1)`` as
``processed_datasets/<base>/<base>_d<d>_<pp>nan.csv``:

- every feature column loses exactly ``round(n * pct)`` cells, the original's count, drawn
  without replacement, independently of the original's;
- the label column is never touched, rows keep their order, and every observed cell equals the
  complete table's, so the training code finds the induced cells' truth in ``<base>_00nan`` as it
  does for the original.

The rate stays the last token of the name, as the MNAR variants have it. An existing file is
rewritten only if identical. ``generate_splits.py`` is not touched; the data directory is
gitignored, so this script is the record.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_splits import RANDOM_SEED, TARGET_COLUMN, inject_nans  # noqa: E402

ROOT = HERE / "processed_datasets"
DEFAULT_VARIANTS = ("pendigits_20nan", "kr-vs-kp_40nan", "credit-g_60nan", "biodeg_80nan")
DRAW_STRIDE = 1000


def _roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as a reader of its CSV sees it, so it compares with a shipped file."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    if not a.isna().equals(b.isna()):
        return False
    return bool(a.astype(str).where(a.notna(), "").equals(b.astype(str).where(b.notna(), "")))


def seed_of(pct: float, draw: int) -> int:
    return RANDOM_SEED + int(pct * 100) + DRAW_STRIDE * (draw - 1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--draws", nargs="+", type=int, default=[2, 3])
    options = parser.parse_args(argv)
    for variant in options.variants:
        base, tag = variant.split("_", 1)
        pp = int(tag.removesuffix("nan"))
        pct = pp / 100
        complete = pd.read_csv(ROOT / base / f"{base}_00nan.csv")
        shipped = pd.read_csv(ROOT / base / f"{variant}.csv")
        regenerated = inject_nans(complete, pct, TARGET_COLUMN, seed=seed_of(pct, 1))
        if regenerated is None or not _same(_roundtrip(regenerated), shipped):
            sys.exit(f"{variant}: inject_nans with seed {seed_of(pct, 1)} does not reproduce the shipped file")
        first = shipped.drop(columns=[TARGET_COLUMN]).isna().to_numpy()
        print(f"{variant}: the shipped draw is reproduced by seed {seed_of(pct, 1)}")
        for draw in options.draws:
            if draw < 2:
                sys.exit("draw 1 is the shipped variant")
            made = inject_nans(complete, pct, TARGET_COLUMN, seed=seed_of(pct, draw))
            assert made is not None
            path = ROOT / base / f"{base}_d{draw}_{pp:02d}nan.csv"
            if path.exists() and not _same(pd.read_csv(path), _roundtrip(made)):
                sys.exit(f"{path} exists and differs; refusing to overwrite")
            made.to_csv(path, index=False)
            gaps = made.drop(columns=[TARGET_COLUMN]).isna().to_numpy()
            counts = gaps.sum(axis=0)
            shared = float((gaps & first).sum() / gaps.sum())
            print(f"  wrote {path.name}: seed {seed_of(pct, draw)}, {gaps.mean():.4f} of feature cells missing, "
                  f"{counts.min()}-{counts.max()} per column (shipped {first.sum(axis=0).min()}-{first.sum(axis=0).max()}); "
                  f"{shared:.1%} of its gaps are gaps in the shipped draw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
