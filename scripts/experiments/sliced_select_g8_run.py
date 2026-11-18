"""E183's last three variants on gorgona8 (docs/tickets/imputation-protocol/72-sliced-checkpoint.md, amendment of
2026-11-18 19:53 GMT-3): ``electricity_40nan``, ``electricity_80nan`` and ``letter_20nan`` at seeds 101/202/303, with
both arms trained here so each (variant, seed) pair stays on one GPU model: ``S`` (``--decode_metric sw_score``) into
E183's state and its control ``Kc`` (E181's arm) into gorgona8's copy of E181's state, both tagged with E183's
experiment. The other 23 variants' runs are medusa1's, copied over for the read.

    uv run --python 3.11 python -B scripts/experiments/sliced_select_g8_run.py train <S|Kc> <variant> <seed>
    uv run --python 3.11 python -B scripts/experiments/sliced_select_g8_run.py --todo <queue> <queues>
    uv run --python 3.11 python -B scripts/experiments/sliced_select_g8_run.py --pairing

``--pairing`` prints, for each finished pair, whether the loss criterion's epoch per fold, the scored cell counts and
hgb's score agree between the arms (the ticket's smoke check); it reads no TRIDENT score. Run from the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from scripts.experiments import classcond_run as e181  # noqa: E402
from scripts.experiments import sliced_select_run as e183  # noqa: E402

MOVED = ("letter_20nan", "electricity_40nan", "electricity_80nan")
ARMS: dict[str, tuple[Path, dict[str, object]]] = {
    "S": (e183.STATE, {"decode_metric": "sw_score"}),
    "Kc": (e181.STATE, e181.ARMS["Kc"]),
}
SAME = ("decode/loss_checkpoint_epoch", "impute/induced/n_num_cells", "impute/induced/n_cat_cells",
        "impute/induced/baseline/hgb/impute_score")


def train(arm: str, variant: str, seed: int) -> None:
    assert variant in MOVED, variant
    state, flags = ARMS[arm]
    e181.EXPERIMENT, e181.STATE, e181.ARMS = e183.EXPERIMENT, state, {arm: flags}
    e181.train(arm, variant, seed)


def todo(which: int, queues: int) -> None:
    cells = [(arm, v, s) for v in MOVED for s in e183.SEEDS for arm in ARMS]  # a pair's two arms side by side
    for k, (arm, variant, seed) in enumerate(cells):
        if k % queues == which - 1 and e183._run(ARMS[arm][0], arm, variant, seed) is None:
            print(arm, variant, seed)


def pairing() -> None:
    import pandas as pd

    for variant in MOVED:
        for seed in e183.SEEDS:
            runs = [e183._run(ARMS[arm][0], arm, variant, seed) for arm in ARMS]
            if None in runs:
                print(f"{variant} seed {seed}: not both arms finished")
                continue
            s, k = (pd.read_csv(run / "metrics" / "raw_fold_metrics.csv").sort_values("fold") for run in runs if run)
            same = {c: bool((s[c].to_numpy() == k[c].to_numpy()).all()) for c in SAME}
            print(f"{variant} seed {seed}: " + ", ".join(f"{c.split('/')[-1]} {'same' if ok else 'DIFFERS'}" for c, ok in same.items()))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and len(argv) == 4 and argv[1] in ARMS:
        train(argv[1], argv[2], int(argv[3]))
    elif argv[:1] == ["--todo"] and len(argv) == 3:
        todo(int(argv[1]), int(argv[2]))
    elif argv == ["--pairing"]:
        pairing()
    else:
        sys.exit(__doc__)
