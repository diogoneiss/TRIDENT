"""One cell of E52, T28's D3: a frozen encoder, pre-trained against random (imputation-pretraining/10).

    uv run --python 3.11 python scripts/experiments/frozen_encoder_run.py \
        <variant> <seed> <FN|FA|FR|C0> <output_dir> <metrics_dir> [--no-mlflow] [--check-frozen]

Today's imputation configuration as a default run reads it (the variant's promoted file, if it
has one, over the task defaults), with ``--score_checkpoints``, plus one change per arm. The
freeze is the variant's own decode budget (``EPOCHS_DECODE``), so the encoder never thaws:

- FN: ``DECODE_FREEZE_EPOCHS`` = the decode budget; the normalised pre-training (today's), heads only;
- FA: the same, after the shipped ``embedding`` objective;
- FR: the same, with ``EPOCHS_PRE 0``: the Xavier-initialised encoder, never trained;
- C0: ``EPOCHS_PRE 0`` alone, a full decode without pre-training (credit-g only; E50 has it on
  pendigits_20nan and kr-vs-kp_40nan).

``--check-frozen`` (with ``--no-mlflow``) also asserts, fold by fold, that the embedder's and
transformer's weights leave the decode stage bit-identical. Five folds. Run from the checkout root.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

ROOT = Path.cwd()
if not (ROOT / "train.py").exists():
    sys.exit(f"run from the TRIDENT checkout root, not {ROOT}")
sys.path.insert(0, str(ROOT))

from src.training import runner  # noqa: E402
from src.training.config import complete_configuration, hyperparameter_file, load_hyperparameters  # noqa: E402
from train import main as train_main  # noqa: E402

EXPERIMENT = "frozen-encoder-2026-10-20"
ARMS = ("FN", "FA", "FR", "C0")


def changes(variant: str, arm: str) -> dict[str, object]:
    resolved, _ = load_hyperparameters(Namespace(dataset_name=variant, task="imputation"))
    frozen: dict[str, object] = {"DECODE_FREEZE_EPOCHS": int(str(complete_configuration(resolved, "imputation")["EPOCHS_DECODE"]))}
    by_arm: dict[str, dict[str, object]] = {
        "FN": frozen,
        "FA": {**frozen, "PRETRAIN_OBJECTIVE": "embedding"},
        "FR": {**frozen, "EPOCHS_PRE": 0},
        "C0": {"EPOCHS_PRE": 0},
    }
    return by_arm[arm]


def _check_frozen() -> None:
    """Wrap the decode stage so each fold asserts its encoder came out unchanged."""
    decode = getattr(runner, "train_and_evaluate_decoder")

    def checked(*args: Any, **kwargs: Any) -> Any:
        model = kwargs["pretraining"].model
        before = {k: v.detach().clone() for k, v in [*model.embedder.state_dict().items(), *model.transformer.state_dict().items()]}
        outcome = decode(*args, **kwargs)
        after = [*outcome.model.embedder.state_dict().items(), *outcome.model.transformer.state_dict().items()]
        changed = [k for k, v in after if not torch.equal(before[k], v)]
        assert not changed, f"the frozen encoder moved: {changed[:5]}"
        print(f"FROZEN_OK {len(after)} tensors unchanged", flush=True)
        return outcome

    setattr(runner, "train_and_evaluate_decoder", checked)


def main(argv: list[str]) -> int:
    variant, seed, arm, output_dir, metrics_dir = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
    check_only = "--no-mlflow" in argv[5:]
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    if "--check-frozen" in argv[5:]:
        if not check_only:
            sys.exit("--check-frozen runs without MLflow only")
        _check_frozen()
    promoted = hyperparameter_file(variant, "imputation")
    config: dict[str, object] = json.loads(promoted.read_text()) if promoted.exists() else {}
    config.update(changes(variant, arm))
    print("cell", variant, seed, arm, "config from", promoted if promoted.exists() else "defaults", json.dumps(config), flush=True)
    args = Namespace(
        dataset_name=variant,
        label_column="class",
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        disable_mlflow=check_only,
        plot_losses=False,
        save_model=False,
        seed=seed,
        cv_folds=5,
        task="imputation",
        hyperparams_override=config,
        score_checkpoints=True,
        mlflow_tags={} if check_only else {"experiment": EXPERIMENT, "arm": arm},
    )
    metrics = train_main(args, return_metrics=True) or {}
    scores = {k: v for k, v in metrics.items() if k.endswith("/impute_score") and "/baseline/" not in k}
    print("CELL_DONE", json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
