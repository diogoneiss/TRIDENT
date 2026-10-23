"""T06 feasibility: a short real run, eager against ``torch.compile``, timed (2026-10-23). Read-only.

    uv run --python 3.11 --with setuptools python scripts/experiments/compile_bench.py <eager|transformer|both> <variant> <out_dir>

Runs today's default with EPOCHS_PRE 30 and EPOCHS_DECODE 60 at seed 404, five folds, without MLflow.
``transformer`` compiles every TabularTransformerEncoder's forward, and ``both`` also every
TabularEmbedder's, by patching their constructors, so ``src/`` is not changed. ``--with setuptools``
is needed only because triton 3.1 imports it, and nothing is added to the lock. Per-epoch rates are
read from the tqdm lines of the output. Run from the checkout root.
"""
import json, sys, time
from typing import Any
sys.path.insert(0, ".")
from argparse import Namespace
import torch
import src.transformer as transformer_module
import src.embedder as embedder_module
from src.training.config import complete_configuration, load_hyperparameters
from train import main as train_main

mode, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]

def compiled_init(cls: type) -> None:
    original = getattr(cls, "__init__")

    def init(self: Any, *a: Any, **k: Any) -> None:
        original(self, *a, **k)
        self.forward = torch.compile(self.forward, dynamic=False)

    setattr(cls, "__init__", init)

if mode in ("transformer", "both"):
    compiled_init(transformer_module.TabularTransformerEncoder)
if mode == "both":
    compiled_init(embedder_module.TabularEmbedder)

resolved, _ = load_hyperparameters(Namespace(dataset_name=variant, task="imputation"))
config = {**complete_configuration(resolved, "imputation"), "EPOCHS_PRE": 30, "EPOCHS_DECODE": 60}
args = Namespace(dataset_name=variant, label_column="class", output_dir=out + f"/{mode}/results", metrics_dir=out + f"/{mode}/metrics",
                 disable_mlflow=True, plot_losses=False, save_model=False, seed=404, cv_folds=5, task="imputation",
                 hyperparams_override=config, score_checkpoints=True, score_calibrated=True)
started = time.time()
metrics = train_main(args, return_metrics=True) or {}
print("BENCH", mode, variant, f"{time.time() - started:.1f} s", json.dumps({k: round(v, 5) for k, v in metrics.items() if k in (
    "impute/induced/uncalibrated/impute_score", "impute/induced/impute_score", "impute/induced/loss_checkpoint/impute_score")}), flush=True)
