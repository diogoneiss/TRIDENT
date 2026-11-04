"""E132's diagnostic: S and B on the four MNAR files, seed 42, MLflow ON (leader), so the per-epoch losses are in the store."""
import sys, os
sys.path.insert(0, os.getcwd())
from argparse import Namespace
from pathlib import Path
from src.training.config import build_training_parser, logged_hyperparameters, resolve_training_request
from train import main
arm, variant = sys.argv[1], sys.argv[2]
base = logged_hyperparameters(resolve_training_request(build_training_parser().parse_args(["--dataset_name", variant])))
over = {**base, "PRETRAIN_OBJECTIVE": "embedding_normalized", "C15_ARM": "A1", **({"EPOCHS_PRE": 0} if arm == "B" else {})}
out = Path("/scratch/diogoneiss/trident-2026-11-04-c1/diag") / arm / f"{variant}_s42"
main(Namespace(dataset_name=variant, label_column="class", output_dir=str(out / "results"), metrics_dir=str(out / "metrics"),
               disable_mlflow=False, plot_losses=False, save_model=False, seed=42, cv_folds=5, task="classification",
               mlflow_tags={"study": "c1_diag_e132", "c1_arm": arm}, hyperparams_override=over), return_metrics=True)
print("CELL_DONE", arm, variant, flush=True)
