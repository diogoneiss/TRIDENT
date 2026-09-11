"""Filesystem artifacts emitted by a training runtime."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .summary import final_metrics_for_tracking, fold_timings_for_tracking
from .types import (
    CLASSIFICATION,
    CrossValidationSummary,
    FoldResult,
    FoldTrackingRecord,
    Hyperparameters,
    PreparedDataset,
    TaskSpec,
    TrackingArtifactPaths,
)


class ArtifactWriter:
    def __init__(self, output_dir: Path, metrics_dir: Path, dataset_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = output_dir / dataset_name / timestamp
        self.metrics_dir = metrics_dir
        self.dataset_name = dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_hyperparameters(
        self, hyperparameters: Hyperparameters, task: TaskSpec = CLASSIFICATION
    ) -> Path:
        """Record the values this run used, and only those.

        A file claiming a fine-tuning rate on a run that never fine-tuned misleads
        whoever reads it later, the same way a logged-but-unused parameter does.
        """
        shared = {
            "DIM": hyperparameters.dimension,
            "HIDDEN_DIM": hyperparameters.hidden_dimension,
            "HEADS": hyperparameters.heads,
            "LAYERS": hyperparameters.layers,
            "DIM_FEED": hyperparameters.feedforward_dimension,
            "DROPOUT": hyperparameters.dropout,
            "EPOCHS_PRE": hyperparameters.pretraining_epochs,
            "BATCH": hyperparameters.batch_size,
            "LR_PRE": hyperparameters.pretraining_learning_rate,
            "WEIGHT_DECAY_PRE": hyperparameters.pretraining_weight_decay,
            "PROB_MASCARA": hyperparameters.mask_probability,
            "LR_SCHEDULER": hyperparameters.lr_scheduler,
        }
        if task.name == "imputation":
            values = {
                **shared,
                "EPOCHS_DECODE": hyperparameters.decode_epochs,
                "LR_DECODE": hyperparameters.decode_learning_rate,
                "WEIGHT_DECAY_DECODE": hyperparameters.decode_weight_decay,
                "LAMBDA_NUM": hyperparameters.lambda_num,
                "EVAL_MASK_RATE": hyperparameters.eval_mask_rate,
            }
        else:
            values = {
                **shared,
                "EPOCH_FINE": hyperparameters.finetuning_epochs,
                "LR_FINE": hyperparameters.finetuning_learning_rate,
                "WEIGHT_DECAY_FINE": hyperparameters.finetuning_weight_decay,
                "LABELS": hyperparameters.labels,
            }
        path = self.results_dir / "hyperparameters.json"
        path.write_text(json.dumps(values, indent=4))
        print(f"Hyperparameters saved to: {path}")
        return path

    def write_metrics(self, fold_results: list[FoldResult]) -> pd.DataFrame:
        rows = [{"fold": result.fold, "dataset": result.dataset_name, **result.metrics} for result in fold_results]
        frame = pd.DataFrame(rows)
        result_path = self.results_dir / "metrics.csv"
        frame.to_csv(result_path, index=False)
        print(f"Metrics saved to: {result_path}")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
        frame.to_csv(root_path, index=False)
        print(f"Metrics also saved to: {root_path}")
        return frame

    def write_cv_tracking_artifacts(
        self,
        records: Sequence[FoldTrackingRecord],
        summary: CrossValidationSummary,
        dataset: PreparedDataset,
        seed: int,
        cv_folds: int,
        task: TaskSpec = CLASSIFICATION,
    ) -> TrackingArtifactPaths:
        """Write the parent-run CSV, summary, diagnostic manifest, and lineage.

        The manifest's ranking section is keyed by the task's fold-ranking metric, so a
        classification run keeps producing ``f1_macro_ranking`` byte for byte.
        """
        paths = TrackingArtifactPaths(
            raw_fold_metrics_csv=self.results_dir / "metrics" / "raw_fold_metrics.csv",
            summary_json=self.results_dir / "metrics" / "cv_summary.json",
            manifest_json=self.results_dir / "tracking" / "diagnostic_manifest.json",
            provenance_json=self.write_tracking_provenance(dataset, seed, cv_folds),
        )
        for path in (
            paths.raw_fold_metrics_csv,
            paths.summary_json,
            paths.manifest_json,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "fold": record.result.fold,
                "dataset": record.result.dataset_name,
                **final_metrics_for_tracking(record),
                # Every fold's timing is kept here for audit; MLflow only
                # replays the best and worst folds as children.
                **fold_timings_for_tracking(record),
            }
            for record in records
        ]
        pd.DataFrame(rows).to_csv(paths.raw_fold_metrics_csv, index=False)

        summary_payload = {
            "interval": "two-sided 95% Student-t",
            "interval_interpretation": (
                "Internal CV uncertainty; not an independent-test generalization guarantee."
            ),
            "formula": "mean ± t(0.975, n - 1) * sample_std / sqrt(n)",
            "metrics": summary.metrics,
            "timings": summary.timings,
            "loss_bands": summary.loss_bands,
        }
        _write_json(paths.summary_json, summary_payload)

        ranking_metric = task.ranking_metric
        short_name = task.ranking_metric_short_name
        ranking = {
            str(result.fold): result.metrics[ranking_metric]
            for result in (record.result for record in records)
            if ranking_metric in result.metrics
        }
        records_by_fold = {record.result.fold: record for record in records}
        manifest_payload = {
            "diagnostic_roles": {str(fold): role for fold, role in summary.diagnostic_roles.items()},
            f"{short_name}_ranking": ranking,
            "selected_folds": [
                {
                    "fold": fold,
                    "role": role,
                    short_name: ranking.get(str(fold)),
                }
                for fold, role in summary.diagnostic_roles.items()
            ],
            "retained_artifact_paths": {
                str(fold): [
                    {
                        "source_path": artifact.path,
                        "artifact_path": artifact.artifact_path,
                    }
                    for artifact in records_by_fold[fold].artifacts
                ]
                for fold in summary.diagnostic_roles
            },
        }
        _write_json(paths.manifest_json, manifest_payload)

        return paths

    def write_tracking_provenance(
        self,
        dataset: PreparedDataset,
        seed: int,
        cv_folds: int | None,
    ) -> Path:
        """Write prepared-dataset lineage for a parent training run."""
        path = self.results_dir / "data" / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": dataset.source_path,
            "splits_path": dataset.splits_path,
            "dataset_name": dataset.frame.attrs.get("dataset_name", self.dataset_name),
            "prepared_schema": {
                "label_column": dataset.label_column,
                "categorical_columns": dataset.categorical_columns,
                "numerical_columns": dataset.numerical_columns,
                "label_classes": dataset.label_classes,
            },
            "preparation": {
                "label_encoding": "LabelEncoder",
                "numerical_scaling": "StandardScaler",
            },
            "split_strategy": (
                "cross_validation" if cv_folds is not None else "single_split"
            ),
            "cv_folds": cv_folds,
            "seed": seed,
        }
        _write_json(path, payload)
        return path

    def write_loss_plot(self, name: str, train_losses: list[float], validation_losses: list[float]) -> Path:
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / name
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(name.removesuffix(".png").replace("_", " ").title())
        plt.legend()
        plt.savefig(path)
        plt.close()
        return path

    def write_imputation_preview(
        self,
        name: str,
        scored_cells: pd.DataFrame,
        seed: int,
        fold: int,
        sample_rows: int = 10,
        scaler: StandardScaler | None = None,
        numerical_columns: Sequence[str] = (),
    ) -> tuple[Path, Path]:
        """A readable sample of what the model filled in, and the full record behind it.

        One sample draw feeds both files, so the preview can never show a row the ledger
        marks as unshown. The preview is for a person, so its numbers are in original
        units; the ledger keeps the scaled values the scoring used as well.
        """
        directory = self.results_dir / "imputation"
        directory.mkdir(parents=True, exist_ok=True)

        rows = sorted(scored_cells["row"].unique())
        generator = np.random.default_rng((seed * 1_000_003 + fold) % (2**32))
        chosen = set(
            generator.choice(rows, size=min(sample_rows, len(rows)), replace=False).tolist()
        )

        ledger = scored_cells.copy()
        ledger["in_preview"] = ledger["row"].isin(chosen)
        ledger["actual_original_units"] = _to_original_units(
            ledger, "actual", scaler, numerical_columns
        )
        ledger["imputed_original_units"] = _to_original_units(
            ledger, "imputed", scaler, numerical_columns
        )
        # It has done its job above, and `actual_original_units` now says the same thing.
        ledger = ledger.drop(columns=["actual_original"], errors="ignore")
        ledger_path = directory / f"{name}_cells.csv"
        ledger.to_csv(ledger_path, index=False)

        preview_path = directory / f"{name}_preview.md"
        preview_path.write_text(
            _render_preview(ledger[ledger["in_preview"]], self.dataset_name), encoding="utf-8"
        )
        return preview_path, ledger_path

    def write_per_column_imputation(
        self, per_fold: Mapping[int | str, Mapping[str, pd.DataFrame]]
    ) -> Path:
        """Every fold's per-column errors in one long-form table on the parent run.

        Pooled metrics rank a fold; this says which column a poor one struggled with. It
        is an artifact rather than tracked metrics because a 57-column table would
        otherwise add hundreds of series per fold to the store.
        """
        rows = [
            {"fold": fold, "population": population, **record}
            for fold, populations in per_fold.items()
            for population, table in populations.items()
            for record in table.to_dict("records")
        ]
        path = self.results_dir / "metrics" / "per_column_imputation.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows, columns=["fold", "population", "column", "metric", "value"]
        ).to_csv(path, index=False)
        return path

    def save_model(self, name: str, model: object) -> Path:
        path = self.results_dir / name
        torch.save(model, path)
        print(f"Final model saved to: {path}")
        return path


def _to_original_units(
    ledger: pd.DataFrame,
    column: str,
    scaler: StandardScaler | None,
    numerical_columns: Sequence[str],
    # ``Any`` rather than ``object``: the list is assigned straight into a pandas column,
    # and pandas declares its own value union that ``object`` is too wide for. The cells
    # really are heterogeneous -- a float for a numerical column, a category label
    # otherwise -- so this is the honest width, not a concession.
) -> list[Any]:
    """Numbers as a person would recognise them; categories are already readable."""
    values = []
    order = list(numerical_columns)
    # The decode stage carries a known truth across at full precision. Preferring it beats
    # re-deriving one, which amplifies the scaled value's float32 rounding by the column's
    # ``scale_`` -- a tenth of a unit on kc2's widest column (ticket 0004). A prediction
    # has no such counterpart, so ``imputed`` falls through to the arithmetic below.
    exact_column = f"{column}_original"
    for _, cell in ledger.iterrows():
        if exact_column in ledger.columns and not pd.isna(cell[exact_column]):
            values.append(cell[exact_column])
            continue
        if cell["kind"] != "numerical" or scaler is None or cell["column"] not in order:
            values.append(cell[column])
            continue
        # inverse_transform wants a whole row, so undo this one column by hand.
        index = order.index(cell["column"])
        scale, mean = scaler.scale_, scaler.mean_
        # Built with the with_mean/with_std defaults, so both are arrays by the time a
        # fitted scaler reaches here. Asserted rather than folded into the guard above:
        # widening that guard would send the column down the "already readable" path and
        # silently change the artifact's contents.
        assert scale is not None and mean is not None
        values.append(float(cell[column]) * float(scale[index]) + float(mean[index]))
    return values


# How many cells of one row share a table before it stops being readable.
_PREVIEW_WIDTH = 6


def _render_preview(previewed: pd.DataFrame, dataset_name: str) -> str:
    """Each sampled row as three lines: what was true, what the model saw, what it said."""
    lines = [
        f"# Imputation preview: {dataset_name}",
        "",
        "One block per sampled row. `model saw` distinguishes a cell hidden for scoring",
        "from one the dataset was already missing. Values are in original units,",
        "shown to three decimals; `*` marks a number too long to show exactly, whose",
        "full value is in the cell ledger beside this file.",
        "",
    ]
    if previewed.empty:
        lines.append("_No cells were scored._")
        return "\n".join(lines) + "\n"

    for row, cells in previewed.groupby("row", sort=True):
        lines.append(f"**row {row}** -- {len(cells)} cell(s) filled in")
        lines.append("")
        # A row can have a dozen cells filled in, and one table that wide reads as noise,
        # so it is split into blocks a reader can take in at a glance.
        for start in range(0, len(cells), _PREVIEW_WIDTH):
            block = cells.iloc[start : start + _PREVIEW_WIDTH]
            lines.append("| | " + " | ".join(block["column"]) + " |")
            lines.append("|---|" + "---|" * len(block))
            lines.append(
                "| actual | "
                + " | ".join(_show(value) for value in block["actual_original_units"])
                + " |"
            )
            lines.append(
                "| model saw | "
                + " | ".join(
                    "[NULL]->[MASK]" if population.startswith("induced") else "[MASK]"
                    for population in block["population"]
                )
                + " |"
            )
            lines.append(
                "| imputed | "
                + " | ".join(_show(value) for value in block["imputed_original_units"])
                + " |"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


# The preview's display precision (ticket 0004). `.4g` used to print an exact zero that
# had round-tripped through float32 as `1.144e-09`, which reads as a real measurement
# rather than the zero it means, and `spambase` is 77% exact zeros.
_DISPLAY_DECIMALS = 3
# Anything below half a unit in the last shown place cannot survive the rounding, so it
# is zero here. Snapping it also keeps a small negative from printing as `-0.000`.
_ZERO_EPSILON = 5e-4
# Trails a shortened number. A digit followed by `*` can only ever close emphasis in
# markdown, never open it, so the tables render as written.
_ROUNDED_MARKER = "*"


def _show(value: object, width: int = 18) -> str:
    if not isinstance(value, (float, np.floating)):
        text = str(value)
        return text if len(text) <= width else text[: width - 2] + ".."
    number = float(value)
    shown = 0.0 if abs(number) < _ZERO_EPSILON else number
    text = f"{shown:.{_DISPLAY_DECIMALS}f}"
    if len(text) > width:
        # Clipping loses more than rounding does, so the marker belongs here too.
        return text[: width - 3] + ".." + _ROUNDED_MARKER
    # The marker earns its place only where the text is not the number. A reader who
    # sees none can quote the value; one who sees it knows to open the ledger instead.
    return text if float(text) == number else text + _ROUNDED_MARKER


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_to_builtin(payload), indent=2))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    # ``is_dataclass`` is true for the class object as well as an instance, and ``asdict``
    # only accepts an instance. The class case never reaches here in practice; excluding it
    # states that rather than letting it through to a runtime TypeError.
    if is_dataclass(value) and not isinstance(value, type):
        return _to_builtin(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value
