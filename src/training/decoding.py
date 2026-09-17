"""Decode stage: train a decoder to reconstruct hidden cells, then score what it fills in.

The imputation task's counterpart to classifier fine-tuning. It reuses the encoder
pre-training produced, trains on masks re-rolled every epoch exactly as pre-training does,
and selects its checkpoint on a validation mask that never changes so that the loss moves
only when the model does. See ADR 0004.
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.embedder import as_category_strings
from src.models import TridentDecoder
from src.utils import preprocess_table

from .data import evaluation_mask
from .imputation_metrics import CATEGORICAL, NUMERICAL, mean_mode_baselines, score_cells
from .schedulers import StageScheduler, batches_per_epoch
from .types import (
    DecodingOutcome,
    FoldResult,
    FoldSplit,
    Hyperparameters,
    PreparedDataset,
    PretrainingOutcome,
    TrainingTracker,
)


def train_and_evaluate_decoder(
    dataset: PreparedDataset,
    fold: FoldSplit,
    pretraining: PretrainingOutcome,
    hyperparameters: Hyperparameters,
    device: torch.device,
    tracker: TrainingTracker,
    seed: int,
    fold_ordinal: int,
    complete_sibling: pd.DataFrame | None = None,
    score_null_path: bool = False,
    score_search_objective: bool = False,
) -> DecodingOutcome:
    """Train the decoder on this fold and score what it reconstructs on the test split.

    ``score_search_objective`` additionally scores the validation split, for a
    hyper-parameter search that must never rank trials on the test split (ADR 0005).
    """
    features = dataset.frame.drop(columns=[dataset.label_column])
    train_frame = features.iloc[fold.train_indices].reset_index(drop=True)
    validation_frame = features.iloc[fold.validation_indices].reset_index(drop=True)

    model = TridentDecoder(
        embedder=pretraining.model.embedder,
        transformer=pretraining.model.transformer,
        lambda_num=hyperparameters.lambda_num,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters.decode_learning_rate,
        weight_decay=hyperparameters.decode_weight_decay,
    )
    scheduler = StageScheduler(
        optimizer,
        hyperparameters.lr_scheduler,
        epochs=hyperparameters.decode_epochs,
        batches_per_epoch=batches_per_epoch(len(train_frame), hyperparameters.batch_size),
    )

    # Clean targets never change, so they are encoded once per fold. The validation mask
    # is drawn once too: a re-rolled one would move the loss for reasons that have
    # nothing to do with the model, making checkpoint selection meaningless.
    embedder = model.embedder
    clean_train = embedder.encode(_clean(train_frame), device)
    clean_validation = embedder.encode(_clean(validation_frame), device)
    hidden_validation = embedder.encode(
        evaluation_mask(
            validation_frame, hyperparameters.eval_mask_rate, seed, fold_ordinal
        ),
        device,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_state = None

    print("\n=== Starting Decoding (reconstructing hidden cells) ===")
    for epoch in tqdm(range(hyperparameters.decode_epochs), desc="Decode epochs"):
        hidden_train = embedder.encode(
            preprocess_table(
                train_frame.copy(), p_base=hyperparameters.mask_probability, fine_tunning=False
            ),
            device,
        )
        model.train()
        order = torch.randperm(len(train_frame)).to(device)
        running = torch.zeros((), dtype=torch.float64, device=device)
        steps = 0
        for start in range(0, len(train_frame), hyperparameters.batch_size):
            batch = order[start : start + hyperparameters.batch_size]
            optimizer.zero_grad()
            loss, _ = model(hidden_train[batch], clean_train[batch])
            loss.backward()
            optimizer.step()
            scheduler.after_batch()
            running += loss.detach().double()
            steps += 1
        train_losses.append(float((running / max(steps, 1)).item()))

        model.eval()
        with torch.no_grad():
            validation_loss, _ = model(hidden_validation, clean_validation)
        validation_losses.append(float(validation_loss.item()))
        tracker.log_metrics(
            {
                "decode/train_loss": train_losses[-1],
                "decode/val_loss": validation_losses[-1],
                "decode/learning_rate": scheduler.learning_rate,
            },
            step=epoch,
        )
        scheduler.after_epoch(validation_losses[-1])
        if validation_losses[-1] < best_validation_loss:
            best_validation_loss = validation_losses[-1]
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_frame = features.iloc[fold.test_indices].reset_index(drop=True)
    baselines = mean_mode_baselines(
        train_frame, dataset.numerical_columns, dataset.categorical_columns
    )
    hidden_test = embedder.encode(
        evaluation_mask(test_frame, hyperparameters.eval_mask_rate, seed, fold_ordinal), device
    )
    raw_variant = (
        dataset.raw_numerical.iloc[fold.test_indices].reset_index(drop=True)
        if dataset.raw_numerical is not None
        else None
    )
    cells = _score_population(
        model, hidden_test, embedder.encode(_clean(test_frame), device), "masked", raw_variant
    )

    metrics: dict[str, float | int | str] = {}
    scored = score_cells(cells, baselines)
    metrics.update({f"impute/masked/{name}": value for name, value in scored.metrics.items()})
    # Nominal is what was asked for; realised is what the masking helper actually hid,
    # which falls as missingness rises because it never hides an already-missing cell.
    eligible = int(hidden_test.masked_positions.numel() - _already_missing(test_frame))
    metrics["impute/masked/realised_rate"] = (
        len(cells) / eligible if eligible else 0.0
    )

    # Extra difficulties, scored on this same checkpoint. Diagnostics only: the primary
    # rate above is what validation selected on and what ranks the fold.
    clean_test = embedder.encode(_clean(test_frame), device)
    for extra in hyperparameters.eval_mask_rates_extra:
        at_rate = _score_population(
            model,
            embedder.encode(evaluation_mask(test_frame, extra, seed, fold_ordinal), device),
            clean_test,
            "masked",
        )
        prefix = f"impute/masked/rate_{round(extra * 100)}"
        metrics.update(
            {f"{prefix}/{name}": value for name, value in score_cells(at_rate, baselines).metrics.items()}
        )

    if complete_sibling is not None:
        induced = _score_induced_missing(
            model, dataset, test_frame, complete_sibling, fold, device
        )
        cells = pd.concat([cells, induced], ignore_index=True)
        scored_induced = score_cells(induced, baselines)
        metrics.update(
            {f"impute/induced/{name}": value for name, value in scored_induced.metrics.items()}
        )
        if score_null_path:
            # The same gaps, left as the [NULL] the variant stores rather than swapped to
            # [MASK]. The head was never trained at a null position, so this measures
            # whether it transfers there. Diagnostic only; it never ranks a fold.
            through_null = _score_induced_missing(
                model, dataset, test_frame, complete_sibling, fold, device, as_mask=False
            )
            cells = pd.concat([cells, through_null], ignore_index=True)
            metrics.update(
                {
                    f"impute/induced/null_token/{name}": value
                    for name, value in score_cells(through_null, baselines).metrics.items()
                }
            )

    # The search objective: the same masked-population score on the fixed validation
    # mask the checkpoint watched, so a search never ranks trials on the test split. Its
    # own family, logged and returned only when asked, so no ordinary or cross-validation
    # run ever carries a validation/ key and the cv/test summariser never sees one.
    validation_metrics: dict[str, float | int | str] = {}
    if score_search_objective:
        validation_cells = _score_population(
            model, hidden_validation, clean_validation, "masked"
        )
        validation_metrics = {
            f"validation/impute/masked/{name}": value
            for name, value in score_cells(validation_cells, baselines).metrics.items()
        }
        tracker.log_metrics({name: float(value) for name, value in validation_metrics.items()})

    tracker.log_metrics({f"test/{name}": float(value) for name, value in metrics.items()})
    return DecodingOutcome(
        model=model,
        result=FoldResult(
            fold=fold_ordinal,
            dataset_name=dataset.frame.attrs.get("dataset_name", ""),
            metrics={**metrics, **validation_metrics},
        ),
        train_losses=train_losses,
        validation_losses=validation_losses,
        scored_cells=cells,
    )


def _score_induced_missing(
    model,
    dataset: PreparedDataset,
    test_frame: pd.DataFrame,
    complete_sibling: pd.DataFrame,
    fold: FoldSplit,
    device: torch.device,
    as_mask: bool = True,
) -> pd.DataFrame:
    """Score the cells the dataset is actually missing, against the complete sibling.

    These are the real imputation benchmark: a generator took them away, so their true
    value is known. By default they are shown to the model as ``[MASK]``, the token the
    decoder was trained to fill. With ``as_mask=False`` they keep the ``[NULL]`` the
    variant stores, which is the diagnostic path.
    """
    hidden = test_frame.mask(test_frame.isna(), "[MASK]" if as_mask else "[NULL]")
    population = "induced" if as_mask else "induced_null_token"

    truth = complete_sibling.iloc[fold.test_indices].reset_index(drop=True)
    truth = truth[[column for column in test_frame.columns]].copy()
    # The sibling as it is stored, kept before the scaling below overwrites it: this is
    # the only place an induced cell's true value exists, the variant holding a gap.
    raw_truth = truth.copy()
    if dataset.scaler is not None and len(dataset.numerical_columns):
        # Into the same scaled space the variant lives in, using the variant's own scaler.
        truth[list(dataset.numerical_columns)] = dataset.scaler.transform(
            truth[list(dataset.numerical_columns)]
        )

    embedder = model.embedder
    encoded = embedder.encode(hidden, device)
    if not as_mask:
        # [NULL] cells are not [MASK] cells, so nothing would be selected for scoring.
        # Point the selection at the gaps explicitly.
        encoded = _select(encoded, torch.tensor(test_frame.isna().to_numpy(), device=device))
    # A category the variant never shows (the generator took its every occurrence) is
    # outside the embedder's vocabulary, so the head can never produce it and the encoder
    # cannot even index it. Such a cell is a miss by construction: its truth is encoded
    # as [NULL], which no prediction ever equals, and its recorded value is put back on
    # the scored cell so the artifact and the metrics see what was really there.
    encodable_truth, unseen = _within_vocabulary(_clean(truth), embedder)
    scored = _score_population(
        model, encoded, embedder.encode(encodable_truth, device), population, raw_truth
    )
    for (row, column), value in unseen.items():
        hit = (scored["row"] == row) & (scored["column"] == column)
        scored.loc[hit, ["actual", "actual_original"]] = value
    return scored


def _within_vocabulary(frame: pd.DataFrame, embedder) -> tuple[pd.DataFrame, dict]:
    """The frame with every category the embedder never saw replaced by ``[NULL]``,
    and the (row, column) -> value map of what was replaced."""
    frame = frame.copy()
    unseen: dict[tuple[int, str], str] = {}
    for column in embedder.categorical_columns:
        known = set(embedder.label_encoders[column].classes_)
        values = as_category_strings(frame[column])
        outside = [row for row, value in enumerate(values) if value not in known]
        if not outside:
            continue
        for row in outside:
            unseen[(row, column)] = str(values[row])
        # ``get_loc`` returns a slice or a boolean mask when column labels repeat, and
        # ``iloc`` needs the positional int. Feature names are unique, so this always is
        # one; asserting states that rather than leaving ``iloc`` to fail obscurely if a
        # duplicated column ever reaches here.
        position = frame.columns.get_loc(column)
        assert isinstance(position, int)
        frame.iloc[outside, position] = "[NULL]"
        print(
            f"Warning: {len(outside)} induced cell(s) in '{column}' hold categories the "
            f"variant never shows ({sorted(set(values[row] for row in outside))}); scored "
            "as misses, since the decoder cannot produce them."
        )
    return frame, unseen


def _select(encoded, positions: torch.Tensor):
    """The same encoding, with these cells marked as the ones to reconstruct."""
    from dataclasses import replace

    return replace(encoded, masked_positions=positions)


def _already_missing(frame: pd.DataFrame) -> int:
    return int(frame.isna().to_numpy().sum())


def _score_population(model, hidden, truth, population: str, raw_truth=None) -> pd.DataFrame:
    """One row per cell the model was asked to fill, with the truth beside its guess."""
    embedder = model.embedder
    with torch.no_grad():
        prediction = model.predict(hidden)
    mask = hidden.masked_positions
    rows: list[dict] = []

    for index, column in enumerate(embedder.categorical_columns):
        selected = mask[:, index]
        if not bool(selected.any()):
            continue
        where = torch.nonzero(selected, as_tuple=False).squeeze(1).tolist()
        encoder = embedder.label_encoders[column]
        actual = encoder.inverse_transform(truth.cat_indices[index][selected].tolist())
        imputed = encoder.inverse_transform(prediction.categorical_ids[index][selected].tolist())
        confidence = prediction.categorical_confidence[index][selected].tolist()
        rows.extend(
            {
                "row": row, "column": column, "kind": CATEGORICAL, "population": population,
                "actual": str(a), "imputed": str(p), "confidence": float(c),
                "actual_original": str(a),
            }
            for row, a, p, c in zip(where, actual, imputed, confidence)
        )

    offset = len(embedder.categorical_columns)
    for index, column in enumerate(embedder.numerical_columns):
        selected = mask[:, offset + index]
        if not bool(selected.any()):
            continue
        where = torch.nonzero(selected, as_tuple=False).squeeze(1).tolist()
        rows.extend(
            {
                "row": row, "column": column, "kind": NUMERICAL, "population": population,
                "actual": float(a), "imputed": float(p), "confidence": float("nan"),
                "actual_original": _exact(raw_truth, row, column),
            }
            for row, a, p in zip(
                where,
                truth.num_values[index][selected].tolist(),
                prediction.numerical_values[index][selected].tolist(),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "row", "column", "kind", "population",
            "actual", "imputed", "confidence", "actual_original",
        ],
    )


def _exact(raw_truth, row: int, column: str) -> float:
    """The number the frame holds before scaling; NaN when it was not kept.

    ``actual`` stays in the scaled float32 space the metrics compare in. This is the
    same cell as a person would read it, and it never reaches a metric.
    """
    if raw_truth is None or column not in raw_truth.columns:
        return float("nan")
    return float(raw_truth.at[row, column])


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as the model should see it when nothing is being hidden on purpose."""
    return preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True)
