# Ticket 0003: Training Loop Performance (Behavior-Preserving)

## Status

Done.

## Problem

`TabularEmbedder.forward` was fed a pandas DataFrame for **every mini-batch**. Each call
re-ran `LabelEncoder.transform` per categorical column, looped in Python over every cell of
every numerical column, and copied the result to the GPU. On top of that, the numerical path
called `Tensor.nonzero()` three times per column per forward, and each of those forces a
device-to-host synchronisation.

Profile of one pre-training epoch on `vehicle_00nan` (CUDA, DIM 16, batch 64, 846 rows,
18 numerical columns), steady state:

| Cost centre | Share |
|---|---|
| `TabularEmbedder.forward` | ~67% |
| ...of which `nonzero` (54 synchronisations per forward) | ~25% |
| ...of which `split_numeric_and_special` (Python loop per cell) | ~16% |
| backward | ~12% |

The model itself is tiny, so the loop was CPU- and launch-bound rather than compute-bound.

## What changed

1. **`EncodedTable` + `TabularEmbedder.encode`** (`src/embedder.py`). A whole frame is converted
   to tensors once; batches are tensor slices. Pre-training re-encodes only the masked frames,
   once per epoch, because the masks are re-rolled; clean targets and all fine-tuning frames are
   encoded once per fold. `forward` still accepts a raw DataFrame, so `model(df)` and saved
   models keep working.
2. **Vectorized `split_numeric_and_special`** (`src/utils.py`). Whole-column comparisons instead
   of a per-cell Python loop. `astype(np.float64)` preserves the old `float(val)` behavior,
   including raising on non-numeric junk.
3. **`torch.where` instead of `nonzero` + index assignment** (`src/embedder.py`). The
   `[MASK]`/`[NULL]`/normal cases are mutually exclusive, so this is the same selection with no
   synchronisations.
4. **Batched per-column MLPs** (`src/embedder.py`). Parameters stay per column (unchanged
   `state_dict`, unchanged initialisation order), but they are gathered and applied with one
   broadcast multiply-add and one `baddbmm` rather than two `Linear` calls per column.
5. **Broadcast positional embedding** instead of gathering an expanded index matrix per row.
6. **`torch.no_grad()` for the pre-training target** instead of building a graph then detaching it.
7. **On-device float64 loss accumulation**, so per-batch `.item()` no longer synchronises. The
   metrics dict from `TridentPretrainer.forward` now carries a detached tensor for the same reason.
8. **`torch.randperm` stays on the CPU generator** and is moved to the device once per epoch.

## Exactness standard and result

HEAD was verified to be bit-reproducible across separate processes first, which allowed an exact
gate rather than a tolerance. A harness ran `prepare_dataset → build_folds → train_pretrainer →
train_and_evaluate_classifier` in `run_training`'s order and recorded every per-epoch loss, every
per-epoch validation metric and every test metric, on five configurations: the reviewed
`vehicle_00nan` fixture; a longer, wider `vehicle_00nan` run to amplify drift;
`credit-g_20nan` (categorical plus `[NULL]` in both column kinds); `kr-vs-kp_00nan`
(35 categorical columns); and `electricity_00nan` (45k rows).

- **Every final test metric is bit-identical** to HEAD across all five configurations (94 metrics).
- Intermediate loss values differ by at most **1.6e-6**, from the changed gradient summation
  order in `torch.where` and `baddbmm`. Both remain float-noise.
- The new code is itself bit-reproducible across processes.
- `tests/fixtures/vehicle_00nan_regression.json` passes **unmodified**.

## Measured speedup

Pre-training / fine-tuning milliseconds per epoch, CUDA, two samples averaged:

| Dataset | Shape | Pre-train | Fine-tune | Wall clock |
|---|---|---|---|---|
| `vehicle_00nan` | 846 rows, 18 numerical | 540 → 92 | 307 → 70 | 5.2x |
| `credit-g_20nan` | 1000 rows, 12 categorical + 8 numerical | 423 → 104 | 230 → 79 | 3.6x |
| `kr-vs-kp_00nan` | 3196 rows, 35 categorical | 1339 → 253 | 809 → 178 | 5.2x |
| `electricity_00nan` | 45312 rows, 8 numerical | 7820 → 1760 | 4091 → 1028 | 4.3x |

## Fixes included

- Removed a `print("a")` debug leftover in `TridentPretrainer.forward`.
- Removed the unused `test_loss` parameter of `build_fold_result`.
- `TabularEmbedder.__init__` now raises `ValueError` when two columns reduce to the same
  sanitized module key. They previously shared one MLP or embedding table silently.
- Aligned the epoch defaults: `Hyperparameters` and `Hyperparameters.from_mapping` disagreed
  (300/150 versus 40/40), which made `test_runner_uses_fold_buffers_and_finalizes_cross_validation_once`
  fail on HEAD. Both are now 100 pre-training and 75 fine-tuning epochs, matching README.
- Replaced `copy.deepcopy(model.state_dict())` with a detached clone.

## Found and deliberately NOT fixed

1. **Scheduler cadence.** `CosineAnnealingLR(T_max=epochs)` is stepped once per *batch*, so the
   learning rate reaches zero after `epochs` batches and then oscillates back up for the rest of
   training. `AGENTS.md` protects scheduler cadence, so this was left alone. It looks unintended.
2. **Fit-before-split leakage.** The `StandardScaler`, the label encoder and the categorical
   vocabularies are all fit on the full dataset before any split exists. `AGENTS.md` protects the
   split/scaling order.
3. **`"nan"` is a real category.** `astype(str)` turns null cells into the literal string `"nan"`,
   which enters each categorical vocabulary alongside `[NULL]`. Pre-training targets are built
   from the raw frame, so numerical nulls produce NaN vectors and categorical nulls hit the
   `"nan"` embedding. Those positions can never be `[MASK]` positions, so the loss never selects
   them and the behavior is harmless today, but it is fragile.
4. **`LABELS` is ignored.** `Hyperparameters.labels` is logged to MLflow but never used; the
   classifier width comes from the dataset's label classes.
5. **Double initialisation.** `train_pretrainer`'s `initialize_weights` re-applies Xavier to
   transformer layers that already initialised themselves, and to the positional table. Harmless,
   but it consumes RNG draws, so changing it would alter every seeded result.
