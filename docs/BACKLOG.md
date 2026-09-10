# TRIDENT Backlog: Bugs, Pendencies and Improvements

Findings collected while optimizing the training loop (see
[ticket 0003](tickets/0003-training-loop-performance.md)). Everything here was
verified against the code, not inferred. Nothing in this file has been fixed;
items already fixed are listed at the bottom for reference.

Each item says whether it is **protected** by `AGENTS.md`, which forbids changing
training behavior — notably split/scaling order and scheduler cadence — unless a
task explicitly authorizes it. Protected items need a decision from the author
before anyone touches them, because they change published results.

| ID | Item | Severity | Protected |
|---|---|---|---|
| ~~B1~~ | ~~Learning-rate schedule completes a full cosine cycle~~ — fixed, see below | High | Yes |
| B2 | `--cv_folds 1` crashes with `ZeroDivisionError` | Medium | No |
| B3 | Optuna proposes head counts that crash, scored as 0.0 | Medium | No |
| C1 | Scaler and encoders fit on the whole dataset before splitting | High | Yes |
| C2 | `"nan"` is a real category next to `[NULL]` | Low today | Yes |
| C3 | `LABELS` hyperparameter is logged but never used | Low | No |
| C4 | Pre-training re-initializes already-initialized layers | Low | Yes |
| P1 | `preprocess_table` row fallback is a Python loop | Medium | No |
| P2 | Hand-rolled attention instead of fused SDPA | Medium | Yes |
| I1 | No way to set hyperparameters from the CLI | Medium | No |
| I2 | Optuna overwrites the dataset's hyperparameter file | Medium | No |
| H1 | `create_pretrain_datasets` is dead code | Low | No |
| H2 | Padding-mask branch is unreachable | Low | No |
| H3 | `torch.save` pickles the whole model object | Medium | No |
| H4 | `opt.py` rebinds the module-level `mlflow` name | Low | No |

---

## Bugs

### B1. The learning-rate schedule completes a full cosine cycle — protected

**Fixed 2026-09-09** by [ADR 0003](adr/0003-selectable-learning-rate-schedule.md): the
schedule is now selectable (`--lr_scheduler` / `LR_SCHEDULER`), the old behaviour
survives as `cosine_legacy` and remains the default, and every MLflow run carries an
`lr_scheduler` tag (backfilled on the 191 pre-existing runs). The original analysis
is kept below for reference.

`CosineAnnealingLR` is built with `T_max=<number of epochs>` but `scheduler.step()`
is called once per **mini-batch**, in both stages
(`src/training/pretraining.py`, `src/training/finetuning.py`).

`T_max` counts scheduler steps, so the cosine period is measured in batches, not
epochs. The learning rate reaches zero after `T_max` batches and then rises again.

Reproduced with the current vehicle configuration (300 epochs, batch 256,
cv_folds 3, roughly 2 batches per epoch, so about 600 steps):

```
after   1 batches: lr = 3.400e-04
after 300 batches: lr = 0.000e+00     <- intended end of the schedule
after 600 batches: lr = 3.400e-04     <- training actually ends here
```

Training therefore finishes at the **starting** learning rate rather than an
annealed one. The severity depends on the batches-per-epoch ratio, so it differs
per dataset: small datasets complete more cycles than large ones, which makes runs
across datasets less comparable than they look.

Likely intent: step once per epoch, or set `T_max` to `epochs * batches_per_epoch`.
Either changes every published number, so it needs an explicit decision.

### B2. `--cv_folds 1` crashes

`build_folds` (`src/training/data.py`) computes
`validation_ratio = 0.1 / (1.0 - 1.0 / cv_folds)`, which divides by zero when
`cv_folds` is 1. `validate_parsed_args` (`src/training/config.py`) does not
constrain the value.

```
ZeroDivisionError: float division by zero
```

Suggested fix: reject `cv_folds < 2` at parse time with a clear message, pointing
users at the predefined-split mode (omit `--cv_folds`) for a single split.

### B3. Optuna proposes head counts that crash, and scores them as 0.0

`define_search_space` (`opt.py`) samples `DIM` over `range(64, 257, 32)` and `HEADS`
over `{4, 8, 12, 16}` independently. `MultiHeadAttention` computes
`head_dim = hidden_size // num_heads` and then reshapes, which requires `HEADS` to
divide `DIM` exactly.

5 of the 28 reachable combinations are invalid, all of them `HEADS=12`:
`(64, 12), (128, 12), (160, 12), (224, 12), (256, 12)`.

```
RuntimeError: shape '[4, 19, 12, 10]' is invalid for input of size 9728
```

`ObjectiveFunctionWrapper.__call__` catches every exception and returns `0.0`, so
Optuna records these as legitimately terrible hyperparameters rather than as
configuration errors. That biases the search away from `HEADS=12` for reasons that
have nothing to do with model quality, and roughly 18% of the grid is wasted.

Suggested fix: constrain the search space so `HEADS` divides `DIM`, or raise
`optuna.TrialPruned` for invalid combinations so they are not scored. Separately,
consider narrowing the bare `except Exception` so genuine crashes stay visible.

---

## Correctness and methodology

### C1. Scaler and encoders are fit before splitting — protected

`prepare_dataset` (`src/training/data.py`) fits `StandardScaler` and `LabelEncoder`
on the **entire** frame before any fold indices exist, and `TabularEmbedder` builds
each categorical vocabulary from the whole feature frame. Test-set statistics
therefore reach training.

This is called out in `docs/ARCHITECTURE.md` and protected by `AGENTS.md`, so it
appears to be a deliberate choice. Worth confirming it is intentional and stating
so in the paper, since it inflates results relative to a per-fold pipeline.

### C2. `"nan"` is a real category next to `[NULL]` — protected

`TabularEmbedder` builds vocabularies with `df[col].astype(str).unique()`, which
turns a missing categorical cell into the literal string `"nan"`. That token gets
its own embedding alongside the dedicated `[NULL]` token, so missingness is
represented two different ways depending on the code path.

Related: pre-training targets come from the raw frame, where nulls are still `NaN`
rather than `[NULL]`. Numerical nulls therefore produce NaN target vectors.

This is currently harmless because masking never overwrites an already-null cell,
so those positions can never be selected by the reconstruction loss. It is fragile:
any change to the masking rule would start feeding NaN into the loss.

### C3. `LABELS` is logged but never used

`Hyperparameters.labels` is parsed, stored and logged to MLflow, but the classifier
width comes from `len(dataset.label_classes)` in `train_and_evaluate_classifier`.
The recorded parameter is therefore misleading: setting `LABELS: 2` on a 4-class
dataset changes nothing but implies otherwise.

Suggested fix: drop the field, or validate it against the dataset and fail loudly
on a mismatch.

### C4. Pre-training re-initializes already-initialized layers — protected

`train_pretrainer` applies `initialize_weights` to the whole model, running Xavier
over every `nn.Linear` and `nn.Embedding`. The transformer's `FeedForwardNetwork`
and `MultiHeadAttention` already Xavier-initialize themselves in their
constructors, and the positional embedding table gets overwritten too.

Harmless in effect, but it consumes random draws, so removing it shifts every
seeded result.

---

## Performance still on the table

### P1. `preprocess_table` falls back to a Python loop per row

After drawing the random mask, `preprocess_table` (`src/utils.py`) guarantees at
least one masked cell per row by looping in Python over every row that got none:

```python
for i in np.where(no_mask_rows)[0]:
    non_null_indices = np.where(~null_matrix.iloc[i].values)[0]
    ...
```

Cost scales with how many rows need that fallback, which explodes at low mask
probabilities. Measured on 45000 rows and 8 columns:

| `p_base` | Rows needing the fallback | Time per call |
|---|---|---|
| 0.50 | 201 | 74 ms |
| 0.05 | 29709 | 1378 ms |

This is called twice per epoch during pre-training. At the default `PROB_MASCARA`
of 0.5 it is not a problem, so it was left alone. It is a latent cliff for anyone
tuning `PROB_MASCARA` downward, which Optuna does (its range starts at 0.2).

Vectorizable by choosing one random non-null column per affected row with
`argmax` over a random matrix masked by `~null_matrix`. Note that this changes the
random draw sequence, so seeded results would move.

### P2. Hand-rolled attention instead of fused SDPA — protected

`MultiHeadAttention` (`src/transformer.py`) materializes the full attention matrix
and applies softmax and dropout by hand.
`torch.nn.functional.scaled_dot_product_attention` would fuse this and cut memory.

Deliberately excluded from the optimization work: fused SDPA draws its dropout mask
differently, so seeded runs would no longer reproduce. The same reasoning excluded
fused AdamW and TF32 matmuls. All three are available whenever exact
reproducibility against existing runs stops being a requirement.

---

## Improvements

### I1. Hyperparameters cannot be set from the CLI

`main.py` exposes no hyperparameter flags. The only ways in are a JSON file at
`datasets/hiperparams/<base>/<dataset>.json`, or the `hyperparams_override`
attribute that only `opt.py` sets.

Two consequences:

- Reproducing a run with specific hyperparameters means creating a file per
  dataset. A nine-dataset batch run needs nine identical files.
- If the path is wrong, `load_hyperparameters` silently falls back to defaults. No
  warning is printed, so a typo looks like a successful run with different results.

Suggested fix: add `--hyperparams <path.json>` that applies to every dataset in the
invocation, and log at INFO which source was used (override, file, or defaults).

### I2. Optuna overwrites the dataset's hyperparameter file

`ObjectiveFunctionWrapper.save_best_params` writes
`datasets/hiperparams/<base>/<dataset>.json` every time a trial improves, and
`run_hyperparameter_optimization` writes it again at the end. Any hand-written
config at that path is destroyed without confirmation, and a partially finished
study leaves the file holding a mid-search best rather than a final one.

Suggested fix: write to the timestamped `optuna_<timestamp>/` directory only, and
require an explicit flag to promote a result into `datasets/hiperparams/`.

---

## Code health

### H1. `create_pretrain_datasets` is dead code

Defined in `src/utils.py`, never called anywhere in the repository. It also
duplicates, in an older form, the split logic that `build_folds` now owns.

### H2. The padding-mask branch is unreachable

`TabularTransformerEncoder.forward` accepts `src_key_padding_mask`, and
`MultiHeadAttention` has a `masked_fill` branch for it, but both call sites
(`src/models.py`) invoke the transformer with a single argument. Every row has the
same fixed column count, so padding never occurs. The parameter and its branch are
untested and untestable as wired.

### H3. `torch.save` pickles the whole model object

`ArtifactWriter.save_model` (`src/training/artifacts.py`) calls
`torch.save(model, path)`, which pickles the class structure rather than weights.
Checkpoints break whenever the module layout changes, need `weights_only=False` to
load, and there is no loader in the repository at all.

Suggested fix: save `model.state_dict()` plus the hyperparameters needed to rebuild
the model, and add the matching load path.

### H4. `opt.py` rebinds the module-level `mlflow` name

When tracking is disabled, `run_hyperparameter_optimization` executes
`global mlflow; mlflow = _DisabledMlflow()`, permanently swapping the real library
for a stub in that module for the rest of the process. It works, but it is
action-at-a-distance, and `_DisabledMlflow` has to keep pace with every MLflow call
`opt.py` makes or it fails with `AttributeError`.

Suggested fix: pass a tracker object explicitly, the way `src/training/tracking.py`
already does with `create_tracker(enabled)`.

---

## Fixed already

- **B1**, the per-batch cosine schedule, via [ADR 0003](adr/0003-selectable-learning-rate-schedule.md)
  (selectable schedule, legacy default, MLflow tag plus backfill script).

These came out of the same review and are done, in
[ticket 0003](tickets/0003-training-loop-performance.md):

- `Hyperparameters` and `Hyperparameters.from_mapping` declared different epoch
  defaults, so the value depended on which path built the request. This made a unit
  test fail. Both now agree.
- A `print("a")` debug statement in the pre-training loss.
- An unused `test_loss` parameter on `build_fold_result`.
- Two column names that sanitized to the same module key silently shared one MLP or
  embedding table. This now raises `ValueError`.

## Open housekeeping

`datasets/hiperparams/vehicle/vehicle_00nan.json` was created to pin the single
vehicle run. It is untracked and its contents now match the defaults exactly, so it
changes nothing today. It will silently keep vehicle at 300/150 if the defaults are
ever lowered again. Delete it or commit it deliberately.
