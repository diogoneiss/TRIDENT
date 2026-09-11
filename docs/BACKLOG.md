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
| ~~B2~~ | ~~`--cv_folds 1` fails with an error naming `n_splits`~~ -- fixed, see below | Medium | No |
| ~~B3~~ | ~~Optuna proposes head counts that crash, scored as 0.0~~ — fixed, see below | Medium | No |
| ~~B4~~ | ~~Optuna with MLflow enabled scored every trial 0.0~~ — fixed, see below | High | No |
| C1 | Scaler and encoders fit on the whole dataset before splitting | High | Yes |
| C2 | `"nan"` is a real category next to `[NULL]` | Low today | Yes |
| C3 | `LABELS` hyperparameter is logged but never used | Low | No |
| ~~C5~~ | ~~Numeric precision and scaling: float32 storage, inverse-scaling error~~ -- fixed, see below | Medium | Partly |
| C4 | Pre-training re-initializes already-initialized layers | Low | Yes |
| P1 | `preprocess_table` row fallback is a Python loop -- **partly fixed**, the rest is protected | Low now | Partly |
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

**Fixed 2026-09-10.** `validate_parsed_args` now refuses `cv_folds < 2` when the arguments
are read, and `build_folds` raises a `ValueError` naming `cv_folds` for callers that reach
it directly -- `train.main` takes a Namespace and Optuna builds its own, so the parse-time
guard alone would not cover them. Both messages point at omitting the flag, which is the
predefined-split mode and the single-split path the user actually wanted.

**The reported symptom below was wrong**, and the correction is worth keeping: the
`ZeroDivisionError` is unreachable. scikit-learn's `KFold`/`StratifiedKFold` refuses
`n_splits=1` before `raw_folds` is ever iterated, so the validation ratio is never
computed. What a user actually hit was `k-fold cross-validation requires at least one
train/test split by setting n_splits=2 or more, got n_splits=1` -- a clear enough error
that names an argument they never typed. The fix is the same either way.

The original analysis is kept below.

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

**Fixed 2026-09-10** by [ADR 0004](adr/0004-imputation-decoder-task.md): `define_search_space` now draws `HEADS` first and `DIM` as a multiple of it, so an invalid pair cannot be sampled at all, and a crashing trial raises `optuna.TrialPruned` instead of being scored. Returning `0.0` was only "the worst possible score" while maximising macro F1; under the imputation task's minimised error ratio it is the *best* achievable value, so the search would have hunted for crashes. The original analysis is kept below.

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

### B4. Optuna with MLflow enabled scored every trial 0.0

**Fixed 2026-09-09.** Each trial opened a nested MLflow run and then called the
trainer, whose tracker called `mlflow.start_run` without `nested=True`. MLflow
raises `Run with UUID ... is already active` for that, the trial's bare
`except Exception` swallowed it, and the trial returned 0.0, so a study with
tracking on was a study of zeros. The store held no Optuna runs, consistent with
this never having worked since the training refactor.

The runner now takes a tracking role: `opt.py` sets `mlflow_run_role =
"optuna_trial"` and the runner logs a lightweight record into the trial run the
objective already has open (`src/training/tracking.py`, `OptunaTrialTracker`),
instead of opening a second top-level run. Trials carry parameters, the
structured tags and final metrics only; the winning trial is tagged
`best_trial = true` after the study, and `--retrain_best` produces a normal
parent run tagged `optuna_study_run_id`. Covered by
`tests/unit/test_opt_tracking.py`.

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

### C5. Numeric precision and scaling

**Fixed 2026-09-10** by [ticket 0004](tickets/0004-imputation-preview-precision.md),
for concerns 1 and 2. The research framing below was wrong about concern 1: it is not a
dtype question but a **provenance** one, and asking it the right way dissolved the
float64 cost question entirely.

`split_numeric_and_special` downcasts to float32 at `src/utils.py:123`, which is correct,
because the model's parameters are float32. The defect was that the *displayed* truth was
then read back out of that tensor (`src/training/decoding.py`) when the exact value was
still available upstream. Preparation now keeps the numerical columns as they stand before
scaling (`PreparedDataset.raw_numerical`), the decode stage carries the exact number across
in `actual_original`, and the preview prefers it. Nothing wants float64 in the tensor; the
`imputed` side stays float32 because that is the model's real output precision.

Measured on a real `kc2_20nan` run, the widest column `e` (`scale_` 123,433) was displayed
as `870848.556` where the dataset holds `870848.580`. Two of its 21 columns moved at the
display precision.

Concern 2 is fixed by the display rules chosen with it: three decimals, anything below
`5e-4` printed as `0.000`, and a `*` marking any number the rendering had to shorten, with
a legend pointing at the cell ledger for the full value.

**Concern 3 stands as recorded, not fixed**: clamping was rejected deliberately, and
rounding to three decimals shows `-38.55` as `-38.550`, still negative and, being exact at
that precision, unmarked -- the marker signals shortening, never implausibility. The interaction
with C1 was checked and is none: display inverts through whatever scaler produced the
scaled space, while C1 is about scoring fairness.

Scoring never moved -- `scored_cells["actual"]` and `["imputed"]` are what the metrics
read and they are untouched -- so `credit-g_20nan` and `vehicle_00nan` both pass unedited.

The original analysis is kept below.

Three related concerns surfaced while reading a real imputation preview
(`spambase_20nan`). None is a wrong number today, but together they decide how far the
original-unit reporting can be trusted, so they want investigating rather than patching.

**1. Values round-trip through float32.** `EncodedTable.num_values` is float32 while the
prepared frame is float64. Inverting the scaler from the float32 value amplifies the
rounding by that column's `scale_`, so a column with a wide spread shows visible error:

| dataset | widest column | `scale_` | error in original units |
|---|---|---|---|
| `kc2_20nan` | `e` | 123,400 | 1.15e-01 |
| `credit-g_20nan` | `credit_amount` | 2,826 | 6.64e-04 |
| `spambase_20nan` | `capital_run_length_longest` | 206.7 | 2.79e-04 |
| `electricity_20nan` | `vicprice` | 0.0109 | 4.03e-08 |

**Scoring is unaffected**: both sides of every comparison are the same float32 values, so
`rmse_num_z` and `impute_score` are exact. It is the *original-unit display* that inherits
the amplification, and on `kc2` that is a tenth of a unit.

**2. Exact zeros print as scientific-notation noise.** `spambase` is 77% exact zeros, and
each prints as e.g. `1.144e-09` in the preview, which reads as a meaningful tiny value
rather than "zero". Cosmetic, but it makes the artifact harder to trust at a glance.

**3. Numerical imputations are unbounded.** The numerical head is a plain regressor, so it
happily predicts a negative count (`capital_run_length_total` imputed at `-38.55`) or a
negative word frequency. **Clamping was considered and deliberately rejected** by the
author: the model should not be handed constraints it did not learn. Recorded so the
behaviour is understood, not so it is silenced.

Questions worth answering: should `num_values` be float64, and what does that cost in
memory and speed on the 45k-row datasets? Should the preview round to the column's own
observed precision instead of a fixed format? Does any of this interact with C1, where the
scaler is fit before splitting?

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

**Partly fixed 2026-09-10, without moving a single draw.** The analysis below concluded
that fixing this "changes the random draw sequence, so seeded results would move". That is
true of the vectorisation it proposed, but it is not true of the cost. Measured at
`p_base` 0.05 on 45000 rows, **94% of the loop's time was the pandas row lookup**
`null_matrix.iloc[i].values`, not the draw: 998 ms against 60 ms for the same lookup
through numpy.

Hoisting one `null_matrix.to_numpy()` out of the loop leaves the `np.random.choice` calls
identical in count, order and argument, so the seeded sequence is untouched. Measured on
one machine, before against after, with the output frames compared by `DataFrame.equals`:

| `p_base` | before | after | speed-up | output identical |
|---|---|---|---|---|
| 0.50 (default) | 96 ms | 71 ms | 1.4x | yes |
| 0.20 (Optuna's floor) | 310 ms | 139 ms | 2.2x | yes |
| 0.05 | 637 ms | 236 ms | 2.7x | yes |

`tests/unit/test_preprocess_table.py` pins the draw as a fixed literal captured from the
pre-change implementation, and `vehicle_00nan` passes unedited.

**What is deliberately left, and is protected**: replacing the loop entirely with one
`argmax` over a random matrix, as suggested below. It would be faster again, but it
redraws the fallback, so it moves every seeded result and the `vehicle_00nan` baseline
with them. That makes it a training-behaviour change, needing a flag, an MLflow tag and a
backfill like [ADR 0003](adr/0003-selectable-learning-rate-schedule.md), and it should not
be done for speed alone. The cliff that motivated the item is largely flattened without it.

The original analysis is kept below.

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
- **B4**, Optuna trials crashing on a second top-level MLflow run, via the
  `optuna_trial` tracking role (see above).
- **B2**, `--cv_folds 1`, via a parse-time rejection plus a `build_folds` guard for the
  programmatic callers. The `ZeroDivisionError` in the original report was unreachable.
- **P1**, partly: the pandas row lookup is hoisted out of the fallback loop, 2.2x at
  Optuna's lowest `PROB_MASCARA`, with the draw sequence untouched. Full vectorisation
  stays open and is protected.
- **B3**, invalid `HEADS`/`DIM` pairs, via [ADR 0004](adr/0004-imputation-decoder-task.md)
  (constrained sampling plus `TrialPruned` on failure).
- **C5**, concerns 1 and 2, via [ticket 0004](tickets/0004-imputation-preview-precision.md)
  (exact truth retained before scaling; three-decimal display with a rounding marker).
  Concern 3, unbounded numerical imputations, stands as recorded behaviour.

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
