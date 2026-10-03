# Add an imputation task with a value decoder, keep classification the default, and tag it in MLflow

## Status

Accepted (2026-09-10). Charted and decided on the wayfinder map at
[`docs/wayfinder/imputation-decoder/map.md`](../wayfinder/imputation-decoder/map.md);
every numbered decision below links the ticket that holds its full reasoning and the
facts verified along the way. Implementation plan:
[`docs/wayfinder/imputation-decoder/plan.md`](../wayfinder/imputation-decoder/plan.md),
to be executed test-first with `mattpocock-skills:tdd`.
Resolves backlog item B3 as a side effect; leaves C1, C2, C4, I1 and I2 untouched and
routes around them, and raises the new item C5 (numeric precision and scaling).

The backfill of decision 7 was **applied on 2026-09-10** to the 251 runs recorded between
2026-07-03 and that date. The six imputation runs already present were skipped, keeping
their own `task = imputation`.

**Amended 2026-10-01** by [ADR 0013](0013-imputation-defaults-from-the-confirmatory-study.md): the decode stage defaults to 450 epochs (`--decode_epochs`),
and the imputation task's pre-training objective and schedule default to `embedding_normalized`
and `cosine`; classification is untouched.

**Amended 2026-10-03** by [ADR 0014](0014-gaps-shown-as-mask-in-the-decode-stage.md): the decode
stage shows a row's real gaps as `[MASK]`, the token the induced scoring shows them as, instead
of `[NULL]` (`--decode_gap_token null` restores it); a gap still never enters the loss.

**Amended 2026-10-03** by [ADR 0015](0015-decode-checkpoint-chosen-on-the-validation-gaps.md): the decode stage keeps the epoch whose validation
rows' own gaps score best instead of the epoch of the lowest validation-mask loss
(`--decode_checkpoint loss` restores it; a variant without a complete table falls back to it).

## Context

TRIDENT trains in two stages that share one embedder and one transformer: masked-cell
pre-training, which regresses the transformer output at `[MASK]` positions against the
detached clean embedding of the same cell, then classifier fine-tuning on the `[CLS]`
token. Imputation is only a by-product of that design. Masking is the pretext, but
nothing in the model maps a token back to a value, so TRIDENT cannot be evaluated as an
imputer at all.

The data already holds the ground truth for one. The `_20nan` through `_80nan` variants
are row-aligned with `_00nan` cell for cell (verified: same shape and columns, every
observed cell equal, `_00nan` equal to the raw CSV), with NaNs injected MCAR per feature
column by `datasets/generate_splits.py`. Every injected cell therefore has a known true
value.

The goal is an **imputation task** that uses the same dynamic-masking technique the
classification pipeline uses, adds a **decoder** that reconstructs actual cell values,
is trackable in MLflow beside classification runs, and leaves every classification run
bit-identical to today. `AGENTS.md` protects the split-and-scale order, the scheduler
cadence, the seeded draw sequence and the `vehicle_00nan` regression fixture; ADR 0003
set the pattern for gating a behaviour change behind a flag, an MLflow tag and a
backfill.

## Decision

1. **Task selection is a command-line enum, never a config key.**
   `--task {classification,imputation}` defaults to `classification`. It is a property of
   the invocation, held on `TrainingRequest`, logged as a run parameter beside
   `dataset_name` and `seed`. A task key in a per-dataset config would silently make
   different datasets train different tasks under `--all`.
   ([ticket 08](../wayfinder/imputation-decoder/issues/08-cli-and-hyperparameter-surface.md))

2. **A decode stage replaces fine-tuning; pre-training is untouched.** The runner's shape
   becomes `train_pretrainer` followed by either `train_and_evaluate_classifier` or
   `train_and_evaluate_decoder` (new module `src/training/decoding.py`). A joint
   pre-training objective and a value-loss-only variant remain reachable later as flag
   values, not rewrites.
   ([ticket 04](../wayfinder/imputation-decoder/issues/04-decoder-architecture-and-stage.md))

3. **The decoder is a set of per-column heads on the encoder output.** For categorical
   column `c`, `Linear(d, |real categories of c|)`: the output space **excludes**
   `[MASK]`, `[NULL]` and the dead literal `"nan"` entry that `astype(str)` puts into
   every vocabulary of a column with missing values (verified: 13 of 13 categorical
   columns on `credit-g_20nan`). For each numerical column, `Linear(d, hidden) -> ReLU ->
   Linear(hidden, 1)`, the embedder's input MLP mirrored, batched across columns the way
   `TabularEmbedder.forward` batches its MLPs. Targets are encoded from the
   fine-tuning-style processed frame (`preprocess_table(..., fine_tunning=True)`), not
   the raw frame pre-training encodes: the raw frame yields 1400 NaN numerical targets on
   `credit-g_20nan`, the processed frame none. ([ticket 04](../wayfinder/imputation-decoder/issues/04-decoder-architecture-and-stage.md);
   literature survey on branch `research/masked-cell-decoder-heads`)

4. **Decode-stage training mirrors fine-tuning.** Loss is the sum of the categorical
   cross-entropy and `LAMBDA_NUM` times the numerical MSE, each first averaged over its own
   count of scored cells, with no auxiliary embedding term. The encoder is fine-tuned, not
   frozen. Training masks are re-rolled every epoch at `PROB_MASCARA`, exactly as
   pre-training does. The stage joins the run's single `lr_scheduler`. The best
   validation-loss epoch is restored. The frame is feature-only. New heads take framework
   defaults; **the decode stage must never re-run pre-training's Xavier sweep**, which
   would erase the pretrained encoder.
   ([ticket 04](../wayfinder/imputation-decoder/issues/04-decoder-architecture-and-stage.md))

5. **Two cell populations are scored, on the test fold only.** *Self-masked cells*
   (observed cells hidden by `preprocess_table`) exist on every variant. *Induced-missing
   cells* (NaN in the variant, observed in the row-aligned `_00nan` sibling) are the
   headline wherever the sibling exists and are **fed to the model as `[MASK]`**, the
   token the decoder trained on. Training never reads the sibling. The evaluation mask is
   one fixed `preprocess_table` draw per fold, seeded from the run seed and fold ordinal,
   at `EVAL_MASK_RATE` (default 0.2, the literature standard); validation uses the same
   draw for checkpoint selection. Cells NaN in `_00nan` itself are never scored.
   ([ticket 03](../wayfinder/imputation-decoder/issues/03-imputation-ground-truth.md);
   protocol survey on branch `research/imputation-eval-protocol`)

6. **Metrics and fold ranking.** Per population: pooled `rmse_num_z` and `mae_num_z` in
   scaled space, pooled `acc_cat`, column-averaged `macro_f1_cat`, and per-type cell
   counts. Folds are ranked by

   ```text
   impute_score = w_num * (rmse_num_z / rmse_num_z_baseline)
                + w_cat * (err_cat    / err_cat_baseline)
   ```

   with baselines from mean/mode imputation computed on the **training fold** and applied
   at the scored test cells, `w_*` the fractions of scored cells per type, lower better,
   1.0 meaning no better than the naive baseline. It degrades correctly on the six
   all-numerical datasets and on all-categorical `kr-vs-kp`. A constant column can yield a
   zero baseline; that division is guarded. **The ranking metric and its direction are an
   explicit per-task property** rather than a maximise-shaped variant. Metric keys are
   `impute/masked/<metric>` and `impute/induced/<metric>`; tracking prefixes `test/` and
   the CV parent wraps them as `cv/test/.../<statistic>`. Per-column values go to an
   artifact, never to tracked metrics.
   ([ticket 05](../wayfinder/imputation-decoder/issues/05-loss-and-error-metrics.md))

7. **MLflow identity: two new dense tags on every run kind, backfilled.** `task`
   (`classification` | `imputation`) on parents, diagnostic children, Optuna studies and
   trials, backfilled as `classification` with `task_backfilled = true`. `is_optuna`
   (`true` | `false`), true only on study parents and trials (a `--retrain_best` parent
   stays false), backfilled as `false` with **no** marker because the store proves the
   value: it holds no Optuna run at all. Imputation runs stay in `TRIDENT/<base>` and are
   filtered by tag; run names are `impute_<dataset>_<timestamp>`. The backfill generalises
   `scripts/backfill_lr_scheduler_tag.py` rather than copying it. Three call sites set
   tags today (`_execution_tags`, `_log_diagnostic_children`, the study parent in
   `opt.py`) and all three need both tags.
   ([ticket 06](../wayfinder/imputation-decoder/issues/06-mlflow-run-identity.md))

8. **Artifacts, one home per layout.** `imputation_preview_fold_N.md`, a row triptych
   (actual / model saw / imputed) over a fixed-seed sample of test rows in original
   units; `imputation_cells_fold_N.csv`, one row per scored cell, a strict superset of the
   preview with an `in_preview` flag, both scalings and the model's confidence;
   `metrics/per_column_imputation.csv` on the parent, long format with a fold column. The
   first two attach where loss plots attach (every fold writes them, only the diagnostic
   children upload them). `ArtifactWriter` draws the sample and writes both files so they
   cannot disagree. Original units require retaining the fitted `StandardScaler` on
   `PreparedDataset`, which does not move the split-and-scale order.
   ([ticket 07](../wayfinder/imputation-decoder/issues/07-imputation-preview-artifact.md);
   prototype on branch `prototype/imputation-preview`)

9. **Hyperparameters and logging.** Five new defaulted keys:

   | Key | Default |
   |---|---|
   | `EPOCHS_DECODE` | 150 |
   | `LR_DECODE` | 0.001 |
   | `WEIGHT_DECAY_DECODE` | 0.0019 |
   | `LAMBDA_NUM` | 1.0 |
   | `EVAL_MASK_RATE` | 0.2 |

   plus the optional list `EVAL_MASK_RATES_EXTRA` (default empty). Every existing config
   keeps loading. **Only the parameters a task uses are logged**: an imputation run omits
   `EPOCH_FINE`, `LR_FINE`, `WEIGHT_DECAY_FINE` and `LABELS`. Every flag combination stays
   legal with one exception (decision 11).
   ([ticket 08](../wayfinder/imputation-decoder/issues/08-cli-and-hyperparameter-surface.md),
   [ticket 11](../wayfinder/imputation-decoder/issues/11-multi-rate-evaluation.md))

10. **Multi-rate evaluation happens within one run, for self-masked cells only.** The
    evaluation rate is not evaluation-only, since validation checkpoint selection uses it,
    so one primary rate drives both selection and ranking and the extra rates score that
    same checkpoint as test-only diagnostics under integer-percent segments
    (`impute/masked/rate_10/...`). Induced-missing cells need no sweep: the dataset ladder
    already is one. Off by default.
    ([ticket 11](../wayfinder/imputation-decoder/issues/11-multi-rate-evaluation.md))

11. **The `[NULL]`-path diagnostic ships behind `--score_null_path`, off by default.** It
    scores induced-missing cells with the model seeing `[NULL]` instead of `[MASK]`, under
    `impute/induced/null_token/...`, never ranking, adding one column to the cell ledger
    and leaving the preview alone. Because null cells never enter the decode loss, the
    number measures whether a readout trained on `[MASK]` transfers to `[NULL]`, not
    whether the null embedding is informative. The flag is **rejected at parse time**
    without `--task imputation` and warns at runtime on a dataset with no induced-missing
    cells. Pre-registered criterion: decision 5's `[MASK]` substitution is overturned only
    if the null path beats it on `impute_score` across a majority of folds on at least two
    datasets at both the 20% and 60% variants.
    ([ticket 12](../wayfinder/imputation-decoder/issues/12-null-path-diagnostic.md))

    **Verdict, measured 2026-09-10: the criterion is not met and decision 5 stands.** The
    `[MASK]` path won on **all eight** folds tested, on both datasets at both levels:

    | dataset | folds | `[MASK]` score | `[NULL]` score |
    |---|---|---|---|
    | `credit-g_20nan` | 2 | 1.008, 1.004 | 1.155, 1.180 |
    | `credit-g_60nan` | 2 | 1.022, 1.121 | 1.227, 1.422 |
    | `kr-vs-kp_20nan` | 2 | 0.984, 0.969 | 1.920, 1.533 |
    | `kr-vs-kp_60nan` | 2 | 1.012, 0.980 | 1.986, 1.640 |

    The margin is widest on all-categorical `kr-vs-kp`, where the `[NULL]` path is roughly
    twice as bad as filling the mode. That is what decision 11 predicted: the head is never
    trained at a null position, and with 36 categorical columns it has nothing to transfer
    from. The diagnostic stays available behind its flag; the primary path does not move.

    **Re-measured 2026-09-30, after the D-1 fix (`918bcd6`).** The imputation critique found
    that the `[NULL]` path chose its cells in the file's column order while the encoding reads
    categorical columns first, so on `credit-g`, which interleaves the two kinds, the credit-g
    rows above scored mostly cells that were never missing (critique D-1). With the selection
    in the encoder's order, the same criterion on the same four variants (two folds, seed 42,
    defaults, `cosine`; runs tagged `experiment=null-path-recheck-2026-09-30`), `[MASK]` wins
    on all eight folds again, so the verdict stands, now on valid credit-g rows:

    | dataset | folds | `[MASK]` score | `[NULL]` score |
    |---|---|---|---|
    | `credit-g_20nan` | 2 | 0.924, 0.902 | 0.985, 1.023 |
    | `credit-g_60nan` | 2 | 1.010, 1.010 | 1.135, 1.045 |
    | `kr-vs-kp_20nan` | 2 | 0.681, 0.676 | 0.793, 0.757 |
    | `kr-vs-kp_60nan` | 2 | 0.955, 0.909 | 1.023, 0.917 |

    The numbers are not comparable with the 2026-09-10 table (another schedule and three weeks
    of changes); only each row's two paths are.

12. **Isolation.** A frozen `TaskSpec` (name, ranking metric, direction) in a small
    registry, validated in `__post_init__`, dispatched with an `if`/`elif` chain like
    `create_tracker`. The spec reaches `summarize_cross_validation`,
    `write_cv_tracking_artifacts` and `opt.py` as an argument **defaulted to
    classification**, and `TrainingRequest.task` and `score_null_path` are defaulted, so
    every existing call site, unit test and the protected regression fixture pass
    unedited. The diagnostic manifest is keyed by the ranking metric's short name, which
    regenerates the classification artifact byte-identically. `DecodingOutcome` mirrors
    `FinetuningOutcome` plus a scored-cell table.
    ([ticket 09](../wayfinder/imputation-decoder/issues/09-isolation-strategy.md))

13. **Optuna.** One `define_search_space` with a task switch: the three fine-tuning
    parameters swap for `EPOCHS_DECODE`, `LR_DECODE`, `WEIGHT_DECAY_DECODE` plus
    `LAMBDA_NUM`; both evaluation rates are never sampled. The objective is the task's
    ranking metric in the task's direction. The failure path **raises `TrialPruned`
    instead of returning `0.0`**, which under a minimised objective would have scored
    crashes as perfect imputers. `HEADS` is constrained to divisors of `DIM` (backlog B3).
    Best-trial tracking takes its comparison from the task direction. An imputation study
    writes only into its own study directory, never into
    `datasets/hiperparams/<base>/<dataset>.json`, which both tasks read and which an
    imputation study would otherwise overwrite for classification.
    ([ticket 13](../wayfinder/imputation-decoder/issues/13-optuna-for-imputation.md))

14. **A second regression fixture on `credit-g_20nan`**, the only variant exercising both
    column types and the induced-missing population, pinning `impute_score`, `rmse_num_z`
    and `acc_cat` per population, in the existing fixture's exact pattern (environment
    block, measured deviation, tolerance well above it). Degenerate compositions and the
    zero-baseline guard are unit tests. `AGENTS.md` names both fixtures.
    ([ticket 14](../wayfinder/imputation-decoder/issues/14-imputation-regression-fixture.md))

## How to compare

Filter `tags.run_role = 'parent'` and `tags.task = 'imputation'` (add
`tags.is_optuna = 'false'` to drop search runs), then compare within one
`tags.lr_scheduler` as ADR 0003 requires. On a `_20nan`..`_80nan` variant the headline is
`cv/test/impute/induced/impute_score/mean` with its `ci95_lower`/`ci95_upper`; on any
variant, `cv/test/impute/masked/impute_score/mean` is comparable across the ladder. Lower
is better; 1.0 is baseline parity. Open the `best_fold`/`worst_fold` children for the
preview and cell ledger. Classification comparisons are unchanged: add `tags.task =
'classification'` to exclude imputation runs, or rely on the backfilled value.

Because the decode stage has no published runs to preserve and the default schedule is the
legacy per-batch cosine, launch imputation experiments with `--lr_scheduler cosine`.

## Considered options

- **Joint pre-training objective, or value loss only.** Rejected for now: both change
  the protected pre-training path. Both remain reachable as flag values.
- **Tied heads (Design B) or an MAE-style decoder (Design C).** Deferred as ablations;
  B tells whether the embedding geometry is already decodable, C is the only pattern the
  literature validates as a tabular imputer.
- **Score only self-masked or only induced-missing cells.** Rejected: the first ignores
  the true benchmark, the second does not exist on `_00nan`.
- **Feed induced-missing cells as `[NULL]` for the primary path.** Rejected pending the
  pre-registered criterion in decision 11.
- **A separate `TRIDENT/<base>_decode` experiment, or overloading `run_type`.** Rejected:
  the first halves each dataset's history, the second conflates execution kind with task
  and sits on only 90 of 251 runs.
- **A `_decode` run-name suffix.** Rejected: a suffix hides behind the timestamp.
- **A neutral manifest key, a list-valued evaluation rate, a maximise-shaped ranking
  metric, per-column MLflow metrics.** Each rejected in its ticket for changing a
  reviewed artifact, hiding the primary rate, hiding the metric's meaning, or adding
  roughly 340 series per fold on spambase.
- **Returning a direction-aware sentinel on Optuna failure.** Rejected in favour of
  pruning, which never scores a failed trial in either direction.

## Consequences

- Classification runs are bit-identical: the task defaults, the fixture is unedited, the
  manifest is byte-identical, and a unit test asserts no decoder is constructed.
- New: `src/training/decoding.py`, `TaskSpec`, `DecodingOutcome`, a scaler reference on
  `PreparedDataset`, five hyperparameter keys, two request fields, two tags at three call
  sites, three artifact writers, a generalised backfill applied to 251 runs.
- Changed for both tasks: the Optuna failure path and search-space constraint. No Optuna
  run exists in the store, so nothing published depends on the old sampling.
- Still fog on the map: what `--save_model` writes for a decoder run, and whether an
  imputation run should export a fully imputed dataset.
- Out of scope, recorded on the map: impute-then-classify, mask-versus-null input for the
  classifier, benchmarking against external imputers, fixing C1/C2, and the ablations
  above.
