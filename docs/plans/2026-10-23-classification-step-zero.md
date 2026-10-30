# Classification step zero: a plan for the user to approve (T27; 2026-10-23)

**Status: proposed.** Nothing here runs or changes `src/` until the user decides the items in
"Decisions" below.

- **Each stage gets its own ticket** under `docs/tickets/classification-step-zero/` and its own
  EXPERIMENTATION_LOG entry, pre-registered before its first cell.
- **The search-objective change** is [ADR 0029](../adr/0029-classification-search-ranked-on-validation.md),
  proposed with this plan.
- **The user's go to write it:** "ok, vá agora" (2026-10-23, about 17:35 GMT-3), on the
  recommendation to start the classification front with a plan and an ADR.

**Sources, cited rather than re-derived:**

- this repository's plan of 2026-10-17, §2g, the review's item 12;
- the 2026-10-16 review's digests (`docs/reviews/2026-10-16-quick-wins/digests/merge_digest.txt`,
  items #13–#16 and its CLASSIFICATION and NOT QUICK sections; `review_digest.txt`, item 10);
- the wiki's thesis goals (`projects/trident/thesis-goals.md`, H1, H1b and H1c).

## Why

- **The anchor exists, and it is not comparable to the paper (review #13, done).** The fork's
  45-variant default classification sweep gives mean test macro-F1 of 0.858 / 0.799 / 0.742 /
  0.661 / 0.531 at 0–80% MCAR. Its setup: 3-fold StratifiedKFold, seed 42, EPOCHS_PRE 300 and
  EPOCH_FINE 150, `cosine_legacy`, untuned defaults, last run 2026-09-10.
- **It differs from the paper's Table 3 on three independent knobs:**
  - the split shape (3-fold trains on 56.7% of rows; the paper uses one 80/10/10 split);
  - the epoch budget (upstream 40/40);
  - the tuning (the paper's TRIDENT was tuned on test macro-F1).
- **Step zero produces the comparable number,** H1c's test: "TRIDENT's lead from 40% missingness
  survives a fair rerun". End-to-end TRIDENT, tuned on validation only, against the best tuned
  GBDT at 40, 60 and 80% MCAR, with the same tuning budget and paired differences.
  - **Refuted** if, at any of those rates, the paired mean difference favours the best GBDT.
  - **Unconfirmed, not refuted,** if the CI spans zero.
- **No classification run has seen ADRs 0011, 0013, 0014, 0015, 0018 or 0027.**
  - `task_defaults('classification') == {}` is pinned, so classification still runs the upstream
    configuration with `cosine_legacy`.
  - Step zero measures that configuration, not imputation's defaults.

## Stages

### Stage 0 · A determinism spot check (review #16, trap 1)

- **What:** reproduce one old sweep run's fold macro-F1 at HEAD before anything is paired with the
  old sweep.
- **Why:**
  - The environment drifted: Python 3.10 → 3.11, sklearn 1.7.2 → 1.9.0, numpy 2.2.6 → 2.4.6, and
    commits landed after the sweep.
  - The yardstick is the sweep's duplicate-run spread, 0.0002–0.009.
- **The folds:** no fold indices are stored. `build_folds(frame, 'class', 3, 42)` reproducing the
  sweep's folds is an inference from determinism, not a verified index match; the ticket says so.
- **Cost:** minutes. No code.

### Stage 1 · The CPU bar on the sweep's folds (review #14, #15)

- **The GBDT-MIA bar:** LightGBM and HistGradientBoosting classifiers with native NaN,
  `class_weight='balanced'`, and early stopping on `fold.validation_indices`.
  - The input matrix comes from the training fold's vocabulary, with NaN for missing or unseen
    values.
  - Test is scored with `build_fold_result`, paired per fold with the sweep's `f1_macro`.
- **Impute-then-learn, with and without `isna()` indicators,** each trained with HGB, LightGBM and
  RandomForest (balanced). Three imputers:
  - mode, then the column-wise LightGBM imputer on numerical columns, the paper's protocol
    approximately;
  - mean/mode;
  - missForest.
- **Traps carried over verbatim:**
  - Never feed `as_category_strings` output to a native arm: it renders NaN as the `nan` category.
  - electricity_80nan has 7,593 rows with every feature missing.
  - `THREADS` is hard-coded in `missforest.py` and `lightgbm_baseline.py` and overrides
    `OMP_NUM_THREADS`, so the script sets them to 4.
  - Three pairs per variant are descriptive only.
  - The indicator contrast under MCAR is a sanity control, not evidence of informative missingness.
- **Code and cost:**
  - an offline script, with no `src/` change and no store write;
  - about 4–6 h of code;
  - a few CPU-hours on gorgona8's free cores, beside any GPU study.

### Stage 2 · The no-pre-training arm (review #16)

- **The arm:** `EPOCHS_PRE=0` through `hyperparams_override`, 3-fold, seed 42, all 45 variants,
  paired with the sweep (B − A on macro-F1, pooled by level).
  - **Optional arm C:** `EPOCHS_PRE=0` with `EPOCH_FINE=450`, for equal epochs as in E24.
- **Always a private `--metrics_dir`:** the default `metrics/<variant>_metrics.csv` gets
  overwritten.
- **What it measures:** the pre-training gain under the fork's defaults. It is not a replication
  of the paper's tuned Table 4.
- **Code and cost:**
  - an override, so no code;
  - about 0.3× the sweep's compute, roughly 1–2 h on the RTX 3090 Ti;
  - the sweep ran on the laptop's RTX 3050, so Stage 0's check must pass on gorgona8 first, or arm
    A is rerun there.

### Stage 3 · Validation-ranked classification search (ADR 0029, proposed)

The code facts it rests on:

- `opt.py:179` sets the classification `search_objective = 'f1_macro'`, and `opt.py:285` reads it
  from the trial's test metrics. Classification Optuna ranks trials on test macro-F1, as upstream
  does.
- `finetuning.py` logs `finetune/val_f1_macro` per epoch, but `FoldResult.metrics` never carries the
  kept epoch or its validation F1.
- `--cv_folds` is silently dropped under `--use_optuna`.
- `--promote_best` for classification writes the shared `<dataset>.json`, which imputation falls
  back to (F-07-4).

ADR 0029 gives the decision and its options. Code: about 1 day with tests. The classification
fixture (vehicle_00nan) checks only `accuracy`, `f1_micro` and `f1_macro`, so adding keys to the
fold metrics leaves it unchanged.

### Stage 4 · Step zero proper (H1c)

- **The split:** `--cv_folds 10` gives exactly 80/10/10 and keeps the predefined batch-count parity
  on all 9 tables.
  - `--cv_folds 5` would be 70/10/20, and it would flip `cosine_legacy`'s end-state on biodeg,
    credit-g and kr-vs-kp.
  - Thesis-goals asks for "5 repeated splits". A repeated 80/10/10 mechanism needs code (decision
    1).
- **The arms:**
  - **TRIDENT, upstream budget:** 40/40 epochs.
  - **TRIDENT, fork budget:** 300/150 epochs.
  - **Each TRIDENT arm is tuned on validation macro-F1** at the kept checkpoint (ADR 0029), with
    the baselines' budget. The paper gave its baselines 20 trials of 50 epochs.
  - **The GBDTs,** tuned on the same budget.
  - **The schedule is fixed per arm and logged** (decision 3). Comparisons stay within one
    `lr_scheduler`.
- **The checkpoint rule, pre-registered:**
  - The kept epoch is logged, and collapsed folds are counted.
  - In the old sweep, 12 of 90 folds kept an epoch below 10, all on credit-g and kc2, and one
    collapsed: credit-g_80nan's worst fold, epoch 1, test F1 0.285.
  - On the small tables the class-weighted validation CE never drops below ln 2.
- **The reading:**
  - Paired differences, pooled across tables per missingness level, with the seed-sign rule.
  - Per-table results are descriptive: test macro-F1 SD is 0.045–0.073 on kc2, vehicle, credit-g
    and biodeg at the paper's test sizes, against margins of about one point.
  - H1c's refutation criterion as written above.
- **Budget, to be timed** (decision 6). Time one electricity cell at 10-fold before committing to a
  number.
  - The old sweep took 539 min for 135 folds (45 variants × 3) on the laptop's RTX 3050.
  - 10 folds × 3 seeds × 45 variants is 1,350 folds, ten times as many, before any tuning trials.
  - On the 3090 Ti with four queues, a rough guess is 10–15 h per untuned 300/150 arm, and about a
    fifth of that at 40/40.
  - Tuning multiplies by the trial count.
  - **Timed on 2026-10-25** (decision 6): one fold of electricity_40nan at `--cv_folds 10`, seed 42,
    the sweep's configuration (300/150, `cosine_legacy`), alone on gorgona8's RTX 3090 Ti, took about
    5 min 50 s (09:49:45–09:55:35 GMT-3, data loading included), so about 1 h for a 10-fold cell
    alone. E71's electricity_40nan arm A, 3-fold under four queues, took 890–960 s a fold. The run
    was stopped after fold 1, to free the GPU for E76; MLflow off.
  - **A reduced option:** tune on a subset and evaluate on all 45. The subset must cover every
    level on every base; the 21-variant subset has 80nan for only 3 bases (review #16, trap 4).

### Later, not step zero

- **H1:** a value-space reconstruction head with classification non-inferior. It needs the joint
  or sequential arm (`--task classification --pretrain_objective value` runs today, sequentially).
- **H1b:** TRIDENT's imputations into a fixed classifier.
- **H2:** `[NULL]` against imputation + indicator into the same backbone, under informative
  missingness.
- **The deep baselines** (FT-Transformer, TabNet, SAINT, NAIM): a 6–10 h strict-mypy harness, out of
  step zero unless the user says otherwise (decision 7).
- **Compilation (T06's measurement):** `torch.compile` with `--embedding_lookup batched` can be
  switched on for these new classification arms from the start. They have no old runs to stay
  comparable with, and it saves about 23% a steady epoch. It is a flag per study (decision 8).

## Decisions (what "approve" means)

1. **Splits:** `--cv_folds 10` (80/10/10, no code), or a repeated-80/10/10 mechanism as
   thesis-goals' "5 repeated splits" (code).
2. **Tuning budget:** the trial count and epochs per trial for TRIDENT and the GBDTs (the paper:
   20 × 50 for baselines). Tune on all 45 variants or on a covering subset.
3. **Schedule per arm:** keep `cosine_legacy` (upstream, and today's classification default), or
   run with `cosine` as imputation does. Either way it is fixed per arm.
4. **ADR 0029:** accept, amend or reject. It also settles `--promote_best` for classification.
5. **XGBoost** in the bar: needs a lock pin, `xgboost>=3.2,<3.3`, since 3.3 needs Python 3.12.
6. **Compute cap:** how many GPU nights step zero may take, after the electricity timing.
7. **Deep baselines:** out of step zero (recommended), or in, with their harness.
8. **Speed flags for new classification arms:** compilation and the batched lookup on from the
   start, or today's eager path.

**Settled on 2026-10-29, 21:20 GMT-3** (the user: "concordo com suas recomendações, podemos rodar
os baselines profundos depois como uma emenda, certo?"):

1. **Splits:** `--cv_folds 10` (80/10/10), no new code.
2. **Tuning budget:** 20 trials × 50 epochs for each TRIDENT arm and each GBDT, the paper's
   baseline budget. It is tuned on a covering subset of the 45 variants (every base and every
   level), and evaluated on all 45. The subset is fixed in Stage 4's pre-registration.
3. **Schedule:** `cosine_legacy` for every TRIDENT arm, fixed and logged.
4. **ADR 0029:** accepted as written. `validation` is passed by flag, not made the default.
5. **XGBoost:** out. The bar is LightGBM and HistGradientBoosting.
6. **Compute cap:** set in Stage 4's pre-registration from a timed estimate, and approved by the
   user there, before its first cell.
7. **Deep baselines:** out of step zero. They may join later as an amendment, with their own
   pre-registration.
8. **Speed flags:** today's eager path, with no compilation (T06: break-even a cell).
- **Machines:** step zero may split between gorgona8 and medusa1 by seed (ADR 0035).

**Recommended order:**

- **Now:** Stage 0, then Stage 1 on the CPU. Neither needs a decision beyond this plan.
- **Next:** Stage 2 on a free GPU night.
- **After the user's go on ADR 0029:** Stage 3's code, then Stage 4.
