# ADR 0029: Classification search ranked on validation macro-F1

- **Status:** **accepted on 2026-10-29, 21:20 GMT-3** by the user, with decisions 1–5 as written
  ("concordo com suas recomendações"). Proposed 2026-10-23. Part of classification step zero
  ([plan](../plans/2026-10-23-classification-step-zero.md), Stage 3). The plan's eight step-zero
  decisions were settled at the same time; see the plan's "Settled" section.
- **Amends:** ADR 0005, which kept classification's test-ranked search on purpose
  (ADR 0005:260-262).

## Context

- **Classification Optuna ranks trials on test macro-F1.**
  - `opt.py:179` sets `search_objective = 'f1_macro'`, and `opt.py:285` reads it from the trial's
    test metrics.
  - It is the upstream behaviour, and the reason the paper's TRIDENT numbers are optimistic: it was
    tuned on test, while its baselines were tuned on validation (thesis-goals, H1c).
- **The code cannot rank on validation today.**
  - `finetuning.py` logs `finetune/val_f1_macro` per epoch, but neither the kept epoch nor its
    validation F1 reaches `FoldResult.metrics`.
  - The checkpoint is the epoch of the lowest class-weighted validation cross-entropy, faithful to
    upstream.
- **Three more defects meet here:**
  - `--cv_folds` is silently dropped under `--use_optuna`: trials and the retrain run on one split.
  - `--promote_best` for classification writes the shared `<dataset>.json`, which imputation
    falls back to (review finding F-07-4).
  - The store holds no classification study, so no fork number depends on the old ranking.
- **Imputation already ranks on validation** (ADR 0005, ADR 0008), so this makes the two tasks
  agree.

## Decision

1. **Each classification fold result carries its validation reading at the kept checkpoint:**
   `val_f1_macro`, `val_f1_micro`, `val_accuracy`, and `kept_epoch`. These are new keys in
   `FoldResult.metrics`; the test metrics are unchanged.
2. **A new flag picks classification's search objective:** `--classification_search_objective
   test|validation` (`CLASSIFICATION_SEARCH_OBJECTIVE`).
   - The default is `test`, today's behaviour, so every earlier study and the
     `task_defaults('classification') == {}` pin stay as they are.
   - `validation` ranks trials on `val_f1_macro`.
   - The `search_objective` tag records which one ran.
3. **Under `--use_optuna`, `--cv_folds` is refused unless it reaches the trials and the retrain.**
   With `validation`, each trial is scored as the mean validation macro-F1 over its folds. The
   no-silent-drop part is a bug fix, and it applies whatever the objective.
4. **`--promote_best` is refused for classification** until a classification-keyed file exists
   (`<dataset>.classification.json`, as ADR 0005 did for imputation). It never again writes the
   shared `<dataset>.json`.
5. **The checkpoint rule does not change here.** The lowest weighted validation CE stays. A
   checkpoint by validation macro-F1 would be a separate, flag-gated choice, decided after step
   zero logs the kept epochs and counts the collapsed folds.

## Consequences

- **Step zero's TRIDENT arms are tuned on validation only,** as the baselines are. That is the
  condition H1c needs.
- **Old classification runs stay readable.** The new keys are absent on them, and readers must
  treat their absence as "not recorded".
- **The vehicle_00nan fixture checks only `accuracy`, `f1_micro` and `f1_macro`,** so it is
  unchanged.
- **Code:** about 1 day with tests (finetuning, opt.py, config and tags), behind the flag, with a
  unit test per decision.
- **Rejected alternatives:**
  - **Switching the default to `validation`:** it would change the meaning of `opt.py` for anyone
    rerunning upstream's protocol. Step zero can pass the flag.
  - **Ranking on the validation CE:** it misaligns with the reported metric.

## Open for the user

- Whether `validation` should become the default once step zero has run with it.
