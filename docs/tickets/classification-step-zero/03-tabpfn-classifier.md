# TabPFN v2's classifier against TRIDENT's classification, on E71's folds (E83; pre-registered 2026-10-27)

This ticket was written and committed before the study's first fit.

- **Why:** T40. Step zero has no foundation-model classifier, while TabPFN v2 is the strongest
  rival on imputation (E55, E57; ADR 0028).
  - Step zero's own folds (`--cv_folds 10`) wait on ADR 0029's decisions, which the user deferred.
  - So this first reading pairs TabPFN with the classification runs that exist: E71's arm A, which
    is TRIDENT under the fork's defaults, `cosine_legacy`, 3 folds.
- **The user's go:** "vamos para o T40: o TabPFN como baseline de classificação" (2026-10-27,
  before 19:40 GMT-3), after deferring step zero ("podemos deixar isso para depois").
- **Code:** no change to `src/`, no training, nothing in the store.
  - `scripts/experiments/tabpfn_classification.py` has `run` and `--tally`.
  - It reuses E55's settings from `tabpfn_baseline`: v2's checkpoints, 8 estimators,
    `random_state` 0, the GPU, and the 10,000-row context cap.
- **TabPFN v2's limits** (the installed `tabpfn` 9.2.0, `inference_config.py`): 10,000 context rows,
  500 features, 10 classes. The class limit "is not ignorable".
  - **letter (26 classes) is out.** A many-class wrapper would need a new dependency.
  - pendigits has exactly 10 classes.
  - electricity's training split (about 25,700 rows) is subsampled to 10,000, stratified, with a
    fixed seed.
- **Smoke, outside the study (seed 1):** vehicle, credit-g, electricity and pendigits at 40nan,
  1 min in all.
  - Only shapes were read. The context equals TRIDENT's training split (vehicle 479 rows) and is
    capped at 10,000 on electricity; every F1 is finite.

## Design (fixed before the first fit)

- **Cells:** 8 tables × 5 levels (0/20/40/60/80nan, MCAR) × seeds 42 and 7 = 80 pairs, 3 folds
  each.
  - The tables are pendigits, electricity, vehicle, kr-vs-kp, spambase, credit-g, biodeg and kc2.
  - Seeds 42 and 7 are E71's complete seeds; seed 13 stopped at 89 of 135 and is not read.
- **Folds:** `build_folds(frame, label, 3, seed)`, the same split E71's runs used.
  - TabPFN's context is the training split only. TRIDENT trained on the same rows and used the
    validation split for its checkpoint.
  - TabPFN is untuned, as it is used, and the test split is never seen.
- **Inputs:** the prepared frame (scaled numbers), with gaps as NaN, which TabPFN reads natively.
  A category is its index in the training split's vocabulary, declared categorical.

## Measures and reading, fixed now (`tabpfn_classification.py --tally`)

- **Per level:** TabPFN − TRIDENT (E71 arm A) test macro-F1, paired fold by fold, pooled over the
  8 tables and 2 seeds (48 pairs), with the verdict rule and a SESOI of 0.01 macro-F1.
  - The reading is TabPFN ahead, TRIDENT ahead, equivalent, or no verdict.
- **It is descriptive, as E71 was:** folds within a table are not independent, so the interval is
  read as a summary.
  - The per-table means are printed beside it.
- **What it leads to:**
  - It answers T40's open decision with data: does TabPFN join step zero's bar, or stand as a
    separate strong tier (as ADR 0028 did for imputation)? That choice stays the user's.
  - Step zero's own folds and tuning come later, with ADR 0029's decisions.
  - No default changes.

## Limits fixed now

- TRIDENT's side is the fork's untuned defaults (`cosine_legacy`, 300/150), not step zero's tuned
  arms, and 3 folds rather than 80/10/10.
- letter is not covered.
- MCAR only.
