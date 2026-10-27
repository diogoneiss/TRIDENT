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

## Outcome (2026-10-27, 20:05 GMT-3): TabPFN ahead at every level

- **Ran:** 80 of 80 pairs (240 fits), 19:53–20:04 GMT-3; none failed.
- **Per level** (TabPFN − TRIDENT test macro-F1, 48 fold pairs, both seeds agreeing):

  | level | difference | 95% interval |
  |---|---|---|
  | 0nan | +0.021 | [+0.014, +0.029] |
  | 20nan | +0.019 | [+0.011, +0.026] |
  | 40nan | +0.017 | [+0.002, +0.032] |
  | 60nan | +0.024 | [+0.006, +0.043] |
  | 80nan | +0.024 | [+0.006, +0.041] |

  - Every level reads "TabPFN ahead", by about 2 points of macro-F1.
- **By table (descriptive):**
  - TabPFN leads on vehicle (+0.04 to +0.12), spambase, biodeg and kc2 at every level.
  - TRIDENT leads on credit-g from 20nan (+0.02 to +0.10) and on electricity from 60nan (+0.015,
    +0.030).
  - pendigits ties at 80nan.
- **Why TRIDENT leads where it does (post hoc, not in the reading; `e83_posthoc` below):**
  credit-g is 70% "good".
  - At 60–80nan TabPFN's accuracy (0.70–0.71) is the majority class's, so it predicts "good"
    almost always, and its macro-F1 falls to 0.44–0.46.
  - TRIDENT trains on a class-weighted loss. It gives up accuracy (0.50–0.59) for the minority's
    recall, so its macro-F1 is higher.
  - electricity shows the same trade: TabPFN higher on accuracy, TRIDENT higher on macro-F1.
  - So these leads come from the class weighting under a macro metric, not from a better
    classifier.
- **What it leads to:**
  - Untuned, TabPFN v2 is stronger than TRIDENT's classification under the fork's defaults, on 8
    of the 9 tables (letter is not covered).
  - For T40's decision, the user's: it qualifies as a strong baseline. Whether it joins step
    zero's bar or stands as a separate tier (as ADR 0028 did for imputation) is the user's choice.
  - **A fair macro-F1 comparison needs TabPFN class-balanced too.** The package exposes
    `balance_probabilities`, and a balanced decision threshold is another route. That belongs in
    step zero's design, not here.
  - No default changes.

The reader's printout (`tabpfn_classification.py --tally`), unedited:

```text
- 0nan: TabPFN − TRIDENT macro-F1 +0.0212 [+0.0136, +0.0288] seeds +0.0178, +0.0246, 48 fold pairs -> TabPFN ahead
  - pendigits TabPFN 0.996 TRIDENT 0.992; electricity TabPFN 0.871 TRIDENT 0.842; vehicle TabPFN 0.836 TRIDENT 0.762; kr-vs-kp TabPFN 0.993 TRIDENT 0.987; spambase TabPFN 0.952 TRIDENT 0.930; credit-g TabPFN 0.676 TRIDENT 0.673; biodeg TabPFN 0.857 TRIDENT 0.841; kc2 TabPFN 0.717 TRIDENT 0.701
- 20nan: TabPFN − TRIDENT macro-F1 +0.0186 [+0.0108, +0.0264] seeds +0.0160, +0.0212, 48 fold pairs -> TabPFN ahead
  - pendigits TabPFN 0.973 TRIDENT 0.961; electricity TabPFN 0.783 TRIDENT 0.775; vehicle TabPFN 0.706 TRIDENT 0.667; kr-vs-kp TabPFN 0.896 TRIDENT 0.878; spambase TabPFN 0.932 TRIDENT 0.906; credit-g TabPFN 0.620 TRIDENT 0.643; biodeg TabPFN 0.846 TRIDENT 0.809; kc2 TabPFN 0.714 TRIDENT 0.682
- 40nan: TabPFN − TRIDENT macro-F1 +0.0170 [+0.0018, +0.0322] seeds +0.0072, +0.0268, 48 fold pairs -> TabPFN ahead
  - pendigits TabPFN 0.916 TRIDENT 0.908; electricity TabPFN 0.727 TRIDENT 0.728; vehicle TabPFN 0.656 TRIDENT 0.572; kr-vs-kp TabPFN 0.810 TRIDENT 0.798; spambase TabPFN 0.915 TRIDENT 0.878; credit-g TabPFN 0.551 TRIDENT 0.608; biodeg TabPFN 0.800 TRIDENT 0.755; kc2 TabPFN 0.692 TRIDENT 0.685
- 60nan: TabPFN − TRIDENT macro-F1 +0.0240 [+0.0055, +0.0426] seeds +0.0268, +0.0213, 48 fold pairs -> TabPFN ahead
  - pendigits TabPFN 0.797 TRIDENT 0.783; electricity TabPFN 0.662 TRIDENT 0.677; vehicle TabPFN 0.579 TRIDENT 0.464; kr-vs-kp TabPFN 0.724 TRIDENT 0.705; spambase TabPFN 0.886 TRIDENT 0.855; credit-g TabPFN 0.460 TRIDENT 0.558; biodeg TabPFN 0.780 TRIDENT 0.717; kc2 TabPFN 0.730 TRIDENT 0.667
- 80nan: TabPFN − TRIDENT macro-F1 +0.0236 [+0.0064, +0.0408] seeds +0.0260, +0.0212, 48 fold pairs -> TabPFN ahead
  - pendigits TabPFN 0.522 TRIDENT 0.523; electricity TabPFN 0.580 TRIDENT 0.610; vehicle TabPFN 0.459 TRIDENT 0.364; kr-vs-kp TabPFN 0.630 TRIDENT 0.626; spambase TabPFN 0.808 TRIDENT 0.769; credit-g TabPFN 0.440 TRIDENT 0.476; biodeg TabPFN 0.701 TRIDENT 0.646; kc2 TabPFN 0.679 TRIDENT 0.615
```

Post hoc, accuracy beside macro-F1 where TRIDENT leads (`e83_posthoc`, not in the reading), unedited:

```text
credit-g_20nan: accuracy TabPFN 0.742 TRIDENT 0.665; macro-F1 TabPFN 0.620 TRIDENT 0.643
credit-g_40nan: accuracy TabPFN 0.715 TRIDENT 0.626; macro-F1 TabPFN 0.551 TRIDENT 0.608
credit-g_60nan: accuracy TabPFN 0.707 TRIDENT 0.586; macro-F1 TabPFN 0.460 TRIDENT 0.558
credit-g_80nan: accuracy TabPFN 0.699 TRIDENT 0.495; macro-F1 TabPFN 0.440 TRIDENT 0.476
electricity_60nan: accuracy TabPFN 0.694 TRIDENT 0.680; macro-F1 TabPFN 0.662 TRIDENT 0.677
electricity_80nan: accuracy TabPFN 0.643 TRIDENT 0.633; macro-F1 TabPFN 0.580 TRIDENT 0.610
```
