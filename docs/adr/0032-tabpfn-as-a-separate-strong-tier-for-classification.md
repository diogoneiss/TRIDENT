# ADR 0032: TabPFN v2 as a separate strong tier for classification

- **Status:** **accepted on 2026-10-27, 20:16 GMT-3** by the user ("nivel forte separado, assim como na
  imputação"). It settles T40's open decision.
- **Builds on:**
  - the classification step-zero plan ([2026-10-23](../plans/2026-10-23-classification-step-zero.md));
  - ADR 0029, still proposed, whose decisions the user deferred on 2026-10-27;
  - the imputation precedent of ADR 0021 (two tiers) and ADR 0028 (TabPFN as an offline
    strong-tier member).
  - It amends none of them.

## Context

- **Step zero tests H1c**, the thesis's claim that TRIDENT's lead from 40% missingness survives a
  fair rerun.
  - Its bar is the best tuned GBDT with native NaN: LightGBM and HistGradientBoosting, and
    XGBoost if decision 5 adds it.
  - The comparison is at 40, 60 and 80% MCAR, with paired differences.
- **E83** ([classification-step-zero/03](../tickets/classification-step-zero/03-tabpfn-classifier.md)):
  untuned TabPFN v2 on E71's 3-fold folds (seeds 42/7, 8 tables) leads TRIDENT's classification
  under the fork's defaults at every level.
  - The lead is +0.017 to +0.024 macro-F1.
  - TRIDENT leads only on credit-g and electricity. There, post hoc, TabPFN predicts the majority
    class, and TRIDENT's class-weighted loss buys macro-F1.
- **Folding TabPFN into H1c's bar would change the hypothesis,** from "against the best GBDT" to
  "against a pre-trained foundation model". It would make H1c's refutation close to certain on
  E83's evidence. Imputation faced the same choice and kept two readings (ADR 0021, ADR 0028).

## Decision

1. **TabPFN v2's classifier is a separate strong tier for classification.**
   - H1c's bar stays the best tuned GBDT, as the plan defines it.
   - TabPFN is read beside it, with its own paired verdict, and never enters "the best baseline"
     H1c is judged against.
2. **It is offline,** like ADR 0028.
   - It needs the `tabpfn` extra (9.2.0, v2's checkpoints) and no `src/` change.
   - The settings are E55's (8 estimators, `random_state` 0, a 10,000-row context cap), and it
     runs on step zero's own folds when step zero runs.
3. **It is untuned,** as TabPFN is used.
   - Whether step zero adds a class-balanced TabPFN arm (`balance_probabilities`, or a balanced
     threshold) for a fair macro-F1 reading is decided with step zero's design. E83 shows why it
     matters.
4. **Tables past v2's limits:**
   - letter (26 classes against v2's 10) is reported as not covered, unless the user later adds a
     many-class wrapper, which needs a new dependency.
   - Context beyond 10,000 rows is a stratified fixed-seed subsample.
5. **Every report states both readings:** TRIDENT against the GBDT bar (H1c), and TRIDENT against
   TabPFN (the strong tier).

## Consequences

- H1c stays the test the thesis wrote.
- The strong tier shows, without hiding it, that an untuned foundation model currently leads
  TRIDENT's classification by about 2 macro-F1 points under the fork's defaults (E83).
- Step zero's tuned TRIDENT arms will be read against both tiers when they run.
