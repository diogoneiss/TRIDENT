# The induced validation objective on the other six variants (pre-registered 2026-09-28)

Written and committed before the first trial ran.

## Why

On `credit-g_80nan` ([02](02-credit-g-80nan-induced-objective.md)), ranking by the
validation rows' own gaps chose a configuration better than the masked objective's on the
induced test cells on every fold, and the two validation objectives turned out unrelated
across trials (Spearman −0.145); the masked winner ranked 40th of 40 by the induced one.
Study [01](01-overnight-full-profile-studies.md) ran six more variants under the masked
objective. This repeats each of them under the induced objective.

## Question

On each of the six variants, does ranking by `validation/impute/induced/impute_score`
choose a configuration that does better on the induced test cells than the one the
masked objective chose in study 01?

## Design (fixed before launch)

- Per variant, study 01's design with `--search_objective induced`: `--search_space
  full`, 40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`, seed 42, the
  predefined single split, no `--promote_best`. Each winner retrained five-fold, seed 42,
  MLflow on, tagged `retrained_from=best_trial` and
  `selected_by=validation/impute/induced/impute_score`.
- Queue, cheapest first so results arrive early, one process at a time:
  1. `biodeg_20nan`
  2. `kr-vs-kp_40nan`
  3. `pendigits_20nan`
  4. `spambase_20nan`
  5. `letter_20nan`
  6. `electricity_20nan`
- One pass; every finished pair reported in queue order; nothing rerun, extended or
  replaced on its result. A variant not finished when the session is resumed is
  reported as unfinished.

## Measures (per variant, as in 02)

- **Primary:** the retrained winner's `cv/test/impute/induced/impute_score/mean` against
  study 01's masked-selected winner of the same variant and against the defaults run,
  fold-paired over the same five folds ("better" only where the paired interval excludes
  zero), and against the run's own best baseline ("beats the bar" only with disjoint
  intervals).
- **Secondary:** the same on the masked test cells.
- **Exploratory:** Spearman between the masked and induced validation scores across the
  40 trials, and, where study 01's winner was one of the first ten (repeated) draws, its
  rank under the induced objective.
- **Across variants:** how many of the six are "better" on the primary measure. No claim
  beyond these seven variants.

## Known limits, stated in advance

One seed, one selected trial per study, a single validation split. The induced
validation population is smaller than the masked one on low-missingness variants
(`_20nan` gaps against a 20% mask of observed cells), so its selection may be noisier there.
