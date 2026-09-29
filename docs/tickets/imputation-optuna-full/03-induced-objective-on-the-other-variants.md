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

## Outcome (2026-09-29; stopped before the last variant)

Five of the six variants finished between 2026-09-28 22:47 and 2026-09-29 11:05.
`electricity_20nan` is **unfinished**: its study had started at 11:05 and was stopped at
11:09, in its first trial, at the user's request, to free the machine. The decision was
taken before any electricity result existed. The stop left
`optuna_electricity_20nan_20260929_110526` and its `optuna_trial_0` as `RUNNING` in the
store.

**Primary, induced test cells**: the induced-selected winner against study 01's
masked-selected winner of the same variant, fold-paired over the same five folds
(negative favours the induced winner), with the folds on which the induced winner was
lower; then against the defaults and the run's own best baseline.

| variant | induced winner | masked winner | induced − masked | verdict | defaults | induced − defaults | best baseline | bar |
|---|---|---|---|---|---|---|---|---|
| `biodeg_20nan` | 0.655 [0.483, 0.827] | 0.686 | −0.031 [−0.060, −0.002], 5/5 | **better** | 0.674 | −0.019 [−0.042, +0.003], 4/5 | `hgb` 0.644 | overlaps |
| `kr-vs-kp_40nan` | 0.784 [0.746, 0.821] | 0.776 | +0.008 [−0.008, +0.023], 1/5 | level | 0.778 | +0.006 [−0.008, +0.020], 2/5 | `hgb` 0.772 | overlaps |
| `pendigits_20nan` | 0.354 [0.339, 0.369] | 0.345 | +0.009 [+0.004, +0.015], 0/5 | **worse** | 0.399 | −0.045 [−0.046, −0.044], 5/5 | `knn10` 0.348 | overlaps |
| `spambase_20nan` | 0.900 [0.878, 0.923] | 0.891 | +0.009 [−0.012, +0.031], 1/5 | level | 0.886 | +0.015 [−0.007, +0.036], 1/5 | `hgb` 0.865 | overlaps |
| `letter_20nan` | 0.497 [0.491, 0.503] | 0.469 | +0.028 [+0.023, +0.034], 0/5 | **worse** | 0.515 | −0.018 [−0.022, −0.014], 5/5 | `knn10` 0.473 | **loses** |
| `electricity_20nan` | unfinished | | | | | | | |

**Secondary, masked test cells** (induced − masked winner): `biodeg_20nan` −0.037
[−0.077, +0.004], `kr-vs-kp_40nan` −0.007 [−0.058, +0.044], `pendigits_20nan` +0.007
[−0.001, +0.015], `spambase_20nan` +0.013 [−0.024, +0.050], `letter_20nan` +0.037
[+0.030, +0.044].

**Across variants**, with `credit-g_80nan` from [02](02-credit-g-80nan-induced-objective.md):
the induced objective chose better on 2 of 6 (`credit-g_80nan`, `biodeg_20nan`), level on
2 (`kr-vs-kp_40nan`, `spambase_20nan`) and worse on 2 (`pendigits_20nan`, `letter_20nan`).
It is not a uniform improvement, and on no variant did it produce a model that beats its
best baseline.

**Exploratory.** The rank correlation between the masked and the induced validation scores
across each study's 40 trials orders the outcomes: −0.15 on `credit-g_80nan` (better by
0.143), 0.80 on `biodeg_20nan` (better by 0.031), 0.57 on `kr-vs-kp_40nan` (level), and
0.92 to 0.99 on `spambase_20nan`, `pendigits_20nan` and `letter_20nan` (level or worse).
Where the two objectives disagree, ranking by the population the headline is scored on
recovers a better configuration; where they agree almost perfectly, the switch only
changes which of several near-equal trials wins, and on these variants that went against
it. Six variants and one seed cannot separate that pattern from chance; it is a
hypothesis for a pre-registered test, not a finding.
