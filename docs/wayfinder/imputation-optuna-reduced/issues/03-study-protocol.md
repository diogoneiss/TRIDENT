# 03. Study protocol: datasets, variants, budget, schedule, seed

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, rounds 1 and 2)
Blocked by: none

## Question

Which dataset variants does a study tune on, and how far does a promoted configuration
travel? Which datasets, at what trial budget, under which fixed schedule, and with what
seeding and splitting per trial?

## Answer

**Variants (user's decision, round 1)**: only `_20nan` and `_40nan`. One study per
(dataset, variant) pair, each writing its own per-variant configuration, so a promoted
configuration travels nowhere: `_20nan` gets its own, `_40nan` its own. That also yields a
within-dataset comparison of whether the optimum shifts with missingness, which a shared
configuration would have hidden. `_00nan`, `_60nan` and `_80nan` are out of scope.

**Datasets (delegated to the charting session, round 2)**: `credit-g`, `kr-vs-kp`,
`spambase`. The axis the imputation metric actually varies on is column-type composition
(`impute_score` degrades to one family's ratio on a single-type table, and `LAMBDA_NUM`
acts only on a mixed one), so the subset covers all three compositions once: credit-g is
the only affordable mixed table (electricity's `day` is declared categorical too, but its
45k rows put it out of scope), kr-vs-kp the only all-categorical one, and spambase the
widest numerical one, with a default-configuration imputation run already in the store. The user
was offered `vehicle` as a cheaper numerical stand-in (40 s a trial against spambase's
290 s, halving the total) and kept spambase.

**Budget (round 2)**: 40 trials per study; TPE's ten random start-up trials leave thirty
guided ones over four or five dimensions.

| Study | Trial cost (single split, defaults, RTX 3050 Laptop GPU) | 40 trials, both variants |
|---|---|---|
| credit-g | 66 s, measured (34 s pre-training, 32 s decode) | about 1.5 h |
| kr-vs-kp | 293 s, measured (141 s pre-training, 152 s decode) | about 6.5 h |
| spambase | about 500 s, projected: 290 s per 2-fold fold times the 1.7 ratio measured on credit-g | about 11 h |

**About nineteen hours for all six at 40 trials** (ticket 07's measurements, 2026-09-10).
The charting estimate of eleven hours assumed a trial costs what a 2-fold fold costs; it
does not, because a fold trains on about 40% of the rows and the single split on 80%,
and the measured ratio is 1.7 on credit-g. The levers if nineteen hours is too much: 30
trials brings it to about fourteen, and `vehicle` in place of spambase to about nine and
a half. The 40-trial decision stands until the user pulls one.

**Schedule (round 1)**: `cosine`, as ADR 0004 prescribes for imputation experiments,
tagged on the study parent and trials through the existing `lr_scheduler` tag so results
compare within one schedule per ADR 0003. The two real imputation runs in the store used
`plateau`; they do not serve as baselines (ticket 06 makes fresh ones).

**Decided by the charting session without a question, stated in round 1 and not
contested**:

- **Seeded sampler**: `TPESampler(seed=<run seed>)`. Today `optuna.create_study` takes no
  sampler, so a study is not reproducible; the run seed already fixes every trial's
  training, so it fixes the sampler too.
- **Predefined single split per trial**, as classification's study does: one fold's cost
  per trial, and the same split for every trial so trials differ only in hyperparameters.
- **No pruner** in the first version; pruning on the validation score is fog on the map.

## Comments
