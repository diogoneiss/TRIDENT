# Overnight full-profile studies against the baseline bar (pre-registered 2026-09-28)

Written and committed before the first trial ran, so the analysis cannot be chosen after
the results are seen.

## Why

With every baseline of ADR 0007 in the store (backfill of 2026-09-28), the decoder is
below the **baseline bar** (the lowest `impute_score` of `mean_mode`, `knn5`, `knn10`,
`hgb`) on only 6 of 21 variants on the induced population, never with disjoint 95%
intervals, while a baseline clears it with disjoint intervals on 3 (`pendigits_20nan`
and `letter_20nan` by `knn10`, `credit-g_80nan` by `mean_mode`). `hgb` sets the bar on
13 of 21.

Twelve `reduced` studies (decode-stage knobs only, default architecture and epochs) did
not beat the defaults (review addendum 2026-09-15, §3 and §8). The `full` profile, which
also samples the architecture and the pre-training knobs, has run on `credit-g_20nan`,
`credit-g_40nan` and `kr-vs-kp_20nan` only, where its best validation objectives were in
the range of the `reduced` studies at a fifth of the epochs. It has never been validated
out of sample, and never on a table where a baseline wins clearly.

## Question

Does a `full`-profile search find a configuration whose five-fold induced
`impute_score` is below the baseline bar of the same run, on the tables where the
decoder loses?

## Design (fixed before launch)

- Profile `full` (as implemented: architecture, pre-training and decode knobs, both
  stages at 20 to 60 epochs), 40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`,
  seed 42, the predefined single split, objective `validation/impute/masked/impute_score`
  (ADR 0005). No `--promote_best`: nothing is written under `datasets/hiperparams/`.
- After each study, its best trial's configuration is retrained **five-fold, seed 42**,
  with MLflow on, as an ordinary parent run tagged `optuna_study_run_id` and
  `retrained_from=best_trial`. That run logs every baseline on the same folds and cells.
- Queue, in this order, one process at a time:
  1. `pendigits_20nan`
  2. `letter_20nan`
  3. `electricity_20nan`
  4. `kr-vs-kp_40nan`
  5. `biodeg_20nan`
  6. `spambase_20nan`
  7. `credit-g_80nan`
- Stopping rule: one pass of the queue. Every study whose five-fold retrain finishes is
  reported, in queue order; nothing is rerun, extended or replaced because of its
  result. A study still running when the user resumes the session is reported as
  unfinished.

## Measures

- **Primary:** per retrained run, `cv/test/impute/induced/impute_score/mean` against the
  run's own baseline bar, with both 95% intervals; "beats the bar" only where the
  model's upper bound is below the bar's lower bound.
- **Secondary:** the same against the default-configuration run of 2026-09-24/25 on the
  same variant (same seed, same folds, fold-paired); the masked population the same way.
- Reported alongside: the winner's validation objective, its rank among the study's
  trials once retrained is not known (one retrain per study), and trial time.

## Known limits, stated in advance

- Selection is on one validation split; the addendum measured a median winner rank of
  18/40 on the test side (§8.3), so one retrained winner per study is a noisy draw, and
  a miss does not show the profile cannot win.
- `full` trains 20 to 60 epochs per stage against the defaults' 300 and 150, so a
  difference from the defaults mixes search space and training budget (§8.4).
- One seed. No claim about seed variance.

## Outcome (2026-09-28; the queue completed at 20:10)

Every study and every retrain in the queue finished, one pass, nothing rerun. The
interim outcome written at 07:41 (commit `69c9868`) is superseded by this one; its two
rows are unchanged. A study started at 00:47 and stopped at 00:50, while the launch was
rearranged, left `optuna_pendigits_20nan_20260928_004728` and its `optuna_trial_1` as
`RUNNING` in the store; they are not part of this record.

| variant | study | median trial | best validation objective | 5-fold retrain |
|---|---|---|---|---|
| `pendigits_20nan` | 2.1 h | 198 s | 0.372 | `impute_pendigits_20nan_20260928_025538` |
| `letter_20nan` | 3.7 h | 271 s | 0.499 | `impute_letter_20nan_20260928_071055` |
| `electricity_20nan` | 5.8 h | 507 s | 0.652 | `impute_electricity_20nan_20260928_135656` |
| `kr-vs-kp_40nan` | 1.2 h | 99 s | 0.665 | `impute_kr-vs-kp_40nan_20260928_155051` |
| `biodeg_20nan` | 0.7 h | 58 s | 0.594 | `impute_biodeg_20nan_20260928_164352` |
| `spambase_20nan` | 2.6 h | 230 s | 0.899 | `impute_spambase_20nan_20260928_192703` |
| `credit-g_80nan` | 0.3 h | 28 s | 0.923 | `impute_credit-g_80nan_20260928_200740` |

**Pre-registered measures.** `cv/test/impute/<population>/impute_score/mean` with 95%
intervals; the best baseline is the same run's `baseline/best`, named by its
`best_baseline` tag; the defaults are the 2026-09-24/25 runs of the same variant (the
2026-09-27 rerun for electricity), same seed and folds; the last column is tuned minus
defaults, fold-paired, with the folds on which the tuned model was lower.

| variant | population | tuned model | best baseline | verdict | defaults | tuned − defaults |
|---|---|---|---|---|---|---|
| `pendigits_20nan` | induced | 0.345 [0.335, 0.355] | `knn10` 0.348 [0.339, 0.357] | overlaps | 0.399 [0.384, 0.414] | −0.054 [−0.060, −0.048], 5/5 |
| `pendigits_20nan` | masked | 0.393 [0.385, 0.401] | `knn10` 0.407 [0.398, 0.416] | overlaps | 0.435 [0.417, 0.452] | −0.042 [−0.055, −0.029], 5/5 |
| `letter_20nan` | induced | 0.469 [0.462, 0.475] | `knn10` 0.473 [0.468, 0.478] | overlaps | 0.515 [0.507, 0.522] | −0.046 [−0.053, −0.039], 5/5 |
| `letter_20nan` | masked | 0.505 [0.500, 0.510] | `knn10` 0.544 [0.541, 0.547] | **beats the bar** | 0.551 [0.544, 0.558] | −0.046 [−0.050, −0.043], 5/5 |
| `electricity_20nan` | induced | 0.653 [0.613, 0.693] | `hgb` 0.593 [0.547, 0.640] | overlaps | 0.669 [0.620, 0.718] | −0.016 [−0.047, +0.015], 4/5 |
| `electricity_20nan` | masked | 0.700 [0.609, 0.791] | `hgb` 0.666 [0.570, 0.763] | overlaps | 0.710 [0.616, 0.805] | −0.010 [−0.017, −0.003], 5/5 |
| `kr-vs-kp_40nan` | induced | 0.776 [0.748, 0.805] | `hgb` 0.772 [0.746, 0.797] | overlaps | 0.778 [0.734, 0.822] | −0.002 [−0.026, +0.022], 3/5 |
| `kr-vs-kp_40nan` | masked | 0.800 [0.747, 0.854] | `hgb` 0.787 [0.718, 0.856] | overlaps | 0.802 [0.744, 0.860] | −0.002 [−0.048, +0.044], 3/5 |
| `biodeg_20nan` | induced | 0.686 [0.520, 0.852] | `hgb` 0.644 [0.452, 0.837] | overlaps | 0.674 [0.509, 0.839] | +0.012 [−0.008, +0.032], 2/5 |
| `biodeg_20nan` | masked | 0.614 [0.584, 0.645] | `hgb` 0.607 [0.572, 0.641] | overlaps | 0.596 [0.547, 0.645] | +0.019 [−0.011, +0.049], 1/5 |
| `spambase_20nan` | induced | 0.891 [0.856, 0.926] | `hgb` 0.865 [0.834, 0.897] | overlaps | 0.886 [0.859, 0.913] | +0.005 [−0.022, +0.032], 2/5 |
| `spambase_20nan` | masked | 0.882 [0.828, 0.936] | `hgb` 0.874 [0.829, 0.919] | overlaps | 0.879 [0.827, 0.930] | +0.004 [−0.025, +0.032], 3/5 |
| `credit-g_80nan` | induced | 1.160 [1.083, 1.237] | `mean_mode` 1.000 [1.000, 1.000] | **loses to the bar** | 1.024 [1.008, 1.040] | +0.137 [+0.045, +0.229], 0/5 |
| `credit-g_80nan` | masked | 1.038 [0.941, 1.135] | `mean_mode` 1.000 [1.000, 1.000] | overlaps | 0.995 [0.968, 1.022] | +0.043 [−0.059, +0.146], 2/5 |

**Verdict.** Of 14 population-level comparisons, the tuned model **beats the bar once**
(`letter_20nan`, masked cells, `knn10`), **loses to it once** (`credit-g_80nan`, induced
cells, to the mean/mode fill) and overlaps it on the other 12. On the induced
population, the headline, it beats the bar nowhere.

**Against the defaults** the search helped where KNN is the bar and nowhere else:
`pendigits_20nan` and `letter_20nan` improve on every fold of both populations by 0.04 to
0.05, far beyond the seed noise measured before (0.007 to 0.014, addendum §8.2), and
`electricity_20nan` by 0.010 on the masked cells (5/5 folds). On `kr-vs-kp_40nan`,
`biodeg_20nan` and `spambase_20nan`, where `hgb` is the bar, the paired differences are
within ±0.02 with intervals across zero. On `credit-g_80nan` the selected configuration
is **worse** than the defaults on the induced cells (+0.137, 0/5 folds): the validation
objective scores the masked population, and at 80% missingness the configuration it
preferred does not carry over to the induced one.

**Exploratory, not pre-registered:** fold-paired against the run's own best baseline,
which shares the folds and cells (bold where the interval excludes zero in the
model's favour).

| variant | population | best baseline | model − baseline, folds model lower |
|---|---|---|---|
| `pendigits_20nan` | induced | `knn10` | −0.003 [−0.009, +0.002], 4/5 |
| `pendigits_20nan` | masked | `knn10` | **−0.014 [−0.020, −0.008], 5/5** |
| `letter_20nan` | induced | `knn10` | −0.005 [−0.013, +0.004], 4/5 |
| `letter_20nan` | masked | `knn10` | **−0.039 [−0.043, −0.036], 5/5** |
| `electricity_20nan` | induced | `hgb` | +0.060 [+0.030, +0.089], 0/5 |
| `electricity_20nan` | masked | `hgb` | +0.034 [+0.012, +0.055], 0/5 |
| `kr-vs-kp_40nan` | induced | `hgb` | +0.005 [−0.024, +0.033], 3/5 |
| `kr-vs-kp_40nan` | masked | `hgb` | +0.013 [−0.037, +0.064], 1/5 |
| `biodeg_20nan` | induced | `hgb` | +0.042 [+0.007, +0.076], 1/5 |
| `biodeg_20nan` | masked | `hgb` | +0.008 [−0.043, +0.058], 3/5 |
| `spambase_20nan` | induced | `hgb` | +0.026 [−0.019, +0.070], 0/5 |
| `spambase_20nan` | masked | `hgb` | +0.009 [−0.002, +0.019], 1/5 |
| `credit-g_80nan` | induced | `mean_mode` | +0.160 [+0.083, +0.237], 0/5 |
| `credit-g_80nan` | masked | `mean_mode` | +0.038 [−0.059, +0.135], 2/5 |

Where the baseline is KNN (clean numerical tables), the tuned model is at or below it and
clearly below it on the masked cells; where it is `hgb` or the mean/mode fill, the model
is above it, clearly so on `electricity_20nan`, `biodeg_20nan` (induced) and
`credit-g_80nan` (induced).

**The winning configurations** share two traits on all seven tables: 4 to 6 transformer
layers against the defaults' 2, and a pre-training learning rate 2 to 34 times below the
default 3.4e-4. Their 20 to 60 epochs per stage are the `full` profile's own range, not a
finding; width, batch size and mask rate vary.

| variant | DIM | LAYERS | HEADS | BATCH | PROB_MASCARA | LR_PRE | epochs pre/decode |
|---|---|---|---|---|---|---|---|
| `pendigits_20nan` | 192 | 6 | 8 | 64 | 0.2 | 1.5e-05 | 40/60 |
| `letter_20nan` | 144 | 5 | 16 | 64 | 0.4 | 1.0e-05 | 50/60 |
| `electricity_20nan` | 112 | 5 | 16 | 512 | 0.5 | 2.7e-05 | 50/60 |
| `kr-vs-kp_40nan` | 208 | 5 | 16 | 64 | 0.4 | 9.7e-05 | 40/40 |
| `biodeg_20nan` | 128 | 5 | 16 | 64 | 0.3 | 1.3e-04 | 20/60 |
| `spambase_20nan` | 160 | 4 | 16 | 256 | 0.3 | 3.5e-05 | 50/60 |
| `credit-g_80nan` | 240 | 4 | 16 | 512 | 0.3 | 1.8e-04 | 40/50 |

Limits, as stated in advance: one seed, one selected trial per study, selection on the
masked validation population, and a training budget that differs from the defaults'.
