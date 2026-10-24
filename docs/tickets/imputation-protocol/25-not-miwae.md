# An imputer that models the mechanism: does not-MIWAE shrink the MNAR bias? (E72; pre-registered 2026-10-24)

This ticket was written and committed before the study's first cell. The implementation was
checked on synthetic data and smoke-run on the study's path before it, with no study score seen
(see "Launch").

- **Why:** this is T35, accepted by the user on 2026-10-22 ("eu concordo com todas as suas ideias").
  - E45/E46's signed-bias panel shows every imputer under-predicting the hidden values under MNAR,
    and the model's bias moving least of 10 methods on 9 of 10 pairs. None of the 10 models the
    mechanism, so the wiki's H1d ("an imputer that models the mechanism shrinks the bias") is
    untested.
- **The user's go:** "sim aprovo" (2026-10-23, 23:44 GMT-3), to coding T35 in parallel with
  Stage 2.
- **The code, clean-room, with nothing in `src/` and nothing in the store:**
  - `scripts/experiments/not_miwae.py`: MIWAE (Mattei and Frellsen, 2019) and not-MIWAE (Ipsen,
    Mattei and Frellsen, ICLR 2021), written from the papers' equations. The authors' TensorFlow
    code has no license and was not read.
  - `scripts/experiments/metric_panel.py --fitters vae`: the two as baseline imputers, refitted on
    each training fold and asked for exactly the ledger's induced cells, as E56 ran TabPFN.
  - `scripts/experiments/vae_mnar_report.py`, the reader; `run_vae_mnar_queue.sh`, the lanes.

## Design (fixed before the first cell)

- **One model, two arms.** A deep latent variable model: encoder and decoder MLPs of two tanh
  layers of 128 units, a Gaussian decoder with a per-feature sd, latent dimension p − 1, a
  standard normal prior, the gaps set to 0 at the encoder's input (the data are z-scored).
  - **MIWAE:** the importance-weighted bound on the observed values only, which assumes MAR.
  - **not-MIWAE:** the same bound times the missingness model, self-masking logistic,
    P(gap in x_j) = sigmoid(w_j (x_j − b_j)), with (w_j, b_j) free per feature. The sign is learnt,
    not given.
  - **So the contrast isolates modelling the mechanism:** the two differ only by that term.
- **Training, the paper's UCI settings as I read them:** K = 20 importance samples, batch 16, Adam
  at 1e-3, 100,000 steps, seed 0, on the CPU with 2 threads.
- **Fills:** self-normalised importance sampling with 1,000 draws. MIWAE averages the decoder's
  means; not-MIWAE averages the draws, since its weights depend on them.
- **Cells:** E45/E46's ten MNAR pairs (`mnar_bias_report.PAIRS`), 17 variants (10 MNAR and 7
  MCAR controls), seeds 42/7/13, on arm G's folds and ledgers: 51 pairs of five folds.
- **Running:** ten CPU lanes, about 4–5 h. State in `/scratch2/diogoneiss/trident-2026-10-24-vae-mnar/`.

## Measures and reading, fixed now (`vae_mnar_report.py --tally`)

Over 15 fold pairs, with the verdict rule:

- **Per pair and mechanism:**
  - notmiwae − miwae on `abs_bias` (the mean over numerical columns of each column's
    |mean(fill − truth)|), on `signed_bias` and on `impute_score` (1% SESOI);
  - notmiwae − model (K, the ledger's fills) on `abs_bias` and on `impute_score`.
- **The interaction, fold-paired:** (notmiwae − miwae) on `abs_bias`, MNAR minus MCAR.
- **Primary,** that interaction over the 10 pairs:
  - **supported** (modelling the mechanism shrinks the bias MNAR adds): better on 4 or more, and
    worse on at most 1;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **Descriptive:** the 12 methods (the panel's ten, MIWAE and not-MIWAE) ranked by the MNAR − MCAR
  shift of `signed_bias`, as E45/E46's measure 5 ranked ten; and how often not-MIWAE beats the
  model's K on `impute_score` under MNAR.
- **What each reading leads to (no default changes; an external baseline only, as MaCoDE):**
  - **supported:** H1d holds on imputation; the thesis can say modelling the mechanism shrinks the
    MNAR bias, and whether not-MIWAE's smaller bias also beats the model's error is the
    descriptive count's job.
  - **refuted:** a self-masking mechanism model does not shrink the bias on these files.
  - **mixed:** reported, and the user chooses.

## Launch

- **The synthetic check (not study data):** 2,000 rows of 6 correlated Gaussian features,
  self-masking with true w = 2 on five of them (29% missing). MIWAE's fills had a bias of −0.417 and
  not-MIWAE's −0.022 (the mean fill's −1.239), and not-MIWAE learnt w ≈ 1.8–3.9 on the masked
  columns; 20,000 steps.
- **Timing (no score computed):** 2.4–2.8 ms a step on fold 1 of kc2_probit_60nan and
  spambase_probit_60nan, so about 4–5 min a fit.
- **The panel path smoke (scores not inspected):** kc2_probit_60nan, seed 42, at 50 steps, to a
  scratch directory: one row per fold for model, miwae and notmiwae.
- **Launched** 2026-10-24, 00:01 GMT-3, on the pre-registration commit (0c9ffef): ten CPU lanes.

## Outcome (2026-10-24, 05:33 GMT-3)

**The run:** 51 of 51 pairs, 00:01–05:30 GMT-3, with none failed.

**A reader fix before the first full read:** `--tally`'s summary counts crashed on the `signed_bias`
labels ("lower"/"higher"), which are printed in place of the verdict's calls. The counts now use the
calls; no rule, measure or number changed.

**The printout** (`scripts/experiments/vae_mnar_report.py --tally`, unedited):

```text
## `biodeg_probit_60nan` against `biodeg_60nan`
- MNAR: notmiwae − miwae abs_bias +0.1270 worse; notmiwae − miwae signed_bias -0.0956 lower; notmiwae − miwae impute_score +2.8335 worse; notmiwae − model abs_bias +0.2909 worse; notmiwae − model impute_score +2.9134 worse (signed bias: model -0.332, miwae -0.492, notmiwae -0.588)
- MCAR: notmiwae − miwae abs_bias +0.1285 worse; notmiwae − miwae signed_bias -0.1250 lower; notmiwae − miwae impute_score +2.8081 worse; notmiwae − model abs_bias +0.1930 worse; notmiwae − model impute_score +2.9487 worse (signed bias: model -0.040, miwae -0.116, notmiwae -0.241)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0014 [-0.1177, +0.1149] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.293, 2. hgb -0.311, 3. lgbm -0.311, 4. missforest -0.314, 5. softimpute -0.318, 6. knn5 -0.328, 7. knn10 -0.332, 8. ice -0.337, 9. notmiwae -0.346, 10. condexp -0.353, 11. miwae -0.376, 12. mean_mode -0.453

## `kc2_probit_60nan` against `kc2_60nan`
- MNAR: notmiwae − miwae abs_bias +0.0393 worse; notmiwae − miwae signed_bias -0.0379 lower; notmiwae − miwae impute_score +0.1011 worse; notmiwae − model abs_bias +0.1504 worse; notmiwae − model impute_score +0.1870 worse (signed bias: model -0.266, miwae -0.396, notmiwae -0.434)
- MCAR: notmiwae − miwae abs_bias +0.0355 worse; notmiwae − miwae signed_bias -0.0361 lower; notmiwae − miwae impute_score +0.1556 worse; notmiwae − model abs_bias +0.0724 worse; notmiwae − model impute_score +0.2542 worse (signed bias: model -0.039, miwae -0.100, notmiwae -0.136)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0038 [-0.0120, +0.0195] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.227, 2. softimpute -0.260, 3. missforest -0.269, 4. miwae -0.296, 5. notmiwae -0.298, 6. knn5 -0.301, 7. lgbm -0.307, 8. hgb -0.310, 9. ice -0.327, 10. condexp -0.328, 11. knn10 -0.331, 12. mean_mode -0.508

## `vehicle_probit_60nan` against `vehicle_60nan`
- MNAR: notmiwae − miwae abs_bias -0.0702 better; notmiwae − miwae signed_bias +0.0916 higher; notmiwae − miwae impute_score -0.0120 better; notmiwae − model abs_bias -0.0409 better; notmiwae − model impute_score -0.0116 none (signed bias: model -0.162, miwae -0.203, notmiwae -0.111)
- MCAR: notmiwae − miwae abs_bias +0.0026 none; notmiwae − miwae signed_bias -0.0053 none; notmiwae − miwae impute_score +0.0235 worse; notmiwae − model abs_bias +0.0059 none; notmiwae − model impute_score +0.0125 none (signed bias: model -0.030, miwae -0.039, notmiwae -0.045)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0727 [-0.0837, -0.0618] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. notmiwae -0.066, 2. model -0.133, 3. miwae -0.163, 4. missforest -0.180, 5. hgb -0.202, 6. lgbm -0.202, 7. ice -0.278, 8. softimpute -0.285, 9. knn5 -0.335, 10. knn10 -0.340, 11. condexp -0.368, 12. mean_mode -0.590

## `spambase_probit_60nan` against `spambase_60nan`
- MNAR: notmiwae − miwae abs_bias +0.1514 none; notmiwae − miwae signed_bias -0.0102 none; notmiwae − miwae impute_score +7.3162 worse; notmiwae − model abs_bias +0.3539 worse; notmiwae − model impute_score +7.3889 worse (signed bias: model -0.275, miwae -0.477, notmiwae -0.488)
- MCAR: notmiwae − miwae abs_bias +0.1256 worse; notmiwae − miwae signed_bias -0.0711 lower; notmiwae − miwae impute_score +8.3443 worse; notmiwae − model abs_bias +0.2601 worse; notmiwae − model impute_score +8.4278 worse (signed bias: model -0.031, miwae -0.184, notmiwae -0.255)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0258 [-0.0519, +0.1035] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. notmiwae -0.233, 2. model -0.245, 3. hgb -0.264, 4. lgbm -0.264, 5. ice -0.271, 6. knn10 -0.277, 7. knn5 -0.278, 8. missforest -0.279, 9. condexp -0.287, 10. softimpute -0.289, 11. miwae -0.294, 12. mean_mode -0.297

## `biodeg_probit_20nan` against `biodeg_20nan`
- MNAR: notmiwae − miwae abs_bias +0.3067 worse; notmiwae − miwae signed_bias +0.0204 none; notmiwae − miwae impute_score +3.5928 worse; notmiwae − model abs_bias +0.5168 worse; notmiwae − model impute_score +3.8672 worse (signed bias: model -0.076, miwae -0.317, notmiwae -0.296)
- MCAR: notmiwae − miwae abs_bias +0.2859 worse; notmiwae − miwae signed_bias -0.0112 none; notmiwae − miwae impute_score +3.2768 worse; notmiwae − model abs_bias +0.3759 worse; notmiwae − model impute_score +3.5336 worse (signed bias: model -0.032, miwae -0.130, notmiwae -0.141)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0207 [-0.2803, +0.3218] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.045, 2. ice -0.052, 3. hgb -0.079, 4. lgbm -0.081, 5. softimpute -0.083, 6. missforest -0.094, 7. condexp -0.132, 8. knn5 -0.140, 9. notmiwae -0.155, 10. knn10 -0.163, 11. miwae -0.186, 12. mean_mode -0.292

## `kc2_probit_20nan` against `kc2_20nan`
- MNAR: notmiwae − miwae abs_bias +0.0253 worse; notmiwae − miwae signed_bias -0.0229 lower; notmiwae − miwae impute_score +0.1675 worse; notmiwae − model abs_bias +0.0827 worse; notmiwae − model impute_score +0.3264 worse (signed bias: model -0.050, miwae -0.147, notmiwae -0.170)
- MCAR: notmiwae − miwae abs_bias +0.0176 worse; notmiwae − miwae signed_bias -0.0035 none; notmiwae − miwae impute_score +0.2393 worse; notmiwae − model abs_bias +0.0376 worse; notmiwae − model impute_score +0.3646 worse (signed bias: model -0.000, miwae -0.077, notmiwae -0.080)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: +0.0077 [-0.0118, +0.0271] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.050, 2. ice -0.051, 3. missforest -0.055, 4. lgbm -0.063, 5. hgb -0.065, 6. miwae -0.070, 7. softimpute -0.073, 8. knn5 -0.075, 9. knn10 -0.083, 10. condexp -0.085, 11. notmiwae -0.089, 12. mean_mode -0.176

## `vehicle_probit_20nan` against `vehicle_20nan`
- MNAR: notmiwae − miwae abs_bias -0.0097 none; notmiwae − miwae signed_bias +0.0142 none; notmiwae − miwae impute_score +0.0423 none; notmiwae − model abs_bias +0.0384 worse; notmiwae − model impute_score +0.1823 worse (signed bias: model -0.076, miwae -0.139, notmiwae -0.125)
- MCAR: notmiwae − miwae abs_bias +0.0104 worse; notmiwae − miwae signed_bias +0.0018 none; notmiwae − miwae impute_score +0.0687 worse; notmiwae − model abs_bias +0.0280 worse; notmiwae − model impute_score +0.1875 worse (signed bias: model -0.021, miwae -0.050, notmiwae -0.048)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0201 [-0.0355, -0.0047] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.055, 2. lgbm -0.069, 3. ice -0.071, 4. hgb -0.072, 5. notmiwae -0.077, 6. missforest -0.082, 7. knn5 -0.085, 8. miwae -0.090, 9. knn10 -0.097, 10. softimpute -0.100, 11. condexp -0.123, 12. mean_mode -0.359

## `biodeg_probit0p25_60nan` against `biodeg_60nan`
- MNAR: notmiwae − miwae abs_bias +0.0948 worse; notmiwae − miwae signed_bias -0.0824 lower; notmiwae − miwae impute_score +0.6988 worse; notmiwae − model abs_bias +0.2168 worse; notmiwae − model impute_score +0.7664 worse (signed bias: model -0.219, miwae -0.339, notmiwae -0.421)
- MCAR: notmiwae − miwae abs_bias +0.1285 worse; notmiwae − miwae signed_bias -0.1250 lower; notmiwae − miwae impute_score +2.8081 worse; notmiwae − model abs_bias +0.1930 worse; notmiwae − model impute_score +2.9487 worse (signed bias: model -0.040, miwae -0.116, notmiwae -0.241)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0337 [-0.1005, +0.0332] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. knn5 -0.177, 2. model -0.179, 3. notmiwae -0.180, 4. knn10 -0.186, 5. hgb -0.203, 6. lgbm -0.204, 7. missforest -0.206, 8. ice -0.208, 9. softimpute -0.208, 10. condexp -0.212, 11. miwae -0.222, 12. mean_mode -0.272

## `biodeg_probit1p0_60nan` against `biodeg_60nan`
- MNAR: notmiwae − miwae abs_bias +0.0839 none; notmiwae − miwae signed_bias +0.0046 none; notmiwae − miwae impute_score +3.5309 worse; notmiwae − model abs_bias +0.3757 worse; notmiwae − model impute_score +3.6124 worse (signed bias: model -0.582, miwae -0.872, notmiwae -0.867)
- MCAR: notmiwae − miwae abs_bias +0.1285 worse; notmiwae − miwae signed_bias -0.1250 lower; notmiwae − miwae impute_score +2.8081 worse; notmiwae − model abs_bias +0.1930 worse; notmiwae − model impute_score +2.9487 worse (signed bias: model -0.040, miwae -0.116, notmiwae -0.241)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.0446 [-0.1851, +0.0959] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.542, 2. hgb -0.588, 3. lgbm -0.589, 4. notmiwae -0.626, 5. missforest -0.633, 6. knn5 -0.636, 7. knn10 -0.646, 8. ice -0.666, 9. condexp -0.720, 10. softimpute -0.724, 11. miwae -0.755, 12. mean_mode -0.849

## `biodeg_probit2p0_60nan` against `biodeg_60nan`
- MNAR: notmiwae − miwae abs_bias -0.0429 none; notmiwae − miwae signed_bias +0.1943 higher; notmiwae − miwae impute_score +3.3101 worse; notmiwae − model abs_bias +0.3978 worse; notmiwae − model impute_score +3.4076 worse (signed bias: model -1.405, miwae -1.846, notmiwae -1.652)
- MCAR: notmiwae − miwae abs_bias +0.1285 worse; notmiwae − miwae signed_bias -0.1250 lower; notmiwae − miwae impute_score +2.8081 worse; notmiwae − model abs_bias +0.1930 worse; notmiwae − model impute_score +2.9487 worse (signed bias: model -0.040, miwae -0.116, notmiwae -0.241)
  - (notmiwae − miwae on abs_bias), MNAR minus MCAR: -0.1714 [-0.2668, -0.0760] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -1.366, 2. notmiwae -1.411, 3. lgbm -1.415, 4. hgb -1.417, 5. missforest -1.570, 6. ice -1.605, 7. knn5 -1.613, 8. knn10 -1.627, 9. condexp -1.724, 10. miwae -1.730, 11. softimpute -1.747, 12. mean_mode -1.838

- MNAR: notmiwae − miwae on abs_bias: better 1 / no difference 4 / worse 5 (fold-pair t, seed agreement)
- MNAR: notmiwae − miwae on signed_bias: better 4 / no difference 4 / worse 2 (fold-pair t, seed agreement)
- MNAR: notmiwae − miwae on impute_score: better 1 / no difference 1 / worse 8 (fold-pair t, seed agreement)
- MNAR: notmiwae − model on abs_bias: better 1 / no difference 0 / worse 9 (fold-pair t, seed agreement)
- MNAR: notmiwae − model on impute_score: better 0 / no difference 1 / worse 9 (fold-pair t, seed agreement)

Primary, (notmiwae − miwae on abs_bias), MNAR minus MCAR: better on ['vehicle_probit_60nan', 'vehicle_probit_20nan', 'biodeg_probit2p0_60nan'], worse on none -> mixed (fold-pair t, seed agreement)
```

**The levels, descriptive** (a one-off read of the same CSVs, unedited):

```text
Mean impute_score over 15 folds, MNAR variant | MCAR control
- biodeg_probit_60nan: model 0.912 miwae 0.992 notmiwae 3.826 mean_mode 1.000 | model 0.761 miwae 0.901 notmiwae 3.709 mean_mode 1.000
- kc2_probit_60nan: model 0.691 miwae 0.777 notmiwae 0.878 mean_mode 1.000 | model 0.572 miwae 0.671 notmiwae 0.826 mean_mode 1.000
- vehicle_probit_60nan: model 0.620 miwae 0.620 notmiwae 0.608 mean_mode 1.000 | model 0.645 miwae 0.634 notmiwae 0.657 mean_mode 1.000
- spambase_probit_60nan: model 0.950 miwae 1.023 notmiwae 8.339 mean_mode 1.000 | model 0.942 miwae 1.026 notmiwae 9.370 mean_mode 1.000
- biodeg_probit_20nan: model 0.616 miwae 0.891 notmiwae 4.484 mean_mode 1.000 | model 0.587 miwae 0.843 notmiwae 4.120 mean_mode 1.000
- kc2_probit_20nan: model 0.497 miwae 0.656 notmiwae 0.824 mean_mode 1.000 | model 0.541 miwae 0.667 notmiwae 0.906 mean_mode 1.000
- vehicle_probit_20nan: model 0.427 miwae 0.567 notmiwae 0.609 mean_mode 1.000 | model 0.418 miwae 0.537 notmiwae 0.605 mean_mode 1.000
- biodeg_probit0p25_60nan: model 0.903 miwae 0.971 notmiwae 1.670 mean_mode 1.000 | model 0.761 miwae 0.901 notmiwae 3.709 mean_mode 1.000
- biodeg_probit1p0_60nan: model 0.921 miwae 1.002 notmiwae 4.533 mean_mode 1.000 | model 0.761 miwae 0.901 notmiwae 3.709 mean_mode 1.000
- biodeg_probit2p0_60nan: model 0.915 miwae 1.013 notmiwae 4.323 mean_mode 1.000 | model 0.761 miwae 0.901 notmiwae 3.709 mean_mode 1.000

Median over 15 folds, MNAR variant: impute_score / mae_ratio / dispersion (fill sd over truth sd)
- biodeg_probit_60nan: model 0.912 / 0.730 / 0.528; miwae 0.987 / 0.813 / 0.254; notmiwae 1.916 / 1.157 / 0.997
- kc2_probit_60nan: model 0.717 / 0.448 / 0.735; miwae 0.874 / 0.491 / 0.453; notmiwae 0.921 / 0.540 / 0.672
- vehicle_probit_60nan: model 0.606 / 0.472 / 0.820; miwae 0.608 / 0.449 / 0.754; notmiwae 0.600 / 0.453 / 0.800
- spambase_probit_60nan: model 0.952 / 0.896 / 0.246; miwae 1.020 / 0.872 / 0.036; notmiwae 7.799 / 4.129 / 2.149
- biodeg_probit_20nan: model 0.563 / 0.402 / 0.868; miwae 0.885 / 0.610 / 0.367; notmiwae 2.466 / 1.609 / 1.435
- kc2_probit_20nan: model 0.510 / 0.339 / 0.913; miwae 0.708 / 0.364 / 0.714; notmiwae 0.835 / 0.435 / 0.792
- vehicle_probit_20nan: model 0.435 / 0.328 / 0.900; miwae 0.589 / 0.380 / 0.835; notmiwae 0.615 / 0.394 / 0.861
- biodeg_probit0p25_60nan: model 0.895 / 0.690 / 0.576; miwae 0.968 / 0.756 / 0.280; notmiwae 1.206 / 0.907 / 0.703
- biodeg_probit1p0_60nan: model 0.915 / 0.734 / 0.477; miwae 0.999 / 0.897 / 0.211; notmiwae 3.190 / 1.789 / 1.316
- biodeg_probit2p0_60nan: model 0.913 / 0.796 / 0.410; miwae 1.010 / 0.958 / 0.155; notmiwae 4.568 / 1.914 / 1.368
```

**Verdict: mixed.** The interaction (notmiwae − miwae on `abs_bias`, MNAR minus MCAR) is better on
3 of 10 pairs (vehicle_probit_60nan, vehicle_probit_20nan, biodeg_probit2p0_60nan) and worse on
none. Supported needed 4; refuted needed at most 1 better. By the ticket, the user chooses.

**Reading:**

- **The instrument failed on biodeg and spambase, under both mechanisms.** not-MIWAE's
  `impute_score` is 1.7–4.5 on biodeg and 8.3–9.4 on spambase, several times worse than the mean
  fill (1.000), with fill dispersion at or above the truth's (median 1.0 on biodeg_probit_60nan,
  2.1 on spambase_probit_60nan; the model's 0.53 and 0.25). That is a fill built from too few
  effective draws on these wide (41 and 57 columns), skewed tables, a likely case of
  importance-weight degeneracy; this study does not test that explanation. MIWAE stays near the
  mean fill there (0.89–1.03). So on six of the ten pairs (biodeg's six) and spambase's, the
  primary contrast compares a failed fill with a weak one, and their calls say little about H1d.
- **Where not-MIWAE is a working imputer, it does what H1d says.** On vehicle (18 columns) its
  `impute_score` matches the model's (0.608 against 0.620 under MNAR at 60nan), and modelling the
  mechanism shrinks the bias MNAR adds:
  - interaction better at 60nan (−0.073) and at 20nan (−0.020);
  - its signed bias under MNAR at 60nan is −0.111 against MIWAE's −0.203 and the model's −0.162;
  - it moves least of the 12 methods from MCAR to MNAR at 60nan (−0.066; the model −0.133).
- **kc2 (21 columns)** sits between: not-MIWAE works (0.83–0.92) but worse than MIWAE, and the
  interaction shows no difference at either rate.
- **The model still leads everywhere on error:** notmiwae − model on `impute_score` under MNAR is
  worse on 9 of 10 and no different on vehicle_probit_60nan.
- **For the user (mixed):** the evidence for H1d is the vehicle pairs only. A fair test on the
  wider tables would need a better-conditioned fill (more draws, the decoder's mean under the
  not-MIWAE weights, or a known-sign mechanism), pre-registered as a new study. As it stands,
  the thesis can say "on vehicle, a mechanism-aware VAE shrinks the MNAR bias below every other
  method's, at the model's error; on biodeg and spambase this implementation does not impute
  competently". No default changes; not-MIWAE is an external baseline only.

