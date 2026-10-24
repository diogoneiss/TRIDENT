# not-MIWAE with a mean-based fill: does modelling the mechanism shrink the MNAR bias on biodeg and spambase? (E73; pre-registered 2026-10-24)

This ticket was written and committed before the study's first cell. A synthetic check and a pilot
on a variant outside the study ran before it (see "Launch").

- **Why:** E72 ([25](25-not-miwae.md)) was mixed. Its not-MIWAE fill failed on biodeg and spambase
  (`impute_score` 1.7–9.4, worse than the mean fill), so those six pairs did not test H1d. On
  vehicle, where the fill worked, modelling the mechanism shrank the MNAR bias.
- **The user's go:** "sim, mas lance o 2 imediatamente em seguida" (2026-10-24, 11:56 GMT-3), to
  "pré-registre e lance o 3 agora" (a better-conditioned not-MIWAE fill), with the blend output (E74)
  right after.
- **The diagnosis, from the pilot:** E72's fill averages decoder draws under not-MIWAE's weights. On
  biodeg it produced fills up to |48–62| (scaled units) for truths within |7|, and more draws made
  it worse (more chances for an extreme draw to dominate). The decoder's means under the same
  weights did not.
- **Code, nothing in `src/` or the store:**
  - `scripts/experiments/not_miwae.py`: `fit_notmiwae_fills` fits not-MIWAE once, with E72's
    settings and seed, and reads it with two fills;
  - `metric_panel.py --fitters notmiwae_fills`;
  - `scripts/experiments/notmiwae_fill_report.py`, the reader; `run_notmiwae_fill_queue.sh`, the lanes.

## Design (fixed before the first cell)

- **One not-MIWAE fit per training fold,** exactly E72's (K 20, batch 16, Adam 1e-3, 100,000 steps,
  seed 0, self-masking logistic with the sign learnt), read two ways with 10,000 draws:
  - **`notmiwae_mean`, the primary fill:** Σ_l w̃_l μ(z_l), the decoder's means under not-MIWAE's
    normalised weights (which include the mechanism term evaluated on each draw);
  - **`notmiwae_draws`, descriptive:** E72's fill with ten times the draws.
- **MIWAE is E72's:** same folds, settings and seed, so it is not refitted.
- **Cells:** the six pairs where E72's fill failed: biodeg's tilt dial at 60nan (slopes 0.25, 0.5,
  1.0, 2.0) against biodeg_60nan, biodeg_probit_20nan against biodeg_20nan, and
  spambase_probit_60nan against spambase_60nan. Nine variants at 42/7/13, 27 pairs, arm G's folds
  and ledgers.
- **Running:** nine CPU lanes, about 1.5–2 h; state in
  `/scratch2/diogoneiss/trident-2026-10-24-notmiwae-fill/`.

## Measures and reading, fixed now (`notmiwae_fill_report.py --tally`)

Over 15 fold pairs, with the verdict rule:

- **Per pair and mechanism:** notmiwae_mean − miwae on `abs_bias`, `signed_bias` and `impute_score`
  (1% SESOI); notmiwae_mean − model (K) on `impute_score`; each method's mean `impute_score`.
- **The instrument check, reported:** a pair is flagged where notmiwae_mean's mean `impute_score`
  is not below the mean fill (1.0) under either mechanism.
- **The interaction, fold-paired:** (notmiwae_mean − miwae on `abs_bias`), MNAR minus MCAR.
- **Primary,** that interaction over the 6 pairs:
  - **supported** (modelling the mechanism shrinks the bias MNAR adds): better on 3 or more, and
    worse on at most 1;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **Descriptive:** the 13 methods ranked by the MNAR − MCAR shift of `signed_bias`.
- **What each reading leads to (no default changes; an external baseline only):**
  - **supported:** with E72's vehicle pairs, H1d holds on biodeg and spambase too.
  - **refuted:** modelling a self-masking mechanism does not shrink the bias on biodeg and spambase
    once the fill works; H1d's support stays the vehicle pairs.
  - **mixed:** reported, and the user chooses.
  - **Flagged pairs** are read as "the instrument still fails" whatever their call.

## Launch

- **A synthetic check (not study data):** 1,000 rows of 40 skewed, correlated features, self-masking
  (38% missing), 20,000 steps. All three fills worked there (RMSE 0.69–0.71 of the mean fill's), so
  the synthetic data did not reproduce E72's failure.
- **The pilot, on biodeg_40nan (MCAR, outside this study and E72), fold 1 of seed 42:** 20% of the
  test fold's observed cells hidden, E72's settings at 100,000 steps. RMSE as a share of the mean
  fill's: E72's fill (1,000 draws) 2.29, max |fill| 48.2; `notmiwae_draws` (10,000) 2.84, max 62.0;
  `notmiwae_mean` 0.865, max 1.7; the truths' max |value| 7.0. This chose the primary fill.
- **Launched** 2026-10-24, 12:04 GMT-3, on the pre-registration commit (ac02a04): nine CPU lanes.

## Outcome (2026-10-24, 13:53 GMT-3)

**The run:** 27 of 27 pairs, 12:04–13:52 GMT-3, with none failed.

**The printout** (`scripts/experiments/notmiwae_fill_report.py --tally`, unedited):

```text
## `biodeg_probit_60nan` against `biodeg_60nan`
- MNAR: notmiwae_mean − miwae abs_bias +0.0068 none; notmiwae_mean − miwae signed_bias -0.0066 none; notmiwae_mean − miwae impute_score +0.0002 none; notmiwae_mean − model impute_score +0.0802 worse
  - impute_score: model 0.912, miwae 0.992, notmiwae 3.826, notmiwae_mean 0.992, notmiwae_draws 3.548; signed bias: model -0.332, miwae -0.492, notmiwae_mean -0.499
- MCAR: notmiwae_mean − miwae abs_bias +0.0426 worse; notmiwae_mean − miwae signed_bias -0.0475 lower; notmiwae_mean − miwae impute_score +0.0085 none; notmiwae_mean − model impute_score +0.1491 worse
  - impute_score: model 0.761, miwae 0.901, notmiwae 3.709, notmiwae_mean 0.910, notmiwae_draws 2.796; signed bias: model -0.040, miwae -0.116, notmiwae_mean -0.164
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.0358 [-0.0548, -0.0168] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.293, 2. notmiwae_draws -0.299, 3. hgb -0.311, 4. lgbm -0.311, 5. missforest -0.314, 6. softimpute -0.318, 7. knn5 -0.328, 8. knn10 -0.332, 9. notmiwae_mean -0.335, 10. ice -0.337, 11. condexp -0.353, 12. miwae -0.376, 13. mean_mode -0.453

## `spambase_probit_60nan` against `spambase_60nan`
- MNAR: notmiwae_mean − miwae abs_bias -0.0024 none; notmiwae_mean − miwae signed_bias +0.0124 none; notmiwae_mean − miwae impute_score +0.0027 none; notmiwae_mean − model impute_score +0.0754 worse
  - impute_score: model 0.950, miwae 1.023, notmiwae 8.339, notmiwae_mean 1.026, notmiwae_draws 7.599; signed bias: model -0.275, miwae -0.477, notmiwae_mean -0.465
- MCAR: notmiwae_mean − miwae abs_bias +0.0187 none; notmiwae_mean − miwae signed_bias -0.0171 none; notmiwae_mean − miwae impute_score +0.0018 none; notmiwae_mean − model impute_score +0.0853 worse
  - impute_score: model 0.942, miwae 1.026, notmiwae 9.370, notmiwae_mean 1.027, notmiwae_draws 8.345; signed bias: model -0.031, miwae -0.184, notmiwae_mean -0.201
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.0211 [-0.0446, +0.0025] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. notmiwae_draws -0.234, 2. model -0.245, 3. notmiwae_mean -0.264, 4. hgb -0.264, 5. lgbm -0.264, 6. ice -0.271, 7. knn10 -0.277, 8. knn5 -0.278, 9. missforest -0.279, 10. condexp -0.287, 11. softimpute -0.289, 12. miwae -0.294, 13. mean_mode -0.297

## `biodeg_probit_20nan` against `biodeg_20nan`
- MNAR: notmiwae_mean − miwae abs_bias +0.0051 none; notmiwae_mean − miwae signed_bias -0.0024 none; notmiwae_mean − miwae impute_score +0.0005 none; notmiwae_mean − model impute_score +0.2750 worse
  - impute_score: model 0.616, miwae 0.891, notmiwae 4.484, notmiwae_mean 0.891, notmiwae_draws 4.703; signed bias: model -0.076, miwae -0.317, notmiwae_mean -0.319
- MCAR: notmiwae_mean − miwae abs_bias +0.0062 none; notmiwae_mean − miwae signed_bias -0.0095 none; notmiwae_mean − miwae impute_score -0.0022 none; notmiwae_mean − model impute_score +0.2545 worse
  - impute_score: model 0.587, miwae 0.843, notmiwae 4.120, notmiwae_mean 0.841, notmiwae_draws 4.029; signed bias: model -0.032, miwae -0.130, notmiwae_mean -0.140
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.0011 [-0.0164, +0.0142] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.045, 2. ice -0.052, 3. hgb -0.079, 4. lgbm -0.081, 5. softimpute -0.083, 6. missforest -0.094, 7. condexp -0.132, 8. knn5 -0.140, 9. knn10 -0.163, 10. notmiwae_mean -0.179, 11. miwae -0.186, 12. notmiwae_draws -0.208, 13. mean_mode -0.292

## `biodeg_probit0p25_60nan` against `biodeg_60nan`
- MNAR: notmiwae_mean − miwae abs_bias +0.0346 worse; notmiwae_mean − miwae signed_bias -0.0351 lower; notmiwae_mean − miwae impute_score +0.0030 none; notmiwae_mean − model impute_score +0.0706 worse
  - impute_score: model 0.903, miwae 0.971, notmiwae 1.670, notmiwae_mean 0.974, notmiwae_draws 1.512; signed bias: model -0.219, miwae -0.339, notmiwae_mean -0.374
- MCAR: notmiwae_mean − miwae abs_bias +0.0426 worse; notmiwae_mean − miwae signed_bias -0.0475 lower; notmiwae_mean − miwae impute_score +0.0085 none; notmiwae_mean − model impute_score +0.1491 worse
  - impute_score: model 0.761, miwae 0.901, notmiwae 3.709, notmiwae_mean 0.910, notmiwae_draws 2.796; signed bias: model -0.040, miwae -0.116, notmiwae_mean -0.164
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.0080 [-0.0227, +0.0068] none
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. knn5 -0.177, 2. model -0.179, 3. knn10 -0.186, 4. hgb -0.203, 5. lgbm -0.204, 6. missforest -0.206, 7. ice -0.208, 8. softimpute -0.208, 9. notmiwae_mean -0.210, 10. condexp -0.212, 11. notmiwae_draws -0.214, 12. miwae -0.222, 13. mean_mode -0.272

## `biodeg_probit1p0_60nan` against `biodeg_60nan`
- MNAR: notmiwae_mean − miwae abs_bias -0.0519 better; notmiwae_mean − miwae signed_bias +0.0514 higher; notmiwae_mean − miwae impute_score -0.0020 none; notmiwae_mean − model impute_score +0.0795 worse
  - impute_score: model 0.921, miwae 1.002, notmiwae 4.533, notmiwae_mean 1.000, notmiwae_draws 4.428; signed bias: model -0.582, miwae -0.872, notmiwae_mean -0.820
- MCAR: notmiwae_mean − miwae abs_bias +0.0426 worse; notmiwae_mean − miwae signed_bias -0.0475 lower; notmiwae_mean − miwae impute_score +0.0085 none; notmiwae_mean − model impute_score +0.1491 worse
  - impute_score: model 0.761, miwae 0.901, notmiwae 3.709, notmiwae_mean 0.910, notmiwae_draws 2.796; signed bias: model -0.040, miwae -0.116, notmiwae_mean -0.164
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.0945 [-0.1284, -0.0605] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -0.542, 2. hgb -0.588, 3. lgbm -0.589, 4. missforest -0.633, 5. knn5 -0.636, 6. knn10 -0.646, 7. notmiwae_mean -0.656, 8. ice -0.666, 9. notmiwae_draws -0.693, 10. condexp -0.720, 11. softimpute -0.724, 12. miwae -0.755, 13. mean_mode -0.849

## `biodeg_probit2p0_60nan` against `biodeg_60nan`
- MNAR: notmiwae_mean − miwae abs_bias -0.1178 better; notmiwae_mean − miwae signed_bias +0.1201 higher; notmiwae_mean − miwae impute_score -0.0084 better; notmiwae_mean − model impute_score +0.0890 worse
  - impute_score: model 0.915, miwae 1.013, notmiwae 4.323, notmiwae_mean 1.005, notmiwae_draws 4.005; signed bias: model -1.405, miwae -1.846, notmiwae_mean -1.726
- MCAR: notmiwae_mean − miwae abs_bias +0.0426 worse; notmiwae_mean − miwae signed_bias -0.0475 lower; notmiwae_mean − miwae impute_score +0.0085 none; notmiwae_mean − model impute_score +0.1491 worse
  - impute_score: model 0.761, miwae 0.901, notmiwae 3.709, notmiwae_mean 0.910, notmiwae_draws 2.796; signed bias: model -0.040, miwae -0.116, notmiwae_mean -0.164
  - (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: -0.1604 [-0.2102, -0.1107] better
  - by |MNAR − MCAR shift of signed bias| (1 moves least): 1. model -1.366, 2. lgbm -1.415, 3. hgb -1.417, 4. notmiwae_draws -1.538, 5. notmiwae_mean -1.562, 6. missforest -1.570, 7. ice -1.605, 8. knn5 -1.613, 9. knn10 -1.627, 10. condexp -1.724, 11. miwae -1.730, 12. softimpute -1.747, 13. mean_mode -1.838

- pairs where notmiwae_mean is not below the mean fill under a mechanism: ['spambase_probit_60nan', 'biodeg_probit1p0_60nan', 'biodeg_probit2p0_60nan']

Primary, (notmiwae_mean − miwae on abs_bias), MNAR minus MCAR: better on ['biodeg_probit_60nan', 'biodeg_probit1p0_60nan', 'biodeg_probit2p0_60nan'], worse on none -> supported (fold-pair t, seed agreement)
```

**Verdict: supported by the count, but by the ticket's flag rule it rests on one informative
pair.** The interaction is better on 3 of 6 (biodeg_probit_60nan, biodeg_probit1p0_60nan,
biodeg_probit2p0_60nan) and worse on none, which the count calls supported. The ticket also fixed
that a flagged pair, where notmiwae_mean is not below the mean fill under a mechanism, reads as
"the instrument still fails" whatever its call. Slopes 1.0 and 2.0 are flagged (notmiwae_mean
1.000 and 1.005 under MNAR), and so is spambase. Of the three unflagged pairs, the interaction is
better on one (biodeg_probit_60nan) and shows no difference on two (biodeg_probit0p25_60nan,
biodeg_probit_20nan).

**Reading:**

- **The mean fill removed E72's failure.** not-MIWAE's `impute_score` went from 1.7–9.4 (E72's
  draw-averaged fill) to 0.84–1.03, level with MIWAE on every variant (notmiwae_mean − miwae on
  `impute_score`: no difference on 11 of 12 mechanism-variants, better on biodeg_probit2p0_60nan).
  The draws read with 10,000 samples stays broken (1.5–8.3), confirming the pilot's diagnosis.
- **But on these tables both VAEs impute at about the mean fill's level.** MIWAE and notmiwae_mean
  sit at 0.84–0.91 on biodeg's MCAR controls and at 0.97–1.03 on biodeg's 60nan dial and spambase,
  against the model's 0.59–0.95. Where a method barely beats the mean fill, a smaller bias says
  little about imputing well; that is why the ticket flagged those pairs.
- **The mechanism term does move the fills the right way as the tilt steepens:** on biodeg's dial
  the interaction goes from no difference at slope 0.25 to −0.036 at 0.5, −0.095 at 1.0 and −0.160
  at 2.0. Under MNAR at slope 2.0 notmiwae_mean's signed bias is −1.73 against MIWAE's −1.85.
  It is a real direction, read on fills that are not competent ones at 1.0 and 2.0.
- **The model's bias still moves least** of the 13 methods on four of the six pairs, and the model
  is the most accurate on all six (notmiwae_mean − model worse everywhere).
- **With E72:** H1d has support where a mechanism-aware VAE imputes competently: vehicle at 60 and
  20nan (E72) and biodeg_probit_60nan (here, notmiwae_mean 0.992 under MNAR, 0.910 under MCAR). On
  the steep biodeg tilts and on spambase the VAEs are not competent imputers, so those pairs cannot
  carry the claim either way.
- **For the user:** the thesis can state H1d with that scope. No default changes; not-MIWAE stays
  an external baseline.

