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
