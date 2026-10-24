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
