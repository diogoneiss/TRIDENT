# Multiple imputation from H's fills: do Rubin's intervals cover? (E175, T56; pre-registered 2026-11-12 23:13 GMT-3)

This ticket was written and committed before its reader was run on any ledger.

- **Why:** every study so far scores one fill per gap. An analyst who uses the completed table needs intervals that
  account for the gaps, which is multiple imputation: M completed tables and Rubin's rules. TRIDENT's readout H (the
  decode stage's fill, checkpoint by loss, no calibration) is a point fill: its numerical head is a scalar trained by
  squared error and its categorical head keeps only the top class, so a single completed table understates every
  variance. ADR 0030 (the spread between re-masked fills ranks errors) declined to turn that spread into an interval
  (E77, E151), and decision 8's interval is per cell, not per analysis. This study asks whether draws built around H's
  fill from residuals the deployment already has give valid intervals for whole-table quantities.
- **The user's go:** "gostei do 4 também" for this front and "de acordo com suas recomendações" for its design
  (2026-11-12, GMT-3): the target is the complete table's value, no ensemble across seeds, MICE as the only
  multiple-imputation comparator, MIMIC-IV tables out of this round.
- **Offline, no training of TRIDENT:** it reads E174's cells (ticket 67: H at the thesis seeds, with the hidden
  observed validation cells of each fold) and writes nothing to the store. No `src/` change; the stochastic MICE is a
  script-side subclass of ICE (a member of ADR 0021's strong tier of baselines).
- **Disclosure:** E174's tally (imputation scores of H, TabPFN and the hybrids) was read on 2026-11-12. No coverage,
  width or any quantity of this study was computed on these cells; `--check` runs on synthetic data only. Theory
  predicts that a single fill undercovers, the more the more is missing, so the first primary is expected to hold at
  40 and 80% missing; whether it does at 20%, and whether the draws are proper enough (second primary), is open.

## Design (fixed now)

- **Cells (45):** `biodeg`, `kc2`, `vehicle`, `credit-g`, `pendigits` at 20, 40 and 80% MCAR × seeds 101/202/303, from
  `/scratch/diogoneiss/trident-2026-11-10-hybrid-thesis-seeds/cells` (copied unchanged to the machine that reads).
  Reader `scripts/experiments/multiple_imputation.py`, state `/scratch/diogoneiss/trident-2026-11-12-multiple-imputation`.
- **Unit:** a fold. The sample is the fold's test rows (n); the target Q is the estimand on the whole complete table
  (N rows, the gaps holding the ledgers' truth), so the complete-data variance carries the factor 1 − n/N.
- **Estimands, features only:** every numerical column's mean; every categorical column's share of the complete
  table's commonest category; the least-squares slope of each numerical column on the next numerical column in file
  order.
- **Arms**, each completing the same test rows:
  - `CD`: the truth in every gap. The control: its intervals must cover near 0.95, or the estimands are misread.
  - `SI`: H's fill once, the complete-data variance alone.
  - `MI-D` (deployable): M = 20 draws. A numerical gap takes H's fill plus a residual of its column among the fold's
    hidden observed validation cells; a categorical gap takes the truth of one of those cells where H predicted the
    same class. The pool is resampled with replacement before each draw, so the draws carry the pool's own
    uncertainty. A cell without a pool keeps H's fill (the share is printed).
  - `MI-O` (oracle, descriptive): the same from the other four folds' induced gaps, whose truth no deployment has;
    numerical residuals per tertile of the row's gap count where a bin holds 30 (ADR 0030 decision 8's bins).
  - `MICE`: ICE fitted once on the training split as in ADR 0021, then M = 20 chains whose fills are drawn from each
    conditional's predictive distribution (`BayesianRidge`'s mean and standard deviation, the logistic model's
    probabilities), clipped as ICE clips.
- **Pooling:** Rubin's rules, T = W + (1 + 1/M)B, Barnard and Rubin's degrees of freedom, 95% t intervals.
- **Randomness:** `default_rng([175, crc32(variant), seed, fold, arm])`; a cell reruns to the same file.
- **Machine:** all 45 cells on medusa1's CPUs (gorgona8's 32 cores are held by another project); every arm of a cell
  on one machine (ADR 0035, pairs on one machine). No GPU.

## Measures and rule (fixed now)

Per (variant, seed, fold) and arm: coverage, the share of estimands whose interval holds Q, and the shortfall
u = max(0, 0.95 − coverage). Fold-pair verdict on the 15 (seed, fold) differences of u, SESOI 0.01, lower is better.

- **Primary 1, u(MI-D) − u(SI) on the 15 variants** (E151's thresholds):
  - **supported** if MI-D is better on at least 10 and worse on at most 1;
  - **refuted** if better on 5 or fewer, or worse on 4 or more;
  - **mixed** otherwise.
- **Primary 2, proper enough:** MI-D's mean coverage is at least 0.90 on at least 10 of the 15 variants (yes or no).
- **Secondary:** u(MI-D) − u(MICE), the same verdict, counted; coverage by family of estimand; interval width relative
  to `CD`; the estimate's shift from `CD`'s in `CD` standard errors; λ, the between-imputation share of the variance.
  `MI-O` and `CD` are descriptive.
- Three seeds, 15 variants, M = 20: fixed; no extension.
- **Read** once with all 45 in: `uv run --python 3.11 python -B scripts/experiments/multiple_imputation.py --tally`.
- **What it leads to:**
  - both primaries hold: a proposal to the user for a multiple-imputation output of the decoder (an ADR is theirs);
  - primary 1 holds and 2 fails: the residual draws improve on a single fill but are not proper; the pre-agreed next
    step is the GPU study (about 15 GPU-hours, 45 cells retrained keeping softmax samples and dropout at fill time,
    no `src/` change), pre-registered on its own;
  - primary 1 refuted or mixed: reported as is, with the family that fails.
- **Limits:** MCAR only; estimands of the features, not of a model with the label; the validation cells are observed
  cells hidden by the trainer, so `MI-D` assumes their residuals resemble the gaps' (true under MCAR by construction;
  `MI-O` measures what that assumption costs).
