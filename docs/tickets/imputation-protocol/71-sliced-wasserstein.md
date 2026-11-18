# TRIDENT against its baselines on a distributional score that is not marginal: the sliced Wasserstein distance between filled and true rows (E182, T59; pre-registered 2026-11-18 13:00 GMT-3)

This ticket was written and committed before the study's first job. One smoke was seen before it (`kc2_20nan`, seed
101, all five groups): only the gate, the control, the timings and the stability under more directions were printed,
not TRIDENT against any baseline.

- **Why:** `impute_score` reads each fill against its truth, and `dist_score` reads each column's fills against that
  column's truths. Both are blind to the relation between columns: a fill that draws every column from the right
  distribution, independently, scores 0 on `dist_score`. The user asked for a distributional score that is not strictly
  marginal and chose the sliced Wasserstein distance among the options offered.
- **The user's go:** "ok, gosto do sliced wasserstein. proponha um experimento com ele e ponha para rodar! a gorgona8
  voltou, voce pode dividir trabalho com ela para acelerar. primeiro va nas tabelas menores" (2026-11-18, about 12:45
  GMT-3). The design below is my proposal.
- **Code (this commit):** `scripts/experiments/sliced_run.py` (`run`, `--jobs`, `--tally`). Nothing under `src/`
  changes: the score lives in the study's script, so no run, default or baseline-cache key moves.
- **What the score is (`sw_score`):**
  - The rows of a test fold that hold at least one scored cell, written twice: as an imputer fills them and as they
    truly are. The observed cells are the same in both.
  - A row is a vector: numerical columns in the scaled space; each categorical column one-hot over the table's
    categories and divided by √2, so a wrong category is as far as one scaled unit.
  - 2048 unit directions, drawn once per (seed, fold) and shared by every imputer of the fold. Every row is projected
    on each direction; per direction, the 1-D Wasserstein distance between the filled rows' projections and the true
    rows' (the mean gap between the two sorted lists); the distance is the mean over the directions.
  - `sw_score` is that distance over the mean/mode fill's own, as `impute_score` and `dist_score` are built: the truth
    scores 0, the mean/mode fill 1.
  - It is one ratio, not one per kind of column, so measuring it against the class-conditional fill would multiply
    every imputer's score of a fold by the same constant: no class-scale arm. The class-conditional fill is scored as
    an imputer instead.
- **Scenarios:** E181's 26 MCAR variants × seeds 101/202/303 × 5 folds, the induced population. No training, no MLflow,
  no GPU.
  - TRIDENT: the fills in E181's `Kc` fold ledgers (the default run; on the mixed tables it kept the default's epoch).
  - Baselines, refitted on the fold's training rows and asked for exactly the ledger's cells: knn5, knn10, hgb, lgbm
    (the bar), missForest, condexp, softimpute, ice (the strong tier).
  - References: the mean/mode fill, the class-conditional mean/mode fill, the truth.
  - Control: the truths of each column shuffled among that column's scored cells. Every marginal is exact, so its
    `dist_score` is 0, and the rows are wrong.
  - Two machines, whole tables to each: gorgona8 takes `vehicle`, `kr-vs-kp`, `pendigits`; medusa1 the other six. Small
    tables first. 390 jobs (variant, seed, group), 4 threads each.
- **Gates, before any number is read:**
  - A refit's `impute_score` per fold against the one E181's run logged for the same baseline on the same cells. In
    the smoke every gap was at most 1e-16. A baseline whose gap passes 0.005 on a variant is not read on it.
  - The truth scores 0 and the mean/mode fill 1 in every fold.
  - Control: the score is read on a variant only if the shuffled truths' mean `sw_score` is at least 0.10 there. Below
    that the score does not see the rows (the observed cells dilute it), and the variant is listed as blind.
- **Measures and decision rule:** 15 (seed, fold) values a variant, fold-pair rule with seed agreement
  (`report_common.verdict`), margin 0.01 of the opponent's mean, as E181.
  - Hypothesis: on `sw_score`, TRIDENT is not behind the best strong-tier baseline on most variants.
  - Primary, over the variants the control passes: TRIDENT − the strong-tier baseline with the lowest mean `sw_score`.
    Supported if worse on at most a quarter of them; refuted if worse on half or more; mixed otherwise; inconclusive if
    the control passes on fewer than 13 of 26.
  - Secondary, no rule: the same against the bar; the same two comparisons on `dist_score` and `impute_score` from the
    same refits, and the variants whose call on `sw_score` differs from each; the levels of the control and of the
    class-conditional fill.
  - Read once, with the 390 files in.
- **Disclosed:**
  - The score's construction (rows, encoding, 2048 directions), the control's threshold, the gate and the rule's
    thresholds are my choices.
  - Projection noise: on `kc2_20nan` seed 101, 128 against 256 directions moved a fold's `sw_score` by up to 0.030
    (mean 0.005); 2048 against 4096 by up to 0.011 (mean 0.002). The directions are shared within a fold, so the noise
    is common to the two sides of every difference read.
  - In the smoke the control scored 1.21 on `kc2_20nan` seed 101 (above the mean/mode fill) with `dist_score` 0.
  - TRIDENT's checkpoints were chosen by a point-error score. TabPFN is not in the study.
  - A post-hoc look of the same day, outside any registration, put `dist_score` on the class-conditional scale for
    TRIDENT alone (below 1 on 24 of 26 variants); it is not part of this study.
- **Amendment (2026-11-18 13:03 GMT-3):** six jobs of `kr-vs-kp_40nan` on gorgona8 stopped on the encoder's own
  check: a true hidden value was a category the variant never shows among its observed cells. The one-hot now spans the
  categories the variant shows and those its hidden truths hold (all five folds' ledgers), the same for every imputer
  and group; on a variant without such a category the numbers are identical (`kc2_20nan` seed 101: 0.0). The failed
  jobs wrote no file and are rerun. No change to design, seeds or rule; nothing read.

## Outcome (2026-11-18 13:36 GMT-3; 390 of 390 files, 225 on medusa1 and 165 on gorgona8)

Read once with `sliced_run.py --tally`; the unedited tally is `/scratch/diogoneiss/trident-2026-11-18-sliced/tally.txt`
on medusa1. **Mixed.**

- **Gates.** The truth scores 0 and the mean/mode fill 1 in every fold. Every refit reproduces E181's logged
  `impute_score` to 3e-6 or better except three on `kr-vs-kp`, refitted on gorgona8 against scores logged on medusa1:
  knn5 (largest gap 0.039), knn10 (0.034) and ice (0.019). By the rule they are not read there: knn5, knn10 and ice
  on `kr-vs-kp_20nan`, `_40nan` and `_60nan`, ice on `_80nan`.
- **Control.** The shuffled truths score at least 0.10 on all 26 variants (0.16 on `kr-vs-kp_80nan` to 1.25 on
  `kc2_40nan`), with `dist_score` 0.0000 everywhere: the score sees the rows on every variant.
- **Primary, TRIDENT − the best strong-tier baseline on `sw_score`, 26 variants:** better 9, no difference 6, worse 11.
  Worse on a quarter or fewer was needed for supported and on half or more for refuted.
  - Better: `letter_20nan` (−0.157), `pendigits_20nan` (−0.055), `_40nan` (−0.096), `_60nan` (−0.053),
    `biodeg_20nan` (−0.051), `_40nan` (−0.052), `electricity_20nan` (−0.023), `kc2_80nan` (−0.035),
    `vehicle_20nan` (−0.009).
  - Worse: `credit-g_20nan` (+0.070), `_40nan` (+0.220), `_60nan` (+0.347), `_80nan` (+0.350), all to ice;
    `biodeg_80nan` (+0.181, softimpute); `pendigits_80nan` (+0.097), `kr-vs-kp_40nan` (+0.047), `_80nan` (+0.036),
    `vehicle_80nan` (+0.045), `spambase_20nan` (+0.026), `electricity_80nan` (+0.026), all to missForest.
- **Secondary, the same refits on the three scores (better / no difference / worse):**

  | opponent | `sw_score` | `dist_score` | `impute_score` |
  |---|---|---|---|
  | bar (knn5, knn10, hgb, lgbm) | 7 / 7 / 12 | 7 / 4 / 15 | 20 / 5 / 1 |
  | best strong-tier baseline | 9 / 6 / 11 | 7 / 6 / 13 | 18 / 6 / 2 |

  The `impute_score` row repeats E181's calls. Against the strong tier the call on `sw_score` differs from
  `impute_score`'s on 15 variants and from `dist_score`'s on 5.
- **Levels.** TRIDENT's `sw_score` runs from 0.129 (`pendigits_20nan`) to 0.925 (`credit-g_80nan`); the
  class-conditional fill's from 0.41 to 0.96. On 11 variants the control scores lower than TRIDENT (for instance
  `electricity_80nan` 0.246 against 0.635, `biodeg_80nan` 0.418 against 0.747): there, fills with exact marginals and
  wrong rows are closer to the true rows than TRIDENT's.
- **What the fills are.** From 40% missing the ledgers hold the default output, the kept checkpoint's fills after
  ADR 0027's calibration, fitted for point error. TabPFN is not in the study.

**Reading.** On point error TRIDENT leads its baselines; on the two distributional scores it does not: it is behind
the best strong baseline on 11 of 26 variants on `sw_score` and on 13 on `dist_score`, mostly at 80% missing and on
`credit-g` at every level, and ahead at 20 to 60% on `pendigits`, `letter`, `biodeg` and `electricity`. The two
distributional scores mostly agree, so most of what `sw_score` charges here is what `dist_score` already charges:
fills shrunk to the centre. Where the control beats TRIDENT, the marginals weigh more in the score than the rows'
structure. No default changes. E183 (ticket 72) asks whether choosing the checkpoint by `sw_score` recovers it.
