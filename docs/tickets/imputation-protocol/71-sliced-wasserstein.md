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
- **Amendment (2026-11-18 13:10 GMT-3):** six jobs of `kr-vs-kp_40nan` on gorgona8 stopped on the encoder's own
  check: a true hidden value was a category the variant never shows among its observed cells. The one-hot now spans the
  categories the variant shows and those its hidden truths hold (all five folds' ledgers), the same for every imputer
  and group; on a variant without such a category the numbers are identical (`kc2_20nan` seed 101: 0.0). The failed
  jobs wrote no file and are rerun. No change to design, seeds or rule; nothing read.

## Outcome

Not read yet.
