# A three-way downstream classification: gaps left missing, the default fills, the fills chosen by `sw_score` (E186, T63; pre-registered 2026-11-22 14:38 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E183 to E185 optimised the fills for the sliced Wasserstein score (`sw_score`) and read that score only.
  This study asks whether that optimisation is useful: does a classifier do better on the table filled that way?
- **The user's go** (2026-11-22): "Quero testar o quão 'útil' é a otimização da imputação por sliced w. Para isso rode
  uma tarefa de classificação 3 vias: dado simplesmente faltante, imputado pelo H otimizado por impute score e H
  optimizado por sliced w. [...] Eu quero um modelo de classificação não tão sofisticado". The design below is mine.
- **Disclosed:**
  - **"H" is read as the decoder's output, not ADR 0024's H.** ADR 0024's H is the checkpoint of the lowest
    validation loss, chosen by neither score. The two arms that fit the request are on disk: **K**, the default run
    (checkpoint by the validation gaps' `impute_score`, ADR 0015), and **S**, the same run with
    `--decode_metric sw_score` (E183). Both carry the default output: calibrated from 40% missing (ADR 0027), raw at
    20nan, the same readout in both arms.
  - **Nothing is trained and no GPU is used.** The fills are the fold ledgers of E184's and E185's runs (tickets 73
    and 74): 26 MCAR variants × seeds 101/202/303 × five folds, on gorgona8. Those studies' `sw_score` and
    `impute_score` were read before this design; no classifier was ever run on these fills.
  - E146 (ticket 52, a pilot on `dist_score` against `impute_score`) was read before this design: +0.014 for the
    distributional checkpoint when the classifier is fitted on complete rows, +0.003 (no difference) when it is
    fitted on the filled rows. The second is this study's primary protocol, so a null is the expected shape.
  - Smoke before this commit: `credit-g_40nan` seed 101, a study cell, run once to check the mechanics. It printed
    `1000 rows, 8000 gaps, 32000 gaps reached the forest's training rows`; its CSV was checked for shape (140 rows,
    no empty score) and deleted without reading a score. The tally was exercised on random numbers.
- **Script (this commit):** [scripts/experiments/sliced_utility.py](../../../scripts/experiments/sliced_utility.py);
  CSVs under `/scratch/diogoneiss/trident-2026-11-22-sliced-utility/csv`.

## Design and rule (fixed before the first cell)

- **Cells:** the 18 variants of E184 and the 8 of E185 × seeds 101/202/303 × five folds: 78 (variant, seed) files.
- **Tables,** per (variant, seed), in the ledgers' scaled space, the variant's features with its gaps:
  - `missing`: left as they are;
  - `K`, `S`: filled by that arm's ledger;
  - beside: `L` and `LS` (K and S with E184's loss term at w = 0.8), `naive` (the fold's mean/mode fill), `truth`.
  - Each row's gaps are filled by the fold in which the row is a test row, so no decoder trained on the row it
    fills, and none saw a label. The arms must hold the same cells and truths, every gap must have a ledger cell,
    and a filled table with a gap left is an error.
- **Classifier, fixed and untuned:**
  - **Primary: a random forest** (500 trees, scikit-learn's defaults otherwise; categories as ordinal codes). It
    takes a gap as it is, so `missing` is truly missing.
  - **Beside: logistic regression** (categories one-hot). It cannot take a gap: in `missing` a number's gap is the
    training mean with an indicator column, a category's gap a level of its own. That arm is a naive fill with
    indicators, stated now; the learners are never averaged.
- **Protocol:** macro-F1 on the imputation's own test folds.
  - **Primary, `own`:** the classifier is fitted on a table's rows outside the test fold and scored on that table's
    test fold (impute, then learn; for `missing`, learn with the gaps). The only protocol in which the missing arm
    is the user's.
  - **Beside, `complete`:** fitted on the truth-filled rows, scored on each table's test fold (E146's primary).
- **Readings,** each a difference pooled over the 26 variants per (seed, fold) replicate, 15 replicates, plain 95% t
  interval (E146's): above 0 the first table is the more useful, below 0 the second, otherwise no difference. The
  Nadeau–Bengio interval is printed beside; a verdict it does not share is reported as fragile.
  - **Primary: F1(S) − F1(K)**, forest, `own`.
  - **Three-way: F1(K) − F1(missing) and F1(S) − F1(missing)**, forest, `own`.
  - Beside, no rule: LS − S and L − K; naive and truth against missing; the levels of all seven tables; the three
    readings by missingness level (20/40/60/80nan), per variant (15 fold values, with counts), for logistic
    regression and under `complete`.
- **Read once:** `--tally`, when all 78 files are in; it refuses a partial read. No default changes whatever the
  result.
- **Limits stated now:** MCAR only; one seed set, already used to read `sw_score`; two untuned learners; a forest
  splits on gaps well, so `missing` is a strong arm under MCAR; macro-F1 only. A claim beyond this needs fresh seeds.

## Outcome

Not read yet.
