# Linear baselines and the missing per-fold baselines, on the decoder's cells (E41; pre-registered 2026-10-16)

This ticket was written and committed before any fold of these arms was scored on gorgona8, on
2026-10-16 at 22:47 GMT-3 (`5278337`). It freezes the rules, the cell map and the reading. The pass
itself waits for `src/training/linear_baselines.py`, planned for 2026-10-17. The plan is
[2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md) (§2b, §3),
which the user accepted on 2026-10-16.

## Why

The 2026-10-16 review (wiki `projects/trident/findings-2026-10-16-baselines-methodology`, findings
2–3) ran a probe on the Windows laptop. It was not pre-registered and not stored, so its numbers
are indicative only (`docs/reviews/2026-10-16-quick-wins/digests/merge_digest.txt`, items 2–4):

- **Two linear arms set the bar.** condexp (a Gaussian conditional expectation) and inductive
  SoftImpute fell below the bar of 5 on about 7 induced variants by seed means.
- **One candidate flip.** SoftImpute was below the model's K on kc2_20nan on 3 of 3 seeds, by
  means only. kc2_20nan has the widest intervals of the 21.
- **ICE needs a sequential transform.** The inherited `MissForestImputer.impute` replay scores up
  to about 19× the mean/mode fill for linear learners. With the transform fixed, ICE never beat
  the model.

The user decided on 2026-10-16 that every baseline is logged in **two tiers**:

- **the weak tier** is today's bar of 5, unchanged;
- **the strong tier** is missForest and these new arms, with its own best and gap.

That needs ADR 0021, a live family and a backfill, all after this study (plan §2b). This study
measures the arms first, offline, on the existing runs' folds.

## Design (fixed before the first score)

### The arms, in a new module `src/training/linear_baselines.py`

Each arm follows ADR 0007's `BaselineImputer` protocol.

**The hidden-mask rule.** NaN is written at `hidden | isna` before anything is fitted or predicted,
because the frame passed to `impute` still holds every hidden cell's truth.

**The clip rule.** Every numerical fill is clipped to the column's [min, max] on the training fold.

- **condexp:**
  - **Moments.** Means and pairwise-complete covariances of the numerical columns, from the
    training fold's observed cells in the frame's scaled space.
  - **Correlation.** The covariance is turned into a correlation R and shrunk as
    R ← 0.9R + 0.1I. Its eigenvalues are floored at 1e-3, and the result is rescaled to a
    covariance.
  - **Fill.** Each test row's hidden numerical cells get E[x_m | x_o] from the row's observed
    numerical cells. A row with none gets the training means.
  - **Categorical columns** get the training mode.
  - **Clip** once at the end.
- **SoftImpute (inductive):**
  - **Centring.** Columns are centred on their training means.
  - **Rank.** k is the smallest k whose leading eigenvalues of the pairwise-complete covariance
    reach 90% of the sum of its positive eigenvalues, with negative eigenvalues counted as 0.
    k ≤ p − 1. **No p//2 cap**: that cap broke the probe on spambase.
  - **Fit.** Hard-impute on the training fold: missing cells start at 0, then iterate a rank-k SVD
    reconstruction, at most 100 iterations, stopping when the relative change is below 1e-5.
  - **Loadings.** W = V_k diag(s_k) / sqrt(n); s² is the mean of the discarded eigenvalues of the
    fitted matrix's covariance.
  - **Fill.** A test row's hidden cells get the PPCA posterior: z = (W_oᵀW_o + s² I)⁻¹ W_oᵀ (x_o − μ_o),
    and then x_m = μ_m + W_m z.
  - **Categorical columns** get the training mode.
  - **Clip** once at the end.
- **ICE:**
  - **Features (the categorical-correct design, D6).** Numerical columns as the current fills;
    categorical columns one-hot over the training fold's vocabulary. A categorical target's own
    block is excluded.
  - **Learners.** `BayesianRidge` for numerical targets; `LogisticRegression` (multinomial, lbfgs,
    `max_iter=1000`) for categorical ones. A column with fewer than two observed classes, or one
    observed value, falls back to a constant, as missForest's does.
  - **Fit.** missForest's round loop:
    - start at the training means and modes;
    - visit the columns by increasing missingness, Gauss–Seidel;
    - stop by `still_converging`, at most 10 rounds;
    - keep every round's models.
  - **Transform: sequential, sklearn's.** The test cells start at the training means and modes.
    Rounds 1 to k are applied in order, k being the fit's kept round, each round column by column.
    The clip is applied inside every round.
  - **Name:** `ice`, not MICE. It draws nothing and is deterministic.
- **Optional arm `missforest_seq`:** missForest with the same sequential transform. It checks
  review finding 4 on the small tables (credit-g, vehicle, biodeg, kc2). It is not run on
  spambase, which is memory-heavy.
- **lgbm and missForest per fold,** recomputed through
  `BaselineScorer(families=("lgbm", "missforest"))` wherever a run's fold CSV lacks them. On
  gorgona8 these are cache hits, from ADR 0017's backfill.

### The cell map (which model numbers pair with which folds)

| seed set | model column (K) | variants |
|---|---|---|
| 101/202/303 | E34 (`selection-2026-10-03`), `impute/<pop>/induced_checkpoint/*` | the 16 variants other than the five below |
| 101/202/303 | E38 (`architecture-large-2026-10-06`, arm ALL), `impute/<pop>/*` | pendigits_20nan, letter_20nan |
| 101/202/303 | E39 (`retune-2026-10-15`, arm D), `impute/<pop>/*` | credit-g_40nan, kr-vs-kp_40nan, spambase_40nan |
| 42/7/13 | E35 (`selection-replication-2026-10-04`), `impute/<pop>/induced_checkpoint/*` | all 21; five labelled and never pooled (below) |
| 404/505/606 | E40 (`calibration-2026-10-16`), `impute/<pop>/*` | its 12 variants at ≥40nan |

**The five labelled variants on 42/7/13.** credit-g_40nan, kr-vs-kp_40nan and spambase_40nan read
the files ADR 0019 removed. pendigits_20nan and letter_20nan used the encoder before ADR 0018.

That makes 162 (variant, seed) pairs, 810 folds, each recomputed with `recompute_folds(variant,
seed, 5, 0.2, [], fold_scorer=...)`. That is EVAL_MASK_RATE 0.2 with no extra rates, as all these
runs logged.

### Integrity, before anything is read

- `mean_mode` equals 1 ± 1e-9 on every fold.
- knn5 and hgb reproduce each run's logged fold values within 1e-9. It is the same machine.

A cell that fails is reported and left out, never patched.

### Running it

- **Where:** gorgona8's CPU, beside whatever the GPU queue runs, in 4 processes.
  - `OMP_NUM_THREADS=4`.
  - `missforest.THREADS` and `lightgbm_baseline.THREADS` are set to 4.
- **The store:** read-only; nothing is written to it.
- **Output:** one CSV row per (variant, seed, fold, population, arm) under
  `/scratch2/diogoneiss/trident-<date>-linear-baselines/`, with every component metric
  (`rmse_num_z`, `mae_num_z`, `acc_cat`, `macro_f1_cat`, `impute_score`).
- **Estimated cost:** 1–2 CPU-h.

## Measures

**Primary,** per variant and seed set, on the induced `impute_score`, each over 15 fold pairs with
the project's verdict rule. A verdict needs the 95% t-interval to exclude zero and all three seeds'
mean differences to agree in sign.

1. **Each arm against the bar of 5** (the weak tier).
2. **Each arm against the model's K.**
3. **The model's K against the strong tier's best:** missForest, condexp, SoftImpute and ICE, the
   best chosen on the CV mean per run, as ADR 0007 decision 8 does.

**Equivalence.** A variant with no verdict is also called "equivalent" when the interval lies inside
±1% of the opponent's mean (the SESOI, D3).

**Reading, fixed now.** It is descriptive: no default changes, and the tiers are already decided
(plan §2b). This study reports:

- on how many variants each arm is better than the weak bar, in each seed set;
- every variant where an arm is better than K. The model loses that variant in the strong tier;
  the kc2_20nan flip is the one to watch;
- the model's tally against the strong tier's best: better / no difference / worse, with the test
  named, per seed set.

Seed sets are never pooled where the configuration differs (the five variants above). Elsewhere,
42/7/13 and 101/202/303 may also be read pooled, as 30 pairs, as a secondary reading.

**Secondary:**

- the masked population;
- the ranks SoftImpute chose;
- ICE's rounds;
- `missforest_seq` against missForest on the small tables;
- the unclipped fills, as a diagnostic, never in a count;
- minutes a fold.

**Analysis:** `scripts/experiments/linear_baselines_report.py`, written through the shared helper
(T21). The bar comes from `summary.OUTSIDE_THE_BAR`, never from a name tuple.

## Known limits, stated in advance

- **These are development cells.** Every default was chosen on them (threat 2).
- **condexp conditions on numerical cells only,** so it understates a mixed-type Gaussian on
  credit-g and electricity. On all-categorical kr-vs-kp, condexp and SoftImpute equal the
  mean/mode fill by construction.
- **SoftImpute's inductive projection departs from the canonical transductive algorithm.** Fitting
  on test rows would break the protocol (ADR 0016, *Bias*).
- **The clip and the rank rule** were shaped by the probe's seed-42 scores, which is why they are
  fixed here before this study scores anything.
- **Multiplicity:** about 3 arms × 21 variants × 3 seed sets of verdicts. The counts are
  descriptive, and each is reported with its test.

## Amendment 2026-10-16, about 23:10 GMT-3 (before any arm's score was read)

- **The module and driver landed.** They are `src/training/linear_baselines.py` and
  `scripts/experiments/linear_baselines_run.py`. Unit tests show:
  - no arm reads a hidden cell's truth;
  - the clip holds;
  - SoftImpute can keep rank 3 of 4 columns;
  - ICE equals scikit-learn's `IterativeImputer(BayesianRidge)` within 1e-6, with the same rounds
    and the training range as bounds;
  - the global streams are untouched.
- **One numerical guard not named above:** the PPCA noise variance is floored at 1e-6, the probe's
  floor. It binds only when the discarded spectrum is empty (k = p).
- **The integrity check on the first cell** (vehicle_20nan, seed 42) against E35's fold CSV: all 24
  shared baseline columns are equal (difference 0.0), and mean/mode is exactly 1. Only these columns
  were looked at; no arm's score was read.

## Outcome (2026-10-17, 01:40 GMT-3)

**Run.** All 162 (variant, seed) pairs ran from 23:06 to 01:26 GMT-3 on the CPU, with no failure
and no store write.

- **Integrity, all passed.** Every merged fold's recomputed baselines equal the run's own logged
  ones within 1e-9, and mean/mode is 1 on every fold. The report refuses a cell otherwise.
- **An error in the shared helper was caught and fixed before this reading.** Its first pass let
  `missforest_seq`, the diagnostic arm, into the weak bar. `report_common` now names every
  baseline's class and refuses unknown names; a test pins the case.

**Numbers.** From `scripts/experiments/linear_baselines_report.py` and
`scripts/experiments/thesis_table.py`; the full tables are in
`docs/reports/2026-10-17-thesis-table-development.md`. Induced `impute_score`, 15 fold pairs, the
project's rule (95% t-interval and seed agreement), equivalence within ±1%. 42/7/13 leaves out its
five labelled variants.

| | 101/202/303 (21 variants) | 42/7/13 (16) | 404/505/606 (12) |
|---|---|---|---|
| condexp better than the weak bar | 4: spambase_60nan, credit-g_60nan, credit-g_80nan, vehicle_60nan | 4: spambase_60nan, credit-g_60nan, vehicle_20nan, vehicle_60nan | 2: spambase_60nan, credit-g_60nan |
| SoftImpute better than the weak bar | 1: kc2_60nan | 0 | 2: credit-g_60nan, kc2_60nan |
| ICE better than the weak bar | 1: kc2_60nan | 0 | 1: kc2_60nan |
| an arm better than K | credit-g_80nan (condexp) | credit-g_80nan (condexp) | credit-g_80nan (condexp) |
| K against the weak bar | better 16 / none 3 / worse 2 | better 12 / none 3 / worse 1 | better 10 / none 1 / worse 1 |
| K against the strong tier's best | better 13 / none 7 / worse 1 | better 8 / none 7 / worse 1 | better 6 / none 4 / worse 2 |

**Reading (descriptive, as fixed).**

- **The arms set the bar on a few small numerical or mixed tables at 60nan and above,** as the
  probe found. condexp replicates on spambase_60nan and credit-g_60nan in every seed set.
- **They beat the model on one variant only, credit-g_80nan,** in all three seed sets. There K is
  above 1.0 (the model already loses to the mean/mode fill), and condexp sits at that fill plus a
  small numerical gain.
- **The probe's candidate flip on kc2_20nan does not hold paired.** SoftImpute − K is +0.0013 (no
  difference) on 101/202/303 and +0.0357 (no difference) on 42/7/13.
- **SoftImpute without the p//2 cap fails worse on spambase** (ranks 33 to 42, +0.54 at 20nan
  against the bar), so dropping the cap did not fix it, as the review had warned it might not.
  SoftImpute never beats K.
- **ICE is a completeness row:** never better than K, and better than the bar only on kc2_60nan.
- **The strong tier costs the model 3 wins of 16 on the primary set** (16 → 13). The new
  "worse" verdicts are credit-g_80nan on every set, plus biodeg_60nan (missForest) on 404/505/606.
  Across tables the model stays below the strong tier on 9 of 9 tables on 101/202/303 (two-sided
  Wilcoxon on per-table mean gaps, p = 0.0039).

**Secondary: missForest with the sequential fill** (`missforest_seq`, small tables).

- It beats the stock fill by at most 0.009 at 20nan, and loses to it at 60nan and 80nan by up to
  0.049 (credit-g_80nan).
- It turns no model win into a loss. The only change is credit-g_40nan on 404/505/606, where the
  sequential fill is weaker and K's "no difference" against missForest becomes "better".
- Review finding 4's worry (near-ties flipping under per-round application) is not borne out
  on these tables.

**Next.** As planned (§2b): ADR 0021 (the two tiers), then the live `linear` family and its
backfill, in a morning gap between studies. This study's CSVs are the check the backfill must
reproduce.
