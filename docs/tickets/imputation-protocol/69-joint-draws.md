# Multiple imputation with draws joint across a row's gaps: do they cover where per-cell draws fail? (E179, T56; pre-registered 2026-11-13 23:23 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E175 ([68](68-multiple-imputation.md)) found that residual draws around H's fill (H: the decode stage's
  fill, checkpoint by loss, no calibration) cover whole-table estimands better than the single fill, but are not
  proper at 80% missing: means and shares stay covered, slopes between columns do not (0.34 to 0.60), because each
  cell is drawn alone. Its ticket named the next step: a GPU study with draws taken from the model at fill time,
  joint across a row's gaps, read with an interval whose control covers. The design is the proposal
  [multiple-imputation-joint-draws](../../proposals/multiple-imputation-joint-draws.md) (ece5004).
- **The user's go:** asked "Posso pré-registrar com as quatro recomendações?", the user answered "ok de acordo"
  (2026-11-13). The four recommendations:
  - the target is the fold's own complete-data estimate and the interval uses the between-draw variance alone, which
    changes the target accepted for E175 (the whole table's value, the total variance);
  - fresh seeds 4545, 5656, 6767;
  - the thresholds below;
  - the draws are read script-side from the trained model, with no `src/` change.
- **Disclosure:**
  - E175's printout was read on these 15 variants at other seeds (101/202/303), and E174's hybrid tally on the same
    cells. No cell of this study has run at its seeds.
  - `--check` runs on synthetic data only: under the new interval proper draws covered the sample's complete-data
    mean 0.9675 of the time and draws with the parameters held fixed 0.8175.
  - Two smoke cells ran with MLflow off at seed 99 (not a study seed), in a scratch state: `kc2_80nan` on the CPU
    and `credit-g_20nan` (categorical columns) on gorgona8's GPU. They checked that the run trains, the draws file
    is written, `plain` equals H's ledger and the reader and tally run. The tallies went to `/dev/null`; no coverage
    was read.
  - `MI-D` and `MI-R` hold the network fixed, so their between-draw variance has no part for the model's own error;
    under the new interval both may sit below E175's levels. That is the baseline the rule compares against, said
    now.
  - **A change from the proposal, made before any cell:** the proposal measured `MI-R`'s correlation on validation
    cells hidden in the same pass. Counted on E174's selection files (seed 101, fold 1; counts of cells, no score),
    a column pair has 30 such rows on `pendigits` at 20 and 40% only, so the arm would be `MI-D` on 13 of 15
    variants. The reader instead pools a row's hidden cells over the five passes (a cell's residual averaged over
    the passes that hid it). Pairs with 30 rows are then: all at 20% but `kc2` (186 of 210); at 40% `biodeg` 802 of
    820, `vehicle` 77 of 153, `credit-g` 15 of 21, `pendigits` all, `kc2` none; at 80% `pendigits` only. When one
    column of a pair was hidden the other may have been visible, which can understate the correlation.
  - **`MI-R` cannot read supported:** at 80% four of the five tables have no pair with a correlation (too few
    validation rows observe both columns), so `MI-R` is `MI-D` there and the rule's "at least 4 of five" is out of
    reach. Said now: such a reading means the correlation is not estimable at 80% on tables of this size, not that
    joint residuals fail; `MI-R` is informative on the variants where pairs exist, counted in the printout. The rule
    is kept as approved; `MI-J` is the arm it can decide.
  - `MI-J` shows the model rows with fewer hidden cells than it was trained on at 80%, and with its own draws among
    the inputs: the chain leaves the training regime. `MI-R`'s correlation is measured on validation rows with more
    of the row visible than a test row at 80% has.

## Design (fixed now)

- **Cells (45):** H (`deployable_hybrid.train`: `--decode_checkpoint loss --decode_calibration none`, the fold's
  hidden observed validation cells written to `selection/`) retrained on `biodeg`, `kc2`, `vehicle`, `credit-g`,
  `pendigits` at 20, 40 and 80% MCAR × seeds 4545/5656/6767, five folds each. Runner
  `scripts/experiments/joint_draws.py`, tag `experiment = joint-draws-2026-11-13`, state
  `/scratch/diogoneiss/trident-2026-11-13-joint-draws`.
- **Draws at fill time (M = 20):** after each fold's decode stage the run's model, in eval mode with the trainer's
  random states saved and restored around it, fills the test split's gaps into
  `draws/<variant>_s<seed>/draws_fold_<k>.csv`. For one draw every `Dropout` module applies one mask to all rows and
  all passes (one draw of the network); a new mask each draw.
- **Arms**, each completing the same test rows:
  - `MI-D`: E175's deployable draws recomputed on these runs. A numerical gap takes H's fill plus a residual of its
    column among the fold's validation cells; a categorical gap the truth of a validation cell where H predicted the
    same class; the pool resampled with replacement a draw.
  - `MI-S` (descriptive): one pass a draw. A categorical gap takes a class sampled from the head's softmax; a
    numerical gap the pass's fill plus a residual of its column's pool (the pool's residuals are H's, in eval mode).
  - `MI-R`: `MI-D` with a row's numerical residuals drawn together, through a Gaussian copula whose correlation is
    that of the residuals' normal scores over validation rows, a cell's residual averaged over the passes that hid
    it (pairs with fewer than 30 rows take 0; the matrix clipped to positive definite). Each column keeps `MI-D`'s distribution.
    Categorical gaps as `MI-D`.
  - `MI-J`: `MI-S` chained. Each row's gaps are filled one at a time in a random order, and each draw is written
    into the row (a class id, or a value with its mask flag cleared) before the next pass.
  - `MICE`: E175's comparator, ICE fitted once on the training split, M chains drawn from each conditional.
  - `CD` (truth in every gap) and `SI` (H once): for the secondary reading only.
- **Estimands:** E175's, features only: every numerical column's mean, every categorical column's share of the
  complete table's commonest category, the slope of each numerical column on the next one in file order.
- **Primary reading:** the target is the estimand on the fold's test rows with the truth in every gap. The interval
  is Q̄ ± t(M−1) sqrt((1 + 1/M) B), B the variance of the M completed tables' estimates: the missing-data part of
  Rubin's total variance, which is what a complete-data target calls for. `CD` and `SI` have no interval under it.
- **Secondary reading:** E175's (the whole table's value, T = W + (1 + 1/M)B, Barnard and Rubin's degrees of
  freedom) on `vehicle`, `credit-g` and `pendigits`, the tables where E175's control covered.
- **Checks in the reader:** the five ledgers hold exactly the table's gaps; the draws file holds exactly the
  ledger's cells; `plain` (the model's fill as the draws code encodes the table) equals H's ledger (numbers within
  0.001, classes on at least 0.999 of cells).
- **Randomness:** draws `default_rng([179, crc32(variant), seed, fold])` and a torch generator seeded from it;
  reader `default_rng([179, crc32(variant), seed, fold, arm])`. The training itself is the seed's.
- **Machine:** all 45 trainings and the reader on gorgona8 (RTX 3090 Ti, shared with another project and, until it
  ends, with E178's offline lane); one machine, as ADR 0035 (the split of studies between servers) asks. Runs log
  to the leader's store on medusa1 through the tunnel, `host = gorgona8`. medusa1 has had no CUDA since its GPU 0
  fell on 2026-11-13.
- **Cost:** E174's runs of these 45 cells summed 718 minutes (`pendigits` 476); about 12 GPU-hours alone, more on
  the shared card, plus the chained passes (at most one pass a column a draw). The reader is CPU only.

## Measures and rule (fixed now)

Per (variant, seed, fold) and arm: coverage, the share of estimands whose interval holds the target, and the
shortfall u = max(0, 0.95 − coverage). Fold-pair verdict on the 15 (seed, fold) differences of u, SESOI 0.01, lower
is better. For `MI-R` and for `MI-J`, each against `MI-D`, on the primary reading:

- **Supported** if better on at least 4 of the five 80% variants and worse on none of the 15.
- **Refuted** if better on at most 1 of the five, or worse on at least 3 of the 15.
- **Mixed** otherwise.
- **Proper enough** (yes or no, per arm): mean coverage at least 0.90 on at least 12 of the 15 variants.
- **Secondary:** u(`MI-S`) − u(`MI-D`), u(`MI-J`) − u(`MI-S`), and `MI-R` and `MI-J` against `MICE`, counted;
  coverage by family of estimand; the mean within-imputation variance over the complete-data one (Rubin's second
  condition, descriptive); E175's reading on the three tables, with `MI-D` − `SI`, `MI-R` − `MI-D`, `MI-J` − `MI-D`.
- Three seeds, 15 variants, M = 20: fixed; no extension.
- **Read** once with all 45 cells in, on gorgona8:
  `uv run --python 3.11 python -B scripts/experiments/joint_draws.py --tally`
- **What it leads to:** an arm supported and proper enough: a proposal to the user for a multiple-imputation output
  of the decoder (an ADR is theirs). Otherwise reported as is, with the family that fails. No ADR changes on this
  study alone.
- **Limits:** MCAR only; estimands of the features, not of a model with the label; the dropout mask is one a draw,
  shared by rows, so it stands for the network's uncertainty only as far as dropout does.
