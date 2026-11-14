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

## Outcome (2026-11-14 08:54 GMT-3; read once 2026-11-14 08:50 GMT-3, on gorgona8 at 3f944d8, all 45 cells in)

The 45 trainings ran on gorgona8's GPU from 01:09 to 08:47 GMT-3 on 2026-11-14, in two queues beside E178's offline
lane and the other project's process: 45 `cell done` lines and no FAILED line, and the leader's store holds 45
FINISHED parent runs tagged `host = gorgona8` at 3f944d8. The reader then ran on gorgona8's CPUs, each of the 45
cells ending on `DONE ... plain equals H's ledger on 5 folds`. The printout, unedited:

```
Primary reading: the fold's complete-data estimate, the between-draw variance alone

- `biodeg_20nan`: MI-D coverage 0.873 u 0.078 within/complete 1.02; MI-S coverage 0.921 u 0.032 within/complete 1.01; MI-R coverage 0.868 u 0.082 within/complete 1.02; MI-J coverage 0.933 u 0.029 within/complete 1.01; MICE coverage 0.612 u 0.338 within/complete 0.99
  - by family: mean MI-D 0.896 MI-S 0.940 MI-R 0.902 MI-J 0.953 MICE 0.525; slope MI-D 0.850 MI-S 0.902 MI-R 0.833 MI-J 0.912 MICE 0.700
  - MI-R: 820.0 of 820 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0041 [-0.0107, +0.0189] none
  - u(MI-J) − u(MI-D) -0.0491 [-0.0679, -0.0302] better
  - u(MI-S) − u(MI-D) -0.0458 [-0.0608, -0.0309] better
  - u(MI-J) − u(MI-S) -0.0032 [-0.0160, +0.0096] none
  - u(MI-R) − u(MICE) -0.2567 [-0.2974, -0.2161] better
  - u(MI-J) − u(MICE) -0.3099 [-0.3448, -0.2750] better
- `biodeg_40nan`: MI-D coverage 0.853 u 0.097 within/complete 1.04; MI-S coverage 0.908 u 0.043 within/complete 1.03; MI-R coverage 0.853 u 0.097 within/complete 1.04; MI-J coverage 0.913 u 0.038 within/complete 1.05; MICE coverage 0.596 u 0.354 within/complete 0.98
  - by family: mean MI-D 0.896 MI-S 0.941 MI-R 0.889 MI-J 0.940 MICE 0.431; slope MI-D 0.808 MI-S 0.873 MI-R 0.815 MI-J 0.885 MICE 0.765
  - MI-R: 795.4 of 820 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0000 [-0.0181, +0.0181] none
  - u(MI-J) − u(MI-D) -0.0592 [-0.0747, -0.0437] better
  - u(MI-S) − u(MI-D) -0.0542 [-0.0694, -0.0391] better
  - u(MI-J) − u(MI-S) -0.0049 [-0.0177, +0.0078] none
  - u(MI-R) − u(MICE) -0.2568 [-0.2842, -0.2294] better
  - u(MI-J) − u(MICE) -0.3160 [-0.3400, -0.2919] better
- `biodeg_80nan`: MI-D coverage 0.756 u 0.194 within/complete 1.14; MI-S coverage 0.791 u 0.159 within/complete 1.13; MI-R coverage 0.755 u 0.195 within/complete 1.15; MI-J coverage 0.759 u 0.191 within/complete 1.21; MICE coverage 0.435 u 0.515 within/complete 0.75
  - by family: mean MI-D 0.857 MI-S 0.907 MI-R 0.863 MI-J 0.824 MICE 0.294; slope MI-D 0.653 MI-S 0.672 MI-R 0.643 MI-J 0.692 MICE 0.580
  - MI-R: 0.0 of 820 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0016 [-0.0136, +0.0169] none
  - u(MI-J) − u(MI-D) -0.0025 [-0.0314, +0.0264] none
  - u(MI-S) − u(MI-D) -0.0346 [-0.0527, -0.0164] better
  - u(MI-J) − u(MI-S) +0.0321 [+0.0097, +0.0544] worse
  - u(MI-R) − u(MICE) -0.3193 [-0.3727, -0.2660] better
  - u(MI-J) − u(MICE) -0.3235 [-0.3782, -0.2687] better
- `kc2_20nan`: MI-D coverage 0.704 u 0.253 within/complete 1.03; MI-S coverage 0.813 u 0.147 within/complete 1.04; MI-R coverage 0.715 u 0.238 within/complete 1.03; MI-J coverage 0.824 u 0.136 within/complete 1.04; MICE coverage 0.541 u 0.409 within/complete 1.00
  - by family: mean MI-D 0.819 MI-S 0.898 MI-R 0.832 MI-J 0.905 MICE 0.559; slope MI-D 0.583 MI-S 0.723 MI-R 0.593 MI-J 0.740 MICE 0.523
  - MI-R: 192.6 of 210 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0146 [-0.0390, +0.0097] none
  - u(MI-J) − u(MI-D) -0.1171 [-0.1653, -0.0689] better
  - u(MI-S) − u(MI-D) -0.1057 [-0.1513, -0.0601] better
  - u(MI-J) − u(MI-S) -0.0114 [-0.0266, +0.0038] none
  - u(MI-R) − u(MICE) -0.1705 [-0.2564, -0.0846] better
  - u(MI-J) − u(MICE) -0.2729 [-0.3298, -0.2161] better
- `kc2_40nan`: MI-D coverage 0.616 u 0.334 within/complete 1.07; MI-S coverage 0.782 u 0.171 within/complete 1.08; MI-R coverage 0.620 u 0.331 within/complete 1.07; MI-J coverage 0.790 u 0.167 within/complete 1.12; MICE coverage 0.504 u 0.446 within/complete 1.02
  - by family: mean MI-D 0.778 MI-S 0.905 MI-R 0.794 MI-J 0.873 MICE 0.486; slope MI-D 0.447 MI-S 0.653 MI-R 0.437 MI-J 0.703 MICE 0.523
  - MI-R: 0.8 of 210 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0032 [-0.0222, +0.0158] none
  - u(MI-J) − u(MI-D) -0.1672 [-0.2158, -0.1187] better
  - u(MI-S) − u(MI-D) -0.1624 [-0.2122, -0.1126] better
  - u(MI-J) − u(MI-S) -0.0048 [-0.0232, +0.0137] none
  - u(MI-R) − u(MICE) -0.1154 [-0.1908, -0.0400] better
  - u(MI-J) − u(MICE) -0.2794 [-0.3295, -0.2293] better
- `kc2_80nan`: MI-D coverage 0.624 u 0.326 within/complete 1.22; MI-S coverage 0.715 u 0.235 within/complete 1.20; MI-R coverage 0.600 u 0.350 within/complete 1.21; MI-J coverage 0.688 u 0.262 within/complete 1.66; MICE coverage 0.369 u 0.581 within/complete 0.95
  - by family: mean MI-D 0.822 MI-S 0.876 MI-R 0.797 MI-J 0.832 MICE 0.406; slope MI-D 0.417 MI-S 0.547 MI-R 0.393 MI-J 0.537 MICE 0.330
  - MI-R: 0.0 of 210 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0244 [+0.0004, +0.0483] worse
  - u(MI-J) − u(MI-D) -0.0634 [-0.1216, -0.0053] better
  - u(MI-S) − u(MI-D) -0.0911 [-0.1206, -0.0615] better
  - u(MI-J) − u(MI-S) +0.0276 [-0.0254, +0.0807] none
  - u(MI-R) − u(MICE) -0.2309 [-0.3078, -0.1540] better
  - u(MI-J) − u(MICE) -0.3187 [-0.3884, -0.2490] better
- `vehicle_20nan`: MI-D coverage 0.884 u 0.066 within/complete 1.01; MI-S coverage 0.956 u 0.011 within/complete 1.01; MI-R coverage 0.891 u 0.059 within/complete 1.01; MI-J coverage 0.960 u 0.010 within/complete 1.01; MICE coverage 0.850 u 0.100 within/complete 1.00
  - by family: mean MI-D 0.922 MI-S 0.967 MI-R 0.926 MI-J 0.974 MICE 0.867; slope MI-D 0.843 MI-S 0.945 MI-R 0.855 MI-J 0.945 MICE 0.831
  - MI-R: 153.0 of 153 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0076 [-0.0279, +0.0126] none
  - u(MI-J) − u(MI-D) -0.0562 [-0.0846, -0.0278] better
  - u(MI-S) − u(MI-D) -0.0552 [-0.0850, -0.0255] better
  - u(MI-J) − u(MI-S) -0.0010 [-0.0071, +0.0052] equivalent
  - u(MI-R) − u(MICE) -0.0419 [-0.0746, -0.0092] better
  - u(MI-J) − u(MICE) -0.0905 [-0.1192, -0.0617] better
- `vehicle_40nan`: MI-D coverage 0.869 u 0.083 within/complete 1.02; MI-S coverage 0.949 u 0.014 within/complete 1.02; MI-R coverage 0.878 u 0.073 within/complete 1.02; MI-J coverage 0.956 u 0.007 within/complete 1.02; MICE coverage 0.787 u 0.163 within/complete 1.00
  - by family: mean MI-D 0.941 MI-S 0.989 MI-R 0.963 MI-J 0.985 MICE 0.907; slope MI-D 0.792 MI-S 0.906 MI-R 0.788 MI-J 0.925 MICE 0.659
  - MI-R: 97.1 of 153 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0095 [-0.0244, +0.0054] none
  - u(MI-J) − u(MI-D) -0.0757 [-0.1020, -0.0494] better
  - u(MI-S) − u(MI-D) -0.0690 [-0.0943, -0.0438] better
  - u(MI-J) − u(MI-S) -0.0067 [-0.0170, +0.0036] none
  - u(MI-R) − u(MICE) -0.0900 [-0.1325, -0.0475] better
  - u(MI-J) − u(MICE) -0.1562 [-0.2044, -0.1079] better
- `vehicle_80nan`: MI-D coverage 0.701 u 0.249 within/complete 1.20; MI-S coverage 0.745 u 0.205 within/complete 1.19; MI-R coverage 0.705 u 0.245 within/complete 1.20; MI-J coverage 0.787 u 0.163 within/complete 1.43; MICE coverage 0.434 u 0.516 within/complete 1.00
  - by family: mean MI-D 0.896 MI-S 0.907 MI-R 0.907 MI-J 0.878 MICE 0.630; slope MI-D 0.494 MI-S 0.573 MI-R 0.490 MI-J 0.690 MICE 0.227
  - MI-R: 0.0 of 153 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) -0.0038 [-0.0318, +0.0242] none
  - u(MI-J) − u(MI-D) -0.0857 [-0.1305, -0.0410] better
  - u(MI-S) − u(MI-D) -0.0438 [-0.0765, -0.0111] none
  - u(MI-J) − u(MI-S) -0.0419 [-0.0782, -0.0056] better
  - u(MI-R) − u(MICE) -0.2705 [-0.3348, -0.2061] better
  - u(MI-J) − u(MICE) -0.3524 [-0.4312, -0.2736] better
- `credit-g_20nan`: MI-D coverage 0.923 u 0.034 within/complete 1.00; MI-S coverage 0.933 u 0.030 within/complete 1.00; MI-R coverage 0.933 u 0.028 within/complete 1.00; MI-J coverage 0.926 u 0.033 within/complete 1.00; MICE coverage 0.854 u 0.099 within/complete 0.99
  - by family: mean MI-D 0.886 MI-S 0.924 MI-R 0.895 MI-J 0.943 MICE 0.752; share MI-D 0.954 MI-S 0.964 MI-R 0.969 MI-J 0.944 MICE 0.949; slope MI-D 0.900 MI-S 0.878 MI-R 0.900 MI-J 0.867 MICE 0.767
  - MI-R: 21.0 of 21 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.002
  - u(MI-R) − u(MI-D) -0.0062 [-0.0218, +0.0095] none
  - u(MI-J) − u(MI-D) -0.0008 [-0.0173, +0.0158] none
  - u(MI-S) − u(MI-D) -0.0044 [-0.0296, +0.0209] none
  - u(MI-J) − u(MI-S) +0.0036 [-0.0113, +0.0185] none
  - u(MI-R) − u(MICE) -0.0715 [-0.0978, -0.0452] better
  - u(MI-J) − u(MICE) -0.0662 [-0.0906, -0.0417] better
- `credit-g_40nan`: MI-D coverage 0.903 u 0.051 within/complete 1.00; MI-S coverage 0.867 u 0.083 within/complete 1.00; MI-R coverage 0.887 u 0.066 within/complete 1.01; MI-J coverage 0.890 u 0.064 within/complete 1.01; MICE coverage 0.656 u 0.294 within/complete 1.01
  - by family: mean MI-D 0.933 MI-S 0.914 MI-R 0.924 MI-J 0.933 MICE 0.410; share MI-D 0.949 MI-S 0.903 MI-R 0.949 MI-J 0.892 MICE 0.795; slope MI-D 0.767 MI-S 0.733 MI-R 0.711 MI-J 0.833 MICE 0.644
  - MI-R: 19.5 of 21 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.003
  - u(MI-R) − u(MI-D) +0.0146 [-0.0131, +0.0423] none
  - u(MI-J) − u(MI-D) +0.0123 [-0.0196, +0.0442] none
  - u(MI-S) − u(MI-D) +0.0321 [+0.0005, +0.0636] worse
  - u(MI-J) − u(MI-S) -0.0197 [-0.0504, +0.0109] none
  - u(MI-R) − u(MICE) -0.2277 [-0.2776, -0.1778] better
  - u(MI-J) − u(MICE) -0.2300 [-0.2809, -0.1791] better
- `credit-g_80nan`: MI-D coverage 0.772 u 0.178 within/complete 0.97; MI-S coverage 0.756 u 0.194 within/complete 0.99; MI-R coverage 0.779 u 0.171 within/complete 0.98; MI-J coverage 0.718 u 0.232 within/complete 1.00; MICE coverage 0.423 u 0.527 within/complete 1.02
  - by family: mean MI-D 0.886 MI-S 0.924 MI-R 0.895 MI-J 0.905 MICE 0.171; share MI-D 0.913 MI-S 0.841 MI-R 0.923 MI-J 0.749 MICE 0.538; slope MI-D 0.333 MI-S 0.378 MI-R 0.333 MI-J 0.433 MICE 0.467
  - MI-R: 0.0 of 21 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.005
  - u(MI-R) − u(MI-D) -0.0077 [-0.0369, +0.0216] none
  - u(MI-J) − u(MI-D) +0.0538 [+0.0199, +0.0878] worse
  - u(MI-S) − u(MI-D) +0.0154 [-0.0166, +0.0474] none
  - u(MI-J) − u(MI-S) +0.0385 [+0.0043, +0.0726] worse
  - u(MI-R) − u(MICE) -0.3564 [-0.4185, -0.2943] better
  - u(MI-J) − u(MICE) -0.2949 [-0.3443, -0.2455] better
- `pendigits_20nan`: MI-D coverage 0.951 u 0.015 within/complete 1.02; MI-S coverage 0.991 u 0.000 within/complete 1.00; MI-R coverage 0.946 u 0.024 within/complete 1.02; MI-J coverage 0.989 u 0.000 within/complete 1.01; MICE coverage 0.624 u 0.326 within/complete 0.98
  - by family: mean MI-D 0.975 MI-S 0.996 MI-R 0.963 MI-J 0.996 MICE 0.546; slope MI-D 0.924 MI-S 0.987 MI-R 0.929 MI-J 0.982 MICE 0.707
  - MI-R: 120.0 of 120 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0086 [-0.0020, +0.0192] none
  - u(MI-J) − u(MI-D) -0.0154 [-0.0285, -0.0022] better
  - u(MI-S) − u(MI-D) -0.0154 [-0.0285, -0.0022] better
  - u(MI-J) − u(MI-S) +0.0000 [+0.0000, +0.0000] equivalent
  - u(MI-R) − u(MICE) -0.3024 [-0.3444, -0.2604] better
  - u(MI-J) − u(MICE) -0.3263 [-0.3606, -0.2921] better
- `pendigits_40nan`: MI-D coverage 0.892 u 0.061 within/complete 1.06; MI-S coverage 0.955 u 0.015 within/complete 1.02; MI-R coverage 0.886 u 0.066 within/complete 1.06; MI-J coverage 0.927 u 0.028 within/complete 1.03; MICE coverage 0.368 u 0.582 within/complete 0.94
  - by family: mean MI-D 0.963 MI-S 0.975 MI-R 0.958 MI-J 0.971 MICE 0.383; slope MI-D 0.818 MI-S 0.933 MI-R 0.809 MI-J 0.880 MICE 0.351
  - MI-R: 120.0 of 120 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0053 [-0.0151, +0.0256] none
  - u(MI-J) − u(MI-D) -0.0332 [-0.0652, -0.0013] better
  - u(MI-S) − u(MI-D) -0.0457 [-0.0799, -0.0115] better
  - u(MI-J) − u(MI-S) +0.0125 [-0.0063, +0.0312] none
  - u(MI-R) − u(MICE) -0.5159 [-0.5838, -0.4480] better
  - u(MI-J) − u(MICE) -0.5544 [-0.6078, -0.5011] better
- `pendigits_80nan`: MI-D coverage 0.637 u 0.313 within/complete 1.09; MI-S coverage 0.658 u 0.292 within/complete 1.08; MI-R coverage 0.581 u 0.369 within/complete 1.09; MI-J coverage 0.665 u 0.285 within/complete 1.13; MICE coverage 0.241 u 0.709 within/complete 0.85
  - by family: mean MI-D 0.967 MI-S 0.971 MI-R 0.946 MI-J 0.846 MICE 0.312; slope MI-D 0.284 MI-S 0.324 MI-R 0.191 MI-J 0.471 MICE 0.164
  - MI-R: 119.4 of 120 column pairs with a residual correlation a fold; cells keeping H's fill under MI-D 0.000
  - u(MI-R) − u(MI-D) +0.0559 [+0.0321, +0.0798] worse
  - u(MI-J) − u(MI-D) -0.0280 [-0.0655, +0.0096] none
  - u(MI-S) − u(MI-D) -0.0215 [-0.0446, +0.0016] none
  - u(MI-J) − u(MI-S) -0.0065 [-0.0548, +0.0419] none
  - u(MI-R) − u(MICE) -0.3398 [-0.3875, -0.2921] better
  - u(MI-J) − u(MICE) -0.4237 [-0.4713, -0.3760] better

- mean coverage at least 0.90 under MI-D: 3 of 15
- mean coverage at least 0.90 under MI-S: 7 of 15
- mean coverage at least 0.90 under MI-R: 2 of 15
- mean coverage at least 0.90 under MI-J: 7 of 15
- mean coverage at least 0.90 under MICE: 0 of 15
- u, MI-R-MI-D: better 0 / no difference 13 / worse 2 (fold-pair t, seed agreement)
- u, MI-J-MI-D: better 10 / no difference 4 / worse 1 (fold-pair t, seed agreement)
- u, MI-S-MI-D: better 10 / no difference 4 / worse 1 (fold-pair t, seed agreement)
- u, MI-J-MI-S: better 1 / no difference 10 / equivalent 2 / worse 2 (fold-pair t, seed agreement)
- u, MI-R-MICE: better 15 / no difference 0 / worse 0 (fold-pair t, seed agreement)
- u, MI-J-MICE: better 15 / no difference 0 / worse 0 (fold-pair t, seed agreement)

E179 PRIMARY, u(MI-R) − u(MI-D): better on 0 of the 5 variants at 80% (none), worse on ['kc2_80nan', 'pendigits_80nan'] of 15 -> refuted (fold-pair t, seed agreement; SESOI 0.01)
E179 PRIMARY, MI-R proper enough (mean coverage at least 0.90 on at least 12 of 15): 2 -> no
E179 PRIMARY, u(MI-J) − u(MI-D): better on 2 of the 5 variants at 80% (['kc2_80nan', 'vehicle_80nan']), worse on ['credit-g_80nan'] of 15 -> mixed (fold-pair t, seed agreement; SESOI 0.01)
E179 PRIMARY, MI-J proper enough (mean coverage at least 0.90 on at least 12 of 15): 7 -> no

Secondary, E175's reading (the whole table's value, the total variance) on vehicle, credit-g, pendigits

- `vehicle_20nan`: CD 0.933; SI 0.933; MI-D 0.954; MI-S 0.960; MI-R 0.952; MI-J 0.958; MICE 0.958
  - u(MI-D) − u(SI) -0.0167 [-0.0329, -0.0004] better
  - u(MI-R) − u(MI-D) +0.0019 [-0.0027, +0.0065] equivalent
  - u(MI-J) − u(MI-D) -0.0038 [-0.0098, +0.0021] equivalent
- `vehicle_40nan`: CD 0.933; SI 0.882; MI-D 0.954; MI-S 0.964; MI-R 0.950; MI-J 0.968; MICE 0.930
  - u(MI-D) − u(SI) -0.0567 [-0.0867, -0.0266] better
  - u(MI-R) − u(MI-D) +0.0024 [-0.0020, +0.0068] equivalent
  - u(MI-J) − u(MI-D) -0.0062 [-0.0123, -0.0001] better
- `vehicle_80nan`: CD 0.933; SI 0.461; MI-D 0.777; MI-S 0.819; MI-R 0.794; MI-J 0.823; MICE 0.613
  - u(MI-D) − u(SI) -0.3162 [-0.3715, -0.2609] better
  - u(MI-R) − u(MI-D) -0.0171 [-0.0368, +0.0025] none
  - u(MI-J) − u(MI-D) -0.0457 [-0.0778, -0.0136] better
- `credit-g_20nan`: CD 0.954; SI 0.826; MI-D 0.974; MI-S 0.977; MI-R 0.979; MI-J 0.979; MICE 0.982
  - u(MI-D) − u(SI) -0.1233 [-0.1553, -0.0914] better
  - u(MI-R) − u(MI-D) +0.0000 [-0.0056, +0.0056] equivalent
  - u(MI-J) − u(MI-D) -0.0018 [-0.0056, +0.0021] equivalent
- `credit-g_40nan`: CD 0.954; SI 0.559; MI-D 0.964; MI-S 0.964; MI-R 0.967; MI-J 0.972; MICE 0.913
  - u(MI-D) − u(SI) -0.3805 [-0.4356, -0.3255] better
  - u(MI-R) − u(MI-D) -0.0026 [-0.0081, +0.0029] equivalent
  - u(MI-J) − u(MI-D) -0.0044 [-0.0153, +0.0066] none
- `credit-g_80nan`: CD 0.954; SI 0.192; MI-D 0.838; MI-S 0.826; MI-R 0.846; MI-J 0.800; MICE 0.587
  - u(MI-D) − u(SI) -0.6462 [-0.6959, -0.5964] better
  - u(MI-R) − u(MI-D) -0.0077 [-0.0260, +0.0107] none
  - u(MI-J) − u(MI-D) +0.0385 [+0.0043, +0.0726] worse
- `pendigits_20nan`: CD 0.991; SI 0.991; MI-D 0.994; MI-S 0.998; MI-R 0.994; MI-J 0.996; MICE 0.976
  - u(MI-D) − u(SI) +0.0000 [+0.0000, +0.0000] equivalent
  - u(MI-R) − u(MI-D) +0.0000 [+0.0000, +0.0000] equivalent
  - u(MI-J) − u(MI-D) +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_40nan`: CD 0.991; SI 0.942; MI-D 0.981; MI-S 0.996; MI-R 0.983; MI-J 0.981; MICE 0.766
  - u(MI-D) − u(SI) -0.0232 [-0.0462, -0.0002] better
  - u(MI-R) − u(MI-D) +0.0000 [+0.0000, +0.0000] equivalent
  - u(MI-J) − u(MI-D) +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_80nan`: CD 0.991; SI 0.327; MI-D 0.675; MI-S 0.705; MI-R 0.617; MI-J 0.690; MICE 0.376
  - u(MI-D) − u(SI) -0.3484 [-0.4107, -0.2861] better
  - u(MI-R) − u(MI-D) +0.0581 [+0.0355, +0.0807] worse
  - u(MI-J) − u(MI-D) -0.0151 [-0.0494, +0.0193] none

- u, MI-D-SI: better 8 / no difference 0 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
- u, MI-R-MI-D: better 0 / no difference 2 / equivalent 6 / worse 1 (fold-pair t, seed agreement)
- u, MI-J-MI-D: better 2 / no difference 2 / equivalent 4 / worse 1 (fold-pair t, seed agreement)
```

- **Reading, by the ticket's own rule:**
  - **`MI-R` (jointly drawn residuals) reads refuted:** better than `MI-D` on none of the five 80% variants, worse on
    `kc2_80nan` and `pendigits_80nan`. The ticket said beforehand that it could not read supported, four of the five
    tables having no residual correlation at 80%. What the printout adds: on `pendigits_80nan`, the one 80% variant
    with correlations (119.4 of 120 pairs), `MI-R` is worse (+0.0559; slopes covered 0.191 against 0.284), and on the
    nine variants at 20 and 40% with correlations (all but `kc2_40nan`) it differs from `MI-D` on none. `kc2_80nan`
    has no pair, so `MI-R` is `MI-D` there and its "worse" (+0.0244 [+0.0004, +0.0483]) can only come from the
    random stream. Where the correlation could be measured, the copula did not help.
  - **`MI-J` (the model's draws chained through the row) reads mixed:** better than `MI-D` on `kc2_80nan` and
    `vehicle_80nan`, worse on `credit-g_80nan`. Over the 15 variants it is better on 10 and worse on 1, but `MI-S`,
    the same model draws without the chain, has the same count, and `MI-J` against `MI-S` is better on 1
    (`vehicle_80nan`) and worse on 2 (`biodeg_80nan`, `credit-g_80nan`). The gain over E175's draws comes from
    drawing from the model (one dropout mask a draw, softmax samples), not from chaining the row. `MI-S` was a
    descriptive arm, so this is a secondary count, not a verdict of the rule.
  - **What the chain does at 80%:** against `MI-S`, slopes are covered more often on four of the five tables
    (`vehicle` 0.690 against 0.573, `pendigits` 0.471 against 0.324) and means less often on all five (`pendigits`
    0.846 against 0.971, `biodeg` 0.824 against 0.907); `credit-g`'s shares fall to 0.749 from 0.841. Each completed
    table is also more variable than the complete data (within/complete 1.66 on `kc2`, 1.43 on `vehicle`).
  - **Proper enough: no, for either arm** (2 and 7 of 15 against 12). At 80% no arm reaches a mean coverage of 0.80
    on any table (the highest is `MI-S` on `biodeg`, 0.791). The family that fails is the slopes: 0.19 to 0.69 at
    80% over the four arms.
  - **`MICE`** has the lowest coverage on all 15 variants; `MI-R` and `MI-J` are each better than it on 15 of 15.
  - **Secondary, E175's reading:** `MI-D` is better than the single fill on 8 of the 9 variants and equivalent on
    the ninth, which is E175's first primary again at fresh seeds. `MI-J` is better than `MI-D` on `vehicle` at 40
    and 80% and worse on `credit-g_80nan`; `MI-R` is worse on `pendigits_80nan`. At 80% the four arms cover 0.62 to
    0.85 where the truth-filled control covers 0.93 to 0.99.
- **What it leads to:** no arm is supported and proper enough, so there is no proposal for a multiple-imputation
  output of the decoder and no ADR changes. Reported as is: draws tied across a row's gaps do not make the intervals
  proper at 80% missing, and the slopes between columns are the family that fails.
- **For the user's judgement:** `MI-R`'s correlation pooled over passes was a change from the approved proposal made
  without asking (disclosed above before any cell). A same-pass correlation would have existed on two variants only.
