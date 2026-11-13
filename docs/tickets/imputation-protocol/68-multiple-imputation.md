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

## Outcome (2026-11-12 23:17 GMT-3; read once 2026-11-12 23:16 GMT-3, on medusa1 at 45cc64b, all 45 cells in)

The 45 cells ran on medusa1 from 23:15 to 23:16 GMT-3, every one ending on its `DONE` line with no cell keeping H's
fill for lack of a pool; the assertions on the ledgers (the five ledgers hold exactly the table's gaps, categories and
units are the table's) passed on all. The printout, unedited:

```
- `biodeg_20nan`: CD coverage 0.883 u 0.067 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.870 u 0.080 width 626856.13 shift 629453.97 lambda 0.00; MI-D coverage 0.906 u 0.044 width 9614947.09 shift 5383867.59 lambda 0.11; MI-O coverage 0.920 u 0.032 width 1192418.02 shift 1209847.13 lambda 0.12; MICE coverage 0.905 u 0.046 width 14059978.03 shift 27329349.31 lambda 0.07
  - by family: mean CD 0.925 SI 0.907 MI-D 0.941 MI-O 0.948 MICE 0.945; slope CD 0.840 SI 0.832 MI-D 0.870 MI-O 0.892 MICE 0.865
  - u(MI-D) − u(SI) -0.0362 [-0.0484, -0.0239] better
  - u(MI-D) − u(MICE) -0.0017 [-0.0235, +0.0202] none
  - u(MI-O) − u(MI-D) -0.0114 [-0.0230, +0.0002] none
  - u(MICE) − u(SI) -0.0345 [-0.0580, -0.0109] better
- `biodeg_40nan`: CD coverage 0.882 u 0.068 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.858 u 0.092 width 149310.68 shift 482619.08 lambda 0.00; MI-D coverage 0.936 u 0.021 width 1402585.50 shift 580185.46 lambda 0.26; MI-O coverage 0.959 u 0.005 width 5018057.80 shift 1633563.65 lambda 0.24; MICE coverage 0.827 u 0.123 width 4227439.26 shift 11728864.85 lambda 0.18
  - by family: mean CD 0.925 SI 0.894 MI-D 0.967 MI-O 0.980 MICE 0.769; slope CD 0.838 SI 0.822 MI-D 0.903 MI-O 0.937 MICE 0.887
  - u(MI-D) − u(SI) -0.0705 [-0.0894, -0.0516] better
  - u(MI-D) − u(MICE) -0.1018 [-0.1262, -0.0774] better
  - u(MI-O) − u(MI-D) -0.0163 [-0.0302, -0.0024] better
  - u(MICE) − u(SI) +0.0313 [+0.0032, +0.0593] worse
- `biodeg_80nan`: CD coverage 0.880 u 0.070 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.507 u 0.443 width 1.13 shift 2.62 lambda 0.00; MI-D coverage 0.810 u 0.140 width 2.96 shift 2.49 lambda 0.65; MI-O coverage 0.835 u 0.115 width 1.70 shift 2.14 lambda 0.51; MICE coverage 0.601 u 0.349 width 209132740688352.00 shift 3.13 lambda 0.40
  - by family: mean CD 0.925 SI 0.446 MI-D 0.867 MI-O 0.878 MICE 0.520; slope CD 0.833 SI 0.570 MI-D 0.752 MI-O 0.790 MICE 0.683
  - u(MI-D) − u(SI) -0.3029 [-0.3571, -0.2487] better
  - u(MI-D) − u(MICE) -0.2091 [-0.2351, -0.1830] better
  - u(MI-O) − u(MI-D) -0.0247 [-0.0570, +0.0076] none
  - u(MICE) − u(SI) -0.0938 [-0.1523, -0.0353] better
- `kc2_20nan`: CD coverage 0.574 u 0.376 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.538 u 0.412 width 1.11 shift 3.01 lambda 0.00; MI-D coverage 0.598 u 0.352 width 1.66 shift 3.06 lambda 0.13; MI-O coverage 0.702 u 0.248 width 9.15 shift 3.28 lambda 0.24; MICE coverage 0.582 u 0.368 width 1.56 shift 2.73 lambda 0.09
  - by family: mean CD 0.854 SI 0.841 MI-D 0.873 MI-O 0.895 MICE 0.854; slope CD 0.280 SI 0.220 MI-D 0.310 MI-O 0.500 MICE 0.297
  - u(MI-D) − u(SI) -0.0602 [-0.0976, -0.0227] better
  - u(MI-D) − u(MICE) -0.0163 [-0.0554, +0.0228] none
  - u(MI-O) − u(MI-D) -0.1041 [-0.1478, -0.0603] better
  - u(MICE) − u(SI) -0.0439 [-0.0854, -0.0024] better
- `kc2_40nan`: CD coverage 0.574 u 0.376 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.571 u 0.379 width 1.09 shift 3.33 lambda 0.00; MI-D coverage 0.663 u 0.287 width 1.79 shift 3.37 lambda 0.23; MI-O coverage 0.803 u 0.147 width 4.57 shift 3.96 lambda 0.38; MICE coverage 0.650 u 0.300 width 1.58 shift 3.33 lambda 0.18
  - by family: mean CD 0.854 SI 0.851 MI-D 0.886 MI-O 0.952 MICE 0.870; slope CD 0.280 SI 0.277 MI-D 0.430 MI-O 0.647 MICE 0.420
  - u(MI-D) − u(SI) -0.0927 [-0.1384, -0.0470] better
  - u(MI-D) − u(MICE) -0.0130 [-0.0535, +0.0275] none
  - u(MI-O) − u(MI-D) -0.1398 [-0.1972, -0.0824] better
  - u(MICE) − u(SI) -0.0797 [-0.1145, -0.0449] better
- `kc2_80nan`: CD coverage 0.574 u 0.376 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.494 u 0.456 width 1.51 shift 6.68 lambda 0.00; MI-D coverage 0.701 u 0.249 width 5.13 shift 6.89 lambda 0.50; MI-O coverage 0.831 u 0.119 width 6.16 shift 7.17 lambda 0.54; MICE coverage 0.611 u 0.339 width 4.77 shift 8.82 lambda 0.43
  - by family: mean CD 0.854 SI 0.768 MI-D 0.917 MI-O 0.965 MICE 0.838; slope CD 0.280 SI 0.207 MI-D 0.473 MI-O 0.690 MICE 0.373
  - u(MI-D) − u(SI) -0.2065 [-0.2638, -0.1492] better
  - u(MI-D) − u(MICE) -0.0894 [-0.1388, -0.0400] better
  - u(MI-O) − u(MI-D) -0.1300 [-0.1714, -0.0886] better
  - u(MICE) − u(SI) -0.1171 [-0.1821, -0.0521] better
- `vehicle_20nan`: CD coverage 0.914 u 0.045 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.905 u 0.053 width 0.99 shift 0.26 lambda 0.00; MI-D coverage 0.928 u 0.031 width 1.07 shift 0.28 lambda 0.09; MI-O coverage 0.926 u 0.035 width 1.06 shift 0.28 lambda 0.08; MICE coverage 0.931 u 0.030 width 1.04 shift 0.24 lambda 0.06
  - by family: mean CD 0.967 SI 0.956 MI-D 0.974 MI-O 0.967 MICE 0.978; slope CD 0.859 SI 0.851 MI-D 0.878 MI-O 0.882 MICE 0.882
  - u(MI-D) − u(SI) -0.0214 [-0.0376, -0.0053] better
  - u(MI-D) − u(MICE) +0.0019 [-0.0128, +0.0166] none
  - u(MI-O) − u(MI-D) +0.0033 [-0.0085, +0.0152] none
  - u(MICE) − u(SI) -0.0233 [-0.0419, -0.0048] better
- `vehicle_40nan`: CD coverage 0.914 u 0.045 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.867 u 0.087 width 0.96 shift 0.62 lambda 0.00; MI-D coverage 0.937 u 0.022 width 1.23 shift 0.68 lambda 0.24; MI-O coverage 0.960 u 0.014 width 1.21 shift 0.64 lambda 0.21; MICE coverage 0.935 u 0.024 width 1.14 shift 0.62 lambda 0.17
  - by family: mean CD 0.967 SI 0.937 MI-D 0.967 MI-O 0.978 MICE 0.978; slope CD 0.859 SI 0.792 MI-D 0.906 MI-O 0.941 MICE 0.890
  - u(MI-D) − u(SI) -0.0648 [-0.1009, -0.0286] better
  - u(MI-D) − u(MICE) -0.0019 [-0.0172, +0.0134] none
  - u(MI-O) − u(MI-D) -0.0081 [-0.0185, +0.0024] none
  - u(MICE) − u(SI) -0.0629 [-0.1057, -0.0200] better
- `vehicle_80nan`: CD coverage 0.914 u 0.045 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.455 u 0.495 width 0.94 shift 2.91 lambda 0.00; MI-D coverage 0.771 u 0.179 width 2.80 shift 4.12 lambda 0.71; MI-O coverage 0.665 u 0.285 width 1.90 shift 4.09 lambda 0.47; MICE coverage 0.583 u 0.367 width 1.84 shift 6.11 lambda 0.37
  - by family: mean CD 0.967 SI 0.570 MI-D 0.933 MI-O 0.893 MICE 0.807; slope CD 0.859 SI 0.333 MI-D 0.600 MI-O 0.424 MICE 0.345
  - u(MI-D) − u(SI) -0.3162 [-0.3705, -0.2619] better
  - u(MI-D) − u(MICE) -0.1886 [-0.2436, -0.1335] better
  - u(MI-O) − u(MI-D) +0.1067 [+0.0621, +0.1512] worse
  - u(MICE) − u(SI) -0.1276 [-0.2067, -0.0485] better
- `credit-g_20nan`: CD coverage 0.941 u 0.026 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.851 u 0.100 width 0.96 shift 0.91 lambda 0.00; MI-D coverage 0.982 u 0.002 width 1.18 shift 0.53 lambda 0.25; MI-O coverage 0.974 u 0.005 width 1.17 shift 0.55 lambda 0.24; MICE coverage 0.979 u 0.004 width 1.09 shift 0.50 lambda 0.16
  - by family: mean CD 0.933 SI 0.943 MI-D 0.962 MI-O 0.971 MICE 0.952; share CD 0.944 SI 0.805 MI-D 0.990 MI-O 0.974 MICE 0.990; slope CD 0.944 SI 0.844 MI-D 0.989 MI-O 0.978 MICE 0.989
  - u(MI-D) − u(SI) -0.0985 [-0.1461, -0.0508] better
  - u(MI-D) − u(MICE) -0.0026 [-0.0129, +0.0078] none
  - u(MI-O) − u(MI-D) +0.0036 [-0.0041, +0.0113] none
  - u(MICE) − u(SI) -0.0959 [-0.1402, -0.0516] better
- `credit-g_40nan`: CD coverage 0.941 u 0.026 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.621 u 0.329 width 0.93 shift 1.82 lambda 0.00; MI-D coverage 0.967 u 0.008 width 1.47 shift 1.02 lambda 0.48; MI-O coverage 0.964 u 0.006 width 1.33 shift 0.97 lambda 0.38; MICE coverage 0.903 u 0.051 width 1.21 shift 1.13 lambda 0.26
  - by family: mean CD 0.933 SI 0.905 MI-D 1.000 MI-O 0.990 MICE 0.762; share CD 0.944 SI 0.385 MI-D 0.969 MI-O 0.995 MICE 0.974; slope CD 0.944 SI 0.800 MI-D 0.922 MI-O 0.867 MICE 0.911
  - u(MI-D) − u(SI) -0.3215 [-0.3884, -0.2546] better
  - u(MI-D) − u(MICE) -0.0426 [-0.0707, -0.0145] better
  - u(MI-O) − u(MI-D) -0.0018 [-0.0171, +0.0135] none
  - u(MICE) − u(SI) -0.2790 [-0.3505, -0.2074] better
- `credit-g_80nan`: CD coverage 0.941 u 0.026 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.218 u 0.732 width 0.73 shift 5.29 lambda 0.00; MI-D coverage 0.841 u 0.109 width 3.15 shift 3.07 lambda 0.80; MI-O coverage 0.844 u 0.106 width 1.62 shift 2.02 lambda 0.55; MICE coverage 0.587 u 0.363 width 1.49 shift 3.05 lambda 0.47
  - by family: mean CD 0.933 SI 0.400 MI-D 0.933 MI-O 0.886 MICE 0.286; share CD 0.944 SI 0.046 MI-D 0.928 MI-O 0.990 MICE 0.744; slope CD 0.944 SI 0.378 MI-D 0.544 MI-O 0.478 MICE 0.600
  - u(MI-D) − u(SI) -0.6231 [-0.6627, -0.5835] better
  - u(MI-D) − u(MICE) -0.2538 [-0.3166, -0.1911] better
  - u(MI-O) − u(MI-D) -0.0026 [-0.0273, +0.0222] none
  - u(MICE) − u(SI) -0.3692 [-0.4355, -0.3029] better
- `pendigits_20nan`: CD coverage 0.981 u 0.002 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.972 u 0.003 width 1.00 shift 0.23 lambda 0.00; MI-D coverage 0.983 u 0.002 width 1.04 shift 0.21 lambda 0.05; MI-O coverage 0.978 u 0.002 width 1.03 shift 0.21 lambda 0.04; MICE coverage 0.966 u 0.013 width 1.03 shift 0.47 lambda 0.06
  - by family: mean CD 1.000 SI 0.996 MI-D 0.996 MI-O 0.996 MICE 0.971; slope CD 0.960 SI 0.947 MI-D 0.969 MI-O 0.960 MICE 0.960
  - u(MI-D) − u(SI) -0.0010 [-0.0030, +0.0011] equivalent
  - u(MI-D) − u(MICE) -0.0115 [-0.0248, +0.0018] none
  - u(MI-O) − u(MI-D) +0.0000 [+0.0000, +0.0000] equivalent
  - u(MICE) − u(SI) +0.0105 [-0.0013, +0.0223] none
- `pendigits_40nan`: CD coverage 0.981 u 0.002 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.953 u 0.018 width 1.00 shift 0.52 lambda 0.00; MI-D coverage 0.981 u 0.002 width 1.14 shift 0.48 lambda 0.19; MI-O coverage 0.998 u 0.000 width 1.08 shift 0.39 lambda 0.11; MICE coverage 0.759 u 0.191 width 1.09 shift 1.46 lambda 0.18
  - by family: mean CD 1.000 SI 0.992 MI-D 0.996 MI-O 1.000 MICE 0.787; slope CD 0.960 SI 0.911 MI-D 0.964 MI-O 0.996 MICE 0.729
  - u(MI-D) − u(SI) -0.0158 [-0.0330, +0.0014] none
  - u(MI-D) − u(MICE) -0.1889 [-0.2199, -0.1579] better
  - u(MI-O) − u(MI-D) -0.0019 [-0.0048, +0.0009] equivalent
  - u(MICE) − u(SI) +0.1731 [+0.1370, +0.2093] worse
- `pendigits_80nan`: CD coverage 0.981 u 0.002 width 1.00 shift 0.00 lambda 0.00; SI coverage 0.353 u 0.597 width 0.83 shift 3.56 lambda 0.00; MI-D coverage 0.669 u 0.281 width 2.30 shift 3.93 lambda 0.70; MI-O coverage 0.535 u 0.415 width 1.47 shift 3.87 lambda 0.48; MICE coverage 0.389 u 0.561 width 1.34 shift 5.84 lambda 0.40
  - by family: mean CD 1.000 SI 0.446 MI-D 0.975 MI-O 0.708 MICE 0.562; slope CD 0.960 SI 0.253 MI-D 0.342 MI-O 0.351 MICE 0.204
  - u(MI-D) − u(SI) -0.3161 [-0.3813, -0.2509] better
  - u(MI-D) − u(MICE) -0.2796 [-0.3326, -0.2265] better
  - u(MI-O) − u(MI-D) +0.1333 [+0.0952, +0.1714] worse
  - u(MICE) − u(SI) -0.0366 [-0.0988, +0.0256] none

- mean coverage at least 0.90 under CD: 9 of 15
- mean coverage at least 0.90 under SI: 3 of 15
- mean coverage at least 0.90 under MI-D: 8 of 15
- mean coverage at least 0.90 under MI-O: 8 of 15
- mean coverage at least 0.90 under MICE: 6 of 15
- u, MI-D-SI: better 13 / no difference 1 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
- u, MI-D-MICE: better 8 / no difference 7 / worse 0 (fold-pair t, seed agreement)
- u, MI-O-MI-D: better 4 / no difference 7 / equivalent 2 / worse 2 (fold-pair t, seed agreement)
- u, MICE-SI: better 11 / no difference 2 / worse 2 (fold-pair t, seed agreement)

E175 PRIMARY 1, u(MI-D) − u(SI): better on 13 of 15, worse on none -> supported (fold-pair t, seed agreement; SESOI 0.01)
E175 PRIMARY 2, MI-D proper enough (mean coverage at least 0.90 on at least 10 of 15): 8 -> no
E175 SECONDARY, u(MI-D) − u(MICE): better on 8, worse on 0 (none)
```

- **Primary 1, supported:** the residual draws (`MI-D`) fall short of 0.95 by less than the single fill (`SI`) on 13
  of 15 variants and by more on none. The two without a call are `pendigits_20nan` (equivalent) and `pendigits_40nan`
  (no call), where the single fill already covers 0.972 and 0.953. The gain grows with missingness: at 80% the single
  fill covers 0.22 to 0.51 and the draws 0.67 to 0.84.
- **Primary 2, not met:** `MI-D`'s mean coverage reaches 0.90 on 8 of 15, below the 10 required. Two different
  failures are inside that count:
  - **The control fails on two tables.** With the truth in every gap, `CD` covers 0.574 on the three `kc2` variants
    (slopes 0.280, means 0.854) and about 0.88 on the three `biodeg` ones (slopes about 0.84). On those six the
    interval itself undercovers before any gap is filled (a t interval with the least-squares variance, on a test fold
    of about 100 to 200 rows; the cause was not examined), so they cannot say whether the draws are proper. `MI-D` is at or above
    `CD` on all six.
  - **Where the control holds** (`vehicle` 0.914, `credit-g` 0.941, `pendigits` 0.981), `MI-D` reaches 0.90 at 20 and
    40% missing on all three tables (0.928 to 0.983) and misses it at 80% on all three (0.771, 0.841, 0.669). At 80%
    the means and shares are still covered (means 0.933, 0.933, 0.975; `credit-g`'s shares 0.928); the slopes are not
    (0.600, 0.544, 0.342). The draws are independent per cell, so they add no relation between columns, and H's fills
    already move the estimates by 3 to 5 complete-data standard errors there.
- **Secondary:**
  - against `MICE`, `MI-D` is better on 8 and worse on none. `MICE` falls below the single fill on `pendigits_40nan`
    (0.759) and `biodeg_40nan` (0.827), through its means (0.787, 0.769);
  - the oracle pool `MI-O` is better than `MI-D` on 4 (the three `kc2`, `biodeg_40nan`) and worse on 2
    (`vehicle_80nan`, `pendigits_80nan`), where its binned residuals give narrower intervals (width 1.90 against 2.80;
    1.47 against 2.30). The validation pool is therefore not what limits `MI-D` at 80%;
  - λ, the between-imputation share of the variance, goes from 0.05–0.25 at 20% to 0.50–0.80 at 80% under `MI-D`;
  - the width and shift columns are unreadable on `biodeg_20nan`, `biodeg_40nan` and for `MICE` on `biodeg_80nan`: a
    mean of ratios to `CD`'s half-width, which is close to zero for some estimand of that table. No rule uses them.
- **Reading, by the ticket's own list:** primary 1 holds and primary 2 fails, so the residual draws improve on a
  single fill but are not proper: at 80% missing they do not cover relations between columns. The pre-agreed next
  step is the GPU study with draws from the model at fill time, pre-registered on its own. Two things this study hands
  it: an interval whose control covers on every table it reads (here it does not on `kc2` and `biodeg`), and draws
  that are joint across a row's gaps, since per-cell noise is what fails here.
- No ADR is proposed. Nothing was rerun; three seeds, 15 variants, M = 20, as fixed.
