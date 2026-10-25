# A test-time re-masking ensemble: does averaging passes help, and does their spread rank the errors? (E75; pre-registered 2026-10-24)

This ticket was written and committed before the study's first cell.

- **Why:** T36 (`docs/TASKS.md`). The imputer gives one fill per cell and no uncertainty. The decode
  stage trains on masks drawn over observed cells, so a test row with some observed cells also shown
  as `[MASK]` is a row shape it has seen. Averaging such passes may sharpen the fill, and their spread
  may say which cells it gets wrong.
- **The user's go:** "Aprovo T36 e T37" (2026-10-23, about 00:03 GMT-3), and the code authorization
  for `src/`: "autorizo o 1, vá" (2026-10-24, about 18:15 GMT-3), on the in-run flag route.
- **Code (`src/`, behind a flag that is off by default):**
  - `--score_remask` (`TrainingRequest.score_remask`), a diagnostic like `--score_checkpoints`; it never
    ranks a fold and changes no training. `src/training/decoding.py::_remask_induced`;
    `EpochMasker.apply` (`src/embedder.py`), split out of `draw`.
  - The passes draw from their own `numpy` generator, seeded by (seed, fold), so the global stream the
    later folds train on does not move. Unit tests: with the flag on, every metric, ledger row and
    weight the run had without it is unchanged, and so is the global stream after the fold; at a
    re-mask rate of 0 the averaged fill equals the single pass exactly.
  - `scripts/experiments/remask_ensemble.py` (cells, queue, reading) and `run_remask_queue.sh`.

## Design (fixed before the first cell)

- **The passes:** R = 20 at the run's kept checkpoint (K) and, with `--score_checkpoints`, at every
  other criterion's (H is the loss's). Each pass shows the test rows' gaps as `[MASK]` and also hides
  each observed cell with probability `PROB_MASCARA` × (the row's observed share), the decode stage's
  own training rule (`draw_dynamic_mask`). Unlike training, no cell is forced per row: every scored row
  already holds a gap. The same 20 draws serve every checkpoint of a fold.
- **The fill:** a number is the mean of its 20 passes; a category is the most voted one (ties to the
  smallest id), with the vote share as its confidence. Only the gaps are scored, on the same cells as
  the single pass.
- **The spread** (ledger column `spread`): the passes' standard deviation of a number (scaled units),
  or 1 − the vote share of a category.
- **ADR 0027:** these rates are all calibrated, so the headline is KC. The passes are read
  uncalibrated against K and H, and once calibrated: a calibration fitted on the validation rows'
  own gaps filled by the same passes, against KC.
- **Logged:** `impute/induced/remask/*`, `impute/induced/loss_checkpoint_remask/*` (and
  `masked_checkpoint_remask/*`), `impute/induced/remask_calibrated/*`. The fold ledger gains the
  populations `induced_remask`, `induced_remask_loss_checkpoint`, `induced_remask_masked_checkpoint`
  and `induced_uncalibrated` (K's single pass, since the ledger's `induced` rows hold KC).
- **Cells:** E44's 15 confirmation variants (pendigits, letter, electricity at 40/60/80nan; vehicle,
  biodeg, kc2 at 40/80nan), seeds 707, 808 and 909, today's default configuration, five folds:
  45 cells.
- **Running:** four GPU queues, MLflow tag `experiment=remask-ensemble-2026-10-24`, state in
  `/scratch2/diogoneiss/trident-2026-10-24-remask/`. About 8–10 h (E44 took 8–9 h on four queues), so
  to about 04:00–05:00 GMT-3 on 2026-10-25; no cutoff, every cell runs.

## Measures and reading, fixed now (`remask_ensemble.py --tally`)

**(a) Accuracy,** induced `impute_score`, 15 fold pairs, the verdict rule with the 1% SESOI:
remask − single pass under K and under H (uncalibrated), and remask calibrated − KC (descriptive).

- **Primary, over the 15 variants:**
  - **supported** (averaging is worth proposing as an output): better on 8 or more and worse on at
    most 1, under both K and H;
  - **refuted:** better on at most 3 under K, or worse on 5 or more under K;
  - **mixed:** otherwise.

**(b) Uncertainty** (descriptive), from the ledger, per fold, on K's and on H's passes:

- `num_rank`: Spearman between a numerical cell's spread and its absolute error;
- `cat_rank`: Spearman between a categorical cell's spread and its miss (0/1); beside it the single
  pass's own signal, 1 − its confidence;
- `coverage90`: the share of numerical cells within 1.645 spreads of the truth (90% nominal).
- A variant **shows a signal** on a measure when the 95% t-interval of its usable folds' mean lies
  above 0. A fold whose spread or error is constant gives no value and is left out, counted.
- **Worth proposing as an uncertainty output:** `num_rank` signals on 12 or more of the 15 under both
  K and H. Otherwise it stays descriptive.

**(c) Dispersion** (descriptive; ADR 0025's panel measure, the fills' SD over the truths' SD per
numerical column): remask − K, verdict rule without a margin, counted lower/higher. Averaging shrinks
fills: where (a) is better and the dispersion is lower on the same variant, the gain is reported as
shrinkage, and E56's dispersion lead over TabPFN is read against it.

**Also printed:** K against E44's run of the same cell and seed, largest fold difference (a
reproduction check, descriptive: commits landed since 2026-10-17).

- **What each reading leads to:** supported, a proposed ADR for an opt-in averaged output, for the
  user; refuted or mixed, reported, and the uncertainty reading (b) is judged on its own.

## Launch checks (2026-10-24, before the first cell)

- **Tests:** the unit suite with the mypy gate (439 passed), and both integration fixtures
  (`credit-g_20nan` imputation on the CPU, `vehicle_00nan` classification), unchanged.
- **Flag on against flag off,** `vehicle_40nan` at 707, five folds, MLflow off, the cell's exact
  arguments (scratch output, not this study's state):
  - all 96 returned metrics identical to the last digit, and 20 new ones;
  - the fold table identical on its 100 columns but the three timing columns (+0.3 s a fold of
    decode time), 20 new columns;
  - every fold ledger's existing rows identical; the five new populations hold 1,219 cells each on
    fold 1, the induced cells' count.
  - For scale only (not a reading): K 0.4914, remask 0.4951; H 0.4944, H remask 0.4980; fold 1–5
    `num_rank` 0.23–0.39, `coverage90` 0.51–0.57.

## Outcome (2026-10-25, 02:38 GMT-3)

**The run:** launched 2026-10-24, 18:31 GMT-3, on the pre-registration commit (5dc83bd); 45 of 45
cells done by 02:35 GMT-3 on 2026-10-25, none failed. K equals E44's run of the same cell and seed on
every fold of all 45 cells (largest difference 0), so training has not moved since 2026-10-17.

**The printout** (`scripts/experiments/remask_ensemble.py --tally`, unedited):

```text
- `pendigits_40nan`: K 0.4259, remask − single +0.0209 [+0.0195, +0.0222] worse; H 0.4269, remask − single +0.0209 [+0.0194, +0.0223] worse; KC 0.4247, remask − single +0.0221 [+0.0209, +0.0232] worse
  - uncertainty, K: num_rank +0.373 [+0.369, +0.377] (15 folds), coverage90 +0.631 [+0.629, +0.634] (15 folds); H: num_rank +0.370 [+0.365, +0.375] (15 folds), coverage90 +0.629 [+0.626, +0.632] (15 folds)
  - dispersion, remask − K: -0.0997 [-0.1008, -0.0986] lower
- `pendigits_60nan`: K 0.5856, remask − single +0.0018 [+0.0011, +0.0024] worse; H 0.5893, remask − single +0.0024 [+0.0018, +0.0029] worse; KC 0.5820, remask − single +0.0054 [+0.0050, +0.0059] worse
  - uncertainty, K: num_rank +0.254 [+0.247, +0.262] (15 folds), coverage90 +0.511 [+0.508, +0.514] (15 folds); H: num_rank +0.245 [+0.234, +0.255] (15 folds), coverage90 +0.503 [+0.496, +0.509] (15 folds)
  - dispersion, remask − K: -0.0850 [-0.0860, -0.0840] lower
- `pendigits_80nan`: K 0.8304, remask − single -0.0019 [-0.0026, -0.0011] better; H 0.8337, remask − single -0.0013 [-0.0024, -0.0001] better; KC 0.8263, remask − single +0.0011 [+0.0008, +0.0013] worse
  - uncertainty, K: num_rank -0.054 [-0.059, -0.048] (15 folds), coverage90 +0.209 [+0.202, +0.217] (15 folds); H: num_rank -0.058 [-0.069, -0.048] (15 folds), coverage90 +0.204 [+0.193, +0.215] (15 folds)
  - dispersion, remask − K: -0.0480 [-0.0495, -0.0465] lower
- `letter_40nan`: K 0.5598, remask − single +0.0233 [+0.0224, +0.0242] worse; H 0.5608, remask − single +0.0235 [+0.0226, +0.0243] worse; KC 0.5584, remask − single +0.0247 [+0.0240, +0.0254] worse
  - uncertainty, K: num_rank +0.217 [+0.212, +0.221] (15 folds), coverage90 +0.562 [+0.560, +0.564] (15 folds); H: num_rank +0.216 [+0.211, +0.221] (15 folds), coverage90 +0.559 [+0.557, +0.561] (15 folds)
  - dispersion, remask − K: -0.1442 [-0.1456, -0.1427] lower
- `letter_60nan`: K 0.7175, remask − single +0.0012 [+0.0005, +0.0019] worse; H 0.7241, remask − single +0.0030 [+0.0020, +0.0039] worse; KC 0.7144, remask − single +0.0043 [+0.0038, +0.0049] worse
  - uncertainty, K: num_rank +0.139 [+0.133, +0.146] (15 folds), coverage90 +0.429 [+0.427, +0.431] (15 folds); H: num_rank +0.129 [+0.121, +0.136] (15 folds), coverage90 +0.412 [+0.403, +0.420] (15 folds)
  - dispersion, remask − K: -0.1038 [-0.1044, -0.1031] lower
- `letter_80nan`: K 0.8902, remask − single -0.0007 [-0.0011, -0.0003] better; H 0.8921, remask − single -0.0016 [-0.0028, -0.0005] better; KC 0.8879, remask − single +0.0008 [+0.0007, +0.0010] worse
  - uncertainty, K: num_rank -0.007 [-0.016, +0.001] (15 folds), coverage90 +0.172 [+0.167, +0.178] (15 folds); H: num_rank -0.004 [-0.012, +0.005] (15 folds), coverage90 +0.180 [+0.169, +0.190] (15 folds)
  - dispersion, remask − K: -0.0399 [-0.0408, -0.0389] lower
- `electricity_40nan`: K 0.7745, remask − single +0.0000 [-0.0175, +0.0175] none; H 0.7727, remask − single +0.0004 [-0.0173, +0.0181] none; KC 0.7650, remask − single +0.0048 [-0.0157, +0.0253] none
  - uncertainty, K: num_rank +0.364 [+0.356, +0.373] (15 folds), cat_rank +0.079 [+0.065, +0.092] (15 folds), coverage90 +0.474 [+0.466, +0.483] (15 folds); H: num_rank +0.361 [+0.350, +0.373] (15 folds), cat_rank +0.081 [+0.069, +0.093] (15 folds), coverage90 +0.477 [+0.468, +0.485] (15 folds); single pass, 1 − confidence: cat_rank +0.248 [+0.242, +0.254] (15 folds)
  - dispersion, remask − K: -0.1404 [-0.1691, -0.1117] lower
- `electricity_60nan`: K 0.8298, remask − single +0.0013 [-0.0028, +0.0055] equivalent; H 0.8389, remask − single -0.0014 [-0.0069, +0.0041] equivalent; KC 0.8293, remask − single +0.0014 [-0.0032, +0.0060] equivalent
  - uncertainty, K: num_rank +0.209 [+0.198, +0.220] (15 folds), cat_rank +0.015 [+0.007, +0.022] (15 folds), coverage90 +0.342 [+0.329, +0.355] (15 folds); H: num_rank +0.207 [+0.200, +0.215] (15 folds), cat_rank +0.020 [+0.013, +0.027] (15 folds), coverage90 +0.340 [+0.333, +0.348] (15 folds); single pass, 1 − confidence: cat_rank +0.173 [+0.164, +0.182] (15 folds)
  - dispersion, remask − K: -0.0829 [-0.0894, -0.0765] lower
- `electricity_80nan`: K 0.9002, remask − single -0.0033 [-0.0080, +0.0014] equivalent; H 0.9009, remask − single -0.0021 [-0.0078, +0.0035] equivalent; KC 0.8986, remask − single -0.0018 [-0.0057, +0.0021] equivalent
  - uncertainty, K: num_rank +0.057 [+0.049, +0.065] (15 folds), cat_rank -0.027 [-0.032, -0.023] (15 folds), coverage90 +0.166 [+0.158, +0.174] (15 folds); H: num_rank +0.047 [+0.037, +0.056] (15 folds), cat_rank -0.033 [-0.042, -0.023] (15 folds), coverage90 +0.156 [+0.151, +0.162] (15 folds); single pass, 1 − confidence: cat_rank +0.094 [+0.087, +0.101] (15 folds)
  - dispersion, remask − K: -0.0745 [-0.1110, -0.0380] lower
- `vehicle_40nan`: K 0.4970, remask − single +0.0024 [-0.0021, +0.0069] none; H 0.5060, remask − single +0.0039 [-0.0008, +0.0086] none; KC 0.4976, remask − single +0.0043 [+0.0002, +0.0084] worse
  - uncertainty, K: num_rank +0.339 [+0.311, +0.367] (15 folds), coverage90 +0.534 [+0.522, +0.546] (15 folds); H: num_rank +0.294 [+0.252, +0.336] (15 folds), coverage90 +0.520 [+0.503, +0.538] (15 folds)
  - dispersion, remask − K: -0.0506 [-0.0553, -0.0460] lower
- `vehicle_80nan`: K 0.8088, remask − single -0.0046 [-0.0062, -0.0029] better; H 0.8167, remask − single -0.0051 [-0.0069, -0.0033] better; KC 0.7872, remask − single +0.0001 [-0.0010, +0.0012] equivalent
  - uncertainty, K: num_rank -0.021 [-0.069, +0.027] (15 folds), coverage90 +0.241 [+0.224, +0.259] (15 folds); H: num_rank -0.036 [-0.083, +0.012] (15 folds), coverage90 +0.239 [+0.217, +0.261] (15 folds)
  - dispersion, remask − K: -0.0348 [-0.0376, -0.0321] lower
- `biodeg_40nan`: K 0.6654, remask − single -0.0017 [-0.0078, +0.0043] none; H 0.6652, remask − single -0.0035 [-0.0095, +0.0024] none; KC 0.6606, remask − single +0.0045 [+0.0001, +0.0089] worse
  - uncertainty, K: num_rank +0.650 [+0.625, +0.675] (15 folds), coverage90 +0.651 [+0.632, +0.671] (15 folds); H: num_rank +0.663 [+0.649, +0.677] (15 folds), coverage90 +0.659 [+0.644, +0.673] (15 folds)
  - dispersion, remask − K: -0.1114 [-0.1179, -0.1048] lower
- `biodeg_80nan`: K 0.9712, remask − single -0.0085 [-0.0110, -0.0060] better; H 0.9825, remask − single -0.0072 [-0.0101, -0.0043] better; KC 0.9466, remask − single -0.0013 [-0.0022, -0.0004] better
  - uncertainty, K: num_rank +0.475 [+0.409, +0.540] (15 folds), coverage90 +0.285 [+0.244, +0.327] (15 folds); H: num_rank +0.393 [+0.290, +0.496] (15 folds), coverage90 +0.235 [+0.173, +0.297] (15 folds)
  - dispersion, remask − K: -0.0420 [-0.0482, -0.0357] lower
- `kc2_40nan`: K 0.5075, remask − single -0.0096 [-0.0216, +0.0023] none; H 0.5190, remask − single -0.0134 [-0.0266, -0.0002] better; KC 0.5071, remask − single -0.0083 [-0.0193, +0.0028] none
  - uncertainty, K: num_rank +0.660 [+0.632, +0.687] (15 folds), coverage90 +0.526 [+0.500, +0.553] (15 folds); H: num_rank +0.645 [+0.624, +0.666] (15 folds), coverage90 +0.538 [+0.518, +0.559] (15 folds)
  - dispersion, remask − K: -0.0316 [-0.0470, -0.0163] lower
- `kc2_80nan`: K 0.6866, remask − single -0.0051 [-0.0114, +0.0011] none; H 0.7107, remask − single -0.0068 [-0.0132, -0.0003] better; KC 0.6793, remask − single -0.0014 [-0.0056, +0.0027] equivalent
  - uncertainty, K: num_rank +0.426 [+0.389, +0.464] (15 folds), coverage90 +0.311 [+0.294, +0.328] (15 folds); H: num_rank +0.401 [+0.350, +0.452] (15 folds), coverage90 +0.291 [+0.267, +0.314] (15 folds)
  - dispersion, remask − K: -0.0492 [-0.0615, -0.0369] lower

- remask − single, K: better 4 / no difference 5 / equivalent 2 / worse 4 (fold-pair t, seed agreement)
- remask − single, H: better 6 / no difference 3 / equivalent 2 / worse 4 (fold-pair t, seed agreement)
- remask − single, KC: better 1 / no difference 2 / equivalent 4 / worse 8 (fold-pair t, seed agreement)
- num_rank, K: signal on 12 of 15 variants with the kind
- cat_rank, K: signal on 2 of 3 variants with the kind
- num_rank, H: signal on 12 of 15 variants with the kind
- cat_rank, H: signal on 2 of 3 variants with the kind
- dispersion, remask − K: lower 15, higher 0, of 15
- K against E44's run of the same cell, largest fold difference: 0.00e+00 over 45 cells

Primary: remask better than the single pass on 4 (K) and 6 (H) of 15, worse on ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan', 'letter_60nan'] (K) and ['pendigits_40nan', 'pendigits_60nan', 'letter_40nan', 'letter_60nan'] (H) -> mixed (fold-pair t, seed agreement)
```

**Reading, by the rule fixed above:**

- **(a) Accuracy: mixed.** Under K the passes are better on 4 and worse on 4 (H: better 6, worse 4).
  They are worse on pendigits and letter at 40 and 60nan (up to +0.023), and better at 80nan on
  pendigits, letter, vehicle and biodeg (−0.0007 to −0.0085), and on kc2 under H.
  - **The gain is shrinkage:** the dispersion is lower on all 15, and calibrated, the passes are
    worse than KC on 8 and better on 1 (biodeg_80nan, −0.0013). ADR 0027's calibration already
    takes what averaging gives at 80nan.
- **(b) Uncertainty: the numerical spread meets the bar.** `num_rank` signals on 12 of 15 under K
  and under H, the pre-stated minimum. The three without a signal are the 80nan numerical tables
  (pendigits −0.054, letter −0.007, vehicle −0.021); electricity_80nan is +0.057. It is strongest on
  the small tables (biodeg_40nan 0.65, kc2_40nan 0.66) and 0.14–0.37 on the large ones at 40–60nan.
  - **It ranks, but it is not an interval:** `coverage90` is 0.17–0.66, never near 0.90. The spread
    is too narrow by a factor that grows with missingness.
  - **Categorical:** the vote share signals on 2 of 3 (electricity 40 and 60nan), but the single
    pass's own 1 − confidence ranks the misses better on all three (0.25, 0.17, 0.09 against 0.08,
    0.02, −0.03).
- **(c) Dispersion:** lower on 15 of 15 (−0.03 to −0.14), so the average spends part of the spread
  E56 credited the model with.

- **What it leads to:**
  - Averaging is not proposed as an output.
  - The numerical spread is worth proposing as an opt-in uncertainty output, a ranking signal and not
    an interval, by the rule fixed above. Whether to write that ADR is the user's call.
  - Categorical uncertainty is better read from the single pass's confidence, which every run
    already logs.

## Post hoc: cells or columns? (2026-10-25, 09:37 GMT-3; not pre-registered)

After the Outcome, for the ADR the user asked for (ADR 0030, proposed). The pooled `num_rank` could
reflect only which columns are hard. `remask_ensemble.py --within-column` computes Spearman within each
numerical column, averaged over columns per fold (`within`), and each column's mean spread against its
RMSE (`columns`). Printout, unedited:

```text
- `pendigits_40nan`: K within +0.336 [+0.332, +0.340] (15); K columns +0.791 [+0.773, +0.809] (15); H within +0.333 [+0.328, +0.339] (15); H columns +0.784 [+0.767, +0.800] (15)
- `pendigits_60nan`: K within +0.228 [+0.221, +0.235] (15); K columns +0.619 [+0.582, +0.655] (15); H within +0.220 [+0.210, +0.229] (15); H columns +0.559 [+0.502, +0.617] (15)
- `pendigits_80nan`: K within -0.055 [-0.061, -0.050] (15); K columns -0.617 [-0.693, -0.542] (15); H within -0.059 [-0.065, -0.052] (15); H columns -0.586 [-0.694, -0.478] (15)
- `letter_40nan`: K within +0.205 [+0.201, +0.209] (15); K columns +0.512 [+0.443, +0.581] (15); H within +0.204 [+0.199, +0.208] (15); H columns +0.495 [+0.425, +0.565] (15)
- `letter_60nan`: K within +0.129 [+0.122, +0.136] (15); K columns +0.331 [+0.246, +0.416] (15); H within +0.120 [+0.113, +0.126] (15); H columns +0.046 [-0.110, +0.202] (15)
- `letter_80nan`: K within -0.005 [-0.012, +0.002] (15); K columns -0.781 [-0.824, -0.738] (15); H within -0.001 [-0.009, +0.006] (15); H columns -0.763 [-0.844, -0.683] (15)
- `electricity_40nan`: K within +0.205 [+0.194, +0.216] (15); K columns -0.693 [-0.887, -0.499] (15); H within +0.198 [+0.182, +0.214] (15); H columns -0.690 [-0.880, -0.501] (15)
- `electricity_60nan`: K within +0.057 [+0.049, +0.066] (15); K columns -0.529 [-0.733, -0.324] (15); H within +0.052 [+0.044, +0.061] (15); H columns -0.626 [-0.808, -0.445] (15)
- `electricity_80nan`: K within -0.075 [-0.081, -0.068] (15); K columns -0.381 [-0.553, -0.209] (15); H within -0.084 [-0.092, -0.075] (15); H columns -0.367 [-0.554, -0.179] (15)
- `vehicle_40nan`: K within +0.138 [+0.121, +0.155] (15); K columns +0.680 [+0.601, +0.759] (15); H within +0.125 [+0.105, +0.146] (15); H columns +0.629 [+0.537, +0.722] (15)
- `vehicle_80nan`: K within -0.008 [-0.031, +0.016] (15); K columns -0.029 [-0.276, +0.219] (15); H within -0.011 [-0.034, +0.012] (15); H columns -0.208 [-0.524, +0.107] (15)
- `biodeg_40nan`: K within +0.423 [+0.409, +0.437] (15); K columns +0.051 [-0.049, +0.151] (15); H within +0.432 [+0.420, +0.443] (15); H columns +0.071 [-0.024, +0.166] (15)
- `biodeg_80nan`: K within +0.219 [+0.178, +0.260] (15); K columns +0.105 [+0.019, +0.191] (15); H within +0.180 [+0.119, +0.241] (15); H columns +0.129 [+0.021, +0.238] (15)
- `kc2_40nan`: K within +0.482 [+0.453, +0.510] (15); K columns +0.704 [+0.632, +0.777] (15); H within +0.464 [+0.438, +0.491] (15); H columns +0.662 [+0.601, +0.723] (15)
- `kc2_80nan`: K within +0.257 [+0.220, +0.295] (15); K columns +0.561 [+0.426, +0.696] (15); H within +0.232 [+0.178, +0.286] (15); H columns +0.477 [+0.346, +0.608] (15)
```

- **Within columns,** the interval lies above 0 on 11 of 15 under K and under H: all 9 variants at
  40–60nan (ρ 0.06–0.48) and biodeg_80nan and kc2_80nan. Not on pendigits, letter, electricity or
  vehicle at 80nan.
- The cell-level signal is weaker than the pooled one where columns add to it (electricity_40nan
  0.21 against 0.36; biodeg_40nan 0.42 against 0.65); on electricity the column-level relation is
  negative.
