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
