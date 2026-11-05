# Choosing the decode checkpoint by a distributional score (E145; pre-registered 2026-11-05)

This ticket was written and committed before the study's first cell (18:49 GMT-3).

- **Why:** the user asked (2026-11-05) for an imputation metric other than `impute_score`, taken from the
  neural-imputation literature, and for a flag to optimise by it. The literature's complaint about cell-by-cell
  error (Shadbahr et al. 2023, Näf et al. 2023 and Zhou et al. 2024 in the vault's review,
  `_raw/litreview-2026-10/D-benchmarks-theory.md`) is that it rewards fills shrunk to the centre; its answer is to
  score the distribution of the fills.
- **What the project already has:** ADR 0025's offline panel reads exactly that for the model, every baseline and
  TabPFN (`metric_panel.py`: per-column Wasserstein distance, category total variation, dispersion), on E40, E44,
  E45/E46 and E56. The per-kind ratios never disagree with `impute_score`; the disagreements are distributional
  only; at 80nan on letter and pendigits the model's fills are more shrunk than the boosters'; against TabPFN, K is
  better on dispersion on 14 of 15 variants and on Wasserstein on 11 (E56). So "does the standing survive a
  distributional metric" is not rerun here. What nobody has tried is *selecting* by the distributional score.
- **The premise that `mlruns` holds the fills, checked:** this machine's `mlruns` has 4478 fold ledgers, all on
  `best_fold` / `worst_fold` children (2235 each): two of a run's five folds, chosen by `impute_score`, with the
  model's fills alone. A paired reading on them would be a selected sample scored by the metric under question. A
  queue study keeps all five ledgers of each cell, so the study trains its own cells.
- **Code** (this commit; the flag off keeps every run as it was):
  - `src/training/distribution_metrics.py`: `wd_num_z` (the mean over numerical columns of the 1-D Wasserstein
    distance between fills and truths, scaled units), `tv_cat` (the mean over categorical columns of the total
    variation between their category frequencies), both the panel's own definitions, and `dist_score`: each kind as
    a ratio to the mean/mode fill's own distance, weighted by its cells, as `impute_score` is built. The naive fill
    scores 1, a perfect one 0.
  - `DECODE_METRIC` / `--decode_metric` (`impute_score`, the default, or `dist_score`): the score the `induced` and
    `masked` checkpoints rank the epochs by, and the validation key an imputation search ranks its trials by. A run
    that asks for `dist_score` logs it, with its two parts, for its own cells. `score_cells` is untouched, so ADR
    0025 stands and the baseline cache stays valid.
  - The tag `decode_metric` on every imputation parent and trial; `scripts/backfill_decode_metric_tag.py` stamps
    `impute_score` on the runs recorded before it (through the leader, after this study's cells).
  - `scripts/experiments/decode_metric_run.py` (cells) and `decode_metric_report.py` (the reading).
  - Unit tests: the score's values on a toy table, its blindness to rows, the per-epoch tensor score equal to the
    pandas path at the kept epoch for both checkpoints, the flag's default and scope, the backfill.
- **A real run before the study** (gorgona8, not a study cell): one pair, kc2_60nan at seed 42, both
  arms, in a state directory of its own. Both cells wrote their five ledgers on the same cells; arm B kept other
  epochs than arm A (381/449/330/323/261 against 223/191/331/373/369); arm B's logged `impute/induced/dist_score`
  equals the reader's offline value fold by fold, and arm A's logged `impute_score` the reader's. 423 and 430 s a
  cell. The unit suite passes (496), with the typing gate, and both regression fixtures are unmoved (vehicle_00nan;
  credit-g_20nan imputation on the CPU).

## Design and rule (fixed before the first cell)

- **Cells:** E40's 12 development variants at 40nan and above (kr-vs-kp, spambase, credit-g at 40/60/80nan;
  vehicle, biodeg, kc2 at 60nan) × seeds 42/7/13 × two arms = 72: seed 42 on gorgona8 (24 cells), seeds 7 and 13 on medusa1
  (48), so every pair's two arms are on one machine (ADR 0035). Each cell is a default imputation run of the checkout, five folds, MLflow off, with
  `--decode_calibration none` in both arms so the fold ledgers hold K, the kept checkpoint's fills. Arm A keeps the
  epoch by `impute_score` (today's K); arm B adds `--decode_metric dist_score`. Both arms of a pair run in one
  queue, on one GPU: the training trajectory is the same and only the kept epoch may differ.
- **Why no stored cell is reused:** the store keeps no fills for three of five folds, and arm A must share arm B's
  trajectory (same commit, torch and machine). The rerun is forced.
- **Readout:** K's induced fills, read from the five fold ledgers of each cell; both arms scored by one offline
  reader (`distribution_scores`, `score_cells`), the mean/mode fills from the rebuilt folds.
- **Primary, two conditions, both needed** (an intersection, so no multiplicity correction), B − A pooled over the
  12 variants per (seed, fold) replicate, 15 replicates, plain 95% t interval:
  1. `dist_score`: the interval is below 0 (the selector improves its own target on the test split);
  2. `impute_score`: the interval's upper bound is below +0.01, the project's smallest effect of interest (no
     relevant cost in point accuracy).
  **Supported** if both hold; **improves the distribution at a cost** if only the first; otherwise **not
  supported**.
- **Beside, descriptive:** per-variant lines (fold-pair intervals as flags), the Wasserstein and total-variation
  parts, the dispersion (sd of fills over sd of truths), the folds where the kept checkpoint differs,
  Nadeau–Bengio intervals.
- **What follows:** *supported* → a proposal to the user: confirmation on fresh seeds, the confirmation set and the
  large tables, and only then the question of the selector (ADR 0025 decision 1 is the user's to amend).
  *At a cost* → recorded as a trade-off, with its size. *Not supported* → the flag stays an option, off. No default
  changes here in any case, and no ADR.
- **Threats stated now:**
  - `dist_score` is marginal, a column at a time: blind to a row's joint distribution, and perfect for the right
    values on the wrong rows (a unit test shows it). It is read beside `impute_score`, never alone.
  - Selection only picks among the epochs of a trajectory trained on squared error and cross-entropy; it cannot
    make the model draw from a distribution.
  - K alone. KC, the default output from 30% missing (ADR 0027), is a calibration fitted for point error on top of
    K and is not read here.
  - Small validation splits make the per-epoch score noisy (kc2 has 522 rows).
  - 12 small and medium tables; no large one, where the panel saw the strongest shrinkage (letter and pendigits
    at 80nan). One seed set.
- **Cut from this step, by name:** energy distance and I-Scores (Näf et al.) and a sliced Wasserstein distance on
  rows (Shadbahr et al.), joint metrics that need full rows or multiple imputations; downstream utility; CRPS
  (T43's ground); a hyperparameter search by the metric (the flag supports it, none is run); the baselines'
  `dist_score` (the panel already reads their Wasserstein distance and total variation offline).
- **Cost, estimated now:** kc2 took 7 minutes a cell on gorgona8 in the real run above; spambase and kr-vs-kp cost
  several times that. About 3 hours, gorgona8 on 3 queues and medusa1 on 4.
- **Read only with all 72 cells done.**
