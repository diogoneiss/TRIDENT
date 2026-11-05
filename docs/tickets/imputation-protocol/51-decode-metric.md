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

## Amendment (2026-11-05, 19:29 GMT-3)

gorgona8, three queues on one GPU, took about 30 minutes a kr-vs-kp cell against medusa1's 4 to 8. On the user's
decision ("cancele as runs da gorgona, use a medusa o máximo possível") gorgona8's queues were stopped at 19:29 GMT-3
and seed 42's 24 cells run on medusa1 too, on four queues added beside the four that hold seeds 7 and 13. gorgona8 had
finished three arm-A cells (kr-vs-kp at 40, 60 and 80nan, seed 42) and was inside their arm-B cells; all of that is
discarded and those pairs are trained whole on medusa1, so every pair still has both arms on one machine (ADR 0035).
The cells, the readout and the rule are unchanged; the reading runs on medusa1's ledgers.

## The tag's backfill (2026-11-05, applied 18:53 to 19:54 GMT-3, verified 20:03 GMT-3)

`scripts/backfill_decode_metric_tag.py --apply`, through the leader's server: `decode_metric=impute_score`, marked
`decode_metric_backfilled=true`, on 3690 imputation runs (parents and Optuna trials, 2286 trees) and their 3675 active
mirrors. A second dry run finds nothing; no active mirror lacks the tag; no run of another task was touched. The 15
stamped runs without an active mirror are 11 failed runs, 3 a kill left `RUNNING`, and one trial of a deleted study.
The study's own cells run with MLflow off and carry no tag.

## Outcome (2026-11-05, 20:22 GMT-3; 72 of 72 cells on medusa1, 0 failed, the last queue done 20:12 GMT-3)

Read with `decode_metric_report.py` on medusa1, both arms from their five fold ledgers. Unedited tally, B − A over 15
(seed, fold) replicates a variant, K's induced fills; the table is built from the reader's lines by script:

| variant | B − A `dist_score` | B − A `impute_score` | A: `dist_score` / `impute_score` | dispersion A → B | mean kept epoch A → B |
|---|---|---|---|---|---|
| kr-vs-kp_40nan | −0.0410 [−0.0607 −0.0213] better | +0.0029 [−0.0073 +0.0131] no difference | 0.2801 / 0.7239 | n/a (no numerical column) | 260 → 259 |
| kr-vs-kp_60nan | −0.1676 [−0.1979 −0.1374] better | +0.0299 [+0.0237 +0.0360] worse | 0.4211 / 0.8668 | n/a (no numerical column) | 103 → 305 |
| kr-vs-kp_80nan | −0.6013 [−0.6552 −0.5474] better | +0.1581 [+0.1483 +0.1678] worse | 0.8018 / 0.9775 | n/a (no numerical column) | 19 → 286 |
| spambase_40nan | −0.0258 [−0.0362 −0.0153] better | +0.0009 [−0.0025 +0.0043] no difference | 0.5350 / 0.9014 | 0.479 → 0.510 | 271 → 356 |
| spambase_60nan | −0.0742 [−0.1004 −0.0480] better | +0.0017 [+0.0002 +0.0031] worse | 0.6226 / 0.9414 | 0.406 → 0.424 | 138 → 338 |
| spambase_80nan | −0.2027 [−0.2357 −0.1697] better | +0.0200 [+0.0160 +0.0240] worse | 0.7526 / 0.9872 | 0.264 → 0.448 | 56 → 392 |
| credit-g_40nan | −0.1525 [−0.1945 −0.1104] better | +0.0300 [+0.0188 +0.0412] worse | 0.5539 / 0.9209 | 0.472 → 0.586 | 77 → 267 |
| credit-g_60nan | −0.2892 [−0.3607 −0.2178] better | +0.0739 [+0.0599 +0.0878] worse | 0.6162 / 0.9886 | 0.396 → 0.646 | 56 → 323 |
| credit-g_80nan | −0.6398 [−0.6695 −0.6101] better | +0.1619 [+0.1519 +0.1719] worse | 0.9222 / 1.0066 | 0.074 → 0.758 | 10 → 262 |
| vehicle_60nan | −0.0227 [−0.0334 −0.0120] better | +0.0011 [−0.0051 +0.0073] no difference | 0.2307 / 0.6392 | 0.828 → 0.856 | 219 → 317 |
| biodeg_60nan | −0.0069 [−0.0116 −0.0023] better | −0.0012 [−0.0033 +0.0009] no difference | 0.3195 / 0.7589 | 0.743 → 0.748 | 333 → 368 |
| kc2_60nan | −0.0257 [−0.0412 −0.0102] better | −0.0022 [−0.0109 +0.0064] no difference | 0.2661 / 0.5750 | 0.897 → 0.913 | 201 → 343 |

```text
Pooled B − A dist_score: -0.1875 [-0.1941, -0.1808] (NB [-0.2026, -0.1723])
Pooled B − A impute_score: +0.0397 [+0.0374, +0.0421] (NB [+0.0343, +0.0452])
Pooled B − A wd_num_z: -0.0960 [-0.1023, -0.0896] (NB [-0.1105, -0.0814])
Pooled B − A tv_cat: -0.1074 [-0.1133, -0.1016] (NB [-0.1209, -0.0940])
Pooled B − A dispersion: +0.1478 [+0.1386, +0.1570] (NB [+0.1267, +0.1689])
Primary: dist_score improves (interval below 0: True); impute_score cost not inside +0.01 (upper bound +0.0421, lower +0.0374) -> improves the distribution at a cost in point accuracy
```

- **By the rule: it improves the distribution at a cost in point accuracy.** `dist_score` is better on 12 of 12
  variants and by −0.19 pooled; both of its parts move (Wasserstein −0.096, total variation −0.107), and the
  dispersion, the fills' sd over the truths', rises by 0.15 toward 1 (the reader's header calls every score "lower is
  better"; for the dispersion 1 is the truth's). `impute_score` is +0.040 worse pooled, four times the margin.
- **The cost is at heavy missingness on three tables.** `impute_score` is worse on 7 variants
  (kr-vs-kp_60nan, kr-vs-kp_80nan, spambase_60nan, spambase_80nan, credit-g_40nan, credit-g_60nan, credit-g_80nan) and no different on 5 (kr-vs-kp_40nan, spambase_40nan, vehicle_60nan, biodeg_60nan, kc2_60nan), where the distribution improves for
  nothing measurable. spambase_60nan's flag is a cost of +0.002, inside the margin; the six others cost +0.020 to
  +0.162.
- **What the two selectors keep differs most where the cost is largest.** At 80nan `impute_score` keeps a very early
  epoch (credit-g 10, kr-vs-kp 19, spambase 56 of 450) whose fills are almost the mean/mode fill: on credit-g_80nan
  `impute_score` 1.007, `dist_score` 0.92, dispersion 0.07. `dist_score` keeps a late epoch (262 to 392) whose fills
  spread like the truths (dispersion 0.45 to 0.76) and are worse than the mean/mode fill cell by cell (credit-g_80nan:
  about 1.17). Neither end is a good output there: one is a collapsed fill that ties the naive one, the other a
  realistic-looking fill that loses to it. This is ADR 0025's caveat on the 80nan wins, measured.
- **What follows, by the ticket:** recorded as a trade-off, with its size. The flag stays an option, off by default;
  no default change and no ADR. Not read here and open: a criterion between the two (for instance the best
  `dist_score` among the epochs within a tolerance of the best `impute_score`), KC, the large tables, fresh seeds.
