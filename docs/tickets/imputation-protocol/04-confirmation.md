# Confirmation on the 15 unused variants (E44; pre-registered 2026-10-16)

This ticket was written and committed before the first cell, and before E43's outcome was read.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2e, §3; D4, D11–D15, D17), accepted by the user on 2026-10-16. Path B, so the cells carry
  E43's deployable readouts.
- **This is the one confirmation read.** It is read once, and nothing is extended or rerun on its
  result.

## Why

Threat 2 of the 2026-10-16 review:

- **Every default was chosen on the same cells.** ADRs 0013, 0014, 0015, 0018 and 0019, and the
  file decisions, were all made on the same 21 variants, each with one MCAR draw.
- **The seed sets are spent.** 42/7/13 and 101/202/303 decided things, and E40 used 404/505/606.
- **Generalisation is untested.** The review bounds the selection optimism on these tables at
  about 0.005, against a mean gap of −0.0335, but new draws and new levels are untested.
- **15 variants on disk have never informed a decision.** 14 have no imputation run at all, and
  electricity_40nan has two runs that failed on the laptop on 2026-09-11.

## Design (fixed before launch)

### Configuration

The frozen post-E40 configuration, as a default run of the checkout at this commit reads it:

- ADRs 0013–0015 and 0019;
- ADR 0018 on pendigits_20nan and letter_20nan only.

**No inheritance (D12).** pendigits and letter at 40–80nan read the defaults, since ADR 0018 decided
per variant. None of the 15 variants has a promoted file. The runner prints every cell's resolved
configuration.

### Readouts, on one trajectory per run

The runner is `scripts/experiments/confirmation_run.py`, with `--score_checkpoints
--score_calibrated`, tagged `experiment=confirmation-2026-10-17`.

- **K**, the headline: the default checkpoint (ADR 0015), an oracle on natural data;
- **H** (`loss_checkpoint`) and **M** (`masked_checkpoint`): the deployable criteria. The
  deployable readout reported for transfer claims is the one E43's reading names;
- **KC** (`calibrated`): an oracle readout, for T20.

### Cells, splits and runs

- **Variants (15):**
  - pendigits, letter and electricity at 40, 60 and 80nan;
  - vehicle, biodeg and kc2 at 40 and 80nan.
- **Seeds:** 707, 808 and 909, unused by any run in the store. 45 cells.
- **Splits (D13):** `StratifiedKFold` with 5 folds, as for the development set, with
  EVAL_MASK_RATE 0.2.
  - No grouped splits.
  - No new OpenML tables. The limit is stated: these are new draws and levels of 6 of the 9
    development tables, not new tables.
- **Running it:** four queues (`scripts/experiments/run_confirmation_queue.sh 1..4`), one trainer
  each, with the CPU threads capped at 4.
  - Cells are assigned longest first, from the store's cell times: about 8.2 h a queue.
  - No cell starts after 18:00 GMT-3 on 2026-10-17.
  - The electricity cells start first in their queues. Both earlier electricity_40nan runs failed,
    on the laptop's 16 GB, so a crash is fixed and its cell rerun from the start, and the rerun is
    noted here.
- **The baseline cache warm-up started before launch** (22:59 GMT-3, on CPU,
  `scripts/experiments/warm_baseline_cache.py`; no score of the model is involved). A cell computes
  whatever is still missing itself, which costs it time and nothing else. The state at launch is
  noted under Launch.
- **The offline contracts:**
  - E41's arms on these folds are written to `/scratch2/diogoneiss/trident-2026-10-17-linear-baselines/csv/`
    by `linear_baselines_run.py`, after this study starts.
  - E42's tuned bar is written to `/scratch2/diogoneiss/trident-2026-10-17-baseline-budget/csv/`
    as `impute/induced/baseline/tuned_bar/impute_score`.
  - The report reads both by (variant, seed). It refuses "strong" and "best of all" while E41's arms
    are missing, and the primary while the tuned bar is.

### Opponents (definitions frozen; scores may come after the GPU pass, since baselines never see the model)

- **The bar of 5 (the weak tier):** live in every cell.
- **missForest:** live.
- **The tuned bar:** E42's grids and selection
  ([02](02-baseline-budget.md)), computed offline on these folds.
- **The strong tier's best:** missForest plus E41's arms ([01](01-linear-baselines.md)), computed
  offline on these folds with `linear_baselines_run.py`.
- **The best of all**, and **the per-type bar** (item 7b, from the fold CSVs).

All of them are read through `scripts/experiments/report_common.py`.

## Measures

**Primary,** per variant, on the induced `impute_score`, over 15 fold pairs. The verdict needs
the 95% t-interval to exclude zero and all three seeds to agree. A call without a verdict whose
interval lies inside ±1% of the opponent's mean is "equivalent" (D3).

- **A win (D4)** is K better than both the bar of 5 and the tuned bar.
- **A loss** is K worse than either.

**Reading, fixed now (D14):**

- **supported:** at least 8 wins of 15 and at most 2 losses;
- **refuted:** 5 or more losses, or 4 or fewer wins;
- **mixed:** anything else.

**Secondary, pre-registered:**

- the tallies of K against the bar of 5, the tuned bar, missForest, the strong tier, the best of
  all and the per-type bar;
- each of those at 80nan apart (6 variants), where on the development set the model sits near the
  mean/mode fill;
- across tables: the mean gap per table, with a two-sided Wilcoxon over the 6 tables, against each
  opponent. It is secondary here, although ADR 0020's draft makes it primary on the development
  set: with 6 tables the smallest p is 0.031, and these are the same tables. What this study adds
  is new draws and levels, so the per-variant tally carries the reading;
- **deployability:** K − H, K − M, M − H, and H and M against the bar;
- **T20's threshold** (KC − K):
  - "threshold 40 supported" if KC is better on some 40nan variant and K better on none of the 15;
  - "threshold 60 suggested" if the only KC verdicts are at 60nan or 80nan;
  - "refuted here" if K is better on any;
- **the metric panel** (D16), computed offline from these runs' ledgers and recomputed baseline
  fills: per-kind ratios, balanced accuracy, signed bias, per-column ratios;
- **context-free rows (D17):** the share of rows with every feature missing (16.8% on
  electricity_80nan), and the scores without them. They stay in the primary score;
- **the twin/no-twin split** on letter and kc2;
- minutes a cell.

**Analysis:** `scripts/experiments/confirmation_report.py`; `--tally` prints the reading. The
primary waits for E42's tuned bar on these folds.

## Known limits, stated in advance

- **The same 6 tables** as the development set, so this tests new draws and levels, not new
  tables.
- **K, the headline, uses the validation gaps' truth.** The deployable readouts sit beside it.
- **The tuned bar selects on the validation gaps,** the same oracle signal as K, which keeps the
  comparison symmetric.
- **One MCAR draw per level,** as for the development set.
- **The reading's thresholds (8, 2, 5, 4) were set before any confirmation cell ran.** At 80nan on
  the development set the model sits near the mean/mode fill, so the 6 80nan variants make
  "supported" harder.

## Launch (2026-10-17, 01:51 GMT-3)

- **Queues:** the four queues started at 01:51:39 GMT-3 on `4273cf6`, with no uncommitted source
  change. E43 had finished at 01:50, and its outcome was read only after this launch.
- **Warm-up:** finished for all 15 variants by 01:35 GMT-3.
- **The deployable readout named by E43** (read after launch, 02:00 GMT-3) is H.

## Outcome (2026-10-17, 10:25 GMT-3)

**Run.** All 45 cells ran from 01:51 to 10:24 GMT-3 (8 h 33 min against about 8.2 h planned).
None failed and none is missing.

**The offline opponents were complete before this reading:**

- E42's tuned bar on all 45 cells, by 03:23 GMT-3;
- E41's arms on the same folds, by 02:39 GMT-3.

**Table.** Built by script from `scripts/experiments/confirmation_report.py`'s functions.

- Induced `impute_score`, K readout, mean over 15 fold pairs; lower is better.
- Each contrast is the mean difference and its verdict (95% t-interval and seed agreement;
  "equivalent" means within ±1% of the opponent's mean).
- **Reading:**
  - **win:** better than both the bar of 5 and the tuned bar;
  - **loss:** worse than either;
  - **neither:** anything else.

| variant | K | bar of 5 | tuned bar | K − bar | K − tuned bar | K − strong tier | H − bar | KC − K | reading |
|---|---|---|---|---|---|---|---|---|---|
| `pendigits_40nan` | 0.4259 | 0.4912 | 0.4786 | -0.0653, better | -0.0528, better | -0.0706, better | -0.0643, better | -0.0012, better | win |
| `pendigits_60nan` | 0.5856 | 0.6424 | 0.6393 | -0.0568, better | -0.0537, better | -0.1215, better | -0.0531, better | -0.0036, better | win |
| `pendigits_80nan` | 0.8304 | 0.8549 | 0.8342 | -0.0245, better | -0.0038, better | -0.0178, better | -0.0212, better | -0.0042, better | win |
| `letter_40nan` | 0.5598 | 0.6154 | 0.6012 | -0.0555, better | -0.0414, better | -0.0554, better | -0.0545, better | -0.0014, better | win |
| `letter_60nan` | 0.7175 | 0.7518 | 0.7499 | -0.0343, better | -0.0324, better | -0.0841, better | -0.0276, better | -0.0030, better | win |
| `letter_80nan` | 0.8902 | 0.9041 | 0.8884 | -0.0139, better | +0.0018, worse | -0.0144, better | -0.0120, better | -0.0023, better | loss |
| `electricity_40nan` | 0.7745 | 0.7309 | 0.7249 | +0.0436, none | +0.0496, worse | -0.0164, none | +0.0417, none | -0.0096, none | loss |
| `electricity_60nan` | 0.8298 | 0.8186 | 0.8190 | +0.0112, worse | +0.0108, worse | -0.0902, better | +0.0203, worse | -0.0004, equivalent | loss |
| `electricity_80nan` | 0.9002 | 0.8999 | 0.8953 | +0.0004, none | +0.0050, none | -0.0569, better | +0.0010, equivalent | -0.0016, equivalent | neither |
| `vehicle_40nan` | 0.4970 | 0.5704 | 0.5708 | -0.0734, better | -0.0738, better | -0.0256, better | -0.0644, better | +0.0005, equivalent | win |
| `vehicle_80nan` | 0.8088 | 0.8165 | 0.8184 | -0.0078, none | -0.0097, none | +0.0425, worse | +0.0002, none | -0.0216, better | neither |
| `biodeg_40nan` | 0.6654 | 0.7413 | 0.7379 | -0.0759, better | -0.0725, better | -0.0242, better | -0.0761, better | -0.0047, none | win |
| `biodeg_80nan` | 0.9712 | 1.0000 | 0.9687 | -0.0288, better | +0.0025, none | +0.0489, worse | -0.0175, better | -0.0246, better | neither |
| `kc2_40nan` | 0.5075 | 0.5931 | 0.6057 | -0.0856, better | -0.0982, better | -0.0604, better | -0.0742, better | -0.0005, equivalent | win |
| `kc2_80nan` | 0.6866 | 0.8431 | 0.8524 | -0.1566, better | -0.1659, better | -0.0587, better | -0.1324, better | -0.0073, none | win |

**Reading, by the rule fixed in advance: mixed.** 9 wins of 15 (≥8 needed), and 3 losses
(≤2 needed for "supported", ≥5 for "refuted").

- **The losses:**
  - electricity_40nan and electricity_60nan, against the boosters;
  - letter_80nan, against the tuned bar only, by +0.0018. That is 0.2% of the bar: inside the 1%
    SESOI in size, but its interval excludes zero, so the rule counts it.
- **Neither:** electricity_80nan, vehicle_80nan, biodeg_80nan.

**What the confirmation says, read beside the development set.**

- **The lead generalises on the large tables and at 40nan.**
  - pendigits and letter at 40 and 60nan are wins by 3–7% against every opponent, missForest and
    the strong tier included.
  - So are vehicle, biodeg and kc2 at 40nan, and kc2 at 80nan.
- **electricity is the table where the model loses,** at 40 and 60nan, as at 20nan in the
  development set (one of its two losses there). The boosters (hgb, lgbm) set the bar there. This
  is a property of the table, now seen at three levels, rather than a failure to generalise.
- **At 80nan the lead mostly vanishes against the tuned bar.** pendigits and kc2 win; electricity,
  vehicle and biodeg are neither, and letter loses by 0.2%. Against missForest (the strong tier)
  the model loses vehicle_80nan and biodeg_80nan.

**Secondary, pre-registered:**

- **Tallies for K:**

  | opponent | tally | at 80nan |
  |---|---|---|
  | the bar of 5 | 11 / none 3 / worse 1 | 4 / 2 / 0 |
  | the tuned bar | 9 / 3 / 3 | 2 / 3 / 1 |
  | missForest | 12 / 1 / 2 | |
  | the strong tier | 12 / 1 / 2 | |
  | the best of all | 10 / 2 / 3 | |
  | the per-type bar | 11 / 3 / 1 | |

  The per-type bar changes no verdict.
- **Across tables** (secondary here, 6 tables): 5 of 6 below the bar, two-sided Wilcoxon p = 0.062
  (the smallest attainable is 0.031); 3 of 6 below the best of all.
- **Deployability:**
  - H against the bar: better 11, no difference 2, equivalent 1, worse 1 (electricity_60nan).
  - K − H: K better on 5, worse on none. On new data the oracle's advantage over the deployable
    readout is small.
  - M − H: equivalent on 12 of 15, H better on 1.
- **T20 (the calibration):**
  - KC − K: KC better on 8 of 15, equivalent on 4, worse on none.
  - At 40nan, KC is better on pendigits_40nan and letter_40nan, so by the secondary rule
    **threshold 40 is supported**: it answers the open question in the proposal (40 rather than
    60).
- **The metric panel, the context-free rows and the twin split** were read offline afterwards;
  see "Offline readings" below.

**For the user to decide:**

- The thesis's confirmation statement: on the confirmation set, the model beats both the untuned
  and the tuned bar on 9 of 15 new variants, ties on 3 and loses on 3, two of them on electricity.
- Whether electricity becomes a stated limit, or an investigation of why the boosters win there.
- Reviving T20 with threshold 40, by an ADR.

### Offline readings (added 2026-10-17, 15:45 GMT-3)

**How they were made:**

- **Source:** `scripts/experiments/metric_panel.py`, run on these runs' fold ledgers, with every
  baseline refitted on the same folds; nothing was written to the store.
- **Opponent:** per (variant, seed), the weak tier's best by mean `impute_score` (D16), not the
  tuned bar of the primary.
- **A bug was fixed before the twin split was read** (c242341): the first pass left the
  non-twin subset empty.
  - kc2 and letter were rerun with the fix.
  - Every other column of the rerun equals the first pass exactly (largest difference 0).

**The panel** (`scripts/experiments/metric_panel_report.py`):

- `biodeg_40nan` (opponent hgb): impute_score -0.076 better, rmse_ratio -0.076 better, mae_ratio -0.109 better, colnorm_ratio -0.125 better, abs_bias +0.001 none, dispersion -0.062 better, wasserstein -0.046 better, impute_score_with_context -0.076 better
- `biodeg_80nan` (opponent mean_mode): impute_score -0.029 better, rmse_ratio -0.029 better, mae_ratio -0.087 better, colnorm_ratio -0.013 better, abs_bias +0.024 worse, dispersion -0.556 better, wasserstein -0.286 better, impute_score_with_context -0.029 better
- `electricity_40nan` (opponent hgb/lgbm): impute_score +0.044 none, rmse_ratio +0.043 none, mae_ratio +0.009 worse, colnorm_ratio +0.022 none, err_ratio +0.044 worse, balacc_error_ratio +0.044 worse, abs_bias +0.008 worse, dispersion +0.039 none, wasserstein -0.009 better, tv +0.083 worse, impute_score_with_context +0.044 none
- `electricity_60nan` (opponent hgb/lgbm): impute_score +0.011 worse, rmse_ratio +0.010 worse, mae_ratio +0.006 worse, colnorm_ratio +0.004 none, err_ratio +0.020 worse, balacc_error_ratio +0.019 worse, abs_bias +0.009 worse, dispersion -0.005 none, wasserstein -0.005 none, tv +0.064 worse, impute_score_with_context +0.013 worse
- `electricity_80nan` (opponent hgb/lgbm): impute_score +0.000 none, rmse_ratio +0.000 none, mae_ratio -0.000 none, colnorm_ratio +0.014 none, err_ratio +0.001 none, balacc_error_ratio +0.002 none, abs_bias +0.009 worse, dispersion +0.017 none, wasserstein +0.021 worse, tv +0.082 worse, impute_score_with_context +0.001 none
- `kc2_40nan` (opponent knn5): impute_score -0.086 better, rmse_ratio -0.086 better, mae_ratio -0.002 none, colnorm_ratio -0.062 better, abs_bias -0.041 better, dispersion -0.191 better, wasserstein -0.013 none, impute_score_with_context -0.086 better
- `kc2_80nan` (opponent knn5): impute_score -0.157 better, rmse_ratio -0.157 better, mae_ratio -0.077 better, colnorm_ratio -0.138 better, abs_bias -0.094 better, dispersion -0.376 better, wasserstein -0.070 better, impute_score_with_context -0.160 better
- `letter_40nan` (opponent lgbm): impute_score -0.056 better, rmse_ratio -0.056 better, mae_ratio -0.059 better, colnorm_ratio -0.054 better, abs_bias +0.001 none, dispersion -0.106 better, wasserstein -0.061 better, impute_score_with_context -0.056 better
- `letter_60nan` (opponent lgbm): impute_score -0.034 better, rmse_ratio -0.034 better, mae_ratio -0.044 better, colnorm_ratio -0.034 better, abs_bias +0.003 worse, dispersion -0.108 better, wasserstein -0.068 better, impute_score_with_context -0.035 better
- `letter_80nan` (opponent hgb/lgbm): impute_score -0.014 better, rmse_ratio -0.014 better, mae_ratio -0.018 better, colnorm_ratio -0.014 better, abs_bias +0.029 worse, dispersion +0.058 worse, wasserstein +0.052 worse, impute_score_with_context -0.015 better
- `pendigits_40nan` (opponent hgb/lgbm): impute_score -0.065 better, rmse_ratio -0.065 better, mae_ratio -0.066 better, colnorm_ratio -0.067 better, abs_bias +0.001 none, dispersion -0.085 better, wasserstein -0.059 better, impute_score_with_context -0.065 better
- `pendigits_60nan` (opponent hgb/lgbm): impute_score -0.057 better, rmse_ratio -0.057 better, mae_ratio -0.081 better, colnorm_ratio -0.059 better, abs_bias -0.000 none, dispersion -0.101 better, wasserstein -0.078 better, impute_score_with_context -0.057 better
- `pendigits_80nan` (opponent hgb/lgbm): impute_score -0.025 better, rmse_ratio -0.025 better, mae_ratio -0.023 better, colnorm_ratio -0.025 better, abs_bias +0.021 worse, dispersion +0.027 worse, wasserstein +0.031 worse, impute_score_with_context -0.026 better
- `vehicle_40nan` (opponent hgb/lgbm): impute_score -0.073 better, rmse_ratio -0.073 better, mae_ratio -0.060 better, colnorm_ratio -0.083 better, abs_bias +0.003 none, dispersion -0.025 better, wasserstein -0.003 none, impute_score_with_context -0.073 better
- `vehicle_80nan` (opponent knn10): impute_score -0.008 none, rmse_ratio -0.008 none, mae_ratio -0.027 better, colnorm_ratio -0.018 better, abs_bias +0.020 worse, dispersion -0.214 better, wasserstein -0.135 better, impute_score_with_context -0.011 none

Disagreements between `impute_score` and a panel metric (opposite verdicts):

- biodeg_80nan: impute_score better, abs_bias worse
- letter_60nan: impute_score better, abs_bias worse
- letter_80nan: impute_score better, abs_bias worse
- letter_80nan: impute_score better, dispersion worse
- letter_80nan: impute_score better, wasserstein worse
- pendigits_80nan: impute_score better, abs_bias worse
- pendigits_80nan: impute_score better, dispersion worse
- pendigits_80nan: impute_score better, wasserstein worse

- **The ratio metrics never disagree with `impute_score`.** The per-kind ratios, the
  column-normalised ratio and balanced accuracy agree with it on all 15 variants, as on E40's
  panel.
- **The disagreements are all distributional:**
  - **abs_bias:** biodeg_80nan, letter_60nan, letter_80nan and pendigits_80nan;
  - **dispersion and Wasserstein:** letter_80nan and pendigits_80nan, against the boosters.
- **On letter_80nan and pendigits_80nan the model's fills are more shrunk than the boosters'.**
  - Dispersion: 0.485 against 0.543 on letter, 0.627 against 0.654 on pendigits (1 is the
    truth's spread).
  - So the fills are closer to the truth cell by cell, but farther from it as a distribution.
  - This is the trade-off a squared-error imputer makes at heavy missingness. It is a caveat on
    these 80nan wins: they hold for point accuracy, not for the distribution.

**The twin split** (`--split`, on the rerun). A twin cell is a cell of a row whose complete
feature vector also appears in the training fold:

- `kc2_40nan`: twins 32.5% of cells; context-free 0.0%; impute_score -0.0856 [-0.1505, -0.0207] better; impute_score_twin +0.2042 [+0.1562, +0.2523] worse; impute_score_other -0.1067 [-0.1693, -0.0440] better; impute_score_with_context -0.0856 [-0.1505, -0.0207] better
- `kc2_80nan`: twins 33.4% of cells; context-free 1.7%; impute_score -0.1566 [-0.1998, -0.1134] better; impute_score_twin +0.0828 [+0.0377, +0.1280] worse; impute_score_other -0.1801 [-0.2236, -0.1367] better; impute_score_with_context -0.1598 [-0.2022, -0.1174] better
- `letter_40nan`: twins 8.8% of cells; context-free 0.0%; impute_score -0.0555 [-0.0573, -0.0538] better; impute_score_twin -0.0800 [-0.0861, -0.0740] better; impute_score_other -0.0538 [-0.0557, -0.0520] better; impute_score_with_context -0.0555 [-0.0573, -0.0538] better
- `letter_60nan`: twins 8.9% of cells; context-free 0.1%; impute_score -0.0343 [-0.0361, -0.0324] better; impute_score_twin -0.0707 [-0.0761, -0.0653] better; impute_score_other -0.0306 [-0.0327, -0.0285] better; impute_score_with_context -0.0345 [-0.0363, -0.0327] better
- `letter_80nan`: twins 8.9% of cells; context-free 3.4%; impute_score -0.0139 [-0.0147, -0.0131] better; impute_score_twin -0.0061 [-0.0102, -0.0020] better; impute_score_other -0.0148 [-0.0156, -0.0140] better; impute_score_with_context -0.0147 [-0.0154, -0.0139] better

- **kc2's wins do not come from twins.**
  - About a third of kc2's cells are twins.
  - On those the model is worse than knn5, by +0.204 at 40nan and +0.083 at 80nan: knn finds the
    duplicate row.
  - On the other cells its lead is larger than overall: −0.107 and −0.180.
- **letter:** better on both subsets at every level.

**Context-free rows (D17)** (`--split`, first pass; the remaining variants). The share of cells in
rows with every feature missing, and the score without them:

- `biodeg_40nan`: twins 0.5% of cells; context-free 0.0%; impute_score -0.0759 [-0.0978, -0.0541] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0759 [-0.0978, -0.0541] better
- `biodeg_80nan`: twins 0.5% of cells; context-free 0.0%; impute_score -0.0288 [-0.0393, -0.0184] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0288 [-0.0393, -0.0184] better
- `electricity_40nan`: twins 0.0% of cells; context-free 0.1%; impute_score +0.0436 [-0.0013, +0.0885] none; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context +0.0437 [-0.0013, +0.0887] none
- `electricity_60nan`: twins 0.0% of cells; context-free 2.9%; impute_score +0.0112 [+0.0059, +0.0164] worse; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context +0.0125 [+0.0058, +0.0193] worse
- `electricity_80nan`: twins 0.0% of cells; context-free 20.9%; impute_score +0.0004 [-0.0093, +0.0100] none; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context +0.0005 [-0.0119, +0.0129] none
- `pendigits_40nan`: twins 0.0% of cells; context-free 0.0%; impute_score -0.0653 [-0.0680, -0.0627] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0653 [-0.0680, -0.0627] better
- `pendigits_60nan`: twins 0.0% of cells; context-free 0.0%; impute_score -0.0568 [-0.0587, -0.0549] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0569 [-0.0588, -0.0550] better
- `pendigits_80nan`: twins 0.0% of cells; context-free 3.7%; impute_score -0.0245 [-0.0264, -0.0227] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0258 [-0.0277, -0.0238] better
- `vehicle_40nan`: twins 0.0% of cells; context-free 0.0%; impute_score -0.0734 [-0.0853, -0.0615] better; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0734 [-0.0853, -0.0615] better
- `vehicle_80nan`: twins 0.0% of cells; context-free 1.9%; impute_score -0.0078 [-0.0202, +0.0047] none; impute_score_twin: a fold has no such cell; impute_score_other: a fold has no such cell; impute_score_with_context -0.0106 [-0.0223, +0.0011] none

- **Removing the context-free cells changes no verdict.**
  - electricity_80nan has the largest share: 20.9% of the cells (16.8% of the rows).
  - There the contrast stays at no difference.

