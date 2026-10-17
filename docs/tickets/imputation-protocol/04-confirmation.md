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
