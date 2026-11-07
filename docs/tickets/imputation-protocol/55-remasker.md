# ReMasker as an external deep baseline (E157, T53; pre-registered 2026-11-06)

This ticket was written and committed before the first scored cell of the study.

- **Why:** ReMasker (Du, Melis and Wang, ICLR 2024) is a transformer masked autoencoder for imputation, the published
  model closest to TRIDENT's decoder after MaCoDE ([07](07-macode.md), E48). It was deferred on 2026-10-06
  ([ADR 0016](../../adr/0016-missforest-baselines.md), the ADR that chose missForest; [BACKLOG I3](../../BACKLOG.md)).
- **The user's go:** "Vamos fazer o 1 e 3 até amanhã a tarde" (2026-11-06), item 3 being "ReMasker in the imputation
  comparison bar". Whether it joins the bar or the strong tier is an ADR and the user's call; this study measures it
  as an external baseline, E48's way.
- **Code:** `src/training/remasker.py`, written here, not vendored: the reference repository
  (github.com/tydusky/remasker, commit `69b4108`, 2023-03-16) states no licence, so no file of it is copied. The
  module keeps the reference's settings and states each departure:
  - **Settings kept:** width 32, encoder depth 6, decoder depth 4, 4 heads, MLP ratio 4, mask ratio 0.5, 600 epochs,
    batch 64, AdamW (0.9, 0.95) at 1e-3 under a per-step half-cosine to 1e-5 after 40 warm-up epochs, fp16 autocast,
    min-max scaling, the shared linear cell embedding with fixed sine-cosine positions, the `tanh / 2 + 0.5` head, the
    two-term loss. The parser's weight decay 0.05 never reaches the reference's optimizer, which runs at AdamW's
    default 0.01; kept.
  - **Inductive:** fit on the training fold, fill any frame (ADR 0007, the baseline protocol).
  - **A gap is never a visible cell.** The reference keeps a fixed half of a row's tokens with gaps sorted last, so a
    row with fewer observed cells than half its columns shows gaps as zeros and scores them against 0. The paper
    re-masks observed values and encodes the observed rest; here gaps are padding in the encoder and the loss reads
    observed cells only. Beyond 50% missing that changes most rows.
  - **Batched imputation** (the reference fills one row at a time; with gaps as padding the fill is the same), the loss
    without per-step host syncs (a non-finite loss stops the fit once an epoch), its own random streams (`fork_rng`).
  - **Numerical tables only,** as the reference states.
  - Unit tests `tests/unit/test_remasker.py`. A toy check on synthetic data (not a study table): on a 2,000-row table
    with three correlated columns and one independent, 30% gaps, 60 epochs, its RMSE was 0.32 / 0.69 / 0.38 against
    the mean fill's 0.99 / 1.98 / 1.01, and 1.01 against 0.99 on the independent column.
- **Driver:** `scripts/experiments/remasker_run.py` (E48's `macode_run.py` with ReMasker in), folds rebuilt by
  `recompute_folds` exactly as each run built them, off-store CSVs in `/scratch2/diogoneiss/trident-2026-11-06-remasker/csv`;
  lanes `scripts/experiments/run_remasker_lanes.py`.

## Pre-launch timing smoke (2026-11-06, timing only, no score computed)

10 epochs on 64% of a table's rows (about a training fold), on gorgona8's GPU beside E156's three trainers, scaled to
600 epochs and five folds. First pass, then again after the per-step host syncs were removed (same computation):

| table | rows | columns | min a seed, first | min a seed, without syncs |
|---|---|---|---|---|
| kc2_40nan | 334 | 21 | 25 | 12 |
| vehicle_40nan | 541 | 18 | 18 | |
| biodeg_40nan | 675 | 41 | 14 | |
| pendigits_40nan | 7,034 | 16 | 123 | 90 |
| letter_40nan | 12,800 | 16 | 226 | |

## Design (fixed before the first scored cell)

- **Arm:** ReMasker as above, untuned, at the reference's settings. TRIDENT's configuration was tuned on the
  development set; said beside every comparison.
- **Scope: E44's confirmation set at seeds 707/808/909, its numerical variants that fit before the read:**
  `pendigits_{40,60,80}nan`, `vehicle_{40,80}nan`, `biodeg_{40,80}nan`, `kc2_{40,80}nan`. 9 variants, 27 (variant,
  seed) pairs, 135 fits. K is the confirmation cells' own (`confirmation_report.cells`, as E48).
  - **Out, and why:** `letter_*` (about 25 process-hours at the paper's batch of 64; a larger batch would be a
    departure from its settings, and fewer variants at its settings is preferred); `electricity_*` (a categorical
    column; the reference takes numbers only). The development set (descriptive in E48) is not run.
- **Cost:** about 17 process-hours from the smoke, four lanes on gorgona8 beside E156 until it ends; expected end
  between 06:00 and 10:00 GMT-3 on 2026-11-07.
- **Read:** once, with all 27 pairs in, by 14:00 GMT-3 on 2026-11-07. If pairs are missing then, the study is
  reported as partial on the variants that have all three seeds, no strong-tier proposal is made from it, and the
  rest is finished before any thesis use.

## Measures (E48's, through `scripts/experiments/remasker_report.py`)

On the induced `impute_score` over 15 fold pairs per variant, the verdict rule with seed agreement and the 1% SESOI
(ADR 0020):

1. **Primary:** K − ReMasker per variant, tallied.
2. ReMasker − the strong tier's best (missForest and E41's arms), − the bar, − the tuned bar.
3. K − the best of all, ReMasker included.
4. ReMasker's fit time per fold.

## Reading, fixed now

- **K − ReMasker is the thesis's second deep-baseline statement,** beside E48's K − MaCoDE.
- **Whether ReMasker joins the strong tier:** E48's rule scaled to the scope: if it is better than the strong tier's
  best on at least 3 of the 9 variants (E48's third), an ADR amending 0021 is proposed for the user's decision;
  otherwise it is an external deep baseline only.
- **No default changes and no bar change either way.**
