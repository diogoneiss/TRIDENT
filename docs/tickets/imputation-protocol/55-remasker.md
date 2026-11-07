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

## Outcome (2026-11-07 06:12 GMT-3): K better than ReMasker on 9 of 9; ReMasker better than the strong tier's best on 1 of 9, so an external deep baseline only. At 60–80% missing the arm barely trained, so only the 40nan readings describe ReMasker

All 27 pairs in, none failed (the last lane ended 2026-11-07 06:05 GMT-3). The first run of `remasker_report.py`
stopped before printing any reading: `report_common.baselines_in` refuses a baseline no class names, and `remasker`
was in none. It was added to `DEEP` beside `macode` (outside the bar and the strong tier, inside the best of all),
which is the design's own reading. Then the report ran once. The unedited printout:

```text
## Confirmation set (E44), seeds 707/808/909, E157's 9 variants
- `pendigits_40nan`: K − remasker -0.0337 [-0.0374, -0.0299] better; remasker − strong -0.0751 [-0.0802, -0.0701] better; remasker − bar -0.0699 [-0.0725, -0.0673] better; remasker − tuned -0.0573 [-0.0601, -0.0545] better; K − best_of_all -0.0337 [-0.0374, -0.0299] better; best of all remasker; fit 1919 s a fold
- `pendigits_60nan`: K − remasker -0.1725 [-0.1785, -0.1665] better; remasker − strong +0.0334 [+0.0278, +0.0391] worse; remasker − bar +0.0981 [+0.0939, +0.1023] worse; remasker − tuned +0.1012 [+0.0971, +0.1052] worse; K − best_of_all -0.0744 [-0.0779, -0.0709] better; best of all hgb/lgbm; fit 1939 s a fold
- `pendigits_80nan`: K − remasker -0.6008 [-0.6476, -0.5541] better; remasker − strong +0.5830 [+0.5353, +0.6307] worse; remasker − bar +0.5763 [+0.5295, +0.6231] worse; remasker − tuned +0.5970 [+0.5500, +0.6440] worse; K − best_of_all -0.0178 [-0.0197, -0.0159] better; best of all condexp; fit 1778 s a fold
- `vehicle_40nan`: K − remasker -0.0741 [-0.0945, -0.0537] better; remasker − strong +0.0485 [+0.0366, +0.0605] worse; remasker − bar +0.0007 [-0.0150, +0.0164] none; remasker − tuned +0.0003 [-0.0126, +0.0132] none; K − best_of_all -0.0256 [-0.0466, -0.0045] better; best of all missforest; fit 141 s a fold
- `vehicle_80nan`: K − remasker -0.6118 [-0.6769, -0.5466] better; remasker − strong +0.6542 [+0.5862, +0.7222] worse; remasker − bar +0.6040 [+0.5363, +0.6716] worse; remasker − tuned +0.6021 [+0.5351, +0.6691] worse; K − best_of_all +0.0425 [+0.0343, +0.0506] worse; best of all missforest; fit 142 s a fold
- `biodeg_40nan`: K − remasker -0.1410 [-0.1765, -0.1054] better; remasker − strong +0.1168 [+0.0889, +0.1447] worse; remasker − bar +0.0651 [+0.0437, +0.0864] worse; remasker − tuned +0.0685 [+0.0464, +0.0905] worse; K − best_of_all -0.0242 [-0.0390, -0.0094] better; best of all missforest; fit 174 s a fold
- `biodeg_80nan`: K − remasker -1.1122 [-1.1991, -1.0252] better; remasker − strong +1.1610 [+1.0741, +1.2479] worse; remasker − bar +1.0833 [+0.9946, +1.1721] worse; remasker − tuned +1.1147 [+1.0263, +1.2030] worse; K − best_of_all +0.0489 [+0.0395, +0.0582] worse; best of all missforest; fit 174 s a fold
- `kc2_40nan`: K − remasker -0.0676 [-0.1045, -0.0306] better; remasker − strong +0.0072 [-0.0567, +0.0710] none; remasker − bar -0.0180 [-0.0887, +0.0527] none; remasker − tuned -0.0306 [-0.1038, +0.0426] none; K − best_of_all -0.0498 [-0.0923, -0.0073] better; best of all knn5/remasker/softimpute; fit 87 s a fold
- `kc2_80nan`: K − remasker -1.1262 [-1.4605, -0.7919] better; remasker − strong +1.0674 [+0.7141, +1.4207] worse; remasker − bar +0.9696 [+0.6213, +1.3178] worse; remasker − tuned +0.9603 [+0.6144, +1.3062] worse; K − best_of_all -0.0587 [-0.0970, -0.0205] better; best of all missforest; fit 86 s a fold
  - K − remasker: better 9 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - remasker − strong: better 1 / no difference 1 / worse 7 (fold-pair t, seed agreement)
  - remasker − bar: better 1 / no difference 2 / worse 6 (fold-pair t, seed agreement)
  - remasker − tuned: better 1 / no difference 2 / worse 6 (fold-pair t, seed agreement)
  - K − best_of_all: better 7 / no difference 0 / worse 2 (fold-pair t, seed agreement)

Decision input: ReMasker better than the strong tier's best on 1 of 9 variants (fewer than 3: an external deep baseline only).
```

- **Primary, K − ReMasker: better on 9 of 9.**
- **Strong-tier rule: 1 of 9** (pendigits_40nan, −0.075); fewer than 3, so ReMasker is an external deep baseline
  only and no ADR is proposed. Restricted to 40nan it would be 1 of 4, so the call does not depend on the defect below.
- **K − the best of all:** better on 7, worse on 2 (vehicle_80nan and biodeg_80nan, where missForest is best).
  ReMasker is the best of all only on pendigits_40nan, and tied for best on kc2_40nan.
- **Fit time:** about 32 min a fold on pendigits, 1.5–3 min on the small tables.

### Why 60–80nan do not describe ReMasker (found at the read; the reading above is not changed by it)

At 80nan ReMasker is worse than the mean fill by 0.58 to 1.11 (`remasker − bar`; the bar contains `mean_mode`). The
cause is the keep count kept from the reference code. Each row keeps `int(L · 0.5)` tokens drawn with gaps sorted last,
and only observed cells beyond that count are re-masked. Once a row has fewer observed cells than half its columns,
nothing is re-masked, so the reconstruction term gets no cells. The fraction from `_remask` on random masks
(20,000 rows; computed after the read, no score involved):

| columns | missing | re-masked / observed | rows with any re-masked |
|---|---|---|---|
| 16 | 0.4 | 0.191 | 0.716 |
| 16 | 0.6 | 0.036 | 0.144 |
| 16 | 0.8 | 0.001 | 0.002 |
| 18 | 0.4 | 0.187 | 0.735 |
| 18 | 0.6 | 0.030 | 0.135 |
| 18 | 0.8 | 0.000 | 0.001 |
| 21 | 0.4 | 0.216 | 0.825 |
| 21 | 0.6 | 0.037 | 0.175 |
| 21 | 0.8 | 0.000 | 0.001 |
| 41 | 0.4 | 0.191 | 0.901 |
| 41 | 0.6 | 0.012 | 0.095 |
| 41 | 0.8 | 0.000 | 0.000 |

- At 80nan the arm learned only to copy visible cells, and the mask token's output never received a gradient. At 60nan
  it trained on 1–4% of observed cells. At 40nan it trained on about a fifth of them, near the ratio's intent.
- **The paper defines the ratio over observed values.** §4.4 reads: "The masking ratio controls the number of
  re-masked values (after excluding missing values)." Its evaluation goes up to 0.7 missing (Appendix B.1). The
  reference code's keep count is what departs from the paper. Under that rule the reference also re-masks no observed
  cell in such rows; it differs only in scoring the gaps it shows as zeros. "Gaps never visible", this module's own
  departure, does not cause the defect.
- A smoke at 80nan before launch (about 10 min on kc2) would have shown it; the timing smoke computed no score.

### What it leads to

- **The thesis's second deep-baseline statement** uses only the 40nan readings, pending the user's decision. On those,
  K beats ReMasker on all four, ReMasker beats the strong tier only on pendigits, and it sits level with the bar on
  vehicle and kc2. The 60–80nan readings are reported as this limit, not as a win for K.
- **Follow-up, proposed, not run:** a re-mask rule over each row's observed cells (half of them, as the paper states)
  on the same 9 variants and seeds, under a new E id and a new pre-registration with an 80nan scored smoke. It is the
  user's call (T53).
- No default changes, no bar change, no ADR.
