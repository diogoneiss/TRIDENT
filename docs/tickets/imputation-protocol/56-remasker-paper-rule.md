# ReMasker with the paper's re-mask count (E158, T53; pre-registered 2026-11-07 08:15 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** [E157](55-remasker.md) found at its read that the reference code's re-mask count, `int(L · 0.5)` tokens kept
  per row with gaps sorted last, re-masks almost no observed cell once a row has fewer observed cells than half its
  columns: 0.0–0.1% of observed cells at 80% missing, 1–4% at 60%, about 20% at 40%. Its 60–80nan readings measure an arm
  with almost no reconstruction signal. The paper counts the ratio over observed values (§4.4: "The masking ratio
  controls the number of re-masked values (after excluding missing values)").
- **The user's go:** "Rode de novo com a regra do artigo para vermos" (2026-11-07), with "Você pode usar um pouco da
  medusa1 para acelerar".
- **Disclosure, known before the first cell:** E157's whole printout (ticket 55), including the strong tier's, the
  bar's and K's values on these folds, which this study shares. The paper-rule arm itself has no score on any study
  table.

## The change (`src/training/remasker.py`, `remask="paper"`; the default stays E157's `reference`)

- A row with `n_obs` observed cells keeps `int(n_obs · 0.5)` of them visible, drawn at random, and re-masks the rest.
  On a complete row this equals the reference's count. Everything else is E157's arm unchanged: settings, departures,
  seed, scaling, fill.
- The driver selects it with `REMASKER_REMASK=paper` and records it in every CSV row (`baseline/remasker/remask_paper`).
  The report refuses a CSV whose rule differs from the one it reads (`remasker_report.py --paper`).
- Unit tests: the paper rule never shows a gap, keeps `n_obs // 2` per row, and matches the reference's count on
  complete rows (`tests/unit/test_remasker.py`; the unit suite passed, 509).
- **Pre-launch toy check** (synthetic, not a study table: 2,000 rows, six columns on one latent and two independent, MCAR,
  60 epochs, RMSE over the gaps):

| missing | mean fill | reference rule | paper rule |
|---|---|---|---|
| 0.4 | 0.743 | 0.442 | 0.431 |
| 0.8 | 0.750 | 0.896 | 0.648 |

  At 80% the reference rule is worse than the mean fill, as in E157; the paper rule is not.

## Design (fixed now, E157's in all else)

- **Scope:** E157's 9 variants (`pendigits_{40,60,80}nan`, `vehicle_`, `biodeg_`, `kc2_{40,80}nan`) at seeds
  707/808/909, 27 pairs, 135 fits; K the confirmation cells' own.
- **Machines (ADR 0035, every arm of a pair on one machine):** medusa1 runs `pendigits_80nan` at all three seeds and
  `pendigits_60nan` s909, four lanes over its two GPUs (they are shared with another user's jobs; about 1.6 GB a
  lane); gorgona8 runs the other 23 pairs in five lanes. The CSVs are pulled to gorgona8 for the read; nothing is written
  to any store. A pair's machine is in its lane log and in this ticket.
  - gorgona8 out: `/scratch2/diogoneiss/trident-2026-11-07-remasker-paper`; medusa1 out:
    `/scratch/diogoneiss/trident-2026-11-07-remasker-paper`.
- **Read:** once, with all 27 pairs in: `remasker_report.py --paper --remasker <csv> --against
  /scratch2/diogoneiss/trident-2026-11-06-remasker/csv`. Expected end about 13:00–15:00 GMT-3 on 2026-11-07. A failed
  pair is rerun on the same machine before the read.

## Measures and reading (E157's, fixed now)

1. **Primary:** K − ReMasker (paper rule) per variant, tallied.
2. ReMasker − the strong tier's best, − the bar, − the tuned bar; K − the best of all (ReMasker paper rule included,
   E157's arm not).
3. **Strong-tier rule:** better than the strong tier's best on at least 3 of 9 → an ADR amending 0021 is proposed for
   the user's decision; otherwise an external deep baseline only.
4. **Descriptive, decides nothing:** paper rule − reference rule per variant (E158 − E157); fit time.

**What it leads to:** E158 replaces E157 as the thesis's ReMasker comparison at every level, since it is the paper's
rule; E157 stays on the record as the reference code's. The thesis sentence is the user's call (asked again later, as
the user requested). No default changes, no bar change, no ADR unless the strong-tier rule fires.

## Limits, stated now

- medusa1's GPUs are RTX 5090s, gorgona8's an RTX 3090 Ti; fills differ per fold between GPUs (E86), so a pair's
  machine is fixed and recorded, and the four pendigits pairs on medusa1 are not compared with gorgona8's seed by seed.
- Untuned, as E157; TRIDENT's configuration was tuned on the development set.

## Outcome (2026-11-07 11:42 GMT-3): K better than ReMasker on 9 of 9; ReMasker better than the strong tier's best on 2 of 9, so an external deep baseline only. The paper's rule removes E157's collapse at 80nan

All 27 pairs in, none failed: gorgona8's five lanes ended 11:37, medusa1's four 09:25 GMT-3. medusa1's four CSVs were
copied to gorgona8's directory with matching SHA-256, and every row of the 27 carries `remask_paper` 1. The first run of
the report stopped at its rule check before printing any reading. It looked up `baseline/remasker/remask_paper`, but
`score_baselines` prefixes the population (`impute/induced/...`). The key was fixed, and the report then ran once. The
unedited printout:

```text
## Confirmation set (E44), seeds 707/808/909, E158, the paper's re-mask rule, 9 variants
- `pendigits_40nan`: K − remasker -0.0392 [-0.0431, -0.0354] better; remasker − strong -0.0695 [-0.0738, -0.0652] better; remasker − bar -0.0643 [-0.0663, -0.0623] better; remasker − tuned -0.0517 [-0.0536, -0.0498] better; K − best_of_all -0.0392 [-0.0431, -0.0354] better; best of all remasker; fit 1807 s a fold
- `pendigits_60nan`: K − remasker -0.0315 [-0.0354, -0.0276] better; remasker − strong -0.1076 [-0.1120, -0.1032] better; remasker − bar -0.0429 [-0.0457, -0.0401] better; remasker − tuned -0.0398 [-0.0426, -0.0370] better; K − best_of_all -0.0315 [-0.0354, -0.0276] better; best of all remasker; fit 1483 s a fold
- `pendigits_80nan`: K − remasker -0.0823 [-0.0863, -0.0783] better; remasker − strong +0.0645 [+0.0601, +0.0689] worse; remasker − bar +0.0578 [+0.0540, +0.0616] worse; remasker − tuned +0.0785 [+0.0743, +0.0827] worse; K − best_of_all -0.0178 [-0.0197, -0.0159] better; best of all condexp; fit 827 s a fold
- `vehicle_40nan`: K − remasker -0.0754 [-0.0995, -0.0514] better; remasker − strong +0.0498 [+0.0313, +0.0684] worse; remasker − bar +0.0020 [-0.0186, +0.0226] none; remasker − tuned +0.0016 [-0.0197, +0.0229] none; K − best_of_all -0.0256 [-0.0466, -0.0045] better; best of all missforest; fit 155 s a fold
- `vehicle_80nan`: K − remasker -0.0695 [-0.0892, -0.0499] better; remasker − strong +0.1120 [+0.0937, +0.1303] worse; remasker − bar +0.0618 [+0.0449, +0.0787] worse; remasker − tuned +0.0598 [+0.0425, +0.0772] worse; K − best_of_all +0.0425 [+0.0343, +0.0506] worse; best of all missforest; fit 151 s a fold
- `biodeg_40nan`: K − remasker -0.2074 [-0.2599, -0.1549] better; remasker − strong +0.1832 [+0.1358, +0.2306] worse; remasker − bar +0.1315 [+0.0928, +0.1701] worse; remasker − tuned +0.1349 [+0.0963, +0.1734] worse; K − best_of_all -0.0242 [-0.0390, -0.0094] better; best of all missforest; fit 183 s a fold
- `biodeg_80nan`: K − remasker -0.0636 [-0.0775, -0.0497] better; remasker − strong +0.1125 [+0.0987, +0.1263] worse; remasker − bar +0.0348 [+0.0224, +0.0472] worse; remasker − tuned +0.0661 [+0.0542, +0.0780] worse; K − best_of_all +0.0489 [+0.0395, +0.0582] worse; best of all missforest; fit 161 s a fold
- `kc2_40nan`: K − remasker -0.1025 [-0.1414, -0.0635] better; remasker − strong +0.0421 [-0.0221, +0.1063] none; remasker − bar +0.0169 [-0.0573, +0.0911] none; remasker − tuned +0.0043 [-0.0693, +0.0780] none; K − best_of_all -0.0550 [-0.1099, -0.0002] better; best of all knn5/softimpute; fit 95 s a fold
- `kc2_80nan`: K − remasker -0.0589 [-0.0989, -0.0189] better; remasker − strong +0.0002 [-0.0490, +0.0493] none; remasker − bar -0.0977 [-0.1394, -0.0560] better; remasker − tuned -0.1070 [-0.1491, -0.0648] better; K − best_of_all -0.0479 [-0.0890, -0.0067] better; best of all missforest/remasker; fit 96 s a fold
  - K − remasker: better 9 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - remasker − strong: better 2 / no difference 2 / worse 5 (fold-pair t, seed agreement)
  - remasker − bar: better 3 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - remasker − tuned: better 3 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - K − best_of_all: better 7 / no difference 0 / worse 2 (fold-pair t, seed agreement)

## ReMasker, the paper's rule − the reference code's (E158 − E157), descriptive
- `pendigits_40nan`: +0.0056 [+0.0033, +0.0079] worse
- `pendigits_60nan`: -0.1410 [-0.1455, -0.1365] better
- `pendigits_80nan`: -0.5185 [-0.5638, -0.4732] better
- `vehicle_40nan`: +0.0013 [-0.0201, +0.0227] none
- `vehicle_80nan`: -0.5422 [-0.6125, -0.4720] better
- `biodeg_40nan`: +0.0664 [+0.0359, +0.0969] worse
- `biodeg_80nan`: -1.0485 [-1.1338, -0.9633] better
- `kc2_40nan`: +0.0349 [-0.0043, +0.0742] none
- `kc2_80nan`: -1.0673 [-1.3909, -0.7437] better
  - better 5 / no difference 2 / worse 2 (fold-pair t, seed agreement)

Decision input: ReMasker better than the strong tier's best on 2 of 9 variants (fewer than 3: an external deep baseline only).
```

- **Primary, K − ReMasker: better on 9 of 9,** at every level now that the arm trains at every level.
- **Strong-tier rule: 2 of 9** (pendigits_40nan −0.070, pendigits_60nan −0.108); fewer than 3, so ReMasker stays an
  external deep baseline and no ADR is proposed.
- ReMasker beats the bar on 3 of 9 (pendigits 40/60, kc2_80nan) and the tuned bar on the same 3; it is the best of all on
  pendigits 40/60, and tied for best on kc2_80nan. K − the best of all is unchanged from E157: better 7, worse 2
  (vehicle_80nan and biodeg_80nan, missForest).
- **Paper − reference (descriptive):** better on all four 80nan variants (−0.52 to −1.07) and on pendigits_60nan
  (−0.141). Worse on pendigits_40nan (+0.006) and biodeg_40nan (+0.066), where the paper's rule re-masks half the
  observed cells against the reference's fifth. At 80nan ReMasker is now within 0.035–0.062 of the bar on three
  variants and better than it on kc2, instead of 0.58–1.11 worse.
- Fit time 14–30 min a fold on pendigits, 1.5–3 min on the small tables.

**What it leads to (fixed in the design):** E158 is the thesis's ReMasker comparison at every level, and E157 stays on
the record as the reference code's count. K − ReMasker is better on all 9 variants under either rule. The thesis
sentence is the user's call, asked after this read. No default changes, no bar change, no ADR.

## The thesis sentence (accepted 2026-11-07 12:00 GMT-3)

On the user's decision ("aprovo a formulação"), E158 is the ReMasker comparison at every level, and E157 is cited as the
reference code's count. The sentence:

> TRIDENT supera o ReMasker nas 9 variantes numéricas do conjunto de confirmação, de 40% a 80% de lacunas; o ReMasker supera a faixa forte apenas no pendigits a 40% e 60%.
