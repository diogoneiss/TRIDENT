# ReMasker on letter, the paper's rule (E159, T53; pre-registered 2026-11-07 21:43 GMT-3)

This ticket was written and committed before the study's first scored cell.

- **Why:** E157 and E158 left letter out for cost (about 25 process-hours at the paper's batch of 64); E158's accepted
  sentence covers 9 numerical confirmation variants. letter, the largest numerical table of the confirmation set, is
  where a masked autoencoder has the most rows to learn from (ReMasker beat the strong tier only on pendigits, the
  other large table).
- **The user's go:** "De acordo, va" (2026-11-07), on the proposal "A: ReMasker no letter 40/60/80, regra do artigo",
  launched together with B (E160) to keep both machines' GPUs busy.
- **Disclosure:** E44's letter cells (K, the strong tier, the bar) are on the record; ReMasker has no score on letter.

## Design (E158's in all else)

- **Arm:** ReMasker, `remask="paper"`, untuned, E158's settings (ticket 56).
- **Scope:** `letter_{40,60,80}nan` at 707/808/909, 9 pairs, 45 fits; K the confirmation cells' own.
- **Machines (ADR 0035):** medusa1 runs `letter_60nan` and `letter_80nan` (6 pairs, six lanes, three on each GPU,
  beside another user's job that holds 27 GB of each); gorgona8 runs `letter_40nan` (3 pairs, three lanes, beside
  E160). Outs: `/scratch/diogoneiss/trident-2026-11-07-remasker-letter` (medusa1) and
  `/scratch2/diogoneiss/trident-2026-11-07-remasker-letter` (gorgona8); medusa1's CSVs are copied over with their
  SHA-256 checked before the read.
- **Read:** once, with all 9 pairs in:
  `remasker_report.py --paper --variants letter_40nan,letter_60nan,letter_80nan --remasker <letter csv> --study E159`,
  then the 12 variants together (E158's nine and these three), from both CSV directories.
  Expected end: about 01:00–03:00 GMT-3 on 2026-11-08.

## Measures and reading (fixed now)

- E158's measures on the three letter variants: K − ReMasker (primary), ReMasker − strong / bar / tuned bar, K − best
  of all, fit time.
- **Strong-tier rule over the 12 numerical variants** (E158's nine plus letter's three): better than the strong
  tier's best on at least 4 of 12 (a third, E48's) → an ADR amending 0021 is proposed for the user's decision;
  otherwise an external deep baseline only. E158 alone gave 2 of 9.
- **The accepted sentence** ("TRIDENT supera o ReMasker nas 9 variantes numéricas..."): if K is better on all three
  letter variants, the sentence's count becomes 12; otherwise the letter variants are stated beside it. The wording
  stays the user's.
- No default changes, no bar change.

## Outcome (2026-11-08 02:38 GMT-3)

All 9 pairs finished without a failure: medusa1's six at 23:57 GMT-3 on 2026-11-07, gorgona8's three at 02:37 GMT-3
on 2026-11-08. medusa1's six CSVs were copied to gorgona8's letter directory and their SHA-256 matched (6/6 OK). Read
once each, as pre-registered.

- **K − ReMasker: better on 3/3 letter variants** (−0.040, −0.012, −0.054), so better on all 12 numerical variants.
- **ReMasker better than the strong tier's best on 2/3 letter variants** (40nan −0.047, 60nan −0.072; worse at 80nan
  +0.040). Over the 12: **4 of 12** (pendigits and letter at 40 and 60nan), the pre-registered threshold. Per this
  ticket, **an ADR amending 0021 (the two baseline tiers) to admit ReMasker to the strong tier is proposed for the
  user's decision.** Not written: ADRs change only on the user's decision. Against it: all four wins are on the two
  large tables at 40–60nan; on the six small-table cells it is worse than the strong tier on 4 and no different on 2
  (kc2), and at 80nan worse on both large tables. A fold costs 58 min on letter_40nan against minutes for the strong tier.
- **The accepted sentence:** K better on all three, so its count becomes 12 and the large-table clause gains letter.
  Proposed wording, the user's to confirm: "TRIDENT supera o ReMasker nas 12 variantes numéricas do conjunto de
  confirmação, de 40% a 80% de lacunas; o ReMasker supera a faixa forte apenas nas duas tabelas grandes, pendigits e
  letter, a 40% e 60%".

### Printout, letter only (unedited)

```
## Confirmation set (E44), seeds 707/808/909, E159, 3 variants
- `letter_40nan`: K − remasker -0.0398 [-0.0424, -0.0373] better; remasker − strong -0.0472 [-0.0512, -0.0433] better; remasker − bar -0.0474 [-0.0502, -0.0445] better; remasker − tuned -0.0332 [-0.0358, -0.0305] better; K − best_of_all -0.0398 [-0.0424, -0.0373] better; best of all remasker; fit 3495 s a fold
- `letter_60nan`: K − remasker -0.0123 [-0.0138, -0.0107] better; remasker − strong -0.0719 [-0.0751, -0.0687] better; remasker − bar -0.0220 [-0.0237, -0.0203] better; remasker − tuned -0.0201 [-0.0218, -0.0184] better; K − best_of_all -0.0123 [-0.0138, -0.0107] better; best of all remasker; fit 1584 s a fold
- `letter_80nan`: K − remasker -0.0539 [-0.0562, -0.0516] better; remasker − strong +0.0395 [+0.0375, +0.0416] worse; remasker − bar +0.0400 [+0.0377, +0.0423] worse; remasker − tuned +0.0557 [+0.0536, +0.0578] worse; K − best_of_all -0.0139 [-0.0147, -0.0131] better; best of all hgb/lgbm; fit 1579 s a fold
  - K − remasker: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - remasker − strong: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
  - remasker − bar: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
  - remasker − tuned: better 2 / no difference 0 / worse 1 (fold-pair t, seed agreement)
  - K − best_of_all: better 3 / no difference 0 / worse 0 (fold-pair t, seed agreement)

Decision input: ReMasker better than the strong tier's best on 2 of 3 variants (at least 1, the third E48 asked for: propose it for the strong tier).
```

### Printout, the 12 numerical variants (E158's CSVs and these; unedited)

```
## Confirmation set (E44), seeds 707/808/909, E159 (12 variants), 12 variants
- `pendigits_40nan`: K − remasker -0.0392 [-0.0431, -0.0354] better; remasker − strong -0.0695 [-0.0738, -0.0652] better; remasker − bar -0.0643 [-0.0663, -0.0623] better; remasker − tuned -0.0517 [-0.0536, -0.0498] better; K − best_of_all -0.0392 [-0.0431, -0.0354] better; best of all remasker; fit 1807 s a fold
- `pendigits_60nan`: K − remasker -0.0315 [-0.0354, -0.0276] better; remasker − strong -0.1076 [-0.1120, -0.1032] better; remasker − bar -0.0429 [-0.0457, -0.0401] better; remasker − tuned -0.0398 [-0.0426, -0.0370] better; K − best_of_all -0.0315 [-0.0354, -0.0276] better; best of all remasker; fit 1483 s a fold
- `pendigits_80nan`: K − remasker -0.0823 [-0.0863, -0.0783] better; remasker − strong +0.0645 [+0.0601, +0.0689] worse; remasker − bar +0.0578 [+0.0540, +0.0616] worse; remasker − tuned +0.0785 [+0.0743, +0.0827] worse; K − best_of_all -0.0178 [-0.0197, -0.0159] better; best of all condexp; fit 827 s a fold
- `vehicle_40nan`: K − remasker -0.0754 [-0.0995, -0.0514] better; remasker − strong +0.0498 [+0.0313, +0.0684] worse; remasker − bar +0.0020 [-0.0186, +0.0226] none; remasker − tuned +0.0016 [-0.0197, +0.0229] none; K − best_of_all -0.0256 [-0.0466, -0.0045] better; best of all missforest; fit 155 s a fold
- `vehicle_80nan`: K − remasker -0.0695 [-0.0892, -0.0499] better; remasker − strong +0.1120 [+0.0937, +0.1303] worse; remasker − bar +0.0618 [+0.0449, +0.0787] worse; remasker − tuned +0.0598 [+0.0425, +0.0772] worse; K − best_of_all +0.0425 [+0.0343, +0.0506] worse; best of all missforest; fit 151 s a fold
- `biodeg_40nan`: K − remasker -0.2074 [-0.2599, -0.1549] better; remasker − strong +0.1832 [+0.1358, +0.2306] worse; remasker − bar +0.1315 [+0.0928, +0.1701] worse; remasker − tuned +0.1349 [+0.0963, +0.1734] worse; K − best_of_all -0.0242 [-0.0390, -0.0094] better; best of all missforest; fit 183 s a fold
- `biodeg_80nan`: K − remasker -0.0636 [-0.0775, -0.0497] better; remasker − strong +0.1125 [+0.0987, +0.1263] worse; remasker − bar +0.0348 [+0.0224, +0.0472] worse; remasker − tuned +0.0661 [+0.0542, +0.0780] worse; K − best_of_all +0.0489 [+0.0395, +0.0582] worse; best of all missforest; fit 161 s a fold
- `kc2_40nan`: K − remasker -0.1025 [-0.1414, -0.0635] better; remasker − strong +0.0421 [-0.0221, +0.1063] none; remasker − bar +0.0169 [-0.0573, +0.0911] none; remasker − tuned +0.0043 [-0.0693, +0.0780] none; K − best_of_all -0.0550 [-0.1099, -0.0002] better; best of all knn5/softimpute; fit 95 s a fold
- `kc2_80nan`: K − remasker -0.0589 [-0.0989, -0.0189] better; remasker − strong +0.0002 [-0.0490, +0.0493] none; remasker − bar -0.0977 [-0.1394, -0.0560] better; remasker − tuned -0.1070 [-0.1491, -0.0648] better; K − best_of_all -0.0479 [-0.0890, -0.0067] better; best of all missforest/remasker; fit 96 s a fold
- `letter_40nan`: K − remasker -0.0398 [-0.0424, -0.0373] better; remasker − strong -0.0472 [-0.0512, -0.0433] better; remasker − bar -0.0474 [-0.0502, -0.0445] better; remasker − tuned -0.0332 [-0.0358, -0.0305] better; K − best_of_all -0.0398 [-0.0424, -0.0373] better; best of all remasker; fit 3495 s a fold
- `letter_60nan`: K − remasker -0.0123 [-0.0138, -0.0107] better; remasker − strong -0.0719 [-0.0751, -0.0687] better; remasker − bar -0.0220 [-0.0237, -0.0203] better; remasker − tuned -0.0201 [-0.0218, -0.0184] better; K − best_of_all -0.0123 [-0.0138, -0.0107] better; best of all remasker; fit 1584 s a fold
- `letter_80nan`: K − remasker -0.0539 [-0.0562, -0.0516] better; remasker − strong +0.0395 [+0.0375, +0.0416] worse; remasker − bar +0.0400 [+0.0377, +0.0423] worse; remasker − tuned +0.0557 [+0.0536, +0.0578] worse; K − best_of_all -0.0139 [-0.0147, -0.0131] better; best of all hgb/lgbm; fit 1579 s a fold
  - K − remasker: better 12 / no difference 0 / worse 0 (fold-pair t, seed agreement)
  - remasker − strong: better 4 / no difference 2 / worse 6 (fold-pair t, seed agreement)
  - remasker − bar: better 5 / no difference 2 / worse 5 (fold-pair t, seed agreement)
  - remasker − tuned: better 5 / no difference 2 / worse 5 (fold-pair t, seed agreement)
  - K − best_of_all: better 10 / no difference 0 / worse 2 (fold-pair t, seed agreement)

Decision input: ReMasker better than the strong tier's best on 4 of 12 variants (at least 4, the third E48 asked for: propose it for the strong tier).
```
