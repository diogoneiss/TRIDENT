# MaCoDE as an external deep baseline (E48; pre-registered 2026-10-19)

This ticket was written and committed before the first scored cell of the study.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2g, item 10; D21).
  - The user chose MaCoDE as the next deep baseline on 2026-10-19 ("vamos agora para t28 e
    macode"). That is the go for vendoring it.
  - TabPFN v2 and its separate venv are not part of this.
- **Why:** every opponent so far is a classical imputer. MaCoDE (An et al., AAAI 2025) is the
  nearest prior art to TRIDENT.
  - It is a masked transformer over discretised rows: one merged mask-or-missing word per column
    (no `[NULL]`/`[MASK]` split) and quantile-bin targets.
  - It handles mixed and categorical tables natively.
  - It has no pre-training stage: the masked model is the imputer.
- **Code:** `src/training/macode.py`, vendored from github.com/an-seunghwan/MaCoDE at commit
  `8867879` (MIT, notice kept), with an inductive wrapper. The changes, each stated in the module:
  - fit on the training fold alone and fill any frame;
  - the bins passed to the data (the released code kept 10);
  - a table of one column type works;
  - the paper's appendix settings: 50 bins, width 128, 4 heads, 2 layers, 500 epochs, batch 1,024,
    AdamW 1e-3 with weight decay 1e-3;
  - its own random streams;
  - the posterior-mean fill for numbers and the argmax for categories.
  - Unit tests: `tests/unit/test_macode.py`. No training path of TRIDENT changes.

## Design (fixed before the first scored cell)

- **Arm:** MaCoDE as above, untuned, as the paper ran it. TRIDENT's configuration was tuned on the
  development set, so this is stated beside every comparison.
- **Cells:**
  - **The development set** at seeds 101/202/303, the thesis table's primary set: 21 variants and
    63 (variant, seed) pairs. K comes from the runs `report_common.development_source` names.
  - **E44's confirmation set** at seeds 707/808/909: 15 variants and 45 pairs. K comes from E44's
    runs.
  - 108 pairs and 540 fits in all, each fold rebuilt by `recompute_folds` exactly as its run
    built it.
- **Driver:** `scripts/experiments/macode_run.py`, on the GPU, off-store. Its CSVs go to
  `/scratch2/diogoneiss/trident-2026-10-19-macode/csv/` (lanes: `scripts/experiments/run_macode_lanes.py`).
- **Pre-launch smoke (2026-10-19), timing only, no score read:**
  - credit-g_60nan at seed 101: 53 s for 5 folds.
  - electricity_20nan at seed 101: 901 s, about 180 s a fold. The largest table sets the pace.
  - The smoke's CSVs were deleted unread.
- **Cost estimate:** by rows and columns, about 8 process-hours for the 108 pairs: about 3.7 h for
  the development set and 4.4 h for the confirmation set.
  - Three processes share the GPU, so the wall time is about 4–5 h.
  - It is off-store and runs in the GPU gap after E47.
  - No cutoff; a relaunch skips finished files.

## Measures

All on the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI
(ADR 0020), through `scripts/experiments/macode_report.py`.

1. **Primary:** K − MaCoDE per variant, tallied on each set.
2. MaCoDE − the strong tier's best (missForest and E41's arms), − the bar, and − the tuned bar
   (E42).
3. K − the best of all, MaCoDE included.
4. MaCoDE's fit time per fold.

## Reading, fixed now

**Descriptive on the development set,** where TRIDENT was tuned and MaCoDE was not.

**On the confirmation set:**

- **K − MaCoDE is the thesis's deep-baseline statement,** reported as a tally beside E44's.
- **Whether MaCoDE joins the strong tier:**
  - If it is better than the strong tier's best on at least 5 of the 15 confirmation variants,
    propose (by an ADR amending 0021) that it join the strong tier as an offline member, like the
    tuned bar.
  - Otherwise it is reported as an external deep baseline only.
- **No default changes either way.**

## Launch

- **Launched at 07:57 GMT-3 on 2026-10-19** on `113ad9a`: three GPU lanes of 36 pairs each, about
  163 estimated minutes per lane. Confirmation folds go first in every lane.
- Progress is in `/scratch2/diogoneiss/trident-2026-10-19-macode/progress.log`.

## Outcome (2026-10-19, 19:01 GMT-3)

**Run.** All 108 (variant, seed) pairs ran from 07:57 to 18:58 GMT-3, in three GPU lanes. None
failed.

- **It took about 11 h, not the 2.7 h estimated.** The three processes saturated the GPU, so they
  ran no faster together than one after another. The smoke had also underestimated the large
  tables: letter took about 38 minutes a seed and electricity about 46.
- **The report first stopped on the fit-time column's name** (the driver prefixes it with the
  population). That was fixed before any score was read.

**The printout** (`scripts/experiments/macode_report.py`): each line is a mean difference over the
15 fold pairs of the induced `impute_score` (lower is better), with the verdict rule and the 1%
SESOI.

```text
## Development set, seeds 101/202/303 (labelled variants: none on this set)
- `kr-vs-kp_20nan`: K − macode -0.0203 [-0.0340, -0.0067] better; macode − strong -0.0546 [-0.0640, -0.0452] better; macode − bar +0.0010 [-0.0133, +0.0153] none; macode − tuned +0.0072 [-0.0072, +0.0216] none; K − best_of_all -0.0186 [-0.0316, -0.0056] better; best of all lgbm/macode; fit 177 s a fold
- `kr-vs-kp_40nan`: K − macode -0.0089 [-0.0157, -0.0021] better; macode − strong -0.0559 [-0.0674, -0.0444] better; macode − bar -0.0356 [-0.0458, -0.0254] better; macode − tuned -0.0135 [-0.0255, -0.0016] better; K − best_of_all -0.0089 [-0.0157, -0.0021] better; best of all macode; fit 176 s a fold
- `kr-vs-kp_60nan`: K − macode -0.0125 [-0.0220, -0.0030] better; macode − strong -0.0228 [-0.0313, -0.0142] better; macode − bar -0.0254 [-0.0355, -0.0153] better; macode − tuned +0.0262 [+0.0166, +0.0358] worse; K − best_of_all -0.0125 [-0.0220, -0.0030] better; best of all macode; fit 176 s a fold
- `kr-vs-kp_80nan`: K − macode -0.0812 [-0.0943, -0.0682] better; macode − strong +0.0628 [+0.0497, +0.0758] worse; macode − bar +0.0628 [+0.0497, +0.0758] worse; macode − tuned +0.0852 [+0.0743, +0.0962] worse; K − best_of_all -0.0185 [-0.0268, -0.0102] better; best of all condexp; fit 176 s a fold
- `spambase_20nan`: K − macode -0.0257 [-0.0379, -0.0136] better; macode − strong +0.0205 [+0.0144, +0.0265] worse; macode − bar +0.0196 [+0.0116, +0.0276] worse; macode − tuned +0.0266 [+0.0202, +0.0330] worse; K − best_of_all -0.0047 [-0.0162, +0.0067] none; best of all hgb/missforest; fit 407 s a fold
- `spambase_40nan`: K − macode -0.0223 [-0.0288, -0.0158] better; macode − strong -0.0135 [-0.0203, -0.0067] better; macode − bar -0.0094 [-0.0170, -0.0018] better; macode − tuned +0.0095 [+0.0043, +0.0146] worse; K − best_of_all -0.0223 [-0.0288, -0.0158] better; best of all macode; fit 406 s a fold
- `spambase_60nan`: K − macode -0.0232 [-0.0286, -0.0177] better; macode − strong +0.0069 [+0.0001, +0.0136] worse; macode − bar -0.0178 [-0.0215, -0.0140] better; macode − tuned +0.0092 [+0.0051, +0.0134] worse; K − best_of_all -0.0163 [-0.0200, -0.0126] better; best of all condexp; fit 406 s a fold
- `spambase_80nan`: K − macode -0.0325 [-0.0380, -0.0270] better; macode − strong +0.0180 [+0.0109, +0.0252] worse; macode − bar +0.0171 [+0.0129, +0.0213] worse; macode − tuned +0.0265 [+0.0215, +0.0315] worse; K − best_of_all -0.0138 [-0.0183, -0.0092] better; best of all condexp/mean_mode; fit 407 s a fold
- `credit-g_20nan`: K − macode -0.1102 [-0.1250, -0.0953] better; macode − strong +0.0946 [+0.0781, +0.1111] worse; macode − bar +0.0715 [+0.0604, +0.0826] worse; macode − tuned +0.0904 [+0.0737, +0.1071] worse; K − best_of_all -0.0156 [-0.0336, +0.0024] none; best of all missforest; fit 31 s a fold
- `credit-g_40nan`: K − macode -0.1083 [-0.1236, -0.0929] better; macode − strong +0.0978 [+0.0828, +0.1129] worse; macode − bar +0.0646 [+0.0470, +0.0822] worse; macode − tuned +0.0908 [+0.0775, +0.1041] worse; K − best_of_all -0.0105 [-0.0193, -0.0016] better; best of all missforest; fit 31 s a fold
- `credit-g_60nan`: K − macode -0.1038 [-0.1206, -0.0869] better; macode − strong +0.1087 [+0.0950, +0.1224] worse; macode − bar +0.0970 [+0.0835, +0.1105] worse; macode − tuned +0.1167 [+0.1052, +0.1283] worse; K − best_of_all +0.0049 [-0.0098, +0.0197] none; best of all condexp; fit 31 s a fold
- `credit-g_80nan`: K − macode -0.1365 [-0.1479, -0.1251] better; macode − strong +0.1489 [+0.1375, +0.1603] worse; macode − bar +0.1470 [+0.1357, +0.1583] worse; macode − tuned +0.1470 [+0.1357, +0.1583] worse; K − best_of_all +0.0124 [+0.0062, +0.0186] worse; best of all condexp; fit 30 s a fold
- `vehicle_20nan`: K − macode -0.1677 [-0.1945, -0.1409] better; macode − strong +0.1330 [+0.1166, +0.1494] worse; macode − bar +0.1152 [+0.0903, +0.1400] worse; macode − tuned +0.1200 [+0.0959, +0.1441] worse; K − best_of_all -0.0348 [-0.0604, -0.0091] better; best of all missforest; fit 26 s a fold
- `vehicle_60nan`: K − macode -0.1506 [-0.1657, -0.1356] better; macode − strong +0.1396 [+0.1216, +0.1575] worse; macode − bar +0.0875 [+0.0785, +0.0964] worse; macode − tuned +0.0939 [+0.0842, +0.1036] worse; K − best_of_all -0.0111 [-0.0247, +0.0025] none; best of all missforest; fit 24 s a fold
- `biodeg_20nan`: K − macode -0.1074 [-0.1355, -0.0793] better; macode − strong +0.0703 [+0.0538, +0.0868] worse; macode − bar +0.0485 [+0.0332, +0.0639] worse; macode − tuned +0.0551 [+0.0369, +0.0733] worse; K − best_of_all -0.0371 [-0.0586, -0.0156] better; best of all missforest; fit 67 s a fold
- `biodeg_60nan`: K − macode -0.1256 [-0.1367, -0.1144] better; macode − strong +0.1371 [+0.1255, +0.1487] worse; macode − bar +0.0278 [+0.0185, +0.0371] worse; macode − tuned +0.0443 [+0.0348, +0.0538] worse; K − best_of_all +0.0115 [-0.0024, +0.0255] none; best of all missforest; fit 66 s a fold
- `kc2_20nan`: K − macode -0.1043 [-0.2239, +0.0153] none; macode − strong +0.1038 [+0.0309, +0.1768] worse; macode − bar +0.1191 [+0.0402, +0.1979] worse; macode − tuned +0.0864 [-0.0020, +0.1748] none; K − best_of_all +0.0148 [-0.0840, +0.1136] none; best of all knn5; fit 24 s a fold
- `kc2_60nan`: K − macode -0.1265 [-0.1680, -0.0850] better; macode − strong +0.0886 [+0.0334, +0.1438] worse; macode − bar -0.0448 [-0.0977, +0.0080] none; macode − tuned -0.0657 [-0.1212, -0.0103] better; K − best_of_all -0.0378 [-0.0613, -0.0144] better; best of all softimpute; fit 18 s a fold
- `pendigits_20nan`: K − macode -0.1093 [-0.1166, -0.1020] better; macode − strong +0.0955 [+0.0880, +0.1031] worse; macode − bar +0.0786 [+0.0723, +0.0849] worse; macode − tuned +0.0857 [+0.0788, +0.0925] worse; K − best_of_all -0.0138 [-0.0182, -0.0094] better; best of all missforest; fit 249 s a fold
- `letter_20nan`: K − macode -0.0041 [-0.0073, -0.0009] better; macode − strong -0.0257 [-0.0292, -0.0222] better; macode − bar -0.0350 [-0.0377, -0.0324] better; macode − tuned -0.0314 [-0.0346, -0.0282] better; K − best_of_all -0.0041 [-0.0073, -0.0009] better; best of all macode; fit 450 s a fold
- `electricity_20nan`: K − macode -0.0341 [-0.0437, -0.0246] better; macode − strong +0.0332 [+0.0232, +0.0431] worse; macode − bar +0.0503 [+0.0432, +0.0573] worse; macode − tuned +0.0729 [+0.0660, +0.0797] worse; K − best_of_all +0.0161 [+0.0056, +0.0267] worse; best of all hgb; fit 547 s a fold
  - K − macode: better 20 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - macode − strong: better 5 / no difference 0 / worse 16 (fold-pair t, seed agreement)
  - macode − bar: better 5 / no difference 2 / worse 14 (fold-pair t, seed agreement)
  - macode − tuned: better 3 / no difference 2 / worse 16 (fold-pair t, seed agreement)
  - K − best_of_all: better 13 / no difference 6 / worse 2 (fold-pair t, seed agreement)

## Confirmation set (E44), seeds 707/808/909
- `pendigits_40nan`: K − macode -0.1409 [-0.1449, -0.1369] better; macode − strong +0.0703 [+0.0659, +0.0746] worse; macode − bar +0.0755 [+0.0720, +0.0790] worse; macode − tuned +0.0881 [+0.0845, +0.0917] worse; K − best_of_all -0.0653 [-0.0680, -0.0627] better; best of all hgb/lgbm; fit 250 s a fold
- `pendigits_60nan`: K − macode -0.1645 [-0.1677, -0.1614] better; macode − strong +0.0430 [+0.0381, +0.0479] worse; macode − bar +0.1077 [+0.1047, +0.1106] worse; macode − tuned +0.1108 [+0.1079, +0.1137] worse; K − best_of_all -0.0568 [-0.0587, -0.0549] better; best of all hgb/lgbm; fit 250 s a fold
- `pendigits_80nan`: K − macode -0.1433 [-0.1460, -0.1405] better; macode − strong +0.1254 [+0.1227, +0.1282] worse; macode − bar +0.1188 [+0.1158, +0.1217] worse; macode − tuned +0.1395 [+0.1369, +0.1420] worse; K − best_of_all -0.0178 [-0.0197, -0.0159] better; best of all condexp; fit 250 s a fold
- `letter_40nan`: K − macode -0.0246 [-0.0264, -0.0227] better; macode − strong -0.0308 [-0.0344, -0.0273] better; macode − bar -0.0310 [-0.0328, -0.0291] better; macode − tuned -0.0168 [-0.0185, -0.0151] better; K − best_of_all -0.0246 [-0.0264, -0.0227] better; best of all macode; fit 449 s a fold
- `letter_60nan`: K − macode -0.0609 [-0.0634, -0.0585] better; macode − strong -0.0232 [-0.0264, -0.0199] better; macode − bar +0.0267 [+0.0250, +0.0284] worse; macode − tuned +0.0285 [+0.0269, +0.0301] worse; K − best_of_all -0.0343 [-0.0361, -0.0324] better; best of all lgbm; fit 450 s a fold
- `letter_80nan`: K − macode -0.0810 [-0.0834, -0.0786] better; macode − strong +0.0667 [+0.0644, +0.0689] worse; macode − bar +0.0671 [+0.0647, +0.0695] worse; macode − tuned +0.0828 [+0.0805, +0.0851] worse; K − best_of_all -0.0139 [-0.0147, -0.0131] better; best of all hgb/lgbm; fit 449 s a fold
- `electricity_40nan`: K − macode -0.0140 [-0.0608, +0.0328] none; macode − strong -0.0024 [-0.0161, +0.0112] none; macode − bar +0.0576 [+0.0462, +0.0689] worse; macode − tuned +0.0636 [+0.0573, +0.0698] worse; K − best_of_all +0.0436 [-0.0013, +0.0885] none; best of all hgb/lgbm; fit 548 s a fold
- `electricity_60nan`: K − macode -0.0479 [-0.0567, -0.0390] better; macode − strong -0.0423 [-0.0516, -0.0331] better; macode − bar +0.0590 [+0.0512, +0.0669] worse; macode − tuned +0.0586 [+0.0513, +0.0660] worse; K − best_of_all +0.0112 [+0.0059, +0.0164] worse; best of all hgb/lgbm; fit 549 s a fold
- `electricity_80nan`: K − macode -0.0380 [-0.0466, -0.0295] better; macode − strong -0.0189 [-0.0212, -0.0165] better; macode − bar +0.0384 [+0.0354, +0.0414] worse; macode − tuned +0.0430 [+0.0403, +0.0457] worse; K − best_of_all +0.0004 [-0.0093, +0.0100] none; best of all hgb/lgbm; fit 548 s a fold
- `vehicle_40nan`: K − macode -0.1718 [-0.1936, -0.1501] better; macode − strong +0.1462 [+0.1285, +0.1640] worse; macode − bar +0.0984 [+0.0829, +0.1140] worse; macode − tuned +0.0980 [+0.0848, +0.1112] worse; K − best_of_all -0.0256 [-0.0466, -0.0045] better; best of all missforest; fit 26 s a fold
- `vehicle_80nan`: K − macode -0.1338 [-0.1488, -0.1189] better; macode − strong +0.1763 [+0.1638, +0.1888] worse; macode − bar +0.1261 [+0.1149, +0.1372] worse; macode − tuned +0.1241 [+0.1135, +0.1347] worse; K − best_of_all +0.0425 [+0.0343, +0.0506] worse; best of all missforest; fit 25 s a fold
- `biodeg_40nan`: K − macode -0.1225 [-0.1470, -0.0980] better; macode − strong +0.0983 [+0.0764, +0.1201] worse; macode − bar +0.0465 [+0.0328, +0.0603] worse; macode − tuned +0.0499 [+0.0360, +0.0639] worse; K − best_of_all -0.0242 [-0.0390, -0.0094] better; best of all missforest; fit 67 s a fold
- `biodeg_80nan`: K − macode -0.0546 [-0.0654, -0.0438] better; macode − strong +0.1035 [+0.0945, +0.1126] worse; macode − bar +0.0258 [+0.0172, +0.0344] worse; macode − tuned +0.0571 [+0.0487, +0.0655] worse; K − best_of_all +0.0489 [+0.0395, +0.0582] worse; best of all missforest; fit 66 s a fold
- `kc2_40nan`: K − macode -0.1339 [-0.2005, -0.0673] better; macode − strong +0.0735 [+0.0179, +0.1292] worse; macode − bar +0.0483 [-0.0285, +0.1251] none; macode − tuned +0.0358 [-0.0440, +0.1155] none; K − best_of_all -0.0550 [-0.1099, -0.0002] better; best of all knn5/softimpute; fit 24 s a fold
- `kc2_80nan`: K − macode -0.1437 [-0.1855, -0.1019] better; macode − strong +0.0850 [+0.0383, +0.1316] worse; macode − bar -0.0129 [-0.0449, +0.0191] none; macode − tuned -0.0221 [-0.0493, +0.0051] none; K − best_of_all -0.0587 [-0.0970, -0.0205] better; best of all missforest; fit 25 s a fold
  - K − macode: better 14 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - macode − strong: better 4 / no difference 1 / worse 10 (fold-pair t, seed agreement)
  - macode − bar: better 1 / no difference 2 / worse 12 (fold-pair t, seed agreement)
  - macode − tuned: better 1 / no difference 2 / worse 12 (fold-pair t, seed agreement)
  - K − best_of_all: better 10 / no difference 2 / worse 3 (fold-pair t, seed agreement)

Decision input: MaCoDE better than the strong tier's best on 4 of 15 confirmation variants (fewer than 5: an external deep baseline only).
```

**The thesis's deep-baseline statement (primary):**

- **TRIDENT beats MaCoDE on every variant with a verdict:**
  - development set: better on 20 of 21, no difference on kc2_20nan;
  - confirmation set: better on 14 of 15, no difference on electricity_40nan.
- **The margins:**
  - large on the small and mixed tables: −0.05 to −0.17 on credit-g, vehicle, biodeg, kc2 and
    pendigits (−0.10 or more on all but biodeg_80nan);
  - small on letter (−0.004 to −0.08), electricity (−0.01 to −0.05) and kr-vs-kp at 20–60nan
    (−0.01 to −0.02).

**By the rule fixed in advance, MaCoDE is an external deep baseline only.** It beats the strong
tier's best on 4 of the 15 confirmation variants, fewer than the 5 that would have led to an ADR
amending 0021.

**Descriptively:**

- **MaCoDE is strong where the tables are large or categorical.**
  - It beats the strong tier's best on kr-vs-kp at 20, 40 and 60nan, spambase_40nan and
    letter_20nan (development), and on letter at 40 and 60nan and electricity at 60 and 80nan
    (confirmation).
  - It is the best of all baselines on kr-vs-kp_40nan, kr-vs-kp_60nan, spambase_40nan,
    letter_20nan and letter_40nan.
- **It is weak on the small tables.** On credit-g, vehicle, biodeg, kc2 and pendigits it is worse
  than the weak bar, by 0.03 to 0.15. On credit-g at 60 and 80nan it is worse than mean/mode.
  - The paper's fixed settings (500 epochs, no dropout, no early stopping) on a few hundred
    training rows may overfit. That is a hypothesis, not measured here.
- **K against the best of all, MaCoDE included,** is unchanged from the table without it:
  - better 13 / no difference 6 / worse 2 on the development set;
  - 10 / 2 / 3 on the confirmation set.
  - Where MaCoDE is the best baseline, K still beats it.
- **The fit took 18–550 s a fold on the RTX 3090 Ti,** from kc2 to electricity, with three fits
  sharing the GPU.

**Limits:** MaCoDE ran untuned, at the paper's settings, as the paper ran it. TRIDENT's
configuration was tuned on the development set, so the development margins favour it. The
confirmation set is the fairer reading, and there the result is the same.

**No default changes.**

