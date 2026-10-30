# Stage 1: the CPU bar on the sweep's folds (E97; pre-registered 2026-10-29)

This ticket was written and committed before the study's first variant. Eight variants ran first
as a smoke check, in a scratch directory outside the study, to time the script and confirm that it
handles electricity_80nan's all-missing rows. Only their per-arm means were printed, and they do
not count.

- **Why:** step zero's plan ([2026-10-23](../../plans/2026-10-23-classification-step-zero.md),
  Stage 1) needs a non-deep bar for classification under missingness before Stage 4 tunes
  anything. It also answers how far untuned GBDTs with native NaN, and impute-then-learn, sit from
  TRIDENT on the same folds.
- **The user's go:** the plan's settled decisions (2026-10-29, 21:20 GMT-3). Stage 1 needs no
  decision beyond the plan, and the bar is LightGBM and HistGradientBoosting (XGBoost is out).
- **Code:** no change to `src/`. `scripts/experiments/cls_stage1_bar.py` (`run`, `--tally`); its
  docstring fixes the arms and settings.

## Design (fixed before the first variant)

- **Folds:** the sweep's 3 folds, `build_folds(frame, 'class', 3, 42)`, on all 45 variants (9 tables
  × 00–80nan).
- **Arms (20 a fold), all untuned, `class_weight='balanced'`, `random_state` 42:**
  - **native NaN:** `native/hgb` and `native/lgbm`, with categories as training-split vocabulary
    codes and NaN for gaps or unseen values;
  - **impute, then learn:** `mode_lgbm`, `mean_mode` and `missforest`, each with and without
    `isna()` indicators, each followed by HGB, LightGBM and RandomForest.
  - The boosters stop early on the fold's validation split: at most 100 rounds, patience 10, with
    the validation loss weighted by the training split's balanced class weights.
- **Running:** gorgona8's CPU only, so every GBDT number comes from one machine. Six parallel
  processes of four threads each, the largest tables first; about 30 min. Nothing is written to the
  store.

## Reading, fixed now (`cls_stage1_bar.py --tally`)

- **Descriptive only:** one seed and three folds a variant, as the plan says. No call is made.
- **Per level:** each arm's paired (arm − TRIDENT) test macro-F1, with a 95% t-interval over the
  pooled fold pairs.
  - **The headline TRIDENT reference is E71's arm A at seed 42:** the sweep's configuration rerun
    at HEAD on gorgona8, on the same folds.
  - The old sweep's own folds are printed beside it as the plan names them. Stage 0 found they do
    not reproduce at HEAD, so they are descriptive only.
- **Also printed:** each arm's mean macro-F1 by level, and the indicator contrast (with − without)
  as an MCAR sanity control, not evidence of informative missingness.
- **What it feeds:** the choice of the GBDT bar and of the impute-then-learn arms that Stage 4
  tunes. That choice is made in Stage 4's pre-registration.

## Outcome

**Done 2026-10-29, 22:14 GMT-3 (descriptive).** All 45 variants ran on gorgona8's CPU, 21:41–22:13
GMT-3, in six processes, with no failure. The tally, unedited (`cls_stage1_bar.py --tally`; also
`tally.txt` in the state directory):

```
Descriptive only: one seed, three folds a variant; intervals are the 95% t over fold pairs, no call.

## 00nan (9 of 9 tables)
- TRIDENT (sweep) mean macro-F1 0.8577 over 9 tables
- TRIDENT (E71 A) mean macro-F1 0.8539 over 9 tables
- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:
  - native/hgb             +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - native/lgbm            +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - mode_lgbm/hgb          +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - mode_lgbm/lgbm         +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - mode_lgbm/rf           +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
  - mode_lgbm+ind/hgb      +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - mode_lgbm+ind/lgbm     +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - mode_lgbm+ind/rf       +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
  - mean_mode/hgb          +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - mean_mode/lgbm         +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - mean_mode/rf           +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
  - mean_mode+ind/hgb      +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - mean_mode+ind/lgbm     +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - mean_mode+ind/rf       +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
  - missforest/hgb         +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - missforest/lgbm        +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - missforest/rf          +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
  - missforest+ind/hgb     +0.0035 [-0.0030, +0.0100] | +0.0073 [-0.0001, +0.0147]; mean 0.8612
  - missforest+ind/lgbm    +0.0001 [-0.0092, +0.0094] | +0.0039 [-0.0035, +0.0112]; mean 0.8577
  - missforest+ind/rf      +0.0075 [-0.0011, +0.0161] | +0.0113 [+0.0030, +0.0196]; mean 0.8652
- indicator contrast (with − without), the MCAR sanity control:
  - mode_lgbm: hgb +0.0000 [+0.0000, +0.0000]; lgbm +0.0000 [+0.0000, +0.0000]; rf +0.0000 [+0.0000, +0.0000]
  - mean_mode: hgb +0.0000 [+0.0000, +0.0000]; lgbm +0.0000 [+0.0000, +0.0000]; rf +0.0000 [+0.0000, +0.0000]
  - missforest: hgb +0.0000 [+0.0000, +0.0000]; lgbm +0.0000 [+0.0000, +0.0000]; rf +0.0000 [+0.0000, +0.0000]

## 20nan (9 of 9 tables)
- TRIDENT (sweep) mean macro-F1 0.7987 over 9 tables
- TRIDENT (E71 A) mean macro-F1 0.7976 over 9 tables
- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:
  - native/hgb             +0.0144 [+0.0053, +0.0234] | +0.0154 [+0.0075, +0.0234]; mean 0.8131
  - native/lgbm            +0.0118 [+0.0028, +0.0208] | +0.0129 [+0.0026, +0.0232]; mean 0.8106
  - mode_lgbm/hgb          +0.0226 [+0.0118, +0.0334] | +0.0237 [+0.0127, +0.0347]; mean 0.8213
  - mode_lgbm/lgbm         +0.0162 [+0.0033, +0.0291] | +0.0173 [+0.0060, +0.0286]; mean 0.8150
  - mode_lgbm/rf           +0.0215 [+0.0123, +0.0308] | +0.0226 [+0.0127, +0.0325]; mean 0.8203
  - mode_lgbm+ind/hgb      +0.0228 [+0.0133, +0.0323] | +0.0239 [+0.0131, +0.0347]; mean 0.8215
  - mode_lgbm+ind/lgbm     +0.0153 [+0.0042, +0.0264] | +0.0164 [+0.0054, +0.0274]; mean 0.8141
  - mode_lgbm+ind/rf       +0.0176 [+0.0080, +0.0271] | +0.0186 [+0.0084, +0.0288]; mean 0.8163
  - mean_mode/hgb          +0.0145 [+0.0045, +0.0245] | +0.0156 [+0.0074, +0.0238]; mean 0.8133
  - mean_mode/lgbm         +0.0120 [+0.0009, +0.0232] | +0.0131 [+0.0044, +0.0218]; mean 0.8107
  - mean_mode/rf           +0.0101 [+0.0007, +0.0195] | +0.0112 [+0.0011, +0.0213]; mean 0.8088
  - mean_mode+ind/hgb      +0.0148 [+0.0037, +0.0259] | +0.0159 [+0.0062, +0.0255]; mean 0.8135
  - mean_mode+ind/lgbm     +0.0081 [-0.0019, +0.0181] | +0.0092 [-0.0000, +0.0184]; mean 0.8069
  - mean_mode+ind/rf       +0.0064 [-0.0022, +0.0151] | +0.0075 [-0.0026, +0.0176]; mean 0.8052
  - missforest/hgb         +0.0242 [+0.0140, +0.0344] | +0.0253 [+0.0162, +0.0343]; mean 0.8229
  - missforest/lgbm        +0.0206 [+0.0110, +0.0303] | +0.0217 [+0.0118, +0.0316]; mean 0.8194
  - missforest/rf          +0.0264 [+0.0180, +0.0348] | +0.0275 [+0.0182, +0.0368]; mean 0.8251
  - missforest+ind/hgb     +0.0254 [+0.0158, +0.0350] | +0.0265 [+0.0174, +0.0356]; mean 0.8241
  - missforest+ind/lgbm    +0.0213 [+0.0106, +0.0320] | +0.0223 [+0.0113, +0.0334]; mean 0.8200
  - missforest+ind/rf      +0.0216 [+0.0125, +0.0308] | +0.0227 [+0.0129, +0.0325]; mean 0.8204
- indicator contrast (with − without), the MCAR sanity control:
  - mode_lgbm: hgb +0.0002 [-0.0046, +0.0051]; lgbm -0.0009 [-0.0057, +0.0039]; rf -0.0040 [-0.0095, +0.0015]
  - mean_mode: hgb +0.0002 [-0.0058, +0.0063]; lgbm -0.0039 [-0.0093, +0.0016]; rf -0.0037 [-0.0086, +0.0013]
  - missforest: hgb +0.0012 [-0.0040, +0.0064]; lgbm +0.0006 [-0.0036, +0.0049]; rf -0.0048 [-0.0103, +0.0007]

## 40nan (9 of 9 tables)
- TRIDENT (sweep) mean macro-F1 0.7424 over 9 tables
- TRIDENT (E71 A) mean macro-F1 0.7405 over 9 tables
- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:
  - native/hgb             +0.0174 [+0.0065, +0.0284] | +0.0193 [+0.0096, +0.0290]; mean 0.7598
  - native/lgbm            +0.0116 [+0.0032, +0.0200] | +0.0135 [+0.0032, +0.0238]; mean 0.7540
  - mode_lgbm/hgb          +0.0259 [+0.0116, +0.0402] | +0.0278 [+0.0149, +0.0407]; mean 0.7683
  - mode_lgbm/lgbm         +0.0190 [+0.0016, +0.0365] | +0.0209 [+0.0040, +0.0379]; mean 0.7614
  - mode_lgbm/rf           +0.0247 [+0.0079, +0.0416] | +0.0267 [+0.0117, +0.0416]; mean 0.7671
  - mode_lgbm+ind/hgb      +0.0259 [+0.0106, +0.0412] | +0.0278 [+0.0122, +0.0435]; mean 0.7683
  - mode_lgbm+ind/lgbm     +0.0204 [+0.0047, +0.0362] | +0.0224 [+0.0076, +0.0371]; mean 0.7628
  - mode_lgbm+ind/rf       +0.0236 [+0.0068, +0.0404] | +0.0255 [+0.0118, +0.0392]; mean 0.7660
  - mean_mode/hgb          +0.0079 [-0.0031, +0.0189] | +0.0098 [-0.0038, +0.0234]; mean 0.7503
  - mean_mode/lgbm         +0.0069 [-0.0055, +0.0194] | +0.0089 [-0.0068, +0.0245]; mean 0.7493
  - mean_mode/rf           +0.0020 [-0.0092, +0.0132] | +0.0040 [-0.0079, +0.0158]; mean 0.7444
  - mean_mode+ind/hgb      +0.0107 [+0.0039, +0.0175] | +0.0126 [+0.0037, +0.0214]; mean 0.7530
  - mean_mode+ind/lgbm     +0.0105 [-0.0028, +0.0239] | +0.0125 [-0.0033, +0.0282]; mean 0.7529
  - mean_mode+ind/rf       +0.0062 [-0.0067, +0.0191] | +0.0081 [-0.0031, +0.0193]; mean 0.7486
  - missforest/hgb         +0.0200 [+0.0024, +0.0375] | +0.0219 [+0.0050, +0.0388]; mean 0.7624
  - missforest/lgbm        +0.0164 [-0.0010, +0.0337] | +0.0183 [+0.0016, +0.0349]; mean 0.7587
  - missforest/rf          +0.0204 [+0.0032, +0.0376] | +0.0223 [+0.0061, +0.0385]; mean 0.7628
  - missforest+ind/hgb     +0.0274 [+0.0113, +0.0435] | +0.0294 [+0.0127, +0.0460]; mean 0.7698
  - missforest+ind/lgbm    +0.0202 [+0.0055, +0.0348] | +0.0221 [+0.0085, +0.0357]; mean 0.7626
  - missforest+ind/rf      +0.0264 [+0.0118, +0.0409] | +0.0283 [+0.0145, +0.0420]; mean 0.7687
- indicator contrast (with − without), the MCAR sanity control:
  - mode_lgbm: hgb +0.0000 [-0.0059, +0.0060]; lgbm +0.0014 [-0.0029, +0.0057]; rf -0.0011 [-0.0067, +0.0045]
  - mean_mode: hgb +0.0028 [-0.0039, +0.0094]; lgbm +0.0036 [-0.0055, +0.0127]; rf +0.0042 [-0.0038, +0.0122]
  - missforest: hgb +0.0075 [+0.0029, +0.0120]; lgbm +0.0038 [-0.0011, +0.0087]; rf +0.0060 [-0.0007, +0.0126]

## 60nan (9 of 9 tables)
- TRIDENT (sweep) mean macro-F1 0.6606 over 9 tables
- TRIDENT (E71 A) mean macro-F1 0.6538 over 9 tables
- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:
  - native/hgb             +0.0263 [+0.0141, +0.0386] | +0.0331 [+0.0217, +0.0445]; mean 0.6869
  - native/lgbm            +0.0197 [+0.0061, +0.0333] | +0.0265 [+0.0135, +0.0396]; mean 0.6803
  - mode_lgbm/hgb          +0.0240 [+0.0087, +0.0394] | +0.0308 [+0.0123, +0.0494]; mean 0.6846
  - mode_lgbm/lgbm         +0.0213 [+0.0019, +0.0407] | +0.0281 [+0.0067, +0.0495]; mean 0.6819
  - mode_lgbm/rf           +0.0233 [+0.0019, +0.0446] | +0.0301 [+0.0058, +0.0544]; mean 0.6839
  - mode_lgbm+ind/hgb      +0.0299 [+0.0136, +0.0462] | +0.0367 [+0.0163, +0.0572]; mean 0.6905
  - mode_lgbm+ind/lgbm     +0.0261 [+0.0084, +0.0438] | +0.0329 [+0.0135, +0.0524]; mean 0.6867
  - mode_lgbm+ind/rf       +0.0269 [+0.0087, +0.0451] | +0.0337 [+0.0124, +0.0550]; mean 0.6875
  - mean_mode/hgb          +0.0183 [+0.0028, +0.0337] | +0.0251 [+0.0112, +0.0389]; mean 0.6789
  - mean_mode/lgbm         +0.0199 [+0.0032, +0.0366] | +0.0267 [+0.0114, +0.0421]; mean 0.6805
  - mean_mode/rf           +0.0110 [-0.0077, +0.0296] | +0.0178 [-0.0026, +0.0381]; mean 0.6716
  - mean_mode+ind/hgb      +0.0215 [+0.0071, +0.0360] | +0.0284 [+0.0154, +0.0413]; mean 0.6821
  - mean_mode+ind/lgbm     +0.0243 [+0.0096, +0.0390] | +0.0311 [+0.0170, +0.0452]; mean 0.6849
  - mean_mode+ind/rf       +0.0131 [-0.0027, +0.0290] | +0.0199 [+0.0005, +0.0394]; mean 0.6737
  - missforest/hgb         +0.0201 [-0.0048, +0.0451] | +0.0270 [+0.0035, +0.0504]; mean 0.6808
  - missforest/lgbm        +0.0206 [-0.0051, +0.0462] | +0.0274 [+0.0037, +0.0511]; mean 0.6812
  - missforest/rf          +0.0153 [-0.0093, +0.0400] | +0.0222 [-0.0028, +0.0471]; mean 0.6759
  - missforest+ind/hgb     +0.0304 [+0.0080, +0.0529] | +0.0373 [+0.0158, +0.0587]; mean 0.6910
  - missforest+ind/lgbm    +0.0248 [+0.0002, +0.0493] | +0.0316 [+0.0081, +0.0551]; mean 0.6854
  - missforest+ind/rf      +0.0192 [-0.0028, +0.0412] | +0.0260 [+0.0034, +0.0487]; mean 0.6798
- indicator contrast (with − without), the MCAR sanity control:
  - mode_lgbm: hgb +0.0059 [+0.0013, +0.0105]; lgbm +0.0048 [-0.0004, +0.0100]; rf +0.0036 [-0.0041, +0.0113]
  - mean_mode: hgb +0.0033 [-0.0031, +0.0096]; lgbm +0.0044 [-0.0021, +0.0108]; rf +0.0022 [-0.0059, +0.0102]
  - missforest: hgb +0.0103 [+0.0035, +0.0171]; lgbm +0.0042 [-0.0034, +0.0119]; rf +0.0039 [-0.0021, +0.0098]

## 80nan (9 of 9 tables)
- TRIDENT (sweep) mean macro-F1 0.5306 over 9 tables
- TRIDENT (E71 A) mean macro-F1 0.5405 over 9 tables
- arm − TRIDENT, test macro-F1 (sweep | E71 A s42), and the arm's mean macro-F1:
  - native/hgb             +0.0420 [+0.0123, +0.0717] | +0.0321 [+0.0162, +0.0479]; mean 0.5726
  - native/lgbm            +0.0414 [+0.0074, +0.0755] | +0.0315 [+0.0127, +0.0502]; mean 0.5720
  - mode_lgbm/hgb          +0.0224 [-0.0116, +0.0564] | +0.0124 [-0.0081, +0.0330]; mean 0.5530
  - mode_lgbm/lgbm         +0.0235 [-0.0100, +0.0570] | +0.0136 [-0.0075, +0.0346]; mean 0.5541
  - mode_lgbm/rf           +0.0254 [-0.0097, +0.0605] | +0.0154 [-0.0072, +0.0380]; mean 0.5559
  - mode_lgbm+ind/hgb      +0.0308 [-0.0014, +0.0631] | +0.0209 [+0.0009, +0.0409]; mean 0.5614
  - mode_lgbm+ind/lgbm     +0.0229 [-0.0099, +0.0557] | +0.0130 [-0.0079, +0.0338]; mean 0.5535
  - mode_lgbm+ind/rf       +0.0281 [-0.0056, +0.0619] | +0.0182 [-0.0043, +0.0407]; mean 0.5587
  - mean_mode/hgb          +0.0363 [+0.0072, +0.0654] | +0.0263 [+0.0107, +0.0419]; mean 0.5668
  - mean_mode/lgbm         +0.0371 [+0.0079, +0.0663] | +0.0272 [+0.0084, +0.0459]; mean 0.5677
  - mean_mode/rf           +0.0298 [-0.0089, +0.0685] | +0.0199 [-0.0058, +0.0455]; mean 0.5604
  - mean_mode+ind/hgb      +0.0386 [+0.0091, +0.0681] | +0.0287 [+0.0129, +0.0445]; mean 0.5692
  - mean_mode+ind/lgbm     +0.0379 [+0.0091, +0.0668] | +0.0280 [+0.0095, +0.0465]; mean 0.5685
  - mean_mode+ind/rf       +0.0244 [-0.0116, +0.0603] | +0.0144 [-0.0105, +0.0393]; mean 0.5549
  - missforest/hgb         +0.0312 [-0.0046, +0.0671] | +0.0213 [-0.0060, +0.0486]; mean 0.5618
  - missforest/lgbm        +0.0270 [-0.0068, +0.0609] | +0.0171 [-0.0077, +0.0418]; mean 0.5576
  - missforest/rf          +0.0288 [-0.0094, +0.0670] | +0.0188 [-0.0096, +0.0473]; mean 0.5594
  - missforest+ind/hgb     +0.0368 [+0.0031, +0.0706] | +0.0269 [+0.0003, +0.0535]; mean 0.5674
  - missforest+ind/lgbm    +0.0330 [+0.0008, +0.0653] | +0.0231 [-0.0007, +0.0469]; mean 0.5636
  - missforest+ind/rf      +0.0351 [-0.0043, +0.0744] | +0.0251 [-0.0057, +0.0559]; mean 0.5656
- indicator contrast (with − without), the MCAR sanity control:
  - mode_lgbm: hgb +0.0084 [-0.0001, +0.0170]; lgbm -0.0006 [-0.0093, +0.0081]; rf +0.0028 [-0.0035, +0.0090]
  - mean_mode: hgb +0.0024 [-0.0051, +0.0098]; lgbm +0.0008 [-0.0093, +0.0110]; rf -0.0054 [-0.0130, +0.0022]
  - missforest: hgb +0.0056 [-0.0024, +0.0136]; lgbm +0.0060 [-0.0014, +0.0135]; rf +0.0063 [-0.0027, +0.0153]
```

- **Untuned GBDTs lead TRIDENT's classification at every rate with gaps.** native/hgb − E71's
  arm A (seed 42):

  | rate | native/hgb − TRIDENT |
  |---|---|
  | 00nan | +0.0073 [−0.0001, +0.0147], about a tie |
  | 20nan | +0.0154 [+0.0075, +0.0234] |
  | 40nan | +0.0193 [+0.0096, +0.0290] |
  | 60nan | +0.0331 [+0.0217, +0.0445] |
  | 80nan | +0.0321 [+0.0162, +0.0479] |

  The lead grows with missingness. These are TRIDENT's untuned old-sweep configuration against
  untuned GBDTs: one seed and three folds, so descriptive.
- **The best arm by level:**
  - missForest + indicators + HGB at 20–60nan (+0.027 to +0.037 against E71 A);
  - native HGB at 80nan (+0.032), where every imputer loses ground.
  - mean/mode is the weakest imputer from 40nan on.
- **The indicator contrast (the MCAR sanity control)** is null at 20–40nan for most arms. It is
  small and positive for HGB at 60–80nan (up to +0.010 with missForest). A tree can use the flag to
  tell imputed values from observed ones, so this is not evidence of informative missingness.
- **Against the old sweep's folds** the picture is the same, within about 0.01.
- **What it feeds into Stage 4's pre-registration:**
  - the GBDT bar is native HGB and LightGBM;
  - the impute-then-learn arm to tune is missForest + indicators with HGB, the strongest at 20–60nan.
  - H1c's question is whether TRIDENT tuned on validation closes a gap of about 2–3 points that
    untuned GBDTs open.
