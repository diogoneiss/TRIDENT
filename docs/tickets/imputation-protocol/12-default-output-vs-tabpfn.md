# The default output against TabPFN: KC and H (E57; pre-registered 2026-10-21)

This ticket was written and committed before the reading was run.

- **Why:** E55 compared TabPFN against K, the uncalibrated checkpoint. Since ADR 0027, the default
  output at a missing rate of 0.3 or more is KC: K after the calibration fitted on the validation
  gaps. Every confirmation variant is at 40nan or above.
  - Calibration helped most at 60 and 80nan (E40), exactly where TabPFN beat K (E55).
  - H, the loss checkpoint, is the deployable readout for claims that must transfer (ADR 0024).
- **The user's go:** "Faça 1 e 2 em sequência agora" (2026-10-21, about 09:10 GMT-3).
- **No run.** The reading uses E44's runs, which logged KC (`--score_calibrated`) and H
  (`--score_checkpoints`), and E55's TabPFN CSVs. Code: a new section in
  `scripts/experiments/tabpfn_report.py`; E55's own printout is unchanged.

## Reading, fixed now

- **The measures:** KC − TabPFN and H − TabPFN, beside K − TabPFN.
  - On the induced `impute_score`, 15 fold pairs, the verdict rule and the 1% SESOI.
  - On E44's 15 confirmation variants, with the tally and every call that changes from K to KC.
  - The development variants at 40nan and above are read where their runs carry KC (descriptive).
- **What it states:** KC − TabPFN is the thesis's statement against TabPFN at the default output
  (ADR 0027). H − TabPFN is the deployable one (ADR 0024).
- **No decision follows.** ADR 0028 stands; this sets which readout the thesis's sentence uses.

## Outcome (2026-10-21, 10:04 GMT-3)

**The printout** (`scripts/experiments/tabpfn_report.py`, its E57 section, unedited):

```text
## E57: confirmation set (E44), seeds 707/808/909, KC and H against TabPFN
- `pendigits_40nan`: K − tabpfn -0.0107 [-0.0137, -0.0078] better; KC − tabpfn -0.0119 [-0.0148, -0.0091] better; H − tabpfn -0.0096 [-0.0129, -0.0064] better
- `pendigits_60nan`: K − tabpfn -0.0180 [-0.0202, -0.0158] better; KC − tabpfn -0.0216 [-0.0238, -0.0193] better; H − tabpfn -0.0143 [-0.0176, -0.0109] better
- `pendigits_80nan`: K − tabpfn +0.0162 [+0.0150, +0.0174] worse; KC − tabpfn +0.0121 [+0.0107, +0.0134] worse; H − tabpfn +0.0195 [+0.0163, +0.0228] worse
- `letter_40nan`: K − tabpfn -0.0344 [-0.0369, -0.0319] better; KC − tabpfn -0.0358 [-0.0382, -0.0334] better; H − tabpfn -0.0334 [-0.0357, -0.0311] better
- `letter_60nan`: K − tabpfn -0.0301 [-0.0317, -0.0284] better; KC − tabpfn -0.0331 [-0.0347, -0.0315] better; H − tabpfn -0.0234 [-0.0264, -0.0204] better
- `letter_80nan`: K − tabpfn +0.0020 [+0.0006, +0.0034] worse; KC − tabpfn -0.0002 [-0.0016, +0.0012] equivalent; H − tabpfn +0.0039 [+0.0018, +0.0061] worse
- `electricity_40nan`: K − tabpfn +0.0303 [-0.0166, +0.0773] none; KC − tabpfn +0.0208 [-0.0125, +0.0541] none; H − tabpfn +0.0285 [-0.0181, +0.0751] none
- `electricity_60nan`: K − tabpfn +0.0038 [-0.0022, +0.0098] none; KC − tabpfn +0.0033 [-0.0021, +0.0087] none; H − tabpfn +0.0129 [-0.0041, +0.0299] none
- `electricity_80nan`: K − tabpfn +0.0103 [+0.0032, +0.0175] worse; KC − tabpfn +0.0087 [+0.0025, +0.0150] worse; H − tabpfn +0.0110 [+0.0042, +0.0178] worse
- `vehicle_40nan`: K − tabpfn -0.0003 [-0.0134, +0.0127] none; KC − tabpfn +0.0002 [-0.0119, +0.0123] none; H − tabpfn +0.0087 [-0.0081, +0.0255] none
- `vehicle_80nan`: K − tabpfn +0.0530 [+0.0443, +0.0618] worse; KC − tabpfn +0.0315 [+0.0246, +0.0383] worse; H − tabpfn +0.0610 [+0.0520, +0.0700] worse
- `biodeg_40nan`: K − tabpfn -0.0414 [-0.0674, -0.0154] better; KC − tabpfn -0.0461 [-0.0722, -0.0200] better; H − tabpfn -0.0416 [-0.0652, -0.0179] better
- `biodeg_80nan`: K − tabpfn +0.0525 [+0.0404, +0.0646] worse; KC − tabpfn +0.0279 [+0.0166, +0.0392] worse; H − tabpfn +0.0638 [+0.0505, +0.0771] worse
- `kc2_40nan`: K − tabpfn -0.0016 [-0.0493, +0.0461] none; KC − tabpfn -0.0021 [-0.0477, +0.0436] none; H − tabpfn +0.0098 [-0.0413, +0.0609] none
- `kc2_80nan`: K − tabpfn -0.0577 [-0.1038, -0.0117] better; KC − tabpfn -0.0650 [-0.1052, -0.0248] better; H − tabpfn -0.0336 [-0.0854, +0.0183] none
  - K − tabpfn: better 6 / no difference 4 / worse 5 (fold-pair t, seed agreement)
  - KC − tabpfn: better 6 / no difference 4 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
  - H − tabpfn: better 5 / no difference 5 / worse 5 (fold-pair t, seed agreement)
  - calls that change from K to KC: ['letter_80nan: K worse -> KC equivalent']

## E57: development set at 40nan and above, seeds 101/202/303 (descriptive), KC and H against TabPFN
- `kr-vs-kp_40nan`: no KC on these runs
- `kr-vs-kp_60nan`: K − tabpfn +0.0432 [+0.0352, +0.0512] worse; KC − tabpfn +0.0311 [+0.0255, +0.0368] worse; H − tabpfn +0.0437 [+0.0362, +0.0513] worse
- `kr-vs-kp_80nan`: K − tabpfn +0.0404 [+0.0330, +0.0479] worse; KC − tabpfn +0.0360 [+0.0262, +0.0458] worse; H − tabpfn +0.0693 [+0.0521, +0.0866] worse
- `spambase_40nan`: no KC on these runs
- `spambase_60nan`: K − tabpfn -0.0337 [-0.0412, -0.0262] better; KC − tabpfn -0.0332 [-0.0404, -0.0259] better; H − tabpfn -0.0303 [-0.0378, -0.0229] better
- `spambase_80nan`: K − tabpfn -0.0074 [-0.0115, -0.0033] better; KC − tabpfn -0.0103 [-0.0139, -0.0067] better; H − tabpfn +0.0025 [-0.0042, +0.0092] equivalent
- `credit-g_40nan`: no KC on these runs
- `credit-g_60nan`: K − tabpfn +0.0422 [+0.0284, +0.0560] worse; KC − tabpfn +0.0326 [+0.0246, +0.0406] worse; H − tabpfn +0.0493 [+0.0372, +0.0615] worse
- `credit-g_80nan`: K − tabpfn +0.0141 [+0.0065, +0.0217] worse; KC − tabpfn +0.0156 [+0.0092, +0.0220] worse; H − tabpfn +0.0341 [+0.0191, +0.0491] worse
- `vehicle_60nan`: K − tabpfn +0.0052 [-0.0049, +0.0152] none; KC − tabpfn +0.0143 [-0.0058, +0.0343] none; H − tabpfn +0.0239 [+0.0052, +0.0427] worse
- `biodeg_60nan`: K − tabpfn -0.0200 [-0.0322, -0.0077] better; KC − tabpfn -0.0087 [-0.0272, +0.0099] none; H − tabpfn -0.0049 [-0.0247, +0.0150] none
- `kc2_60nan`: K − tabpfn -0.0684 [-0.1178, -0.0189] better; KC − tabpfn -0.0574 [-0.1138, -0.0010] better; H − tabpfn -0.0514 [-0.1101, +0.0072] none
  - K − tabpfn: better 4 / no difference 1 / worse 4 (fold-pair t, seed agreement)
  - KC − tabpfn: better 3 / no difference 2 / worse 4 (fold-pair t, seed agreement)
  - H − tabpfn: better 1 / no difference 2 / equivalent 1 / worse 5 (fold-pair t, seed agreement)
  - calls that change from K to KC: ['biodeg_60nan: K better -> KC none']
```

**Reading:**

- **At the default output (KC), on confirmation:** better 6, no difference 4, equivalent 1, worse
  4. Against K: better 6, no difference 4, worse 5.
  - **The one call that changes is letter_80nan:** worse under K (+0.0020), equivalent under KC
    (−0.0002).
- **Calibration narrows TabPFN's 80nan leads without closing them:**
  - vehicle_80nan: +0.053 → +0.032;
  - biodeg_80nan: +0.053 → +0.028;
  - pendigits_80nan: +0.016 → +0.012;
  - electricity_80nan: +0.010 → +0.009.
  - At 40 and 60nan, KC widens K's leads a little: pendigits_60nan −0.018 → −0.022, letter_60nan
    −0.030 → −0.033.
- **The deployable readout (H):** better 5, no difference 5, worse 5 on confirmation. kc2_80nan's win
  under K becomes no difference.
- **Development, descriptive:** KC − TabPFN better 3, no difference 2, worse 4 on the nine variants
  with KC.

**What the thesis's sentence uses:** KC − TabPFN, the default output (ADR 0027).

- On the 15 confirmation variants, the default output beats TabPFN on 6 and loses on 4. All 4
  losses are at 80nan: pendigits, electricity, vehicle and biodeg.
- At 40 and 60nan it never loses.

