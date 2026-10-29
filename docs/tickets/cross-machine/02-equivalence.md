# Is the machine's shift inside the SESOI? (E94; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** E86 ([01](01-gorgona8-medusa1.md)) found no verdict. The mean shift between the
  machines was small (+0.59% on imputation, +0.0039 on classification), but 60 and 36 fold pairs
  could not put it inside the SESOI. By its tie-break, each study runs whole on one machine. The
  user wants to split work between the two machines, "principalmente coisas altamente
  paralelizaveis" (2026-10-29, about 14:45 GMT-3). This check has the power to show equivalence,
  if it holds.
- **The user's go:** "sim, vamos" (2026-10-29, about 15:00 GMT-3).
- **Code:** no change to `src/`. `cross_machine.py --study e94` runs E86's code on more cells;
  `run_cross_machine_e94_queue.sh` runs four queues per machine.

## Design (fixed before the first cell)

- **MLflow off on both machines,** as in E86. Nothing reaches the store, and nothing is mirrored.
- **Imputation,** today's default with five folds: credit-g at 20 and 60nan, kc2 at 20 and 60nan,
  vehicle at 40 and 80nan, and biodeg_40nan, × seeds 1 to 10. That is 70 cells and 350 fold pairs.
- **Classification,** E71's arm A with three folds: vehicle at 20 and 40nan, kc2_40nan,
  credit-g_20nan and biodeg_60nan, × seeds 1 to 10. That is 50 cells and 150 fold pairs.
- **Sizing:** E86's per-fold sd (about 0.046 relative on imputation, about 0.03 on classification)
  puts the 95% half-width near 0.005 on imputation and 0.005 on classification. So a shift of
  E86's size would sit inside ±0.01.
- **Running:** 120 cells on each machine, four queues each.
  - medusa1 runs on GPU1, beside E95 on GPU0.
  - gorgona8 starts when E93's refits end.
  - About 1.5–2.5 h a machine.
  - No cell starts after 2026-10-30 02:00 GMT-3. A study cut there is partial.

## Measures and rule, fixed now (`cross_machine.py --study e94 --tally`)

- **The measure is E86's:** per fold, medusa − gorgona8.
  - Imputation: relative, on the default output's induced `impute_score`, SESOI 1%.
  - Classification: macro-F1, SESOI 0.01.
- **The rule, per task:**
  - **equivalent** (the 95% interval inside ±SESOI): later studies may split their cells between
    the machines by seed. Both arms of a pair stay on one machine, and the `host` tag (ADR 0034)
    records which.
  - **better or worse** (a systematic shift): each study runs whole on one machine.
  - **none:** each study runs whole on one machine. E86's tie-break by per-fold size is not
    reused: a fold can move as much as a seed change, and that is the seed noise every study
    already carries.
- **A split is allowed for a task only if that task is equivalent.** The two tasks are read
  separately.
- **Descriptive:** the largest per-fold difference, and the CPU baselines (hgb, knn5).

## Outcome

**Done 2026-10-29, 17:53 GMT-3: classification equivalent, imputation no verdict.**
- The 120 cells ran on each machine with none failed: medusa1 15:03–16:16 GMT-3 (GPU1), gorgona8
  16:02–17:51 GMT-3.
- medusa1's cells were copied to gorgona8 for the tally.

The tally, unedited (`cross_machine.py --study e94 --tally`; the run-to-run section is empty,
since E94 has no duplicates):

```
## imputation: medusa − gorgona8 on impute/induced/impute_score (relative)
- credit-g_20nan s1: +0.0221, +0.0260, -0.0202, -0.0031, +0.0427
- credit-g_20nan s2: -0.0283, -0.0297, -0.0125, -0.0076, +0.0524
- credit-g_20nan s3: -0.0178, -0.0185, +0.0135, -0.0289, -0.0573
- credit-g_20nan s4: -0.0092, -0.0034, +0.0107, +0.0108, -0.0268
- credit-g_20nan s5: +0.0181, +0.0209, +0.0526, -0.0197, +0.0037
- credit-g_20nan s6: -0.0459, +0.0025, +0.0099, +0.0031, -0.0036
- credit-g_20nan s7: -0.0203, -0.0382, +0.0209, +0.0242, -0.0149
- credit-g_20nan s8: +0.0155, -0.0023, -0.0312, +0.0490, +0.0096
- credit-g_20nan s9: +0.0195, -0.0238, +0.0056, +0.0072, -0.0302
- credit-g_20nan s10: -0.0123, -0.0178, -0.0207, +0.0059, -0.0157
- credit-g_60nan s1: -0.0027, -0.0043, +0.0147, +0.0336, +0.0250
- credit-g_60nan s2: +0.0149, -0.0169, -0.0003, -0.0054, +0.0145
- credit-g_60nan s3: -0.0149, +0.0010, -0.0284, +0.0264, +0.0069
- credit-g_60nan s4: -0.0390, +0.0207, +0.0047, -0.0241, +0.0169
- credit-g_60nan s5: +0.0027, +0.0026, +0.0475, -0.0067, +0.0225
- credit-g_60nan s6: +0.0076, -0.0052, +0.0215, -0.0081, +0.0177
- credit-g_60nan s7: -0.0028, -0.0106, -0.0483, +0.0049, +0.0001
- credit-g_60nan s8: +0.0002, -0.0277, +0.0022, +0.0328, -0.0226
- credit-g_60nan s9: -0.0048, -0.0107, -0.0130, -0.0060, -0.0072
- credit-g_60nan s10: +0.0026, +0.0206, +0.0247, -0.0034, -0.0099
- kc2_20nan s1: +0.0092, +0.0193, -0.0215, -0.0749, +0.0113
- kc2_20nan s2: +0.0566, +0.0155, +0.0727, -0.0482, -0.0015
- kc2_20nan s3: -0.0646, +0.0436, -0.0474, +0.0343, -0.1116
- kc2_20nan s4: -0.2322, -0.0003, +0.0085, -0.0026, -0.0530
- kc2_20nan s5: +0.0183, +0.0183, +0.0048, +0.4381, +0.0806
- kc2_20nan s6: +0.1489, +0.0048, +0.1085, +0.3925, +0.0418
- kc2_20nan s7: +0.1100, -0.0261, -0.0268, -0.0613, -0.0244
- kc2_20nan s8: +0.0198, +0.0380, +0.0059, +0.0149, -0.0733
- kc2_20nan s9: +0.1555, +0.5149, -0.0044, +0.0161, -0.0403
- kc2_20nan s10: -0.0035, -0.0076, -0.0584, +0.0597, -0.0054
- kc2_60nan s1: +0.0409, -0.0513, -0.0082, +0.1086, -0.0015
- kc2_60nan s2: -0.0066, +0.0090, +0.0829, -0.0352, -0.0337
- kc2_60nan s3: +0.0455, +0.0006, +0.2077, -0.0045, -0.0977
- kc2_60nan s4: -0.0366, +0.0166, +0.2078, +0.0418, +0.0487
- kc2_60nan s5: +0.0316, +0.0762, -0.0427, +0.0062, +0.0074
- kc2_60nan s6: +0.0771, +0.0011, -0.0514, -0.0496, +0.0545
- kc2_60nan s7: -0.0421, -0.0220, -0.0070, -0.0081, +0.0940
- kc2_60nan s8: -0.0838, +0.0125, -0.0180, +0.0374, +0.0271
- kc2_60nan s9: +0.0095, -0.0331, -0.0346, -0.0505, -0.1021
- kc2_60nan s10: -0.0211, +0.0227, -0.0335, +0.0647, -0.0069
- vehicle_40nan s1: +0.0024, +0.0008, +0.0077, +0.0167, +0.0148
- vehicle_40nan s2: -0.0154, -0.0305, +0.0226, +0.0041, +0.0043
- vehicle_40nan s3: -0.0126, +0.0189, +0.0205, -0.0105, +0.0102
- vehicle_40nan s4: +0.0088, -0.0877, +0.0217, +0.0090, +0.0416
- vehicle_40nan s5: +0.0076, +0.0100, +0.0093, +0.0314, -0.0246
- vehicle_40nan s6: -0.0051, -0.0013, -0.0270, -0.0185, -0.0007
- vehicle_40nan s7: +0.0173, -0.0078, -0.0440, +0.0018, -0.0086
- vehicle_40nan s8: +0.0178, -0.0577, +0.0250, +0.0434, -0.0964
- vehicle_40nan s9: -0.0120, +0.0030, +0.0174, +0.0180, -0.0233
- vehicle_40nan s10: +0.0088, +0.0011, -0.0181, -0.0135, +0.0080
- vehicle_80nan s1: +0.0002, -0.0101, +0.0020, -0.0126, -0.0213
- vehicle_80nan s2: -0.0059, +0.0120, +0.0271, -0.0047, +0.0060
- vehicle_80nan s3: -0.0006, -0.0142, -0.0056, +0.0023, -0.0222
- vehicle_80nan s4: -0.0130, +0.0193, -0.0013, -0.0149, -0.0151
- vehicle_80nan s5: +0.0136, +0.0089, +0.0328, +0.0190, +0.0196
- vehicle_80nan s6: -0.0011, -0.0102, +0.0035, +0.0153, +0.0274
- vehicle_80nan s7: -0.0023, -0.0035, -0.0062, -0.0214, -0.0011
- vehicle_80nan s8: +0.0092, +0.0038, -0.0148, +0.0051, +0.0049
- vehicle_80nan s9: +0.0020, -0.0014, +0.0079, -0.0079, +0.0298
- vehicle_80nan s10: +0.0077, +0.0277, -0.0040, +0.0029, +0.0022
- biodeg_40nan s1: +0.0122, +0.0206, +0.0314, +0.0065, -0.0246
- biodeg_40nan s2: -0.0129, -0.0286, +0.0128, -0.0182, +0.0381
- biodeg_40nan s3: -0.0203, -0.0358, +0.0349, -0.0460, +0.0216
- biodeg_40nan s4: -0.0130, -0.0281, -0.0105, +0.0130, +0.0113
- biodeg_40nan s5: +0.0103, +0.0146, -0.0007, +0.0315, -0.0175
- biodeg_40nan s6: -0.0497, +0.0004, -0.0431, +0.0310, +0.0022
- biodeg_40nan s7: +0.0257, -0.0098, +0.0182, +0.0195, -0.0192
- biodeg_40nan s8: +0.0152, -0.0057, -0.0021, +0.0065, +0.0050
- biodeg_40nan s9: +0.0023, -0.0125, -0.0124, +0.0453, -0.0610
- biodeg_40nan s10: +0.0233, +0.0526, -0.0298, -0.0036, -0.0083
- largest |difference| per metric: impute/induced/impute_score 0.5149; impute/masked/impute_score 0.3375; impute/induced/baseline/hgb/impute_score 0.0000; impute/induced/baseline/knn5/impute_score 0.0104
- pooled +0.0054 [-0.0006, +0.0113], seeds +0.0061, +0.0030, -0.0049, -0.0028, +0.0271, +0.0186, -0.0040, -0.0010, +0.0104, +0.0012, 350 fold pairs -> none (SESOI 0.01; fold-pair t, seed agreement)

## classification: medusa − gorgona8 on f1_macro
- vehicle_20nan s1: +0.0436, -0.0353, +0.0399
- vehicle_20nan s2: +0.0097, +0.0068, -0.0698
- vehicle_20nan s3: +0.0267, -0.0121, -0.0662
- vehicle_20nan s4: +0.0503, -0.0087, +0.0144
- vehicle_20nan s5: +0.0058, +0.0133, -0.0301
- vehicle_20nan s6: +0.0426, +0.0459, +0.0126
- vehicle_20nan s7: -0.0072, +0.0053, -0.0313
- vehicle_20nan s8: -0.0046, +0.0043, +0.0040
- vehicle_20nan s9: +0.0295, -0.0067, -0.0225
- vehicle_20nan s10: +0.0585, +0.0280, -0.0181
- vehicle_40nan s1: +0.0149, -0.0023, -0.0204
- vehicle_40nan s2: +0.0159, -0.0375, +0.0517
- vehicle_40nan s3: +0.0306, -0.0019, +0.0287
- vehicle_40nan s4: +0.0054, -0.0154, +0.0059
- vehicle_40nan s5: +0.0650, +0.0229, -0.0265
- vehicle_40nan s6: +0.0207, +0.0029, -0.0005
- vehicle_40nan s7: +0.0185, +0.0059, +0.0247
- vehicle_40nan s8: +0.0146, +0.0031, -0.0558
- vehicle_40nan s9: +0.0527, +0.0422, +0.0298
- vehicle_40nan s10: +0.0271, -0.0118, +0.0419
- kc2_40nan s1: -0.0413, -0.0262, +0.0474
- kc2_40nan s2: +0.0300, +0.0252, +0.0586
- kc2_40nan s3: -0.0158, -0.0220, -0.0238
- kc2_40nan s4: -0.0234, +0.0783, -0.0308
- kc2_40nan s5: -0.0134, +0.0610, -0.0099
- kc2_40nan s6: -0.0403, +0.0145, -0.0162
- kc2_40nan s7: +0.0167, +0.0481, +0.0019
- kc2_40nan s8: +0.0298, +0.0931, -0.0339
- kc2_40nan s9: -0.0369, +0.0394, +0.0111
- kc2_40nan s10: +0.0261, -0.1537, -0.2343
- credit-g_20nan s1: -0.0345, -0.0074, +0.0127
- credit-g_20nan s2: -0.0699, -0.0350, -0.0018
- credit-g_20nan s3: -0.0271, -0.0351, -0.0143
- credit-g_20nan s4: +0.0103, -0.0182, -0.0016
- credit-g_20nan s5: +0.0185, -0.0362, -0.0079
- credit-g_20nan s6: -0.0301, -0.0153, +0.0049
- credit-g_20nan s7: -0.0615, -0.0271, +0.0071
- credit-g_20nan s8: +0.0156, -0.0218, -0.0069
- credit-g_20nan s9: -0.0591, -0.0213, +0.0005
- credit-g_20nan s10: -0.0532, +0.0622, +0.0137
- biodeg_60nan s1: +0.0244, +0.0456, +0.0055
- biodeg_60nan s2: +0.0461, +0.0374, -0.0541
- biodeg_60nan s3: +0.0160, -0.0166, +0.0295
- biodeg_60nan s4: +0.0012, -0.0244, +0.0808
- biodeg_60nan s5: +0.0033, +0.0331, +0.0106
- biodeg_60nan s6: +0.0408, -0.0196, -0.0597
- biodeg_60nan s7: +0.0095, +0.0046, +0.0343
- biodeg_60nan s8: +0.0185, -0.0008, +0.0252
- biodeg_60nan s9: +0.0083, +0.0215, -0.0508
- biodeg_60nan s10: +0.0097, -0.0255, -0.0270
- largest |difference| per metric: f1_macro 0.2343; accuracy 0.2874
- pooled +0.0009 [-0.0056, +0.0073], seeds +0.0044, +0.0009, -0.0069, +0.0083, +0.0073, +0.0002, +0.0033, +0.0056, +0.0025, -0.0171, 150 fold pairs -> equivalent (SESOI 0.01; fold-pair t, seed agreement)

## run to run, same machine
```

- **Classification: equivalent.**
  - medusa − gorgona8 on macro-F1 is +0.0009 [−0.0056, +0.0073] over 150 fold pairs, inside ±0.01.
  - **By the rule, a classification study may split its cells between the machines by seed.** Both
    arms of a pair stay on one machine, and the `host` tag records which.
- **Imputation: no verdict.**
  - +0.54% [−0.06%, +1.13%] over 350 fold pairs. The upper bound passes the 1% SESOI, so
    imputation studies still run whole on one machine.
  - The mean is close to E86's (+0.59% there), and six of ten seeds lean positive.
- **Descriptive:**
  - kc2_20nan carries the heavy tail, with folds at +0.51, +0.44 and +0.39. kc2 is the smallest
    table, and at 20nan a fold scores few cells.
  - The CPU baselines agree on hgb (0), and knn5 differs by up to 0.0104.
  - These are read, not acted on: no variant is dropped after the fact.
- **What follows:**
  - Classification studies (step zero, T41's paper half, E92-like TabPFN runs) can use both
    servers by seed.
  - Imputation keeps one machine a study. A rule on medians or with kc2 excluded would be a new
    pre-registered check, and that is the user's call.
