# Is the imputation shift inside the SESOI once kc2 is left out? (E96; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** E94 ([02](02-equivalence.md)) found classification equivalent across the servers, and
  imputation without a verdict: +0.54% [−0.06%, +1.13%] over 350 fold pairs.
  - The tail came from kc2_20nan, with folds at +0.51, +0.44 and +0.39.
  - Read post hoc, the other five variants give −0.02% [−0.30%, +0.26%] over 250 pairs. That read
    chose its variants after seeing the data, so it changes no rule.
  - This study repeats those five variants on fresh seeds, with the rule fixed in advance.
- **The exclusion is declared, and motivated by E94:** kc2 is left out because E94 showed its folds
  to be heavy-tailed. A positive result covers the tables tested here, not kc2.
- **The user's go:** "rode o mais barato então!" (2026-10-29, about 18:15 GMT-3), choosing this
  version over one with spambase and kr-vs-kp (about 5–6 h).
- **Code:** no change to `src/`.
  - `cross_machine.py --study e96` runs E86's code.
  - `run_cross_machine_e94_queue.sh` takes `STUDY=e96`; E94's default is unchanged.

## Design (fixed before the first cell)

- **MLflow off on both machines,** as E86 and E94.
- **Cells:** imputation only, today's default with five folds. The variants are credit-g at 20 and
  60nan, vehicle at 40 and 80nan, and biodeg_40nan, × seeds 11 to 20: 50 cells and 250 fold pairs a
  machine.
- **Running:** four queues on each machine.
  - medusa1 uses GPU1, while E95 keeps GPU0.
  - About 1 h on gorgona8, by these variants' times in E94.
  - No cell starts after 2026-10-30 02:00 GMT-3.

## Measures and rule, fixed now (`cross_machine.py --study e96 --tally`)

- **The measure:** E86's, per fold, medusa − gorgona8, relative on the default output's induced
  `impute_score`. The SESOI is 1%.
- **Equivalent** (the 95% interval inside ±1%): imputation studies whose tables are among
  credit-g, vehicle and biodeg may split their cells between the machines by seed.
  - Both arms of a pair stay on one machine, and the `host` tag records which.
  - Other tables keep one machine a study until they are checked. That holds above all for kc2.
- **Anything else:** imputation keeps one machine a study.

## Outcome

**Done 2026-10-29, 19:36 GMT-3: equivalent.** The 50 cells ran on each machine with none failed,
18:13–19:36 GMT-3; medusa1's were copied to gorgona8 for the tally. Unedited
(`cross_machine.py --study e96 --tally`):

```
## imputation: medusa − gorgona8 on impute/induced/impute_score (relative)
- credit-g_20nan s11: +0.0184, -0.0121, +0.0168, -0.0366, -0.0109
- credit-g_20nan s12: +0.0036, +0.0054, +0.0011, +0.0048, +0.0058
- credit-g_20nan s13: +0.0233, -0.0120, +0.0040, -0.0239, +0.0224
- credit-g_20nan s14: +0.0378, +0.0009, +0.0203, +0.0115, -0.0120
- credit-g_20nan s15: -0.0148, +0.0169, -0.0296, +0.0273, +0.0031
- credit-g_20nan s16: -0.0196, +0.0225, +0.0038, -0.0129, -0.0050
- credit-g_20nan s17: -0.0077, +0.0014, +0.0016, -0.0273, -0.0136
- credit-g_20nan s18: +0.0138, -0.0433, +0.0492, +0.0208, -0.0181
- credit-g_20nan s19: -0.0123, -0.0099, +0.0500, -0.0170, +0.0166
- credit-g_20nan s20: +0.0664, -0.0225, -0.0465, -0.0412, -0.0157
- credit-g_60nan s11: -0.0042, -0.0268, +0.0223, -0.0032, -0.0053
- credit-g_60nan s12: -0.0030, -0.0346, +0.0537, +0.0093, -0.0150
- credit-g_60nan s13: +0.0062, +0.0010, -0.0031, -0.0031, -0.0015
- credit-g_60nan s14: -0.0317, +0.0134, -0.0147, +0.0124, +0.0001
- credit-g_60nan s15: +0.0241, -0.0216, -0.0128, +0.0361, -0.0339
- credit-g_60nan s16: -0.0010, +0.0264, -0.0047, +0.0107, +0.0087
- credit-g_60nan s17: +0.0090, -0.0048, -0.0017, -0.0031, -0.0219
- credit-g_60nan s18: -0.0246, -0.0215, +0.0166, +0.0273, +0.0205
- credit-g_60nan s19: +0.0001, +0.0011, +0.0029, -0.0218, -0.0086
- credit-g_60nan s20: -0.0023, +0.0128, +0.0009, -0.0197, +0.0270
- vehicle_40nan s11: -0.0927, -0.0729, -0.0232, +0.0587, -0.1033
- vehicle_40nan s12: +0.0158, +0.0126, -0.0047, +0.0066, +0.0170
- vehicle_40nan s13: -0.0242, -0.0686, -0.0224, -0.0126, -0.0007
- vehicle_40nan s14: +0.0002, +0.0173, -0.0187, -0.0180, +0.0022
- vehicle_40nan s15: +0.0066, -0.0714, -0.0004, -0.1047, +0.0139
- vehicle_40nan s16: +0.0348, +0.0278, +0.0795, -0.0502, +0.0515
- vehicle_40nan s17: +0.0150, +0.0332, +0.0119, +0.0029, +0.0198
- vehicle_40nan s18: +0.0325, +0.0601, -0.0042, +0.0833, +0.0155
- vehicle_40nan s19: +0.0326, +0.0336, +0.0068, +0.0248, -0.0611
- vehicle_40nan s20: +0.0189, +0.0202, +0.0514, -0.0543, +0.0215
- vehicle_80nan s11: -0.0108, -0.0245, +0.0054, +0.0304, -0.0077
- vehicle_80nan s12: +0.0006, +0.0272, +0.0202, +0.0012, +0.0099
- vehicle_80nan s13: +0.0044, +0.0195, +0.0047, -0.0122, +0.0156
- vehicle_80nan s14: -0.0155, +0.0149, -0.0091, -0.0129, -0.0058
- vehicle_80nan s15: +0.0094, +0.0013, -0.0026, +0.0126, -0.0244
- vehicle_80nan s16: +0.0023, -0.0189, +0.0127, +0.0001, -0.0121
- vehicle_80nan s17: -0.0135, -0.0058, -0.0026, +0.0047, -0.0055
- vehicle_80nan s18: -0.0087, -0.0028, -0.0113, +0.0035, +0.0081
- vehicle_80nan s19: -0.0103, +0.0027, -0.0177, +0.0094, -0.0032
- vehicle_80nan s20: +0.0187, -0.0036, -0.0142, +0.0059, +0.0088
- biodeg_40nan s11: +0.0329, -0.0018, +0.0200, -0.0295, -0.0099
- biodeg_40nan s12: -0.0372, +0.0237, -0.0136, -0.0496, +0.0159
- biodeg_40nan s13: -0.0030, -0.0459, +0.0181, +0.0019, +0.0888
- biodeg_40nan s14: +0.0265, -0.0178, -0.0049, -0.0221, +0.0312
- biodeg_40nan s15: +0.0353, +0.0067, +0.0054, +0.0103, +0.0090
- biodeg_40nan s16: +0.0136, +0.0139, +0.0050, -0.0368, +0.0027
- biodeg_40nan s17: -0.0206, -0.0145, -0.0242, +0.0262, -0.0144
- biodeg_40nan s18: -0.0036, -0.0064, +0.0190, +0.0589, -0.0034
- biodeg_40nan s19: +0.0332, -0.0237, +0.0076, -0.0366, -0.0058
- biodeg_40nan s20: -0.0274, -0.0210, +0.0639, -0.0100, -0.0055
- largest |difference| per metric: impute/induced/impute_score 0.1047; impute/masked/impute_score 0.1487; impute/induced/baseline/hgb/impute_score 0.0000; impute/induced/baseline/knn5/impute_score 0.0096
- pooled +0.0004 [-0.0030, +0.0038], seeds -0.0108, +0.0031, -0.0009, +0.0002, -0.0039, +0.0062, -0.0022, +0.0113, -0.0003, +0.0013, 250 fold pairs -> equivalent (SESOI 0.01; fold-pair t, seed agreement)

## run to run, same machine
```

- **Equivalent:** medusa − gorgona8 is +0.04% [−0.30%, +0.38%] over 250 fold pairs, well inside
  ±1%. Its seeds split 5 to 5 in sign, and the largest fold difference is 0.105.
- **By the rule:** imputation studies whose tables are among credit-g, vehicle and biodeg may split
  their cells between the machines by seed.
  - Both arms of a pair stay on one machine, and the `host` tag records which.
  - kc2 and the unchecked tables (spambase, kr-vs-kp, pendigits, letter, electricity) keep one
    machine a study.
- **With E94:** classification may split, and imputation may split on these three tables.
