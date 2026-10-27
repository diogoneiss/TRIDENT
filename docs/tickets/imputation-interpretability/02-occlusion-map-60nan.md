# Does the occlusion map hold at 60nan? (E79; pre-registered 2026-10-26)

This ticket was written and committed before the study's first cell, and before any result of
E78 ([01](01-occlusion-map.md)) was read.

- **Why:** E78 tests the map at 40nan only. ADR 0030's scope runs to 60nan, and at 60nan a target
  has fewer observed sources in each row, so a map could stop being faithful there.
- **The user's go:** "voce pode executar coisas até 7h30, estarei fora até lá, aproveite o tempo
  extra de gpu para rodar mais variantes por exemplo" (2026-10-26, before 23:22 GMT-3).
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e79` runs and reads this study with E78's code; E78's default path is
    unchanged.
  - `generate_synthblock.py` now also writes `synthblock_60nan` with seed 42 + 60. The 40nan file
    is rewritten byte for byte (same md5).
  - `run_occlusion_queue.sh` takes `STUDY=e79` and a `DEADLINE`.

## Design (fixed before the first cell)

- **Cells:** E78's ten tables at 60nan, 30 cells. Each table runs at the seed set whose earlier
  runs K is checked against:
  - `synthblock_60nan`, pendigits, letter and electricity at 707, 808 and 909 (E44 for the real
    ones);
  - vehicle, biodeg, kc2, kr-vs-kp, spambase and credit-g at 101, 202 and 303 (the thesis table,
    `development_source`).
  - Today's default configuration, five folds, `--score_remask`, read in place exactly as E78.
- **Running:** four GPU queues (`STUDY=e79`). Each starts when the E78 queue of the same number
  completes. MLflow tag `experiment=occlusion-map-60nan-2026-10-27`; state in
  `/scratch2/diogoneiss/trident-2026-10-27-occlusion-60/`.
- **The deadline, fixed now and by the clock alone:** no queue starts a cell after
  2026-10-27, 07:00 GMT-3, so the GPU is free when the user returns at 07:30.
  - Cells not started by then run later, when the user decides; the reading waits for all 30.
  - The deadline never depends on a result: no cell is read before all 30 have run.

## Measures and reading

E78's, unchanged, through `occlusion_map.py --study e79 --tally`:
- **B, primary:** D with the top 3 sources ranked on the validation gaps, 20 random draws, the 1%
  SESOI. Supported if faithful on 7 or more of the 9 real variants and not faithful on none; refuted
  if faithful on 4 or fewer; mixed otherwise.
- **C, gate:** the block AUROC on `synthblock_60nan`. It recovers the blocks if the interval lies
  above 0.8 with every seed above 0.8. It is inconclusive (floor) if K is not below `mean_mode`.
- **Descriptive:** the same list as E78.

**What each reading leads to:**
- **E78 and E79 both supported:** the map's faithfulness holds over the scope ADR 0030 uses (40 and
  60nan, MCAR).
- **E79 weaker than E78:** the claim is limited to 40nan, and the tables where it fails are named.
- Neither study changes a default.

## Outcome (2026-10-27, 07:00 GMT-3): supported, at the bar

- **Ran:** 30 of 30 cells, from 2026-10-27, 02:46 GMT-3 to 06:57 GMT-3, before the 07:00
  deadline; none failed. K reproduced the earlier run of the same cell to 0.0 on all 27 real cells.
- **B, primary: supported, exactly at the bar.** The map is faithful on 7 of 9 and not faithful on
  none.
  - **Faithful:** D with the top 3 is +0.13 to +0.24 on pendigits, letter, electricity, vehicle
    and biodeg, and +0.04 to +0.08 on kr-vs-kp and spambase.
  - **Inconclusive:** kc2, +0.012 [+0.001, +0.023], with two seeds near 0. credit-g, +0.019
    [+0.009, +0.029], whose interval reaches below the 1% SESOI.
  - **Descriptive:** kc2's top-1 D is "no", at −0.003 [−0.014, +0.007].
- **C, gate: recovers the blocks.** The AUROC is 0.914 [0.901, 0.926], with every seed at 0.908 or
  above. K is 0.818 against `mean_mode`'s 1.0. |Spearman| reaches 0.864, closer to the map than at
  40nan (0.847 against 0.979).
- **Against E78:** every effect is smaller at 60nan.
  - D (top 3) falls by 29–43% on pendigits, letter, electricity, vehicle and biodeg. It falls by
    51% on spambase and by 67–74% on kc2, kr-vs-kp and credit-g.
  - kc2, the weakest at 40nan, and credit-g, already small there, fall to inconclusive.
  - Seed stability stays 0.75–0.995; attention against occlusion stays 0.69–0.96.
- **What it leads to:** E78 and E79 are both supported, so the map's faithfulness holds over 40 and
  60nan, MCAR, by the rule fixed before either ran.
  - It holds with a qualification the numbers force: at 60nan it is at the bar, and on kc2 and
    credit-g it is not shown.
  - A report states the claim as "faithful on 9 of 9 at 40nan and 7 of 9 at 60nan", names the two
    tables and says the effect shrinks with missingness.
  - No default changes.

The reader's printout (`occlusion_map.py --study e79 --tally`), unedited:

```text
- `synthblock_60nan`: D top3 +0.280 [+0.270, +0.290] seeds +0.270, +0.283, +0.287 -> faithful: yes; D top1 +0.100 [+0.094, +0.105] seeds +0.093, +0.102, +0.105 -> faithful: yes
  - truth (C): AUROC 0.914 [0.901, 0.926] seeds 0.911, 0.922, 0.908; K 0.8176 against mean_mode 1.0000 -> recovers the blocks: yes
  - descriptive: |Spearman| on the training split, same AUROC (numerical pairs only): 0.864 [0.849, 0.880]
  - map: seed stability +0.769; attention against occlusion +0.701; |Spearman| against occlusion +0.446 (within target, mean)
  - top sources: a1 <- a3 0.246, b3 0.070, n1 0.064; a2 <- a4 0.434, a1 0.424, n1 0.061; a3 <- a1 0.584, c2 0.098, a4 0.080; a4 <- a1 0.477, a2 0.463, c2 0.085; b1 <- b4 0.405, b2 0.402, b3 0.370; b2 <- b1 0.363, b4 0.300, b3 0.265; b3 <- b1 0.280, b2 0.227, b4 0.194; b4 <- b1 0.395, b2 0.324, b3 0.297; c1 <- c2 0.487, c3 0.383, c4 0.364; c2 <- c1 0.504, c3 0.492, c4 0.261; c3 <- c2 0.244, c1 0.219, c4 0.085; c4 <- c1 0.338, c2 0.194, c3 0.146 ...
- `pendigits_60nan`: D top3 +0.190 [+0.183, +0.197] seeds +0.191, +0.190, +0.190 -> faithful: yes; D top1 +0.076 [+0.074, +0.078] seeds +0.075, +0.078, +0.075 -> faithful: yes
  - map: seed stability +0.990; attention against occlusion +0.934; |Spearman| against occlusion +0.549 (within target, mean)
  - top sources: input1 <- input5 0.282, input3 0.237, input15 0.197; input10 <- input8 0.292, input12 0.241, input16 0.190; input11 <- input15 0.265, input9 0.255, input13 0.229; input12 <- input10 0.221, input14 0.202, input8 0.192; input13 <- input15 0.333, input11 0.318, input7 0.211; input14 <- input16 0.225, input8 0.173, input12 0.167; input15 <- input11 0.295, input13 0.238, input10 0.209; input16 <- input14 0.278, input8 0.174, input10 0.173; input2 <- input4 0.235, input6 0.226, input1 0.188; input3 <- input1 0.278, input5 0.243, input7 0.223; input4 <- input8 0.187, input6 0.179, input12 0.162; input5 <- input7 0.258, input1 0.229, input9 0.194 ...
- `letter_60nan`: D top3 +0.157 [+0.154, +0.159] seeds +0.156, +0.157, +0.157 -> faithful: yes; D top1 +0.063 [+0.061, +0.065] seeds +0.064, +0.063, +0.062 -> faithful: yes
  - map: seed stability +0.995; attention against occlusion +0.956; |Spearman| against occlusion +0.631 (within target, mean)
  - top sources: high <- y-box 0.404, onpix 0.260, x-box 0.249; onpix <- width 0.287, high 0.217, x-ege 0.208; width <- x-box 0.381, onpix 0.305, x-ege 0.187; x-bar <- x2ybr 0.304, xybar 0.282, y-bar 0.247; x-box <- width 0.352, y-box 0.271, onpix 0.200; x-ege <- y2bar 0.272, onpix 0.266, width 0.222; x2bar <- xybar 0.283, y-ege 0.223, x-ege 0.212; x2ybr <- y-bar 0.297, xegvy 0.288, x-bar 0.254; xegvy <- y-bar 0.321, x2ybr 0.317, xy2br 0.187; xy2br <- x-bar 0.295, y-bar 0.283, x2ybr 0.227; xybar <- x2bar 0.252, x-bar 0.245, x2ybr 0.241; y-bar <- x2ybr 0.277, xegvy 0.253, x-bar 0.216 ...
- `electricity_60nan`: D top3 +0.131 [+0.126, +0.136] seeds +0.132, +0.132, +0.128 -> faithful: yes; D top1 +0.072 [+0.068, +0.077] seeds +0.073, +0.074, +0.070 -> faithful: yes
  - map: seed stability +0.994; attention against occlusion +0.924; |Spearman| against occlusion +0.658 (within target, mean)
  - top sources: date <- vicprice 0.301, transfer 0.294, vicdemand 0.291; day <- nswdemand 0.208, vicdemand 0.166, period 0.143; nswdemand <- period 0.449, vicdemand 0.320, day 0.211; nswprice <- vicprice 0.295, date 0.154, nswdemand 0.129; period <- nswdemand 0.398, vicdemand 0.112, nswprice 0.095; transfer <- vicdemand 0.328, date 0.258, vicprice 0.253; vicdemand <- nswdemand 0.357, transfer 0.296, vicprice 0.233; vicprice <- nswprice 0.088, vicdemand 0.041, date 0.039
- `vehicle_60nan`: D top3 +0.235 [+0.211, +0.260] seeds +0.238, +0.225, +0.244 -> faithful: yes; D top1 +0.074 [+0.060, +0.087] seeds +0.087, +0.058, +0.076 -> faithful: yes
  - map: seed stability +0.942; attention against occlusion +0.852; |Spearman| against occlusion +0.548 (within target, mean)
  - top sources: CIRCULARITY <- SCALED_RADIUS_OF_GYRATION 0.231, MAX.LENGTH_RECTANGULARITY 0.225, SCATTER_RATIO 0.092; COMPACTNESS <- KURTOSIS_ABOUT_MINOR 0.175, SCATTER_RATIO 0.139, HOLLOWS_RATIO 0.138; DISTANCE_CIRCULARITY <- MAX.LENGTH_ASPECT_RATIO 0.125, SCALED_VARIANCE_MAJOR 0.103, SCALED_VARIANCE_MINOR 0.101; ELONGATEDNESS <- SCATTER_RATIO 0.133, PR.AXIS_RECTANGULARITY 0.118, SCALED_VARIANCE_MINOR 0.113; HOLLOWS_RATIO <- SKEWNESS_ABOUT_MAJOR 0.395, KURTOSIS_ABOUT_MINOR 0.317, PR.AXIS_ASPECT_RATIO 0.222; KURTOSIS_ABOUT_MAJOR <- HOLLOWS_RATIO 0.318, SCALED_RADIUS_OF_GYRATION 0.291, CIRCULARITY 0.247; KURTOSIS_ABOUT_MINOR <- SKEWNESS_ABOUT_MAJOR 0.368, HOLLOWS_RATIO 0.351, COMPACTNESS 0.227; MAX.LENGTH_ASPECT_RATIO <- PR.AXIS_ASPECT_RATIO 0.175, SKEWNESS_ABOUT_MAJOR 0.161, SCALED_VARIANCE_MINOR 0.148; MAX.LENGTH_RECTANGULARITY <- CIRCULARITY 0.274, SCALED_RADIUS_OF_GYRATION 0.202, MAX.LENGTH_ASPECT_RATIO 0.174; PR.AXIS_ASPECT_RATIO <- RADIUS_RATIO 0.285, HOLLOWS_RATIO 0.208, KURTOSIS_ABOUT_MINOR 0.168; PR.AXIS_RECTANGULARITY <- SCATTER_RATIO 0.127, SCALED_VARIANCE_MINOR 0.103, ELONGATEDNESS 0.096; RADIUS_RATIO <- PR.AXIS_ASPECT_RATIO 0.259, COMPACTNESS 0.138, SCALED_VARIANCE_MAJOR 0.133 ...
- `biodeg_60nan`: D top3 +0.156 [+0.146, +0.165] seeds +0.157, +0.155, +0.154 -> faithful: yes; D top1 +0.067 [+0.062, +0.073] seeds +0.074, +0.066, +0.062 -> faithful: yes
  - map: seed stability +0.877; attention against occlusion +0.869; |Spearman| against occlusion +0.448 (within target, mean)
  - top sources: V1 <- V15 0.269, V27 0.220, V39 0.111; V10 <- V16 0.270, V30 0.227, V22 0.196; V11 <- V38 0.219, V34 0.160, V5 0.149; V12 <- V10 0.167, V30 0.159, V17 0.156; V13 <- V39 0.218, V15 0.199, V36 0.132; V14 <- V31 0.301, V9 0.170, V18 0.142; V15 <- V1 0.179, V27 0.169, V13 0.155; V16 <- V10 0.343, V30 0.213, V15 0.156; V17 <- V37 0.387, V41 0.137, V2 0.132; V18 <- V22 0.230, V8 0.171, V37 0.116; V19 <- V4 0.050, V32 0.048, V18 0.038; V2 <- V17 0.204, V18 0.166, V23 0.164 ...
- `kc2_60nan`: D top3 +0.012 [+0.001, +0.023] seeds +0.034, -0.000, +0.003 -> faithful: inconclusive; D top1 -0.003 [-0.014, +0.007] seeds +0.003, -0.008, -0.004 -> faithful: no
  - map: seed stability +0.866; attention against occlusion +0.691; |Spearman| against occlusion +0.181 (within target, mean)
  - top sources: b <- loc 0.043, i 0.042, lOCode 0.039; branchCount <- iv(g) 0.081, v(g) 0.064, l 0.064; d <- uniq_Op 0.078, total_Op 0.064, loc 0.061; e <- branchCount 0.034, v 0.034, b 0.032; ev(g) <- iv(g) 0.097, n 0.075, i 0.070; i <- uniq_Op 0.064, d 0.064, uniq_Opnd 0.061; iv(g) <- branchCount 0.069, total_Op 0.066, loc 0.058; l <- uniq_Op 0.217, d 0.201, iv(g) 0.162; lOBlank <- iv(g) 0.082, i 0.082, loc 0.077; lOCode <- loc 0.057, d 0.057, uniq_Op 0.052; lOCodeAndComment <- branchCount 0.155, loc 0.109, total_Op 0.077; lOComment <- d 0.137, lOCode 0.100, ev(g) 0.099 ...
- `kr-vs-kp_60nan`: D top3 +0.075 [+0.063, +0.088] seeds +0.084, +0.067, +0.074 -> faithful: yes; D top1 +0.043 [+0.036, +0.051] seeds +0.047, +0.039, +0.045 -> faithful: yes
  - map: seed stability +0.937; attention against occlusion +0.855; |Spearman| against occlusion +nan (within target, mean)
  - top sources: bkblk <- cntxt 0.061, katri 0.043, wkpos 0.032; bknwy <- wkovl 0.040, bkxcr 0.036, r2ar8 0.028; bkon8 <- simpl 0.021, bkxcr 0.017, wkovl 0.016; bkona <- wkovl 0.081, bkxbq 0.032, skrxp 0.024; bkspr <- bkxcr 0.111, blxwp 0.067, dsopp 0.059; bkxbq <- wkpos 0.079, bxqsq 0.062, qxmsq 0.042; bkxcr <- bkxwp 0.147, bkspr 0.109, blxwp 0.103; bkxwp <- blxwp 0.222, bkxcr 0.134, bxqsq 0.070; blxwp <- bkxwp 0.236, rkxwp 0.217, bkxcr 0.089; bxqsq <- bkxwp 0.099, rkxwp 0.064, wkpos 0.053; cntxt <- skewr 0.156, wkpos 0.119, wtoeg 0.117; dsopp <- bkspr 0.043, cntxt 0.029, bkxcr 0.025 ...
- `spambase_60nan`: D top3 +0.041 [+0.031, +0.050] seeds +0.036, +0.043, +0.042 -> faithful: yes; D top1 +0.020 [+0.015, +0.026] seeds +0.019, +0.022, +0.020 -> faithful: yes
  - map: seed stability +0.747; attention against occlusion +0.820; |Spearman| against occlusion +0.344 (within target, mean)
  - top sources: capital_run_length_average <- capital_run_length_longest 0.057, word_freq_mail 0.030, word_freq_credit 0.024; capital_run_length_longest <- capital_run_length_total 0.183, capital_run_length_average 0.140, word_freq_order 0.069; capital_run_length_total <- capital_run_length_longest 0.108, word_freq_receive 0.106, word_freq_order 0.103; char_freq_%21 <- word_freq_you 0.102, word_freq_over 0.052, word_freq_hp 0.047; char_freq_%23 <- capital_run_length_average 0.050, word_freq_font 0.049, capital_run_length_longest 0.030; char_freq_%24 <- word_freq_000 0.065, word_freq_money 0.060, capital_run_length_total 0.055; char_freq_%28 <- word_freq_857 0.052, word_freq_hpl 0.038, word_freq_415 0.036; char_freq_%3B <- word_freq_font 0.034, capital_run_length_average 0.033, word_freq_credit 0.032; char_freq_%5B <- word_freq_you 0.041, word_freq_1999 0.036, word_freq_original 0.034; word_freq_000 <- char_freq_%24 0.190, word_freq_all 0.084, char_freq_%21 0.070; word_freq_1999 <- word_freq_hp 0.077, word_freq_hpl 0.072, word_freq_george 0.064; word_freq_3d <- word_freq_85 0.018, word_freq_your 0.016, capital_run_length_average 0.016 ...
- `credit-g_60nan`: D top3 +0.019 [+0.009, +0.029] seeds +0.012, +0.028, +0.017 -> faithful: inconclusive; D top1 +0.018 [+0.011, +0.025] seeds +0.011, +0.021, +0.021 -> faithful: yes
  - map: seed stability +0.840; attention against occlusion +0.787; |Spearman| against occlusion +0.271 (within target, mean)
  - top sources: age <- employment 0.275, duration 0.101, residence_since 0.100; checking_status <- savings_status 0.080, age 0.063, duration 0.059; credit_amount <- duration 0.308, property_magnitude 0.171, purpose 0.151; credit_history <- existing_credits 0.143, installment_commitment 0.068, num_dependents 0.065; duration <- credit_amount 0.321, own_telephone 0.108, purpose 0.108; employment <- age 0.118, residence_since 0.078, property_magnitude 0.069; existing_credits <- credit_history 0.310, duration 0.136, other_payment_plans 0.097; foreign_worker <- duration 0.016, other_payment_plans 0.012, employment 0.011; housing <- property_magnitude 0.096, age 0.047, other_payment_plans 0.045; installment_commitment <- credit_amount 0.169, employment 0.120, age 0.086; job <- own_telephone 0.078, employment 0.077, credit_amount 0.066; num_dependents <- purpose 0.140, other_parties 0.121, existing_credits 0.117 ...

- K against the earlier run of the same cell, largest fold difference: 0.00e+00 over 27 cells
- truth (C, gate): yes

Primary (B): faithful on 7 of 9, not faithful on none -> supported (fold-pair t, seed agreement, SESOI 0.01)
```
