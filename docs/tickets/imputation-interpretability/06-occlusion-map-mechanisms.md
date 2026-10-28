# Does the occlusion map hold under MNAR and MAR? (E88; pre-registered 2026-10-28)

This ticket was written and committed before the study's first cell.

- **Why:** the map's claim is "faithful from 20 to 60nan, MCAR, K" (E78, E79, E85). Every one of
  those cells was MCAR. E84 showed that ADR 0030's spread survives MNAR and MAR at 60nan; whether
  the map survives them too is untested.
- **The user's go:** "aprovo! rode os 2 em paralelo" (2026-10-28, about 16:50 GMT-3), for this study
  on gorgona8 and E87 on medusa1.
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e88` runs and reads this study with E78's code. The older studies'
    paths are unchanged: E85's tally printed the same two Primary lines after the edit.
  - `run_occlusion_queue.sh` takes `STUDY=e88`, and it now exports `MLFLOW_TRACKING_URI` (the
    leader's server, ADR 0033) when the shell has none.

## Design (fixed before the first cell)

- **Cells:** E84's 12 shifted variants at 60nan × seeds 101, 202, 303, so 36 cells.
  - **MNAR (self-masking probit):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2 and
    spambase at 0.5.
  - **MAR (`marf`):** biodeg, kc2, vehicle, spambase and credit-g.
  - Today's default, five folds, `--score_remask`, read in place exactly as E78.
- **No synthetic table:** the truth measure (C) needs `synthblock`, which is MCAR by construction,
  so C is not read here. The tally skips its gate when no synthetic variant is registered.
- **Running:** four GPU queues on gorgona8 (`STUDY=e88`), MLflow tag
  `experiment=occlusion-mechanisms-2026-10-28`, logged through the leader's server; state in
  `/scratch2/diogoneiss/trident-2026-10-28-occlusion-mechanisms/`. About 3–4 h. Each cell runs on
  gorgona8, as E79's MCAR twins did (E86: a study stays on one machine).
- **Smoke:** one cell runs first, and the queues start only after it lands in the leader's store
  with `host=gorgona8`. It counts.

## Measures and rule, fixed now (`occlusion_map.py --study e88 --tally`)

- **Primary, B (deletion faithfulness):** E78's D on top 3. Per variant, over 15 folds:
  - faithful if the 95% interval lies above the SESOI (0.01) with every seed above 0;
  - not faithful if it lies below the SESOI;
  - inconclusive otherwise.
- **The call:** E79's 7 and 4 of 9, scaled to 12.
  - **supported** if faithful on 10 or more of 12 and not faithful on none;
  - **refuted** if faithful on 5 or fewer;
  - **mixed** otherwise.
- **Descriptive:**
  - D top 1, the seed stability of the map, attention against occlusion, |Spearman| against
    occlusion, and the top sources, as E78 prints them;
  - D on each shifted variant against its MCAR twin `<table>_60nan` at the same seeds (E79's
    cells, also on gorgona8);
  - K against E84's run of the same cell. E84's first 40 cells ran under torch 2.5.1 with
    `--score_checkpoints`, so a difference there is expected and not read.
- **Stale reads:** `--todo` reads the follower's copy of the store, up to 10 minutes old, so a
  relaunch within that window can deal a just-finished cell again. That is harmless: the second
  run is a duplicate parent, and the tally takes one.

## Outcome

**Done 2026-10-28, 19:48 GMT-3.** The 36 cells ran 17:05–19:47 GMT-3 on gorgona8, with none
failed. The smoke cell (kc2_probit_60nan s101) is one of them.

The tally, unedited (`occlusion_map.py --study e88 --tally`):

```
- `biodeg_probit0p25_60nan`: D top3 +0.127 [+0.117, +0.136] seeds +0.132, +0.127, +0.122 -> faithful: yes; D top1 +0.055 [+0.049, +0.061] seeds +0.061, +0.051, +0.054 -> faithful: yes
  - map: seed stability +0.872; attention against occlusion +0.831; |Spearman| against occlusion +0.438 (within target, mean)
  - top sources: V1 <- V15 0.185, V27 0.166, V28 0.140; V10 <- V16 0.335, V30 0.232, V17 0.202; V11 <- V38 0.235, V34 0.193, V5 0.137; V12 <- V30 0.153, V10 0.134, V37 0.126; V13 <- V15 0.224, V39 0.195, V31 0.176; V14 <- V31 0.277, V22 0.237, V9 0.191; V15 <- V1 0.182, V13 0.168, V27 0.160; V16 <- V10 0.352, V30 0.239, V27 0.199; V17 <- V37 0.369, V2 0.152, V36 0.146; V18 <- V22 0.209, V8 0.209, V7 0.149; V19 <- V13 0.037, V4 0.032, V27 0.032; V2 <- V23 0.277, V14 0.223, V27 0.200 ...
- `biodeg_probit_60nan`: D top3 +0.102 [+0.095, +0.109] seeds +0.106, +0.106, +0.093 -> faithful: yes; D top1 +0.045 [+0.040, +0.049] seeds +0.047, +0.047, +0.040 -> faithful: yes
  - map: seed stability +0.856; attention against occlusion +0.833; |Spearman| against occlusion +0.441 (within target, mean)
  - top sources: V1 <- V27 0.218, V15 0.177, V39 0.105; V10 <- V12 0.314, V16 0.275, V22 0.235; V11 <- V38 0.256, V34 0.159, V5 0.139; V12 <- V30 0.208, V17 0.155, V10 0.124; V13 <- V39 0.210, V15 0.194, V31 0.154; V14 <- V23 0.306, V31 0.232, V9 0.222; V15 <- V1 0.168, V27 0.158, V13 0.143; V16 <- V10 0.265, V12 0.196, V30 0.158; V17 <- V37 0.305, V22 0.203, V12 0.143; V18 <- V22 0.275, V8 0.255, V7 0.128; V19 <- V14 0.036, V28 0.032, V8 0.032; V2 <- V23 0.289, V37 0.207, V14 0.181 ...
- `biodeg_probit1p0_60nan`: D top3 +0.081 [+0.071, +0.091] seeds +0.075, +0.085, +0.083 -> faithful: yes; D top1 +0.043 [+0.038, +0.048] seeds +0.040, +0.047, +0.043 -> faithful: yes
  - map: seed stability +0.869; attention against occlusion +0.774; |Spearman| against occlusion +0.360 (within target, mean)
  - top sources: V1 <- V15 0.177, V27 0.152, V36 0.124; V10 <- V12 0.461, V22 0.441, V16 0.373; V11 <- V5 0.264, V34 0.230, V38 0.204; V12 <- V30 0.187, V37 0.160, V10 0.152; V13 <- V39 0.183, V31 0.168, V15 0.162; V14 <- V31 0.274, V23 0.268, V9 0.268; V15 <- V1 0.137, V13 0.128, V39 0.126; V16 <- V12 0.311, V10 0.304, V30 0.246; V17 <- V12 0.277, V37 0.269, V10 0.225; V18 <- V22 0.396, V8 0.262, V36 0.193; V19 <- V22 0.006, V17 0.006, V8 0.006; V2 <- V23 0.381, V37 0.215, V8 0.171 ...
- `biodeg_probit2p0_60nan`: D top3 +0.062 [+0.053, +0.071] seeds +0.063, +0.062, +0.061 -> faithful: yes; D top1 +0.042 [+0.037, +0.048] seeds +0.042, +0.041, +0.044 -> faithful: yes
  - map: seed stability +0.872; attention against occlusion +0.649; |Spearman| against occlusion +0.248 (within target, mean)
  - top sources: V1 <- V27 0.184, V15 0.179, V36 0.142; V10 <- V12 0.625, V30 0.449, V16 0.420; V11 <- V12 0.278, V35 0.247, V9 0.234; V12 <- V30 0.234, V35 0.158, V10 0.133; V13 <- V31 0.342, V18 0.279, V41 0.211; V14 <- V2 0.316, V23 0.313, V9 0.275; V15 <- V27 0.157, V13 0.148, V1 0.141; V16 <- V10 0.352, V35 0.299, V30 0.286; V17 <- V12 0.358, V10 0.314, V37 0.275; V18 <- V22 0.479, V8 0.267, V39 0.179; V19 <- V17 0.005, V35 0.005, V15 0.004; V2 <- V23 0.714, V37 0.223, V31 0.200 ...
- `vehicle_probit_60nan`: D top3 +0.283 [+0.255, +0.312] seeds +0.294, +0.292, +0.264 -> faithful: yes; D top1 +0.131 [+0.118, +0.145] seeds +0.121, +0.136, +0.136 -> faithful: yes
  - map: seed stability +0.900; attention against occlusion +0.809; |Spearman| against occlusion +0.506 (within target, mean)
  - top sources: CIRCULARITY <- SCALED_RADIUS_OF_GYRATION 0.263, MAX.LENGTH_RECTANGULARITY 0.211, ELONGATEDNESS 0.206; COMPACTNESS <- ELONGATEDNESS 0.227, HOLLOWS_RATIO 0.195, SKEWNESS_ABOUT_MAJOR 0.170; DISTANCE_CIRCULARITY <- ELONGATEDNESS 0.175, SCALED_VARIANCE_MINOR 0.113, MAX.LENGTH_RECTANGULARITY 0.100; ELONGATEDNESS <- SCATTER_RATIO 0.147, SCALED_VARIANCE_MINOR 0.134, SCALED_VARIANCE_MAJOR 0.118; HOLLOWS_RATIO <- SKEWNESS_ABOUT_MAJOR 0.479, KURTOSIS_ABOUT_MINOR 0.334, COMPACTNESS 0.184; KURTOSIS_ABOUT_MAJOR <- CIRCULARITY 0.200, MAX.LENGTH_RECTANGULARITY 0.190, SKEWNESS_ABOUT_MAJOR 0.173; KURTOSIS_ABOUT_MINOR <- SKEWNESS_ABOUT_MAJOR 0.447, HOLLOWS_RATIO 0.318, COMPACTNESS 0.237; MAX.LENGTH_ASPECT_RATIO <- ELONGATEDNESS 0.172, SKEWNESS_ABOUT_MINOR 0.157, KURTOSIS_ABOUT_MINOR 0.141; MAX.LENGTH_RECTANGULARITY <- CIRCULARITY 0.261, ELONGATEDNESS 0.228, SCALED_RADIUS_OF_GYRATION 0.210; PR.AXIS_ASPECT_RATIO <- RADIUS_RATIO 0.325, ELONGATEDNESS 0.213, HOLLOWS_RATIO 0.207; PR.AXIS_RECTANGULARITY <- ELONGATEDNESS 0.255, SCALED_VARIANCE_MINOR 0.145, SCATTER_RATIO 0.126; RADIUS_RATIO <- PR.AXIS_ASPECT_RATIO 0.217, SCALED_VARIANCE_MINOR 0.165, SKEWNESS_ABOUT_MAJOR 0.154 ...
- `kc2_probit_60nan`: D top3 -0.003 [-0.018, +0.012] seeds -0.005, -0.014, +0.010 -> faithful: inconclusive; D top1 -0.005 [-0.011, +0.001] seeds -0.007, -0.007, -0.000 -> faithful: no
  - map: seed stability +0.695; attention against occlusion +0.343; |Spearman| against occlusion +0.177 (within target, mean)
  - top sources: b <- total_Opnd 0.099, lOBlank 0.086, loc 0.080; branchCount <- lOBlank 0.139, loc 0.091, total_Opnd 0.088; d <- iv(g) 0.089, b 0.087, lOBlank 0.087; e <- lOBlank 0.087, b 0.080, t 0.080; ev(g) <- loc 0.134, e 0.117, t 0.114; i <- n 0.130, uniq_Opnd 0.101, loc 0.099; iv(g) <- v(g) 0.117, l 0.070, b 0.050; l <- uniq_Op 0.194, v 0.192, b 0.186; lOBlank <- branchCount 0.101, l 0.091, i 0.086; lOCode <- loc 0.089, lOBlank 0.077, iv(g) 0.064; lOCodeAndComment <- v(g) 0.174, t 0.093, branchCount 0.092; lOComment <- lOBlank 0.117, loc 0.079, v(g) 0.075 ...
- `spambase_probit_60nan`: D top3 +0.028 [+0.024, +0.033] seeds +0.026, +0.023, +0.035 -> faithful: yes; D top1 +0.009 [+0.006, +0.011] seeds +0.007, +0.008, +0.011 -> faithful: inconclusive
  - map: seed stability +0.689; attention against occlusion +0.755; |Spearman| against occlusion +0.333 (within target, mean)
  - top sources: capital_run_length_average <- capital_run_length_longest 0.034, char_freq_%21 0.034, capital_run_length_total 0.033; capital_run_length_longest <- capital_run_length_average 0.217, capital_run_length_total 0.158, char_freq_%21 0.088; capital_run_length_total <- capital_run_length_longest 0.139, capital_run_length_average 0.116, word_freq_all 0.065; char_freq_%21 <- word_freq_you 0.033, word_freq_george 0.033, word_freq_your 0.029; char_freq_%23 <- word_freq_000 0.017, char_freq_%21 0.017, word_freq_hp 0.016; char_freq_%24 <- word_freq_000 0.106, capital_run_length_average 0.092, word_freq_business 0.075; char_freq_%28 <- word_freq_lab 0.035, word_freq_telnet 0.034, word_freq_technology 0.033; char_freq_%3B <- word_freq_you 0.031, word_freq_font 0.023, word_freq_money 0.022; char_freq_%5B <- word_freq_hpl 0.032, word_freq_hp 0.031, capital_run_length_total 0.028; word_freq_000 <- char_freq_%24 0.127, word_freq_business 0.070, word_freq_email 0.067; word_freq_1999 <- word_freq_hp 0.059, word_freq_technology 0.057, word_freq_george 0.050; word_freq_3d <- word_freq_money 0.032, word_freq_you 0.019, word_freq_650 0.017 ...
- `biodeg_marf_60nan`: D top3 +0.316 [+0.287, +0.345] seeds +0.343, +0.283, +0.321 -> faithful: yes; D top1 +0.188 [+0.160, +0.217] seeds +0.208, +0.170, +0.187 -> faithful: yes
  - map: seed stability +0.914; attention against occlusion +0.767; |Spearman| against occlusion +0.370 (within target, mean)
  - top sources: V1 <- V15 0.163, V27 0.123, V23 0.098; V11 <- V38 0.217, V34 0.195, V5 0.120; V12 <- V30 0.097, V10 0.082, V17 0.076; V13 <- V31 0.206, V36 0.102, V15 0.101; V14 <- V31 0.369, V23 0.177, V24 0.108; V15 <- V13 0.131, V1 0.130, V27 0.109; V16 <- V10 0.269, V4 0.102, V30 0.100; V17 <- V37 0.211, V14 0.094, V36 0.092; V18 <- V22 0.136, V8 0.062, V7 0.057; V19 <- V1 0.045, V34 0.042, V18 0.041; V20 <- V30 0.161, V32 0.156, V34 0.145; V21 <- V34 0.044, V15 0.044, V18 0.041 ...
- `kc2_marf_60nan`: D top3 +0.098 [+0.059, +0.136] seeds +0.102, +0.088, +0.104 -> faithful: yes; D top1 +0.048 [+0.022, +0.075] seeds +0.019, +0.073, +0.053 -> faithful: yes
  - map: seed stability +0.878; attention against occlusion +0.351; |Spearman| against occlusion +0.225 (within target, mean)
  - top sources: b <- lOBlank 0.079, t 0.076, lOCode 0.066; branchCount <- lOBlank 0.051, v(g) 0.049, t 0.041; d <- uniq_Op 0.094, lOBlank 0.062, uniq_Opnd 0.046; e <- lOBlank 0.043, total_Opnd 0.042, uniq_Opnd 0.042; ev(g) <- lOComment 0.189, lOBlank 0.184, v(g) 0.119; i <- uniq_Op 0.040, uniq_Opnd 0.040, d 0.036; iv(g) <- lOBlank 0.144, total_Opnd 0.113, v(g) 0.105; l <- uniq_Op 0.228, iv(g) 0.078, d 0.063; lOBlank <- t 0.055, uniq_Op 0.041, lOCode 0.040; lOComment <- uniq_Opnd 0.075, lOBlank 0.065, branchCount 0.048; n <- lOCode 0.080, t 0.076, lOBlank 0.074; t <- d 0.012, loc 0.011, v 0.010 ...
- `vehicle_marf_60nan`: D top3 +0.430 [+0.396, +0.464] seeds +0.422, +0.446, +0.421 -> faithful: yes; D top1 +0.239 [+0.211, +0.267] seeds +0.236, +0.210, +0.271 -> faithful: yes
  - map: seed stability +0.921; attention against occlusion +0.788; |Spearman| against occlusion +0.426 (within target, mean)
  - top sources: CIRCULARITY <- MAX.LENGTH_RECTANGULARITY 0.222, SCALED_RADIUS_OF_GYRATION 0.190, KURTOSIS_ABOUT_MAJOR 0.130; DISTANCE_CIRCULARITY <- SCALED_VARIANCE_MINOR 0.100, SCATTER_RATIO 0.088, SKEWNESS_ABOUT_MAJOR 0.073; ELONGATEDNESS <- SCALED_VARIANCE_MINOR 0.142, SCATTER_RATIO 0.056, PR.AXIS_RECTANGULARITY 0.052; HOLLOWS_RATIO <- SKEWNESS_ABOUT_MAJOR 0.191, KURTOSIS_ABOUT_MINOR 0.154, MAX.LENGTH_ASPECT_RATIO 0.064; KURTOSIS_ABOUT_MAJOR <- HOLLOWS_RATIO 0.195, MAX.LENGTH_ASPECT_RATIO 0.177, SCALED_RADIUS_OF_GYRATION 0.159; MAX.LENGTH_ASPECT_RATIO <- SKEWNESS_ABOUT_MAJOR 0.107, SCALED_VARIANCE_MAJOR 0.066, RADIUS_RATIO 0.046; MAX.LENGTH_RECTANGULARITY <- CIRCULARITY 0.220, SCALED_RADIUS_OF_GYRATION 0.157, MAX.LENGTH_ASPECT_RATIO 0.112; PR.AXIS_ASPECT_RATIO <- RADIUS_RATIO 0.194, SCALED_VARIANCE_MAJOR 0.132, ELONGATEDNESS 0.123; PR.AXIS_RECTANGULARITY <- SCALED_VARIANCE_MINOR 0.122, ELONGATEDNESS 0.064, SCALED_VARIANCE_MAJOR 0.050; RADIUS_RATIO <- PR.AXIS_ASPECT_RATIO 0.147, SCALED_VARIANCE_MINOR 0.096, SKEWNESS_ABOUT_MAJOR 0.091; SCALED_RADIUS_OF_GYRATION <- CIRCULARITY 0.209, MAX.LENGTH_RECTANGULARITY 0.179, PR.AXIS_RECTANGULARITY 0.134; SCALED_VARIANCE_MAJOR <- SCALED_VARIANCE_MINOR 0.088, PR.AXIS_RECTANGULARITY 0.048, ELONGATEDNESS 0.046 ...
- `spambase_marf_60nan`: D top3 +0.040 [+0.031, +0.049] seeds +0.039, +0.040, +0.041 -> faithful: yes; D top1 +0.033 [+0.027, +0.038] seeds +0.032, +0.033, +0.033 -> faithful: yes
  - map: seed stability +0.909; attention against occlusion +0.769; |Spearman| against occlusion +0.359 (within target, mean)
  - top sources: capital_run_length_longest <- capital_run_length_average 0.080, capital_run_length_total 0.042, word_freq_you 0.034; capital_run_length_total <- capital_run_length_average 0.112, word_freq_receive 0.109, word_freq_money 0.068; char_freq_%21 <- capital_run_length_average 0.089, word_freq_free 0.063, char_freq_%24 0.062; char_freq_%23 <- word_freq_technology 0.033, word_freq_hp 0.029, word_freq_you 0.029; char_freq_%24 <- word_freq_000 0.063, capital_run_length_average 0.060, char_freq_%21 0.060; char_freq_%3B <- word_freq_font 0.038, word_freq_people 0.036, word_freq_your 0.035; word_freq_000 <- char_freq_%24 0.131, word_freq_free 0.070, word_freq_you 0.069; word_freq_415 <- word_freq_technology 0.053, word_freq_direct 0.050, word_freq_857 0.046; word_freq_650 <- word_freq_85 0.048, word_freq_hp 0.046, word_freq_hpl 0.044; word_freq_857 <- word_freq_direct 0.049, word_freq_technology 0.031, word_freq_415 0.028; word_freq_all <- char_freq_%21 0.089, word_freq_you 0.076, word_freq_your 0.070; word_freq_business <- word_freq_make 0.068, word_freq_our 0.062, word_freq_000 0.058 ...
- `credit-g_marf_60nan`: D top3 +0.031 [+0.022, +0.039] seeds +0.031, +0.031, +0.029 -> faithful: yes; D top1 +0.030 [+0.021, +0.039] seeds +0.029, +0.028, +0.032 -> faithful: yes
  - map: seed stability +0.921; attention against occlusion +0.755; |Spearman| against occlusion +0.617 (within target, mean)
  - top sources: credit_amount <- duration 0.308, purpose 0.149, installment_commitment 0.104; credit_history <- existing_credits 0.228, savings_status 0.073, residence_since 0.058; employment <- age 0.105, purpose 0.086, residence_since 0.084; foreign_worker <- residence_since 0.023, other_parties 0.016, duration 0.015; installment_commitment <- credit_amount 0.169, savings_status 0.147, duration 0.136; job <- own_telephone 0.078, property_magnitude 0.060, credit_amount 0.052; num_dependents <- age 0.156, purpose 0.093, residence_since 0.091; other_parties <- other_payment_plans 0.028, property_magnitude 0.025, credit_amount 0.024; other_payment_plans <- foreign_worker 0.033, residence_since 0.032, credit_history 0.029; own_telephone <- job 0.081, purpose 0.053, property_magnitude 0.053; personal_status <- purpose 0.088, employment 0.079, num_dependents 0.074; purpose <- credit_amount 0.099, residence_since 0.096, credit_history 0.074 ...

- K against the earlier run of the same cell, largest fold difference: 7.61e-02 over 36 cells

Primary (B): faithful on 11 of 12, not faithful on none -> supported (fold-pair t, seed agreement, SESOI 0.01)
```

The descriptive twin comparison, D top 3 minus the MCAR twin's (E79's cells, same seeds;
`e88_twins` reading through `occlusion_map._faithfulness`):

```
- `biodeg_probit0p25_60nan` minus `biodeg_60nan`: -0.029 [-0.040, -0.018]
- `biodeg_probit_60nan` minus `biodeg_60nan`: -0.054 [-0.065, -0.042]
- `biodeg_probit1p0_60nan` minus `biodeg_60nan`: -0.074 [-0.087, -0.062]
- `biodeg_probit2p0_60nan` minus `biodeg_60nan`: -0.094 [-0.107, -0.080]
- `vehicle_probit_60nan` minus `vehicle_60nan`: +0.048 [+0.010, +0.086]
- `kc2_probit_60nan` minus `kc2_60nan`: -0.015 [-0.036, +0.005]
- `spambase_probit_60nan` minus `spambase_60nan`: -0.012 [-0.023, -0.001]
- `biodeg_marf_60nan` minus `biodeg_60nan`: +0.160 [+0.133, +0.187]
- `kc2_marf_60nan` minus `kc2_60nan`: +0.086 [+0.044, +0.127]
- `vehicle_marf_60nan` minus `vehicle_60nan`: +0.194 [+0.164, +0.224]
- `spambase_marf_60nan` minus `spambase_60nan`: -0.001 [-0.013, +0.011]
- `credit-g_marf_60nan` minus `credit-g_60nan`: +0.012 [-0.001, +0.024]
```

- **Primary: supported.** The map is faithful on 11 of 12 variants and not faithful on none. The
  exception, kc2_probit_60nan, is inconclusive (D top 3 −0.003 [−0.018, +0.012]). kc2 is the
  smallest table; in E85 its 20nan cells were inconclusive and its 80nan cells not faithful.
- **Descriptive:**
  - **Self-masking MNAR weakens the map:** on biodeg, D falls with the slope, from −0.029 at
    0.25 to −0.094 at 2.0 against the MCAR twin. It stays faithful at every slope.
  - **MAR strengthens it on three tables** (biodeg +0.160, vehicle +0.194, kc2 +0.086). Under MAR,
    a column's gaps depend on observed columns, so those sources carry more of the fill.
  - K differs from E84's runs of the same cells by up to 0.076. That is expected (torch 2.5.1 and
    `--score_checkpoints` there), as stated before the run, and is not read.
- **The map's claim becomes:** faithful from 20 to 60nan under MCAR (E78, E79, E85), and at 60nan
  under MNAR and MAR (E88), read at K. The 80nan limit stands (E85).
