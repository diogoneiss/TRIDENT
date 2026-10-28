# Does the occlusion map hold at 20nan and 80nan? (E85; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** E78 (40nan, supported) and E79 (60nan, supported at the bar) leave the grid's two ends
  untested. The thesis's claim is stated per rate.
- **The user's go:** "de acordo em fazer 1 depois o 2, com 4 em paralelo" (2026-10-27, before
  20:40 GMT-3). This is item 2.
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e85` runs and reads this study with E78's code.
  - The reader now applies the rule to each rate on its own and gives one truth call per synthetic
    variant. E79's full printout is byte for byte identical after the change, checked with diff.
  - `generate_synthblock.py` also writes `synthblock_20nan` and `synthblock_80nan`. The 40/60nan
    files are unchanged (md5).
  - `run_occlusion_queue.sh` takes `STUDY=e85`.

## Design (fixed before the first cell)

- **Cells:** the ten tables at 20nan and 80nan, 60 cells. Each runs at the seed set whose earlier
  runs K is checked against:
  - **20nan:** the nine real tables at 101/202/303 (the thesis table, `development_source`;
    pendigits and letter there use ADR 0018's bigger encoder);
  - **80nan:** pendigits, letter, electricity, vehicle, biodeg and kc2 at 707/808/909 (E44), and
    kr-vs-kp, spambase and credit-g at 101/202/303;
  - **synthblock** at 707/808/909 at both rates.
  - Today's default, five folds, `--score_remask`, read in place exactly as E78.
- **Running:** four GPU queues (`STUDY=e85`). Each starts when E84's queue of the same number
  completes. MLflow tag `experiment=occlusion-map-grid-2026-10-27`; state in
  `/scratch2/diogoneiss/trident-2026-10-27-occlusion-grid/`. About 8 h.

## Measures and reading

E78's, unchanged, **applied to each rate separately** (`occlusion_map.py --study e85 --tally`):
- **B, primary, per rate:** supported if faithful on 7 or more of 9 and not faithful on none;
  refuted if faithful on 4 or fewer; mixed otherwise.
- **C, gate, per rate:** the AUROC on `synthblock_<rate>` lies above 0.8 with every seed above 0.8.
  It is inconclusive (floor) if K is not below `mean_mode`.
- **Expected thin support at 80nan (fixed now, not a reason to stop):**
  - A pair needs 10 cells. On the small tables at 80nan, the validation map may leave a target
    with fewer than 3 ranked sources, and that target is skipped.
  - A variant with no usable target reads inconclusive.
- **Descriptive:** as in E78.

**What it leads to:**
- The claim gains its 20nan and 80nan entries as each rate reads.
- No default changes.

## Amendment (2026-10-27, 20:44 GMT-3): deferred before any cell

- The user asked for only part of E84 and a database migration at about 23:00.
- The chained waiters were stopped; no E85 cell started.
- The design and the rule are unchanged, and the study waits for the user's go.

## Second amendment (2026-10-28, 00:30 GMT-3): running, 80nan first, on torch 2.11.0, deadline 08:30

- **The user's go:** "voce pode voltar ao que tinha planejado antes: o resto do 1, 2 e 4 em
  paralelo. vá" (2026-10-28, before 00:30 GMT-3).
- **Torch:** every E85 cell runs under torch 2.11.0+cu128 (installed by the ADR 0033 preparation,
  2026-10-27). E78 and E79 ran under 2.5.1.
  - The study is consistent within itself.
  - The descriptive "K against the earlier run of the same cell" line will no longer read 0.0,
    because of the torch change; that is not a fault of the in-place reading.
- **Order and deadline, fixed now and by the clock only:**
  - Each queue deals its 80nan cells first, then its 20nan cells (`occlusion_map.todo`; E78 and
    E79's single-rate dealing is unchanged).
  - No cell starts after 2026-10-28, 08:30 GMT-3, since the user stops experiments after 09:00.
- **If a rate does not complete:** it stays **partial, not complete**, and gets no verdict. Its
  cells finish later on the user's go, and its rule is read only then.

## Outcome (2026-10-28, 09:14 GMT-3): 80nan refuted; 20nan partial, no verdict

- **Ran:** 40 of 60 cells, 2026-10-28 from 01:27 to 09:11 GMT-3, all under torch 2.11.0, none
  failed.
  - The deadline stopped new cells at 08:30. The last one ended at 09:11, past the user's 09:00.
  - All 30 cells at 80nan ran. 10 of 30 at 20nan ran: pendigits and electricity complete; letter
    and spambase at 2 of 3 seeds; the rest and synthblock_20nan not started.
- **80nan, complete: refuted** by the rule fixed for each rate.
  - **Faithful on 4 of 9:** pendigits, letter, electricity and vehicle. D (top 3) is +0.028 to
    +0.084. That is a quarter to a third of the 40nan values on pendigits, letter and electricity,
    and about a fourteenth on vehicle.
  - **Not faithful on 4:** biodeg (−0.002), kc2 (−0.011), spambase (−0.021) and credit-g (−0.004).
    Each interval lies below the 1% SESOI, and on spambase and credit-g below 0.
  - **Inconclusive:** kr-vs-kp (+0.008).
  - The thin-support caveat fixed in the ticket does not explain it all. spambase (4,601 rows) has
    enough cells per pair and reads below 0.
- **C at 80nan: recovers the blocks.** The AUROC is 0.853 [0.825, 0.881], every seed above 0.8, and
  K (0.933) is below `mean_mode`. But |Spearman| reads 0.878, so at 80nan the map no longer beats a
  linear reading as it did at 40nan (0.979 against 0.847).
- **20nan: partial, not complete, and no verdict.** pendigits_20nan (+0.312) and
  electricity_20nan (+0.240) read faithful; that is descriptive only. The rule is read only when
  all its cells have run, on the user's go.
- **K against the earlier runs:** the largest fold difference is 0.021 over 33 cells. That is the
  torch change (2.5.1 → 2.11.0), as the second amendment foresaw, and not a check of the in-place
  reading.
- **What it leads to:** the map's faithfulness has a scope. It holds at 40nan (E78, 9 of 9) and at
  60nan (E79, 7 of 9), and not at 80nan, where it holds only on pendigits, letter, electricity
  and vehicle.
  - A report states the claim as "up to 60nan".
  - No default changes.

The reader's printout (`occlusion_map.py --study e85 --tally`), unedited:

```text
- `synthblock_20nan`: incomplete (0 of 3 seeds)
- `synthblock_80nan`: D top3 +0.093 [+0.086, +0.099] seeds +0.087, +0.092, +0.099 -> faithful: yes; D top1 +0.035 [+0.030, +0.040] seeds +0.032, +0.037, +0.036 -> faithful: yes
  - truth (C): AUROC 0.853 [0.825, 0.881] seeds 0.830, 0.852, 0.877; K 0.9330 against mean_mode 1.0000 -> recovers the blocks: yes
  - descriptive: |Spearman| on the training split, same AUROC (numerical pairs only): 0.878 [0.861, 0.895]
  - map: seed stability +0.763; attention against occlusion +0.782; |Spearman| against occlusion +0.276 (within target, mean)
  - top sources: a1 <- c4 0.102, a3 0.099, b1 0.094; a2 <- a4 0.278, a1 0.173, c1 0.168; a3 <- a1 0.201, c2 0.162, c1 0.143; a4 <- a2 0.209, a1 0.117, b4 0.097; b1 <- b3 0.605, b4 0.539, b2 0.467; b2 <- b1 0.472, b3 0.424, b4 0.362; b3 <- b1 0.349, b2 0.275, b4 0.215; b4 <- b1 0.524, b3 0.415, b2 0.411; c1 <- c2 0.675, c3 0.482, c4 0.360; c2 <- c3 0.734, c1 0.647, c4 0.320; c3 <- c2 0.358, c1 0.282, c4 0.129; c4 <- c1 0.344, c3 0.290, c2 0.262 ...
- `pendigits_20nan`: D top3 +0.312 [+0.303, +0.322] seeds +0.316, +0.309, +0.313 -> faithful: yes; D top1 +0.125 [+0.121, +0.129] seeds +0.123, +0.124, +0.128 -> faithful: yes
  - map: seed stability +0.993; attention against occlusion +0.903; |Spearman| against occlusion +0.322 (within target, mean)
  - top sources: input1 <- input3 0.150, input5 0.099, input11 0.058; input10 <- input8 0.088, input12 0.079, input6 0.050; input11 <- input13 0.108, input9 0.084, input15 0.074; input12 <- input14 0.079, input10 0.073, input16 0.051; input13 <- input15 0.194, input11 0.191, input9 0.077; input14 <- input12 0.052, input16 0.052, input10 0.029; input15 <- input13 0.131, input11 0.090, input1 0.046; input16 <- input14 0.063, input12 0.036, input13 0.029; input2 <- input4 0.113, input6 0.091, input1 0.066; input3 <- input1 0.171, input5 0.160, input6 0.085; input4 <- input6 0.104, input2 0.073, input8 0.069; input5 <- input7 0.122, input3 0.109, input1 0.085 ...
- `letter_20nan`: incomplete (2 of 3 seeds)
- `electricity_20nan`: D top3 +0.240 [+0.219, +0.261] seeds +0.217, +0.245, +0.258 -> faithful: yes; D top1 +0.167 [+0.154, +0.181] seeds +0.166, +0.186, +0.151 -> faithful: yes
  - map: seed stability +0.997; attention against occlusion +0.835; |Spearman| against occlusion +0.404 (within target, mean)
  - top sources: date <- nswprice 0.114, transfer 0.095, vicprice 0.092; day <- nswdemand 0.177, period 0.164, vicdemand 0.094; nswdemand <- period 0.371, vicdemand 0.173, day 0.165; nswprice <- vicprice 0.218, nswdemand 0.093, date 0.090; period <- nswdemand 0.395, day 0.124, vicdemand 0.105; transfer <- date 0.185, vicprice 0.167, nswprice 0.132; vicdemand <- nswdemand 0.209, period 0.149, date 0.134; vicprice <- nswprice 0.047, date 0.028, nswdemand 0.022
- `vehicle_20nan`: incomplete (0 of 3 seeds)
- `biodeg_20nan`: incomplete (0 of 3 seeds)
- `kc2_20nan`: incomplete (0 of 3 seeds)
- `kr-vs-kp_20nan`: incomplete (0 of 3 seeds)
- `spambase_20nan`: incomplete (2 of 3 seeds)
- `credit-g_20nan`: incomplete (0 of 3 seeds)
- `pendigits_80nan`: D top3 +0.084 [+0.080, +0.088] seeds +0.084, +0.085, +0.083 -> faithful: yes; D top1 +0.035 [+0.033, +0.037] seeds +0.037, +0.036, +0.032 -> faithful: yes
  - map: seed stability +0.945; attention against occlusion +0.910; |Spearman| against occlusion +0.712 (within target, mean)
  - top sources: input1 <- input5 0.383, input7 0.332, input15 0.293; input10 <- input8 0.466, input12 0.372, input9 0.362; input11 <- input15 0.474, input9 0.409, input10 0.296; input12 <- input4 0.339, input10 0.324, input6 0.313; input13 <- input11 0.273, input7 0.260, input12 0.238; input14 <- input16 0.455, input8 0.406, input6 0.331; input15 <- input11 0.511, input9 0.322, input10 0.287; input16 <- input14 0.465, input8 0.381, input6 0.297; input2 <- input6 0.303, input5 0.261, input4 0.259; input3 <- input5 0.254, input14 0.246, input4 0.245; input4 <- input12 0.340, input8 0.304, input6 0.297; input5 <- input6 0.324, input7 0.283, input1 0.277 ...
- `letter_80nan`: D top3 +0.068 [+0.066, +0.070] seeds +0.068, +0.069, +0.068 -> faithful: yes; D top1 +0.029 [+0.028, +0.030] seeds +0.030, +0.030, +0.029 -> faithful: yes
  - map: seed stability +0.953; attention against occlusion +0.932; |Spearman| against occlusion +0.650 (within target, mean)
  - top sources: high <- y-box 0.494, x-box 0.412, width 0.371; onpix <- width 0.398, high 0.346, x-box 0.331; width <- x-box 0.501, onpix 0.432, high 0.357; x-bar <- x2ybr 0.344, y-bar 0.253, xegvy 0.236; x-box <- width 0.474, y-box 0.393, high 0.364; x-ege <- onpix 0.349, width 0.324, x-box 0.262; x2bar <- xybar 0.245, x-ege 0.197, y-ege 0.193; x2ybr <- xegvy 0.341, y-bar 0.317, y-ege 0.230; xegvy <- y-bar 0.386, x2ybr 0.347, x-bar 0.162; xy2br <- y-bar 0.238, xybar 0.187, y2bar 0.161; xybar <- x2bar 0.262, x-bar 0.236, xy2br 0.211; y-bar <- x2ybr 0.358, xegvy 0.302, x-bar 0.221 ...
- `electricity_80nan`: D top3 +0.054 [+0.050, +0.058] seeds +0.054, +0.054, +0.054 -> faithful: yes; D top1 +0.027 [+0.022, +0.031] seeds +0.026, +0.027, +0.028 -> faithful: yes
  - map: seed stability +0.990; attention against occlusion +0.817; |Spearman| against occlusion +0.769 (within target, mean)
  - top sources: date <- vicdemand 0.493, vicprice 0.484, transfer 0.475; day <- nswdemand 0.210, vicdemand 0.203, transfer 0.166; nswdemand <- period 0.510, vicdemand 0.408, vicprice 0.250; nswprice <- vicprice 0.393, vicdemand 0.220, date 0.209; period <- nswdemand 0.390, nswprice 0.154, vicdemand 0.137; transfer <- vicdemand 0.449, vicprice 0.363, date 0.278; vicdemand <- nswdemand 0.451, transfer 0.387, vicprice 0.331; vicprice <- nswdemand 0.050, nswprice 0.045, transfer 0.041
- `vehicle_80nan`: D top3 +0.028 [+0.012, +0.044] seeds +0.025, +0.032, +0.028 -> faithful: yes; D top1 +0.005 [-0.003, +0.013] seeds +0.004, +0.007, +0.004 -> faithful: inconclusive
  - map: seed stability +0.748; attention against occlusion +0.788; |Spearman| against occlusion +0.442 (within target, mean)
  - top sources: CIRCULARITY <- ELONGATEDNESS 0.329, MAX.LENGTH_RECTANGULARITY 0.294, SCALED_RADIUS_OF_GYRATION 0.271; COMPACTNESS <- SCATTER_RATIO 0.298, PR.AXIS_RECTANGULARITY 0.292, SCALED_VARIANCE_MAJOR 0.287; DISTANCE_CIRCULARITY <- SCALED_RADIUS_OF_GYRATION 0.290, RADIUS_RATIO 0.278, ELONGATEDNESS 0.278; ELONGATEDNESS <- SCALED_RADIUS_OF_GYRATION 0.291, SCATTER_RATIO 0.289, SCALED_VARIANCE_MAJOR 0.269; HOLLOWS_RATIO <- SKEWNESS_ABOUT_MAJOR 0.413, KURTOSIS_ABOUT_MINOR 0.366, CIRCULARITY 0.357; KURTOSIS_ABOUT_MAJOR <- SCALED_VARIANCE_MAJOR 0.308, CIRCULARITY 0.290, KURTOSIS_ABOUT_MINOR 0.246; KURTOSIS_ABOUT_MINOR <- HOLLOWS_RATIO 0.375, ELONGATEDNESS 0.325, DISTANCE_CIRCULARITY 0.320; MAX.LENGTH_ASPECT_RATIO <- SCALED_RADIUS_OF_GYRATION 0.373, SCALED_VARIANCE_MAJOR 0.262, HOLLOWS_RATIO 0.234; MAX.LENGTH_RECTANGULARITY <- CIRCULARITY 0.377, SCATTER_RATIO 0.288, DISTANCE_CIRCULARITY 0.263; PR.AXIS_ASPECT_RATIO <- COMPACTNESS 0.360, RADIUS_RATIO 0.236, HOLLOWS_RATIO 0.210; PR.AXIS_RECTANGULARITY <- SCATTER_RATIO 0.296, SCALED_VARIANCE_MAJOR 0.276, COMPACTNESS 0.269; RADIUS_RATIO <- DISTANCE_CIRCULARITY 0.278, HOLLOWS_RATIO 0.276, SCALED_VARIANCE_MAJOR 0.269 ...
- `biodeg_80nan`: D top3 -0.002 [-0.010, +0.006] seeds +0.001, -0.004, -0.003 -> faithful: no; D top1 -0.003 [-0.009, +0.004] seeds +0.003, -0.004, -0.007 -> faithful: no
  - map: seed stability +0.796; attention against occlusion +0.721; |Spearman| against occlusion +0.232 (within target, mean)
  - top sources: V1 <- V15 0.395, V27 0.298, V36 0.237; V10 <- V16 0.297, V21 0.247, V15 0.208; V11 <- V1 0.296, V38 0.231, V34 0.209; V12 <- V34 0.190, V17 0.176, V24 0.161; V13 <- V15 0.298, V39 0.283, V7 0.236; V14 <- V8 0.283, V31 0.262, V16 0.258; V15 <- V1 0.343, V13 0.287, V27 0.283; V16 <- V10 0.418, V13 0.257, V11 0.237; V17 <- V1 0.222, V33 0.180, V37 0.173; V18 <- V22 0.225, V2 0.200, V8 0.197; V19 <- V15 0.006, V25 0.006, V39 0.006; V2 <- V3 0.300, V21 0.253, V29 0.247 ...
- `kc2_80nan`: D top3 -0.011 [-0.022, +0.000] seeds -0.011, +0.002, -0.025 -> faithful: no; D top1 -0.007 [-0.013, -0.001] seeds -0.008, -0.000, -0.013 -> faithful: no
  - map: seed stability +0.844; attention against occlusion +0.604; |Spearman| against occlusion +0.121 (within target, mean)
  - top sources: b <- v 0.185, d 0.143, uniq_Opnd 0.138; branchCount <- i 0.139, n 0.124, lOComment 0.104; d <- v 0.223, uniq_Op 0.220, v(g) 0.214; e <- n 0.076, v(g) 0.073, total_Op 0.065; ev(g) <- n 0.193, v(g) 0.172, v 0.169; i <- d 0.167, n 0.150, lOComment 0.138; iv(g) <- v(g) 0.105, lOComment 0.082, branchCount 0.070; l <- uniq_Op 0.417, d 0.408, b 0.312; lOBlank <- v(g) 0.107, iv(g) 0.105, v 0.101; lOCode <- n 0.204, loc 0.158, d 0.153; lOCodeAndComment <- n 0.143, d 0.133, uniq_Op 0.129; lOComment <- b 0.122, v(g) 0.110, iv(g) 0.098 ...
- `kr-vs-kp_80nan`: D top3 +0.008 [+0.005, +0.012] seeds +0.009, +0.010, +0.007 -> faithful: inconclusive; D top1 +0.005 [+0.003, +0.008] seeds +0.003, +0.006, +0.006 -> faithful: no
  - map: seed stability +0.879; attention against occlusion +0.766; |Spearman| against occlusion +nan (within target, mean)
  - top sources: bkblk <- katri 0.013, cntxt 0.011, wkcti 0.011; bknwy <- cntxt 0.013, wkcti 0.013, r2ar8 0.010; bkon8 <- dsopp 0.004, cntxt 0.003, skewr 0.003; bkona <- bkspr 0.014, wtoeg 0.009, bkxcr 0.009; bkspr <- bkxcr 0.077, bkxwp 0.056, wkcti 0.042; bkxbq <- reskd 0.109, katri 0.036, rimmx 0.022; bkxcr <- bkxwp 0.155, blxwp 0.093, bkspr 0.063; bkxwp <- blxwp 0.161, bkxcr 0.121, qxmsq 0.059; blxwp <- bkxwp 0.159, rkxwp 0.149, bkxcr 0.071; bxqsq <- bkxwp 0.060, rkxwp 0.051, stlmt 0.036; cntxt <- skewr 0.183, wkcti 0.101, katri 0.100; dsopp <- bkxcr 0.017, blxwp 0.010, wkpos 0.010 ...
- `spambase_80nan`: D top3 -0.021 [-0.025, -0.016] seeds -0.021, -0.020, -0.021 -> faithful: no; D top1 -0.018 [-0.021, -0.015] seeds -0.017, -0.019, -0.018 -> faithful: no
  - map: seed stability +0.590; attention against occlusion +0.746; |Spearman| against occlusion +0.224 (within target, mean)
  - top sources: capital_run_length_average <- char_freq_%24 0.055, word_freq_over 0.046, word_freq_all 0.035; capital_run_length_longest <- word_freq_order 0.151, word_freq_000 0.114, capital_run_length_total 0.107; capital_run_length_total <- word_freq_000 0.135, char_freq_%24 0.119, word_freq_make 0.118; char_freq_%21 <- word_freq_project 0.087, word_freq_857 0.067, word_freq_you 0.057; char_freq_%23 <- word_freq_hpl 0.088, char_freq_%24 0.067, capital_run_length_longest 0.052; char_freq_%24 <- word_freq_you 0.139, word_freq_000 0.133, word_freq_all 0.104; char_freq_%28 <- word_freq_hpl 0.073, word_freq_telnet 0.071, word_freq_415 0.070; char_freq_%3B <- char_freq_%28 0.061, word_freq_technology 0.046, word_freq_order 0.041; char_freq_%5B <- word_freq_george 0.073, char_freq_%28 0.069, word_freq_all 0.059; word_freq_000 <- char_freq_%24 0.189, capital_run_length_total 0.101, word_freq_telnet 0.096; word_freq_1999 <- word_freq_you 0.095, word_freq_hp 0.082, word_freq_hpl 0.072; word_freq_3d <- word_freq_make 0.038, word_freq_cs 0.035, word_freq_000 0.030 ...
- `credit-g_80nan`: D top3 -0.004 [-0.006, -0.002] seeds -0.002, -0.004, -0.006 -> faithful: no; D top1 -0.003 [-0.004, -0.001] seeds -0.001, -0.003, -0.004 -> faithful: no
  - map: seed stability +0.593; attention against occlusion +0.319; |Spearman| against occlusion +0.055 (within target, mean)
  - top sources: age <- residence_since 0.080, employment 0.049, housing 0.042; checking_status <- installment_commitment 0.037, duration 0.033, age 0.027; credit_amount <- installment_commitment 0.133, duration 0.074, other_parties 0.049; credit_history <- age 0.047, foreign_worker 0.030, residence_since 0.030; duration <- residence_since 0.094, age 0.066, credit_amount 0.059; employment <- age 0.038, residence_since 0.035, installment_commitment 0.031; existing_credits <- duration 0.060, residence_since 0.023, credit_amount 0.022; foreign_worker <- checking_status 0.006, age 0.006, residence_since 0.004; housing <- num_dependents 0.022, credit_amount 0.021, residence_since 0.019; installment_commitment <- num_dependents 0.049, job 0.039, duration 0.033; job <- installment_commitment 0.028, age 0.025, residence_since 0.022; num_dependents <- age 0.059, residence_since 0.044, credit_amount 0.038 ...

- K against the earlier run of the same cell, largest fold difference: 2.06e-02 over 33 cells
- truth (C, gate) at 20nan: incomplete
- truth (C, gate) at 80nan: yes

Primary at 20nan: incomplete (2 of 9)

Primary (B) at 80nan: faithful on 4 of 9, not faithful on ['biodeg_80nan', 'kc2_80nan', 'spambase_80nan', 'credit-g_80nan'] -> refuted (fold-pair t, seed agreement, SESOI 0.01)
```
