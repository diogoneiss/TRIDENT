# E184's sliced Wasserstein loss term on the eight larger MCAR variants (E185, T62; pre-registered 2026-11-20 19:10 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E184 (ticket 73) was mixed on the 18 small and medium MCAR variants: the term lowered the output's
  `sw_score` on 5 (with the `sw_score` selector) and 4 (with the default), the same few under both, and by much only
  on `spambase_20nan` (−0.23 / −0.20), the largest table in that set. This study asks whether the term does more on
  the larger tables E184 left out.
- **The user's go:** "teste em mais tabelas!" (2026-11-20), answering my offer after E184 of the larger or more-missing
  tables (`spambase` 40–80nan, `pendigits`, `letter`) or a weight sweep. The design below is mine.
- **Disclosed:**
  - E184's results were read before this design, including the `spambase_20nan` gain that motivates it.
  - The variants are every MCAR variant of E182/E183's set that E184 did not run (no choice by result):
    `pendigits_20/40/60/80nan`, `electricity_20/40/80nan`, `letter_20nan`. `spambase` 40–80nan are not in that set
    and have no E182 baseline files, so they are left out.
  - No new code in `src/`; the weight is E184's (0.8), with no new smoke.
- **Runner (this commit):** `scripts/experiments/sliced_loss_large_run.py` runs `sliced_loss_run.py`'s functions with
  this study's variants, state `/scratch/diogoneiss/trident-2026-11-20-sliced-loss-large`, tag
  `experiment = sliced-loss-large-2026-11-20`, and the rule below. `pendigits` and `letter` use their promoted
  files (ADR 0018 / 0037 bigger encoder where they apply), as in E183.
- **Scenarios:** the 8 variants × seeds 101/202/303 × five folds, all on gorgona8's RTX 3090 Ti, four queues
  (`c15_queue.sh`, two threads each), a pair's two arms side by side.
  - Core (the primary): **S** (`--decode_metric sw_score`) and **LS** (S plus `--decode_sw_weight 0.8`). 48 runs.
  - Second batch, after the core: **K** (the default run) and **L** (K plus the term). 48 runs.
  - Cost: about 14–18 h a batch (E183's gorgona8 clocks: `letter_20nan` about 115 min a run, `electricity_20nan`
    about 70; the term adds about 20% to decode).
- **Checks before the read:** `--gate` and `--pairing`, as in E184. A failing check is reported, not dropped.
- **Measures and decision rule:** as E184: each arm's test-fold ledgers scored by E182's code (`sw_score`,
  `dist_score`, `impute_score`) on the calibrated output, fold-pair rule with seed agreement, margin 0.01 of the
  control's mean.
  - Hypothesis: on the larger tables the term lowers the output's `sw_score`.
  - Primary, LS − S on `sw_score` over the variants whose shuffled control reaches 0.10: supported if better on half
    or more and worse on at most 1 (E184's 2 of 18 scaled to 8); refuted if better on a sixth or fewer (at most 1 of
    8); mixed otherwise. The same rule is printed for L − K once the second batch is in.
  - Secondary, no rule: `impute_score` and `dist_score` for both pairs; kept epochs; each arm against E182's best
    strong-tier baseline on `sw_score` and `impute_score`.
  - Read once: `--tally core` when the core is in, `--tally` when all four arms are. No default changes whatever the
    result; pooling with E184 is not planned.
- **Not in this study:** other weights; `spambase` 40–80nan; pre-training; calibration off.

## Outcome

### Final read, both batches (2026-11-22 08:48 GMT-3): mixed, no default change

All 96 runs were in at 2026-11-22 08:35 GMT-3, none failed. `--gate`: 96 finished runs checked, 0 failed.
`--pairing`: cell counts and hgb's score the same on all 48 pairs (24 S/LS, 24 K/L). Then `--tally`, once.

- **The core line stands, mixed:** LS − S on `sw_score` better on 4 of 8, worse on 2 (unchanged from the core read).
- **Second batch, L − K on `sw_score`, mixed:** better on 3 (`pendigits_40nan` −0.006, `pendigits_60nan` −0.004,
  `electricity_20nan` −0.075), worse on 2 (`pendigits_20nan` +0.087, `letter_20nan` +0.153), no difference on 3.
- **One large gain and two large losses, under both selectors.** `electricity_20nan` gains about 0.075 with either
  selector; the other gains are 0.03 or smaller. The two bigger-encoder 20nan variants (ADR 0018) lose 0.08–0.15
  with either selector.
- **The `letter_20nan` loss is the term's, not the selector's.** With K the kept epoch barely moves (544 to 491), yet
  `sw_score` still doubles (0.139 to 0.292). The early epoch under LS (513 to 154) is a symptom, not the cause.
- **Cost:** L − K on `impute_score` worse on 6 (up to +0.066 on `letter_20nan`, +0.037 on `pendigits_20nan`), better
  on none. `dist_score`: better 5, worse 2.
- **Against the best strong baseline on `sw_score`:** S 5 better / 1 worse, LS 5 / 2, K 6 / 2, L 4 / 3; on
  `impute_score` S 7 / 0, LS 5 / 3, K 7 / 0, L 5 / 2. The term costs standing on both scores.
- With E184, 26 MCAR variants: the term helps much on two (`spambase_20nan`, `electricity_20nan`), hurts much on two
  (`pendigits_20nan`, `letter_20nan`) and moves the rest by 0.03 or less. No default changes and no ADR:
  `--decode_sw_weight` stays at 0.

Printout (624 lines of scorer chatter omitted: 48 `DONE ... reference.csv` lines, the label-mapping blocks
and the categorical-columns loader lines; everything else unedited):

```
- `pendigits_20nan` (seen): mean kept epoch S 494, LS 423, K 528, L 510
  - LS − S on sw_score +0.0841 [+0.0807, +0.0876] worse (S 0.1278, LS 0.2119)
  - LS − S on dist_score +0.0496 [+0.0469, +0.0522] worse (S 0.0759, LS 0.1255)
  - LS − S on impute_score +0.0397 [+0.0350, +0.0444] worse (S 0.3012, LS 0.3409)
  - L − K on sw_score +0.0868 [+0.0826, +0.0910] worse (K 0.1283, L 0.2151)
  - L − K on dist_score +0.0514 [+0.0489, +0.0539] worse (K 0.0758, L 0.1272)
  - L − K on impute_score +0.0365 [+0.0337, +0.0394] worse (K 0.3004, L 0.3369)
  - S − strong (missforest) on sw_score -0.0559 [-0.0603, -0.0515] better
  - LS − strong (missforest) on sw_score +0.0282 [+0.0220, +0.0343] worse
  - K − strong (missforest) on sw_score -0.0555 [-0.0600, -0.0509] better
  - L − strong (missforest) on sw_score +0.0314 [+0.0240, +0.0387] worse
  - S − strong (missforest) on impute_score -0.0328 [-0.0390, -0.0267] better
  - LS − strong (missforest) on impute_score +0.0068 [+0.0012, +0.0124] worse
  - K − strong (missforest) on impute_score -0.0337 [-0.0402, -0.0272] better
  - L − strong (missforest) on impute_score +0.0028 [-0.0034, +0.0091] none
- `pendigits_40nan` (seen): mean kept epoch S 505, LS 425, K 539, L 530
  - LS − S on sw_score +0.0028 [-0.0023, +0.0078] none (S 0.1425, LS 0.1452)
  - LS − S on dist_score +0.0020 [-0.0015, +0.0055] none (S 0.0937, LS 0.0958)
  - LS − S on impute_score +0.0093 [+0.0056, +0.0130] worse (S 0.3819, LS 0.3912)
  - L − K on sw_score -0.0055 [-0.0079, -0.0030] better (K 0.1419, L 0.1364)
  - L − K on dist_score -0.0027 [-0.0043, -0.0011] better (K 0.0933, L 0.0906)
  - L − K on impute_score +0.0035 [+0.0015, +0.0054] worse (K 0.3805, L 0.3839)
  - S − strong (missforest) on sw_score -0.0971 [-0.1022, -0.0920] better
  - LS − strong (missforest) on sw_score -0.0943 [-0.1017, -0.0870] better
  - K − strong (missforest) on sw_score -0.0976 [-0.1029, -0.0924] better
  - L − strong (missforest) on sw_score -0.1031 [-0.1089, -0.0972] better
  - S − strong (missforest) on impute_score -0.1158 [-0.1213, -0.1104] better
  - LS − strong (missforest) on impute_score -0.1065 [-0.1129, -0.1001] better
  - K − strong (missforest) on impute_score -0.1172 [-0.1228, -0.1117] better
  - L − strong (missforest) on impute_score -0.1138 [-0.1185, -0.1090] better
- `pendigits_60nan` (seen): mean kept epoch S 524, LS 478, K 474, L 463
  - LS − S on sw_score -0.0027 [-0.0058, +0.0004] none (S 0.2608, LS 0.2581)
  - LS − S on dist_score -0.0027 [-0.0054, -0.0000] better (S 0.1786, LS 0.1759)
  - LS − S on impute_score +0.0047 [+0.0018, +0.0075] worse (S 0.5550, LS 0.5597)
  - L − K on sw_score -0.0044 [-0.0077, -0.0011] better (K 0.2629, L 0.2586)
  - L − K on dist_score -0.0039 [-0.0062, -0.0017] better (K 0.1796, L 0.1756)
  - L − K on impute_score +0.0024 [+0.0000, +0.0048] worse (K 0.5575, L 0.5599)
  - S − strong (softimpute) on sw_score -0.0566 [-0.0636, -0.0495] better
  - LS − strong (softimpute) on sw_score -0.0592 [-0.0663, -0.0522] better
  - K − strong (softimpute) on sw_score -0.0544 [-0.0625, -0.0464] better
  - L − strong (softimpute) on sw_score -0.0588 [-0.0670, -0.0505] better
  - S − strong (missforest) on impute_score -0.1532 [-0.1620, -0.1444] better
  - LS − strong (missforest) on impute_score -0.1486 [-0.1560, -0.1412] better
  - K − strong (missforest) on impute_score -0.1507 [-0.1590, -0.1424] better
  - L − strong (missforest) on impute_score -0.1483 [-0.1560, -0.1405] better
- `pendigits_80nan` (seen): mean kept epoch S 371, LS 366, K 57, L 40
  - LS − S on sw_score -0.0067 [-0.0098, -0.0037] better (S 0.5949, LS 0.5881)
  - LS − S on dist_score -0.0091 [-0.0118, -0.0064] better (S 0.4721, LS 0.4630)
  - LS − S on impute_score +0.0053 [+0.0044, +0.0062] worse (S 0.8321, LS 0.8374)
  - L − K on sw_score -0.0040 [-0.0121, +0.0040] none (K 0.6054, L 0.6014)
  - L − K on dist_score -0.0086 [-0.0148, -0.0024] better (K 0.4814, L 0.4728)
  - L − K on impute_score +0.0078 [+0.0060, +0.0096] worse (K 0.8254, L 0.8332)
  - S − strong (missforest) on sw_score +0.0858 [+0.0688, +0.1028] worse
  - LS − strong (missforest) on sw_score +0.0791 [+0.0628, +0.0954] worse
  - K − strong (missforest) on sw_score +0.0964 [+0.0811, +0.1117] worse
  - L − strong (missforest) on sw_score +0.0924 [+0.0762, +0.1085] worse
  - S − strong (condexp) on impute_score -0.0157 [-0.0175, -0.0140] better
  - LS − strong (condexp) on impute_score -0.0104 [-0.0121, -0.0087] better
  - K − strong (condexp) on impute_score -0.0224 [-0.0239, -0.0209] better
  - L − strong (condexp) on impute_score -0.0146 [-0.0160, -0.0132] better
- `electricity_20nan` (seen): mean kept epoch S 341, LS 377, K 347, L 333
  - LS − S on sw_score -0.0744 [-0.0780, -0.0709] better (S 0.2649, LS 0.1904)
  - LS − S on dist_score -0.0261 [-0.0295, -0.0226] better (S 0.2455, LS 0.2194)
  - LS − S on impute_score +0.0119 [+0.0061, +0.0178] worse (S 0.6189, LS 0.6308)
  - L − K on sw_score -0.0752 [-0.0820, -0.0684] better (K 0.2779, L 0.2027)
  - L − K on dist_score -0.0247 [-0.0302, -0.0193] better (K 0.2536, L 0.2289)
  - L − K on impute_score +0.0160 [+0.0110, +0.0210] worse (K 0.6178, L 0.6339)
  - S − strong (missforest) on sw_score -0.0359 [-0.0398, -0.0321] better
  - LS − strong (missforest) on sw_score -0.1103 [-0.1150, -0.1057] better
  - K − strong (missforest) on sw_score -0.0229 [-0.0294, -0.0164] better
  - L − strong (missforest) on sw_score -0.0981 [-0.1044, -0.0917] better
  - S − strong (missforest) on impute_score -0.0004 [-0.0096, +0.0088] none
  - LS − strong (missforest) on impute_score +0.0115 [+0.0029, +0.0202] worse
  - K − strong (missforest) on impute_score -0.0014 [-0.0102, +0.0073] none
  - L − strong (missforest) on impute_score +0.0146 [+0.0052, +0.0240] worse
- `electricity_40nan` (seen): mean kept epoch S 313, LS 372, K 315, L 308
  - LS − S on sw_score -0.0257 [-0.0362, -0.0152] better (S 0.3738, LS 0.3481)
  - LS − S on dist_score -0.0245 [-0.0330, -0.0161] better (S 0.3348, LS 0.3103)
  - LS − S on impute_score +0.0096 [-0.0130, +0.0322] none (S 0.7371, LS 0.7467)
  - L − K on sw_score -0.0096 [-0.0245, +0.0054] none (K 0.3689, L 0.3593)
  - L − K on dist_score -0.0119 [-0.0213, -0.0024] better (K 0.3277, L 0.3159)
  - L − K on impute_score -0.0035 [-0.0129, +0.0059] none (K 0.7468, L 0.7433)
  - S − strong (missforest) on sw_score -0.0035 [-0.0130, +0.0060] none
  - LS − strong (missforest) on sw_score -0.0292 [-0.0381, -0.0203] better
  - K − strong (missforest) on sw_score -0.0084 [-0.0156, -0.0011] better
  - L − strong (missforest) on sw_score -0.0179 [-0.0344, -0.0015] better
  - S − strong (missforest) on impute_score -0.0512 [-0.0663, -0.0360] better
  - LS − strong (missforest) on impute_score -0.0416 [-0.0662, -0.0169] better
  - K − strong (missforest) on impute_score -0.0415 [-0.0600, -0.0230] better
  - L − strong (missforest) on impute_score -0.0450 [-0.0591, -0.0309] better
- `electricity_80nan` (seen): mean kept epoch S 233, LS 250, K 155, L 103
  - LS − S on sw_score -0.0127 [-0.0252, -0.0003] better (S 0.6381, LS 0.6254)
  - LS − S on dist_score -0.0194 [-0.0287, -0.0100] better (S 0.5724, LS 0.5530)
  - LS − S on impute_score -0.0024 [-0.0105, +0.0057] none (S 0.9039, LS 0.9015)
  - L − K on sw_score -0.0036 [-0.0150, +0.0078] none (K 0.6442, L 0.6405)
  - L − K on dist_score -0.0116 [-0.0213, -0.0019] none (K 0.5788, L 0.5672)
  - L − K on impute_score -0.0110 [-0.0318, +0.0098] none (K 0.9119, L 0.9009)
  - S − strong (missforest) on sw_score +0.0286 [-0.0008, +0.0580] none
  - LS − strong (missforest) on sw_score +0.0159 [-0.0125, +0.0442] none
  - K − strong (missforest) on sw_score +0.0347 [+0.0031, +0.0663] worse
  - L − strong (missforest) on sw_score +0.0311 [+0.0012, +0.0610] worse
  - S − strong (condexp) on impute_score -0.0536 [-0.0645, -0.0428] better
  - LS − strong (condexp) on impute_score -0.0560 [-0.0618, -0.0502] better
  - K − strong (condexp) on impute_score -0.0456 [-0.0670, -0.0242] better
  - L − strong (condexp) on impute_score -0.0566 [-0.0596, -0.0537] better
- `letter_20nan` (seen): mean kept epoch S 513, LS 154, K 544, L 491
  - LS − S on sw_score +0.1269 [+0.1195, +0.1342] worse (S 0.1375, LS 0.2644)
  - LS − S on dist_score +0.0627 [+0.0529, +0.0726] worse (S 0.1632, LS 0.2259)
  - LS − S on impute_score +0.1073 [+0.0907, +0.1240] worse (S 0.4228, LS 0.5301)
  - L − K on sw_score +0.1528 [+0.1457, +0.1599] worse (K 0.1392, L 0.2920)
  - L − K on dist_score +0.0582 [+0.0555, +0.0609] worse (K 0.1635, L 0.2217)
  - L − K on impute_score +0.0658 [+0.0645, +0.0672] worse (K 0.4206, L 0.4865)
  - S − strong (missforest) on sw_score -0.1594 [-0.1636, -0.1553] better
  - LS − strong (missforest) on sw_score -0.0326 [-0.0408, -0.0243] better
  - K − strong (missforest) on sw_score -0.1577 [-0.1616, -0.1538] better
  - L − strong (missforest) on sw_score -0.0049 [-0.0137, +0.0039] none
  - S − strong (missforest) on impute_score -0.0401 [-0.0434, -0.0369] better
  - LS − strong (missforest) on impute_score +0.0672 [+0.0508, +0.0836] worse
  - K − strong (missforest) on impute_score -0.0423 [-0.0457, -0.0389] better
  - L − strong (missforest) on impute_score +0.0236 [+0.0198, +0.0274] worse

Primary, LS − S on sw_score over the 8 variants the control passes: better on 4, worse on 2 ['pendigits_20nan', 'letter_20nan'] -> mixed
Secondary, LS − S on dist_score, all variants: better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement)
Secondary, LS − S on impute_score, all variants: better 0 / no difference 2 / worse 6 (fold-pair t, seed agreement)

Second batch, L − K on sw_score over the 8 variants the control passes: better on 3, worse on 2 ['pendigits_20nan', 'letter_20nan'] -> mixed
Secondary, L − K on dist_score, all variants: better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement)
Secondary, L − K on impute_score, all variants: better 0 / no difference 2 / worse 6 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on sw_score: S better 5 / no difference 2 / worse 1 (fold-pair t, seed agreement); LS better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement); K better 6 / no difference 0 / worse 2 (fold-pair t, seed agreement); L better 4 / no difference 1 / worse 3 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on impute_score: S better 7 / no difference 1 / worse 0 (fold-pair t, seed agreement); LS better 5 / no difference 0 / worse 3 (fold-pair t, seed agreement); K better 7 / no difference 1 / worse 0 (fold-pair t, seed agreement); L better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement)
```

### Core read, S against LS (2026-11-21 18:22 GMT-3): mixed

The 48 core runs were in at 2026-11-21 18:08 GMT-3, none failed. `--gate`: 63 finished runs checked (the 48 core
runs and 15 second-batch runs already done), 0 failed. `--pairing`: cell counts and hgb's score the same on all 24
S/LS pairs. Then `--tally core`, once.

- **Primary, LS − S on `sw_score`, mixed.** The control passes on all 8. Better on 4 (`electricity_20nan` −0.074,
  `electricity_40nan` −0.026, `electricity_80nan` −0.013, `pendigits_80nan` −0.007), worse on 2 (`pendigits_20nan`
  +0.084, `letter_20nan` +0.127), no difference on 2. Supported needed 4 better and at most 1 worse.
- **Split by table.** The term helps on all three `electricity` variants (at 20nan LS beats missforest by 0.110, S by
  0.036). It hurts on the two 20nan variants that use ADR 0018's bigger encoder: on `letter_20nan` the kept epoch falls
  from 513 to 154 and `sw_score` nearly doubles (0.138 to 0.264).
- **Cost on `impute_score`:** worse on 6 (up to +0.107 on `letter_20nan`, +0.040 on `pendigits_20nan`, +0.012 on
  `electricity_20nan`), better on none. `dist_score`: better 5, worse 2.
- **Against the best strong baseline on `sw_score`:** S better 5 / worse 1, LS better 5 / worse 2; on `impute_score`
  S 7 / 0, LS 5 / 3.
- The second batch (K against L) runs on; its read gives the study's final line. No default changes.

Printout (624 lines of scorer chatter omitted: 48 `DONE ... reference.csv` lines, the label-mapping blocks
and the categorical-columns loader lines; everything else unedited):

```
# 6f6027f / read 2026-11-21 18:08 GMT-3
- `pendigits_20nan` (seen): mean kept epoch S 494, LS 423
  - LS − S on sw_score +0.0841 [+0.0807, +0.0876] worse (S 0.1278, LS 0.2119)
  - LS − S on dist_score +0.0496 [+0.0469, +0.0522] worse (S 0.0759, LS 0.1255)
  - LS − S on impute_score +0.0397 [+0.0350, +0.0444] worse (S 0.3012, LS 0.3409)
  - S − strong (missforest) on sw_score -0.0559 [-0.0603, -0.0515] better
  - LS − strong (missforest) on sw_score +0.0282 [+0.0220, +0.0343] worse
  - S − strong (missforest) on impute_score -0.0328 [-0.0390, -0.0267] better
  - LS − strong (missforest) on impute_score +0.0068 [+0.0012, +0.0124] worse
- `pendigits_40nan` (seen): mean kept epoch S 505, LS 425
  - LS − S on sw_score +0.0028 [-0.0023, +0.0078] none (S 0.1425, LS 0.1452)
  - LS − S on dist_score +0.0020 [-0.0015, +0.0055] none (S 0.0937, LS 0.0958)
  - LS − S on impute_score +0.0093 [+0.0056, +0.0130] worse (S 0.3819, LS 0.3912)
  - S − strong (missforest) on sw_score -0.0971 [-0.1022, -0.0920] better
  - LS − strong (missforest) on sw_score -0.0943 [-0.1017, -0.0870] better
  - S − strong (missforest) on impute_score -0.1158 [-0.1213, -0.1104] better
  - LS − strong (missforest) on impute_score -0.1065 [-0.1129, -0.1001] better
- `pendigits_60nan` (seen): mean kept epoch S 524, LS 478
  - LS − S on sw_score -0.0027 [-0.0058, +0.0004] none (S 0.2608, LS 0.2581)
  - LS − S on dist_score -0.0027 [-0.0054, -0.0000] better (S 0.1786, LS 0.1759)
  - LS − S on impute_score +0.0047 [+0.0018, +0.0075] worse (S 0.5550, LS 0.5597)
  - S − strong (softimpute) on sw_score -0.0566 [-0.0636, -0.0495] better
  - LS − strong (softimpute) on sw_score -0.0592 [-0.0663, -0.0522] better
  - S − strong (missforest) on impute_score -0.1532 [-0.1620, -0.1444] better
  - LS − strong (missforest) on impute_score -0.1486 [-0.1560, -0.1412] better
- `pendigits_80nan` (seen): mean kept epoch S 371, LS 366
  - LS − S on sw_score -0.0067 [-0.0098, -0.0037] better (S 0.5949, LS 0.5881)
  - LS − S on dist_score -0.0091 [-0.0118, -0.0064] better (S 0.4721, LS 0.4630)
  - LS − S on impute_score +0.0053 [+0.0044, +0.0062] worse (S 0.8321, LS 0.8374)
  - S − strong (missforest) on sw_score +0.0858 [+0.0688, +0.1028] worse
  - LS − strong (missforest) on sw_score +0.0791 [+0.0628, +0.0954] worse
  - S − strong (condexp) on impute_score -0.0157 [-0.0175, -0.0140] better
  - LS − strong (condexp) on impute_score -0.0104 [-0.0121, -0.0087] better
- `electricity_20nan` (seen): mean kept epoch S 341, LS 377
  - LS − S on sw_score -0.0744 [-0.0780, -0.0709] better (S 0.2649, LS 0.1904)
  - LS − S on dist_score -0.0261 [-0.0295, -0.0226] better (S 0.2455, LS 0.2194)
  - LS − S on impute_score +0.0119 [+0.0061, +0.0178] worse (S 0.6189, LS 0.6308)
  - S − strong (missforest) on sw_score -0.0359 [-0.0398, -0.0321] better
  - LS − strong (missforest) on sw_score -0.1103 [-0.1150, -0.1057] better
  - S − strong (missforest) on impute_score -0.0004 [-0.0096, +0.0088] none
  - LS − strong (missforest) on impute_score +0.0115 [+0.0029, +0.0202] worse
- `electricity_40nan` (seen): mean kept epoch S 313, LS 372
  - LS − S on sw_score -0.0257 [-0.0362, -0.0152] better (S 0.3738, LS 0.3481)
  - LS − S on dist_score -0.0245 [-0.0330, -0.0161] better (S 0.3348, LS 0.3103)
  - LS − S on impute_score +0.0096 [-0.0130, +0.0322] none (S 0.7371, LS 0.7467)
  - S − strong (missforest) on sw_score -0.0035 [-0.0130, +0.0060] none
  - LS − strong (missforest) on sw_score -0.0292 [-0.0381, -0.0203] better
  - S − strong (missforest) on impute_score -0.0512 [-0.0663, -0.0360] better
  - LS − strong (missforest) on impute_score -0.0416 [-0.0662, -0.0169] better
- `electricity_80nan` (seen): mean kept epoch S 233, LS 250
  - LS − S on sw_score -0.0127 [-0.0252, -0.0003] better (S 0.6381, LS 0.6254)
  - LS − S on dist_score -0.0194 [-0.0287, -0.0100] better (S 0.5724, LS 0.5530)
  - LS − S on impute_score -0.0024 [-0.0105, +0.0057] none (S 0.9039, LS 0.9015)
  - S − strong (missforest) on sw_score +0.0286 [-0.0008, +0.0580] none
  - LS − strong (missforest) on sw_score +0.0159 [-0.0125, +0.0442] none
  - S − strong (condexp) on impute_score -0.0536 [-0.0645, -0.0428] better
  - LS − strong (condexp) on impute_score -0.0560 [-0.0618, -0.0502] better
- `letter_20nan` (seen): mean kept epoch S 513, LS 154
  - LS − S on sw_score +0.1269 [+0.1195, +0.1342] worse (S 0.1375, LS 0.2644)
  - LS − S on dist_score +0.0627 [+0.0529, +0.0726] worse (S 0.1632, LS 0.2259)
  - LS − S on impute_score +0.1073 [+0.0907, +0.1240] worse (S 0.4228, LS 0.5301)
  - S − strong (missforest) on sw_score -0.1594 [-0.1636, -0.1553] better
  - LS − strong (missforest) on sw_score -0.0326 [-0.0408, -0.0243] better
  - S − strong (missforest) on impute_score -0.0401 [-0.0434, -0.0369] better
  - LS − strong (missforest) on impute_score +0.0672 [+0.0508, +0.0836] worse

Primary, LS − S on sw_score over the 8 variants the control passes: better on 4, worse on 2 ['pendigits_20nan', 'letter_20nan'] -> mixed
Secondary, LS − S on dist_score, all variants: better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement)
Secondary, LS − S on impute_score, all variants: better 0 / no difference 2 / worse 6 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on sw_score: S better 5 / no difference 2 / worse 1 (fold-pair t, seed agreement); LS better 5 / no difference 1 / worse 2 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on impute_score: S better 7 / no difference 1 / worse 0 (fold-pair t, seed agreement); LS better 5 / no difference 0 / worse 3 (fold-pair t, seed agreement)
```
