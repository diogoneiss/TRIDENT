# A sliced Wasserstein term in the decode loss: do fills trained on the rows' joint distribution score better on it? (E184, T61; pre-registered 2026-11-19 18:18 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E183 (ticket 72) showed that choosing the decode checkpoint by `sw_score` (the sliced Wasserstein distance
  between filled and true rows, over the mean/mode fill's) lowers the output's `sw_score` on 15 of 26 MCAR variants.
  Ticket 72 left out the loss, which is per cell (cross-entropy, `lambda_num` × squared error) and has no term on the
  joint distribution of a row's fills. This study adds one.
- **The user's go:** "quero fazer o 1 e 2" (2026-11-19), answering my two offers after E183: 1, offer the `sw_score`
  selector as a reporting option; 2, test a term in the loss on the joint distribution of the rows ("Isso muda o
  algoritmo de treino e precisa do seu sim"). Item 1 is documentation only (README, this commit). The design below is mine.
- **Disclosed:**
  - E183's results were read before this design, and the variants were chosen by cost knowing them; they include
    E183's largest effects (`credit-g_60nan`, `_80nan`, `kr-vs-kp_80nan`).
  - The design was drafted with a multi-agent review (three designs, two judges; the minimal one chosen), then
    implemented and reviewed adversarially; nothing of it trained a study seed.
  - A weight smoke ran on 2026-11-19 at seed 999 (outside the study), MLflow off, first fold's decode only, on
    `credit-g_40nan` and `kr-vs-kp_40nan` at weight 1 and 0. Only the term's and the per-cell loss's training
    magnitudes and the clock were read; no score. A first attempt at 16:54 GMT-3 ran the full two folds with four
    processes oversubscribing the CPU and was killed at 18:05 unread.
  - On this machine's CPU the `vehicle_00nan` classification fixture fails on HEAD as on this change (it was recorded
    on CUDA); it passes on gorgona8's GPU with this change; the `credit-g_20nan` imputation fixture passes on CPU.
- **Code (this commit):** `--decode_sw_weight w` (`DECODE_SW_WEIGHT`, default 0, imputation only, tag
  `decode_sw_weight` with a backfill script, not yet applied); at 0 nothing runs and the run is as before.
  `src/models.py` (`TridentDecoder._sliced_term`, used by both head paths and the short-batch fallback),
  `src/training/decoding.py` (passes it; logs `decode/train_sliced`), plumbing in `types.py`, `config.py`,
  `mlflow_utils.py`, `runner.py`; tests `tests/unit/test_decode_sliced_loss.py`; runner
  `scripts/experiments/sliced_loss_run.py`; README entries for the flag and for `--decode_metric`'s
  `impute_score_classcond` and `sw_score` values (item 1).
- **The term, in training batches only:** the rows holding a hidden cell, as filled (numerical cells by the head's
  value, categorical cells by the head's softmax probabilities at their vocabulary slots) and as true, in `sw_score`'s
  layout (scaled numbers; each categorical column one-hot over its ids plus an out-of-vocabulary slot, /√2); 512 fixed
  unit directions from `default_rng(1)` (the selector uses seed 0); T = √width × mean |sort(filled·A) − sort(true·A)|.
  Loss = per-cell loss + w × T. The validation pass runs in eval mode without it, so `decode/val_loss`, the loss
  checkpoint, the plateau scheduler and patience are as before. Writes go to distinct positions (deterministic on CUDA).
- **The weight, fixed by a rule stated before the smoke:** r = T / per-cell loss, median over decode epochs 0–49 per
  variant; w = 0.25 / (median of the two r), one significant figure; stop if the two r differ by more than 3×. The
  smoke's printout:

  ```
  credit-g_40nan median r ep0-49 0.1672 | T ep0 0.3893 ep49 0.3144 ep449 0.2218 | cell loss ep0 2.4764 ep449 1.1942 | decode s on 51 off 41
  kr-vs-kp_40nan median r ep0-49 0.4835 | T ep0 0.3111 ep49 0.1655 ep449 0.1456 | cell loss ep0 0.6022 ep449 0.2662 | decode s on 64 off 53
  r ratio 2.89 (stop if > 3); median r 0.3254; w = 0.25/median r = 0.7684 -> one significant figure 0.8
  ```

  **w = 0.8.** The term adds about 20% to the decode clock (51 s against 41 s, 64 against 53 s a fold).
- **Scenarios:** 18 MCAR variants (`kc2`, `vehicle`, `biodeg` at 20/40/80nan; `credit-g`, `kr-vs-kp` at 20/40/60/80nan;
  `spambase_20nan`), seeds 101/202/303, five folds. All on gorgona8's RTX 3090 Ti, two queues, both arms of a pair
  side by side, tag `experiment = sliced-loss-2026-11-19`.
  - Core (the primary): **S**, `--decode_metric sw_score` (E183's arm, retrained here), and **LS**, S plus
    `--decode_sw_weight 0.8`. 108 runs. Reading the term at the `sw_score` selector keeps E183's early-epoch effect of
    the default checkpoint out of the contrast.
  - Second batch, after the core: **K**, the default run, and **L**, K plus the term. 108 runs.
  - Cost: about 15 h a batch (E183's and E174's clocks; the term's +20% on decode only).
- **Checks before the read:** `--gate` (every LS/L run carries `decode_sw_weight` 0.8 and a nonzero
  `decode/train_sliced`; S and K carry 0.0); `--pairing` (per pair, scored cell counts and hgb's score the same per
  fold; the loss epoch differs by design). A failing check is reported, not dropped.
- **Measures and decision rule:** each arm's test-fold ledgers scored by E182's code (`sw_score`, `dist_score`,
  `impute_score`) on the calibrated output (ADR 0027's default; no uncalibrated read), 15 (seed, fold) values a
  variant, fold-pair rule with seed agreement, margin 0.01 of the control's mean.
  - Hypothesis: the term lowers the output's `sw_score`.
  - Primary, LS − S on `sw_score` over the variants whose shuffled control (from this study's S files) reaches 0.10:
    supported if better on half or more and worse on at most 2; refuted if better on a sixth or fewer; mixed otherwise.
    The same rule is printed for L − K once the second batch is in.
  - Secondary, no rule: LS − S and L − K on `impute_score` (the cost) and `dist_score`; the mean kept epoch per arm;
    each arm against E182's best strong-tier baseline (medusa1's fits) on `sw_score` and `impute_score`.
  - A null at 60–80nan reads "no gain under this training mask" (about one hidden cell a training row there), not
    "a joint term does not help".
  - Read once: `sliced_loss_run.py --tally core` when the core is in, `--tally` when all four arms are. No default
    changes whatever the result.
- **Not in this study:** pre-training (unchanged); other weights; calibration off.

## Outcome

### Core read, S against LS (2026-11-20 05:36 GMT-3): mixed

The 108 core runs were in at 2026-11-20 05:30 GMT-3, none failed. `--gate`: 108 finished runs checked, 0 failed.
`--pairing`: scored cell counts and hgb's score the same on all 54 (variant, seed) pairs. Then `--tally core`, once.

- **Primary, LS − S on `sw_score`, mixed.** The control passes on all 18 variants. The term is better on 5
  (`kc2_40nan` −0.010, `vehicle_40nan` −0.009, `biodeg_40nan` −0.021, `credit-g_20nan` −0.014, `spambase_20nan`
  −0.228), worse on 2 (`biodeg_20nan` +0.011, `credit-g_80nan` +0.015), no difference on 11. Supported needed 9,
  refuted 3 or fewer.
- **One large gain.** On `spambase_20nan` the term takes the output's `sw_score` from 0.713 to 0.485; S tied
  missforest there, LS beats it by 0.229. Elsewhere the moves are 0.02 or smaller.
- **Cost on `impute_score`:** worse on 5 (`vehicle_20nan` +0.007, `vehicle_80nan` +0.004, `biodeg_20nan` +0.016,
  `credit-g_20nan` +0.021, `spambase_20nan` +0.038), better on 1 (`kc2_40nan` −0.011), equivalent on 6, no
  difference on 6. `dist_score`: better 6, worse 3.
- **Against the best strong baseline on `sw_score`:** S better 8 / no difference 4 / worse 6; LS 10 / 3 / 5.
- **Kept epoch:** about the same in both arms (within 52 epochs on every variant).
- The second batch (K against L, the default selector) started 05:30 GMT-3 and runs on; its read gives the study's
  final line. No default changes.

Printout (684 lines of scorer chatter omitted: 108 `DONE ... reference.csv` lines, the label-mapping blocks and
the categorical-columns loader lines; everything else unedited):

```
# 40a90d8 / read 2026-11-20 05:31 GMT-3
- `kc2_20nan` (seen): mean kept epoch S 289, LS 290
  - LS − S on sw_score +0.0020 [-0.0057, +0.0096] none (S 0.3956, LS 0.3976)
  - LS − S on dist_score +0.0022 [-0.0057, +0.0101] none (S 0.2524, LS 0.2546)
  - LS − S on impute_score +0.0044 [-0.0071, +0.0160] none (S 0.5352, LS 0.5396)
  - S − strong (missforest) on sw_score -0.0014 [-0.0501, +0.0473] none
  - LS − strong (missforest) on sw_score +0.0006 [-0.0440, +0.0451] none
  - S − strong (softimpute) on impute_score -0.0025 [-0.1034, +0.0984] none
  - LS − strong (softimpute) on impute_score +0.0019 [-0.0905, +0.0943] none
- `kc2_40nan` (seen): mean kept epoch S 264, LS 316
  - LS − S on sw_score -0.0101 [-0.0155, -0.0047] better (S 0.3670, LS 0.3569)
  - LS − S on dist_score -0.0080 [-0.0117, -0.0043] better (S 0.2348, LS 0.2268)
  - LS − S on impute_score -0.0113 [-0.0225, -0.0001] better (S 0.4935, LS 0.4823)
  - S − strong (missforest) on sw_score -0.0078 [-0.0412, +0.0256] none
  - LS − strong (missforest) on sw_score -0.0179 [-0.0498, +0.0140] none
  - S − strong (softimpute) on impute_score -0.0560 [-0.1088, -0.0033] better
  - LS − strong (softimpute) on impute_score -0.0673 [-0.1129, -0.0217] better
- `kc2_80nan` (seen): mean kept epoch S 224, LS 252
  - LS − S on sw_score -0.0073 [-0.0163, +0.0017] none (S 0.4200, LS 0.4128)
  - LS − S on dist_score -0.0099 [-0.0173, -0.0025] better (S 0.3131, LS 0.3032)
  - LS − S on impute_score +0.0004 [-0.0062, +0.0069] none (S 0.6669, LS 0.6672)
  - S − strong (missforest) on sw_score -0.0496 [-0.0747, -0.0245] better
  - LS − strong (missforest) on sw_score -0.0569 [-0.0806, -0.0331] better
  - S − strong (missforest) on impute_score -0.0703 [-0.0912, -0.0494] better
  - LS − strong (missforest) on impute_score -0.0699 [-0.0957, -0.0442] better
- `vehicle_20nan` (seen): mean kept epoch S 285, LS 309
  - LS − S on sw_score -0.0041 [-0.0121, +0.0039] none (S 0.3594, LS 0.3553)
  - LS − S on dist_score -0.0114 [-0.0192, -0.0035] better (S 0.1985, LS 0.1872)
  - LS − S on impute_score +0.0074 [+0.0020, +0.0128] worse (S 0.4145, LS 0.4220)
  - S − strong (ice) on sw_score -0.0121 [-0.0228, -0.0014] better
  - LS − strong (ice) on sw_score -0.0162 [-0.0245, -0.0080] better
  - S − strong (missforest) on impute_score -0.0316 [-0.0555, -0.0076] better
  - LS − strong (missforest) on impute_score -0.0241 [-0.0496, +0.0014] none
- `vehicle_40nan` (seen): mean kept epoch S 297, LS 303
  - LS − S on sw_score -0.0093 [-0.0157, -0.0028] better (S 0.3431, LS 0.3338)
  - LS − S on dist_score -0.0062 [-0.0108, -0.0016] better (S 0.2136, LS 0.2074)
  - LS − S on impute_score +0.0107 [-0.0004, +0.0218] none (S 0.4812, LS 0.4919)
  - S − strong (softimpute) on sw_score -0.0093 [-0.0225, +0.0039] none
  - LS − strong (softimpute) on sw_score -0.0186 [-0.0324, -0.0047] better
  - S − strong (missforest) on impute_score -0.0379 [-0.0663, -0.0095] better
  - LS − strong (missforest) on impute_score -0.0272 [-0.0535, -0.0010] better
- `vehicle_80nan` (seen): mean kept epoch S 253, LS 228
  - LS − S on sw_score -0.0070 [-0.0142, +0.0003] none (S 0.4390, LS 0.4321)
  - LS − S on dist_score -0.0039 [-0.0099, +0.0020] none (S 0.3424, LS 0.3385)
  - LS − S on impute_score +0.0038 [+0.0009, +0.0067] worse (S 0.7841, LS 0.7879)
  - S − strong (missforest) on sw_score -0.0230 [-0.0387, -0.0073] better
  - LS − strong (missforest) on sw_score -0.0299 [-0.0475, -0.0124] better
  - S − strong (missforest) on impute_score +0.0136 [+0.0010, +0.0261] worse
  - LS − strong (missforest) on impute_score +0.0174 [+0.0051, +0.0297] worse
- `biodeg_20nan` (seen): mean kept epoch S 322, LS 345
  - LS − S on sw_score +0.0105 [+0.0047, +0.0163] worse (S 0.4806, LS 0.4911)
  - LS − S on dist_score +0.0328 [+0.0271, +0.0385] worse (S 0.2648, LS 0.2976)
  - LS − S on impute_score +0.0164 [+0.0081, +0.0247] worse (S 0.5906, LS 0.6070)
  - S − strong (missforest) on sw_score -0.0593 [-0.0690, -0.0497] better
  - LS − strong (missforest) on sw_score -0.0488 [-0.0611, -0.0365] better
  - S − strong (missforest) on impute_score -0.0363 [-0.0567, -0.0159] better
  - LS − strong (missforest) on impute_score -0.0199 [-0.0386, -0.0013] better
- `biodeg_40nan` (seen): mean kept epoch S 309, LS 313
  - LS − S on sw_score -0.0210 [-0.0306, -0.0114] better (S 0.4562, LS 0.4351)
  - LS − S on dist_score -0.0051 [-0.0138, +0.0036] none (S 0.2941, LS 0.2890)
  - LS − S on impute_score -0.0000 [-0.0071, +0.0071] none (S 0.6590, LS 0.6590)
  - S − strong (missforest) on sw_score -0.0529 [-0.0665, -0.0393] better
  - LS − strong (missforest) on sw_score -0.0739 [-0.0867, -0.0611] better
  - S − strong (missforest) on impute_score -0.0358 [-0.0540, -0.0176] better
  - LS − strong (missforest) on impute_score -0.0358 [-0.0497, -0.0219] better
- `biodeg_80nan` (seen): mean kept epoch S 253, LS 241
  - LS − S on sw_score -0.0047 [-0.0110, +0.0017] none (S 0.7144, LS 0.7097)
  - LS − S on dist_score -0.0073 [-0.0151, +0.0004] none (S 0.5786, LS 0.5712)
  - LS − S on impute_score +0.0017 [-0.0017, +0.0051] equivalent (S 0.9384, LS 0.9401)
  - S − strong (softimpute) on sw_score +0.1491 [+0.1331, +0.1651] worse
  - LS − strong (softimpute) on sw_score +0.1444 [+0.1281, +0.1607] worse
  - S − strong (missforest) on impute_score +0.0129 [+0.0049, +0.0209] worse
  - LS − strong (missforest) on impute_score +0.0146 [+0.0068, +0.0224] worse
- `credit-g_20nan` (seen): mean kept epoch S 177, LS 175
  - LS − S on sw_score -0.0141 [-0.0228, -0.0053] better (S 0.6716, LS 0.6575)
  - LS − S on dist_score -0.0218 [-0.0343, -0.0093] better (S 0.4396, LS 0.4178)
  - LS − S on impute_score +0.0214 [+0.0096, +0.0331] worse (S 0.8803, LS 0.9016)
  - S − strong (ice) on sw_score +0.0118 [+0.0034, +0.0202] worse
  - LS − strong (ice) on sw_score -0.0023 [-0.0123, +0.0078] none
  - S − strong (missforest) on impute_score -0.0105 [-0.0308, +0.0099] none
  - LS − strong (missforest) on impute_score +0.0109 [-0.0088, +0.0306] none
- `credit-g_40nan` (seen): mean kept epoch S 260, LS 272
  - LS − S on sw_score -0.0073 [-0.0215, +0.0069] none (S 0.6609, LS 0.6537)
  - LS − S on dist_score -0.0027 [-0.0238, +0.0185] none (S 0.5317, LS 0.5291)
  - LS − S on impute_score +0.0050 [-0.0022, +0.0123] none (S 0.9294, LS 0.9344)
  - S − strong (ice) on sw_score +0.1745 [+0.1575, +0.1915] worse
  - LS − strong (ice) on sw_score +0.1672 [+0.1473, +0.1871] worse
  - S − strong (missforest) on impute_score -0.0019 [-0.0121, +0.0084] none
  - LS − strong (missforest) on impute_score +0.0032 [-0.0080, +0.0143] none
- `credit-g_60nan` (seen): mean kept epoch S 278, LS 296
  - LS − S on sw_score -0.0044 [-0.0225, +0.0138] none (S 0.6573, LS 0.6529)
  - LS − S on dist_score -0.0132 [-0.0373, +0.0108] none (S 0.5646, LS 0.5514)
  - LS − S on impute_score -0.0015 [-0.0075, +0.0045] equivalent (S 0.9930, LS 0.9915)
  - S − strong (ice) on sw_score +0.2071 [+0.1846, +0.2295] worse
  - LS − strong (ice) on sw_score +0.2027 [+0.1874, +0.2180] worse
  - S − strong (condexp) on impute_score +0.0047 [-0.0045, +0.0138] none
  - LS − strong (condexp) on impute_score +0.0032 [-0.0055, +0.0119] none
- `credit-g_80nan` (seen): mean kept epoch S 237, LS 220
  - LS − S on sw_score +0.0147 [+0.0010, +0.0283] worse (S 0.6389, LS 0.6535)
  - LS − S on dist_score +0.0187 [+0.0029, +0.0344] worse (S 0.5987, LS 0.6174)
  - LS − S on impute_score -0.0015 [-0.0074, +0.0044] equivalent (S 1.0458, LS 1.0443)
  - S − strong (ice) on sw_score +0.0633 [+0.0469, +0.0796] worse
  - LS − strong (ice) on sw_score +0.0779 [+0.0614, +0.0944] worse
  - S − strong (condexp) on impute_score +0.0477 [+0.0401, +0.0553] worse
  - LS − strong (condexp) on impute_score +0.0462 [+0.0390, +0.0535] worse
- `kr-vs-kp_20nan` (seen): mean kept epoch S 294, LS 306
  - LS − S on sw_score -0.0035 [-0.0128, +0.0057] none (S 0.2989, LS 0.2954)
  - LS − S on dist_score -0.0058 [-0.0198, +0.0082] none (S 0.1786, LS 0.1728)
  - LS − S on impute_score +0.0030 [-0.0058, +0.0118] none (S 0.5874, LS 0.5904)
  - S − strong (missforest) on sw_score -0.0228 [-0.0325, -0.0131] better
  - LS − strong (missforest) on sw_score -0.0263 [-0.0381, -0.0145] better
  - S − strong (missforest) on impute_score -0.0778 [-0.0885, -0.0671] better
  - LS − strong (missforest) on impute_score -0.0748 [-0.0837, -0.0658] better
- `kr-vs-kp_40nan` (seen): mean kept epoch S 275, LS 263
  - LS − S on sw_score -0.0186 [-0.0412, +0.0039] none (S 0.4336, LS 0.4150)
  - LS − S on dist_score -0.0114 [-0.0329, +0.0101] none (S 0.3820, LS 0.3707)
  - LS − S on impute_score +0.0009 [-0.0046, +0.0064] equivalent (S 0.7255, LS 0.7264)
  - S − strong (missforest) on sw_score +0.0480 [+0.0248, +0.0712] worse
  - LS − strong (missforest) on sw_score +0.0294 [+0.0152, +0.0436] worse
  - S − strong (missforest) on impute_score -0.0780 [-0.0873, -0.0687] better
  - LS − strong (missforest) on impute_score -0.0771 [-0.0880, -0.0662] better
- `kr-vs-kp_60nan` (seen): mean kept epoch S 292, LS 285
  - LS − S on sw_score +0.0246 [-0.0025, +0.0517] none (S 0.4723, LS 0.4969)
  - LS − S on dist_score +0.0321 [+0.0049, +0.0594] worse (S 0.4612, LS 0.4934)
  - LS − S on impute_score -0.0001 [-0.0079, +0.0078] equivalent (S 0.8629, LS 0.8628)
  - S − strong (missforest) on sw_score -0.0891 [-0.1139, -0.0642] better
  - LS − strong (missforest) on sw_score -0.0645 [-0.0883, -0.0406] better
  - S − strong (missforest) on impute_score -0.0767 [-0.0873, -0.0660] better
  - LS − strong (missforest) on impute_score -0.0767 [-0.0882, -0.0653] better
- `kr-vs-kp_80nan` (seen): mean kept epoch S 281, LS 277
  - LS − S on sw_score -0.0088 [-0.0312, +0.0136] none (S 0.5793, LS 0.5705)
  - LS − S on dist_score -0.0138 [-0.0395, +0.0120] none (S 0.6457, LS 0.6320)
  - LS − S on impute_score -0.0017 [-0.0050, +0.0016] equivalent (S 1.0057, LS 1.0040)
  - S − strong (missforest) on sw_score -0.1931 [-0.2124, -0.1739] better
  - LS − strong (missforest) on sw_score -0.2019 [-0.2223, -0.1816] better
  - S − strong (condexp) on impute_score +0.0057 [-0.0022, +0.0135] none
  - LS − strong (condexp) on impute_score +0.0040 [-0.0019, +0.0099] equivalent
- `spambase_20nan` (seen): mean kept epoch S 293, LS 296
  - LS − S on sw_score -0.2283 [-0.2395, -0.2171] better (S 0.7128, LS 0.4845)
  - LS − S on dist_score -0.0540 [-0.0677, -0.0403] better (S 0.5006, LS 0.4467)
  - LS − S on impute_score +0.0378 [+0.0236, +0.0520] worse (S 0.8520, LS 0.8898)
  - S − strong (missforest) on sw_score -0.0010 [-0.0127, +0.0107] none
  - LS − strong (missforest) on sw_score -0.2293 [-0.2425, -0.2160] better
  - S − strong (missforest) on impute_score -0.0062 [-0.0141, +0.0018] none
  - LS − strong (missforest) on impute_score +0.0317 [+0.0192, +0.0441] worse

Primary, LS − S on sw_score over the 18 variants the control passes: better on 5, worse on 2 ['biodeg_20nan', 'credit-g_80nan'] -> mixed
Secondary, LS − S on dist_score, all variants: better 6 / no difference 9 / worse 3 (fold-pair t, seed agreement)
Secondary, LS − S on impute_score, all variants: better 1 / no difference 6 / equivalent 6 / worse 5 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on sw_score: S better 8 / no difference 4 / worse 6 (fold-pair t, seed agreement); LS better 10 / no difference 3 / worse 5 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on impute_score: S better 9 / no difference 6 / worse 3 (fold-pair t, seed agreement); LS better 8 / no difference 5 / equivalent 1 / worse 4 (fold-pair t, seed agreement)
```
