# The decode checkpoint chosen by the sliced Wasserstein score of the validation rows (E183, T60; pre-registered 2026-11-18 13:35 GMT-3)

This ticket was written and committed before the study's first cell. E182 (ticket 71), which measures the same score
on the default run and its baselines, had its 390 files in and had not been read; only its gates had been printed. One smoke was seen (`kc2_20nan`, seed
101, MLflow off): its duration, its GPU memory and the quantities that do not depend on the arm, not its scores.

- **Why:** E182 measures `sw_score`, a distributional score that sees the rows, on fills the default run chose by
  point error. The user asked whether the score should enter training. The first step, as E145 did for `dist_score`
  and E181 for the class-conditional score, is to let it choose the decode checkpoint and train nothing differently.
- **The user's go:** "sim, pre registre!" (2026-11-18, about 13:08 GMT-3), to my offer to register the choice of
  checkpoint by `sw_score` and run it on GPUs. The design is my proposal. The user also asked whether the loss should
  count the total decode error rather than the per-cell one: the loss is not changed here (see "Not in this study").
- **Code (this commit):** `--decode_metric sw_score` (`DECODE_METRIC`), a fourth value; no default changes, and a run
  that does not ask is as it was (both regression fixtures pass).
  - `src/training/decoding.py`: the validation scorer's `sw_score`, for the `induced` and `masked` checkpoints.
  - `src/training/types.py`, `config.py`: the value and its help; a search cannot rank by it (the run does not log it).
  - `scripts/experiments/sliced_select_run.py`: `train`, `--todo`, `--tally`.
  - `tests/unit/test_sliced_checkpoint_score.py`.
- **What the selector computes, every epoch:** the validation rows that hold a scored cell, as the current weights
  fill them and as they truly are; numbers in the scaled space, each categorical column one-hot over the embedder's
  categories plus one slot for an id outside them, divided by √2; 512 fixed unit directions; the mean gap between the
  sorted projections of the two; that distance over the mean/mode fill's. The epoch with the lowest value is kept.
  - It differs from E182's offline score in the directions (512 fixed against 2048 per fold) and in the one-hot's
    vocabulary. The read uses E182's offline score on the test folds for both arms, never the selector's own value.
- **Scenarios:** S, the default run with `--decode_metric sw_score`, on E181's 26 MCAR variants at seeds 101/202/303
  (78 runs), on medusa1.
  - Control: E181's `Kc` run of the same variant and seed, which kept the default's epoch (K and Kc differed by 0.0000
    on six of the seven mixed variants and +0.0012 on `credit-g_20nan`). No new control arm is trained.
  - Pairing, checked in the smoke: on `kc2_20nan` seed 101 the epoch the loss criterion names is the same in all five
    folds as in E181's run, and so are the cell counts and hgb's score. The trajectory is the same; only the kept
    epoch moves (S 212/161/220/293/251 against 50/16/370/321/253).
  - medusa1 only. The arms pair because they train on the same GPU model: CUDA dropout masks differ between GPU
    models, so a cell trained on gorgona8 would not pair with E181's. This reverses my offer of both machines.
  - Calibration stays the default (`DECODE_CALIBRATION gaps`, ADR 0027): from 40% missing the output is the kept
    checkpoint's fills after a calibration fitted for point error. So the study measures the deployable output, and
    the calibration can mute what the selection changes. E145 ran with the calibration off; this does not.
- **Measures and decision rule:** both arms' test-fold ledgers scored by E182's code (`sw_score`, `dist_score`,
  `impute_score`), 15 (seed, fold) values a variant, fold-pair rule with seed agreement, margin 0.01 of the control's
  mean.
  - Hypothesis: choosing the checkpoint by `sw_score` lowers the output's `sw_score`.
  - Primary, over the variants E182's control passes: S − K on `sw_score`. Supported if better on half or more of
    them and worse on at most 2; refuted if better on a sixth or fewer; mixed otherwise.
  - Secondary, no rule: S − K on `impute_score` (the cost) and on `dist_score`; S and K against E182's best
    strong-tier baseline on `sw_score` and `impute_score`, and the calls that move; the mean kept epoch.
  - Read once, with the 78 runs and E182's 390 files in. No default changes whatever the result.
- **Disclosed:** the selector's construction, the variants, seeds and thresholds are my choices. E145's precedent:
  selecting by `dist_score` improved it on 12 of 12 variants and cost `impute_score` on 7.
- **Not in this study:** the loss. Today it is per cell: cross-entropy for a categorical cell and `lambda_num` × squared
  error for a numerical one, averaged over the masked cells (`src/models.py`). For point error the sum over a row's
  cells already is the row's error; what the loss lacks is a term on the joint distribution of a row's fills. Adding
  one changes the training algorithm and needs the user's explicit go.
- **Amendment (2026-11-18 16:10 GMT-3):** at 14:34:29 GMT-3 every process of the study was stopped with 57 of 78
  runs in (all of `kc2`, `vehicle`, `credit-g`, `biodeg`, `kr-vs-kp`, and `pendigits` at 20 and 40). Cause, from the
  user's journal: the user's last SSH session on medusa1 closed at 14:34:01, lingering is off for the account, and
  systemd's user manager exited 28 s later, stopping the tmux scopes the queues ran under (`setsid nohup` does not
  leave that scope). The machine did not restart. Not relaunched, on the user's order ("nao reinicie as filas mortas
  aqui, rodaremos apenas na gorgona por enquanto"). The 21 runs left (`pendigits` 60 and 80, `spambase_20nan`,
  `letter_20nan`, `electricity`) cannot move to gorgona8 as designed: its GPU is another model (RTX 3090 Ti), so they
  would not pair with E181's runs. No change to design, seeds or rule; nothing read.
- **Amendment (2026-11-18 16:15 GMT-3):** relaunched at 16:11 GMT-3 on the user's order, with lingering now on for
  the account: the 21 runs left, as four queues on GPU 0 only (GPU 1 is another user's), shortest first. The 57
  finished runs are kept. No change to design, seeds or rule; nothing read.
- **Amendment (2026-11-18 18:25 GMT-3):** paused on the user's order, because another user needs medusa1 ("consegue
  parar as execucoes na medusa1 assim que a celula atual parar?"). The `STOP` file was created at 17:23 GMT-3, so each
  queue ended after its running cell; on a second order ("interrompa electricity_40nan s101") that cell was killed by
  hand at 17:36, and its `FAILED` line in `progress.log` is that kill, not an error. The last queue left at 18:19.
  69 of 78 runs are in; the 9 left are `electricity_40nan`, `electricity_80nan` and `letter_20nan` at the three
  seeds. To resume: remove `STOP` and relaunch the queues, on the user's order only. No change to design, seeds or
  rule; nothing read.

- **Amendment (2026-11-18 19:53 GMT-3), written in another session before any cell below trained:** the last three variants
  move whole to gorgona8, with their control retrained there. This departs from "medusa1 only" in the Scenarios.
  - **The user's words:** "rode as coisas pendentes do e181 aqu" (2026-11-18, in the session on gorgona8, which leads
    the store since 19:34 GMT-3, epoch 3). E181 has no cell left (99 of 99, read), so I read it as this study's nine
    runs, whose control is E181's `Kc`. E181's one undone item, TabPFN v2 on the class scale, is not run.
  - **What moves:** `letter_20nan`, `electricity_40nan` and `electricity_80nan` at seeds 101/202/303. Both arms are
    trained on gorgona8's RTX 3090 Ti: S (`--decode_metric sw_score`) and its control `Kc` (E181's arm,
    `--decode_metric impute_score_classcond`, which kept the default's epoch on every mixed variant). That makes 18 runs
    for the 9 the ticket left, so each (variant, seed) pair stays on one GPU model, as the Scenarios require. Retraining
    the control is my choice, as in E180. Both are tagged `experiment = sliced-checkpoint-2026-11-18` (arm `S` or
    `Kc`), not E181's name. Runner: `scripts/experiments/sliced_select_g8_run.py`, two queues, one per arm, a pair's
    arms side by side.
  - **State:** the three state directories were copied from medusa1 to the same paths on gorgona8 at 19:50 GMT-3.
    Left out of the copy: `STOP`, every run directory of the three variants (E181's nine `Kc` and `K`; E183's killed
    `electricity_40nan` seed 101), and E182's nine `reference` files for them. Copied: 69 S summaries, 69 `Kc`, 381 of
    E182's 390 files. gorgona8's copy of E181's directory now mixes machines: 23 variants are medusa1's and the three
    above are gorgona8's.
  - **The read changes in one place:** for the three variants, K's `reference` files are re-scored by E182's code
    from gorgona8's `Kc` ledgers (`SLICED_LEDGERS` on gorgona8's copy). K's levels there, and possibly whether
    E182's shuffled control passes (which sets the primary's denominator), can differ from ticket 71's printout. The
    re-scored files govern this study's read, and any control status that flips against ticket 71 is reported in the
    Outcome. The strong-tier baselines stay medusa1's fits (E182's files), which touches only the secondary lines.
    The read runs on gorgona8 with `sliced_select_run.py --tally`, unchanged.
  - **Pairing check, redone here:** the smoke ran on medusa1, so on each finished pair `sliced_select_g8_run.py
    --pairing` compares the loss criterion's epoch per fold, the scored cell counts and hgb's score between the arms.
    No TRIDENT score is read. A pair that differs is reported, not dropped.
  - **Already read before this amendment:** E181's Outcome (ticket 70) and E182's Index headline. No score of this
    study's runs.
  - No change to the rule, seeds, thresholds or what the read prints.

## Outcome (2026-11-19 15:14 GMT-3): supported

The 18 runs of the fifth amendment finished on gorgona8 at 15:01 GMT-3 on 2026-11-19, none failed; with medusa1's 69,
78 of 78 S runs are in. Read once, on gorgona8, after two steps the amendment named:

- **Pairing, redone on gorgona8** (`sliced_select_g8_run.py --pairing`): on all nine pairs of the three moved variants
  the loss criterion's epoch per fold, both cell counts and hgb's score are the same in S and `Kc`.
- **K's reference files for the three variants** were re-scored by E182's code from gorgona8's `Kc` ledgers between
  15:02 and 15:04 GMT-3, none failed.

The printout, unedited below its first two lines (the commit and the clock, added here). Left out above the report:
78 `DONE …__reference.csv` lines, one per S ledger the tally scored, with the data loader's label-mapping and
categorical-column notices.

```
3a5c811
read 2026-11-19 15:04 GMT-3
- `biodeg_80nan` (seen): mean kept epoch K 184 → S 257
  - S − K on sw_score -0.0315 [-0.0512, -0.0119] better (K 0.7465, S 0.7149)
  - S − K on dist_score -0.0279 [-0.0498, -0.0060] better (K 0.6052, S 0.5773)
  - S − K on impute_score -0.0061 [-0.0103, -0.0019] better (K 0.9457, S 0.9397)
  - S − strong (softimpute) on sw_score +0.1496 [+0.1340, +0.1652] worse
  - K − strong (softimpute) on sw_score +0.1811 [+0.1636, +0.1987] worse
  - S − strong (missforest) on impute_score +0.0142 [+0.0075, +0.0208] worse
  - K − strong (missforest) on impute_score +0.0202 [+0.0137, +0.0268] worse
- `vehicle_80nan` (seen): mean kept epoch K 69 → S 269
  - S − K on sw_score -0.0694 [-0.0936, -0.0452] better (K 0.5068, S 0.4374)
  - S − K on dist_score -0.0622 [-0.0842, -0.0402] better (K 0.4019, S 0.3397)
  - S − K on impute_score -0.0058 [-0.0116, +0.0000] none (K 0.7870, S 0.7812)
  - S − strong (missforest) on sw_score -0.0246 [-0.0426, -0.0067] better
  - K − strong (missforest) on sw_score +0.0448 [+0.0069, +0.0827] worse
  - S − strong (missforest) on impute_score +0.0107 [-0.0021, +0.0235] none
  - K − strong (missforest) on impute_score +0.0165 [+0.0034, +0.0296] worse
- `credit-g_40nan` (seen): mean kept epoch K 85 → S 269
  - S − K on sw_score -0.0462 [-0.0609, -0.0314] better (K 0.7059, S 0.6598)
  - S − K on dist_score -0.0604 [-0.0798, -0.0409] better (K 0.5915, S 0.5311)
  - S − K on impute_score +0.0132 [+0.0023, +0.0242] worse (K 0.9211, S 0.9344)
  - S − strong (ice) on sw_score +0.1733 [+0.1577, +0.1889] worse
  - K − strong (ice) on sw_score +0.2195 [+0.2026, +0.2364] worse
  - S − strong (missforest) on impute_score +0.0031 [-0.0094, +0.0156] none
  - K − strong (missforest) on impute_score -0.0101 [-0.0233, +0.0030] none
- `electricity_20nan` (seen): mean kept epoch K 349 → S 323
  - S − K on sw_score -0.0122 [-0.0177, -0.0066] better (K 0.2775, S 0.2654)
  - S − K on dist_score -0.0079 [-0.0138, -0.0019] better (K 0.2524, S 0.2446)
  - S − K on impute_score +0.0004 [-0.0042, +0.0051] equivalent (K 0.6153, S 0.6157)
  - S − strong (missforest) on sw_score -0.0354 [-0.0398, -0.0310] better
  - K − strong (missforest) on sw_score -0.0233 [-0.0308, -0.0157] better
  - S − strong (missforest) on impute_score -0.0035 [-0.0131, +0.0061] none
  - K − strong (missforest) on impute_score -0.0039 [-0.0130, +0.0051] none
- `letter_20nan` (seen): mean kept epoch K 544 → S 513
  - S − K on sw_score -0.0017 [-0.0026, -0.0009] better (K 0.1392, S 0.1375)
  - S − K on dist_score -0.0003 [-0.0008, +0.0001] equivalent (K 0.1635, S 0.1632)
  - S − K on impute_score +0.0022 [+0.0008, +0.0035] worse (K 0.4206, S 0.4228)
  - S − strong (missforest) on sw_score -0.1594 [-0.1636, -0.1553] better
  - K − strong (missforest) on sw_score -0.1577 [-0.1616, -0.1538] better
  - S − strong (missforest) on impute_score -0.0401 [-0.0434, -0.0369] better
  - K − strong (missforest) on impute_score -0.0423 [-0.0457, -0.0389] better
- `spambase_20nan` (seen): mean kept epoch K 278 → S 301
  - S − K on sw_score -0.0254 [-0.0365, -0.0142] better (K 0.7395, S 0.7142)
  - S − K on dist_score -0.0101 [-0.0206, +0.0004] none (K 0.5068, S 0.4967)
  - S − K on impute_score -0.0021 [-0.0083, +0.0041] equivalent (K 0.8491, S 0.8470)
  - S − strong (missforest) on sw_score +0.0004 [-0.0122, +0.0130] none
  - K − strong (missforest) on sw_score +0.0258 [+0.0109, +0.0407] worse
  - S − strong (missforest) on impute_score -0.0111 [-0.0188, -0.0034] better
  - K − strong (missforest) on impute_score -0.0090 [-0.0168, -0.0013] better
- `electricity_40nan` (seen): mean kept epoch K 315 → S 313
  - S − K on sw_score +0.0049 [-0.0061, +0.0159] none (K 0.3689, S 0.3738)
  - S − K on dist_score +0.0071 [-0.0019, +0.0160] none (K 0.3277, S 0.3348)
  - S − K on impute_score -0.0096 [-0.0197, +0.0005] none (K 0.7468, S 0.7371)
  - S − strong (missforest) on sw_score -0.0035 [-0.0130, +0.0060] none
  - K − strong (missforest) on sw_score -0.0084 [-0.0156, -0.0011] better
  - S − strong (missforest) on impute_score -0.0512 [-0.0663, -0.0360] better
  - K − strong (missforest) on impute_score -0.0415 [-0.0600, -0.0230] better
- `electricity_80nan` (seen): mean kept epoch K 155 → S 233
  - S − K on sw_score -0.0061 [-0.0214, +0.0092] none (K 0.6442, S 0.6381)
  - S − K on dist_score -0.0064 [-0.0187, +0.0059] none (K 0.5788, S 0.5724)
  - S − K on impute_score -0.0080 [-0.0204, +0.0044] none (K 0.9119, S 0.9039)
  - S − strong (missforest) on sw_score +0.0286 [-0.0008, +0.0580] none
  - K − strong (missforest) on sw_score +0.0347 [+0.0031, +0.0663] worse
  - S − strong (condexp) on impute_score -0.0536 [-0.0645, -0.0428] better
  - K − strong (condexp) on impute_score -0.0456 [-0.0670, -0.0242] better
- `kr-vs-kp_40nan` (seen): mean kept epoch K 225 → S 257
  - S − K on sw_score -0.0028 [-0.0222, +0.0167] none (K 0.4327, S 0.4299)
  - S − K on dist_score +0.0044 [-0.0226, +0.0314] none (K 0.3810, S 0.3854)
  - S − K on impute_score -0.0019 [-0.0098, +0.0060] none (K 0.7285, S 0.7266)
  - S − strong (missforest) on sw_score +0.0443 [+0.0250, +0.0636] worse
  - K − strong (missforest) on sw_score +0.0471 [+0.0264, +0.0678] worse
  - S − strong (missforest) on impute_score -0.0769 [-0.0861, -0.0677] better
  - K − strong (missforest) on impute_score -0.0750 [-0.0855, -0.0645] better
- `kr-vs-kp_60nan` (seen): mean kept epoch K 87 → S 300
  - S − K on sw_score -0.0679 [-0.1082, -0.0276] better (K 0.5411, S 0.4731)
  - S − K on dist_score -0.0660 [-0.1089, -0.0231] better (K 0.5318, S 0.4659)
  - S − K on impute_score +0.0064 [-0.0004, +0.0133] none (K 0.8592, S 0.8656)
  - S − strong (missforest) on sw_score -0.0883 [-0.1075, -0.0690] better
  - K − strong (missforest) on sw_score -0.0203 [-0.0554, +0.0147] none
  - S − strong (missforest) on impute_score -0.0739 [-0.0839, -0.0640] better
  - K − strong (missforest) on impute_score -0.0804 [-0.0931, -0.0676] better
- `kr-vs-kp_80nan` (seen): mean kept epoch K 19 → S 262
  - S − K on sw_score -0.2255 [-0.2594, -0.1915] better (K 0.8088, S 0.5834)
  - S − K on dist_score -0.1908 [-0.2199, -0.1618] better (K 0.8382, S 0.6474)
  - S − K on impute_score +0.0293 [+0.0196, +0.0391] worse (K 0.9719, S 1.0013)
  - S − strong (missforest) on sw_score -0.1891 [-0.2086, -0.1696] better
  - K − strong (missforest) on sw_score +0.0364 [+0.0035, +0.0693] worse
  - S − strong (condexp) on impute_score +0.0013 [-0.0075, +0.0100] equivalent
  - K − strong (condexp) on impute_score -0.0281 [-0.0362, -0.0200] better
- `credit-g_60nan` (seen): mean kept epoch K 42 → S 274
  - S − K on sw_score -0.1461 [-0.1924, -0.0999] better (K 0.7973, S 0.6512)
  - S − K on dist_score -0.1804 [-0.2417, -0.1192] better (K 0.7391, S 0.5586)
  - S − K on impute_score +0.0132 [+0.0037, +0.0228] worse (K 0.9828, S 0.9960)
  - S − strong (ice) on sw_score +0.2009 [+0.1807, +0.2211] worse
  - K − strong (ice) on sw_score +0.3470 [+0.3022, +0.3919] worse
  - S − strong (condexp) on impute_score +0.0077 [-0.0023, +0.0178] none
  - K − strong (condexp) on impute_score -0.0055 [-0.0143, +0.0033] none
- `credit-g_80nan` (seen): mean kept epoch K 14 → S 281
  - S − K on sw_score -0.2740 [-0.3062, -0.2418] better (K 0.9253, S 0.6513)
  - S − K on dist_score -0.3063 [-0.3418, -0.2707] better (K 0.9176, S 0.6113)
  - S − K on impute_score +0.0424 [+0.0323, +0.0526] worse (K 1.0047, S 1.0471)
  - S − strong (ice) on sw_score +0.0757 [+0.0562, +0.0952] worse
  - K − strong (ice) on sw_score +0.3497 [+0.3184, +0.3809] worse
  - S − strong (condexp) on impute_score +0.0490 [+0.0394, +0.0586] worse
  - K − strong (condexp) on impute_score +0.0066 [+0.0014, +0.0118] none
- `biodeg_40nan` (seen): mean kept epoch K 268 → S 325
  - S − K on sw_score -0.0077 [-0.0142, -0.0011] better (K 0.4569, S 0.4492)
  - S − K on dist_score -0.0062 [-0.0120, -0.0005] better (K 0.2978, S 0.2916)
  - S − K on impute_score -0.0027 [-0.0091, +0.0036] none (K 0.6548, S 0.6521)
  - S − strong (missforest) on sw_score -0.0598 [-0.0722, -0.0474] better
  - K − strong (missforest) on sw_score -0.0522 [-0.0651, -0.0393] better
  - S − strong (missforest) on impute_score -0.0428 [-0.0574, -0.0282] better
  - K − strong (missforest) on impute_score -0.0401 [-0.0574, -0.0228] better
- `kc2_40nan` (seen): mean kept epoch K 222 → S 290
  - S − K on sw_score -0.0143 [-0.0321, +0.0035] none (K 0.3774, S 0.3631)
  - S − K on dist_score -0.0148 [-0.0300, +0.0004] none (K 0.2492, S 0.2344)
  - S − K on impute_score -0.0063 [-0.0220, +0.0095] none (K 0.4836, S 0.4773)
  - S − strong (missforest) on sw_score -0.0117 [-0.0419, +0.0185] none
  - K − strong (missforest) on sw_score +0.0026 [-0.0307, +0.0360] none
  - S − strong (softimpute) on impute_score -0.0722 [-0.1080, -0.0365] better
  - K − strong (softimpute) on impute_score -0.0660 [-0.0987, -0.0333] better
- `kc2_80nan` (seen): mean kept epoch K 223 → S 269
  - S − K on sw_score -0.0150 [-0.0354, +0.0054] none (K 0.4349, S 0.4199)
  - S − K on dist_score -0.0101 [-0.0266, +0.0065] none (K 0.3252, S 0.3151)
  - S − K on impute_score -0.0075 [-0.0188, +0.0037] none (K 0.6732, S 0.6657)
  - S − strong (missforest) on sw_score -0.0498 [-0.0726, -0.0270] better
  - K − strong (missforest) on sw_score -0.0348 [-0.0686, -0.0010] better
  - S − strong (missforest) on impute_score -0.0715 [-0.0932, -0.0497] better
  - K − strong (missforest) on impute_score -0.0639 [-0.0916, -0.0363] better
- `pendigits_40nan` (seen): mean kept epoch K 517 → S 521
  - S − K on sw_score -0.0001 [-0.0012, +0.0009] equivalent (K 0.1435, S 0.1433)
  - S − K on dist_score -0.0002 [-0.0011, +0.0006] none (K 0.0938, S 0.0936)
  - S − K on impute_score -0.0000 [-0.0007, +0.0006] equivalent (K 0.3815, S 0.3815)
  - S − strong (missforest) on sw_score -0.0962 [-0.1016, -0.0908] better
  - K − strong (missforest) on sw_score -0.0961 [-0.1016, -0.0905] better
  - S − strong (missforest) on impute_score -0.1162 [-0.1212, -0.1112] better
  - K − strong (missforest) on impute_score -0.1162 [-0.1214, -0.1109] better
- `pendigits_60nan` (seen): mean kept epoch K 459 → S 506
  - S − K on sw_score -0.0013 [-0.0032, +0.0007] none (K 0.2642, S 0.2629)
  - S − K on dist_score -0.0008 [-0.0022, +0.0005] none (K 0.1807, S 0.1799)
  - S − K on impute_score -0.0017 [-0.0037, +0.0004] equivalent (K 0.5567, S 0.5550)
  - S − strong (softimpute) on sw_score -0.0544 [-0.0613, -0.0476] better
  - K − strong (softimpute) on sw_score -0.0531 [-0.0606, -0.0457] better
  - S − strong (missforest) on impute_score -0.1532 [-0.1613, -0.1451] better
  - K − strong (missforest) on impute_score -0.1515 [-0.1597, -0.1434] better
- `pendigits_80nan` (seen): mean kept epoch K 61 → S 370
  - S − K on sw_score -0.0106 [-0.0164, -0.0048] better (K 0.6057, S 0.5951)
  - S − K on dist_score -0.0094 [-0.0148, -0.0041] better (K 0.4805, S 0.4711)
  - S − K on impute_score +0.0058 [+0.0046, +0.0070] worse (K 0.8258, S 0.8316)
  - S − strong (missforest) on sw_score +0.0861 [+0.0697, +0.1024] worse
  - K − strong (missforest) on sw_score +0.0967 [+0.0800, +0.1134] worse
  - S − strong (condexp) on impute_score -0.0162 [-0.0179, -0.0144] better
  - K − strong (condexp) on impute_score -0.0220 [-0.0238, -0.0202] better
- `vehicle_40nan` (seen): mean kept epoch K 268 → S 304
  - S − K on sw_score -0.0055 [-0.0121, +0.0010] none (K 0.3494, S 0.3439)
  - S − K on dist_score -0.0043 [-0.0089, +0.0003] none (K 0.2183, S 0.2140)
  - S − K on impute_score -0.0024 [-0.0074, +0.0025] none (K 0.4875, S 0.4850)
  - S − strong (softimpute) on sw_score -0.0085 [-0.0241, +0.0071] none
  - K − strong (softimpute) on sw_score -0.0030 [-0.0212, +0.0152] none
  - S − strong (missforest) on impute_score -0.0341 [-0.0576, -0.0107] better
  - K − strong (missforest) on impute_score -0.0317 [-0.0544, -0.0090] better
- `pendigits_20nan` (seen): mean kept epoch K 517 → S 501
  - S − K on sw_score -0.0006 [-0.0020, +0.0008] none (K 0.1287, S 0.1281)
  - S − K on dist_score -0.0000 [-0.0008, +0.0008] none (K 0.0762, S 0.0762)
  - S − K on impute_score +0.0003 [-0.0013, +0.0018] equivalent (K 0.3017, S 0.3020)
  - S − strong (missforest) on sw_score -0.0557 [-0.0602, -0.0511] better
  - K − strong (missforest) on sw_score -0.0551 [-0.0591, -0.0510] better
  - S − strong (missforest) on impute_score -0.0321 [-0.0377, -0.0265] better
  - K − strong (missforest) on impute_score -0.0323 [-0.0381, -0.0266] better
- `kr-vs-kp_20nan` (seen): mean kept epoch K 232 → S 304
  - S − K on sw_score -0.0287 [-0.0460, -0.0114] better (K 0.3291, S 0.3003)
  - S − K on dist_score -0.0341 [-0.0510, -0.0173] better (K 0.2139, S 0.1797)
  - S − K on impute_score -0.0022 [-0.0136, +0.0092] none (K 0.5943, S 0.5921)
  - S − strong (missforest) on sw_score -0.0214 [-0.0384, -0.0044] better
  - K − strong (missforest) on sw_score +0.0073 [-0.0144, +0.0290] none
  - S − strong (missforest) on impute_score -0.0731 [-0.0891, -0.0571] better
  - K − strong (missforest) on impute_score -0.0709 [-0.0833, -0.0585] better
- `biodeg_20nan` (seen): mean kept epoch K 278 → S 326
  - S − K on sw_score -0.0089 [-0.0164, -0.0013] better (K 0.4887, S 0.4798)
  - S − K on dist_score -0.0100 [-0.0183, -0.0017] better (K 0.2757, S 0.2657)
  - S − K on impute_score -0.0039 [-0.0102, +0.0025] none (K 0.5916, S 0.5877)
  - S − strong (missforest) on sw_score -0.0601 [-0.0693, -0.0509] better
  - K − strong (missforest) on sw_score -0.0513 [-0.0618, -0.0408] better
  - S − strong (missforest) on impute_score -0.0392 [-0.0562, -0.0223] better
  - K − strong (missforest) on impute_score -0.0353 [-0.0545, -0.0162] better
- `credit-g_20nan` (seen): mean kept epoch K 103 → S 203
  - S − K on sw_score -0.0605 [-0.0881, -0.0330] better (K 0.7294, S 0.6688)
  - S − K on dist_score -0.0920 [-0.1393, -0.0447] better (K 0.5295, S 0.4375)
  - S − K on impute_score -0.0087 [-0.0224, +0.0050] none (K 0.8897, S 0.8810)
  - S − strong (ice) on sw_score +0.0090 [+0.0003, +0.0177] worse
  - K − strong (ice) on sw_score +0.0696 [+0.0432, +0.0960] worse
  - S − strong (missforest) on impute_score -0.0097 [-0.0244, +0.0049] none
  - K − strong (missforest) on impute_score -0.0010 [-0.0152, +0.0131] none
- `vehicle_20nan` (seen): mean kept epoch K 291 → S 303
  - S − K on sw_score -0.0014 [-0.0055, +0.0026] none (K 0.3623, S 0.3608)
  - S − K on dist_score -0.0020 [-0.0065, +0.0026] none (K 0.1998, S 0.1979)
  - S − K on impute_score -0.0014 [-0.0052, +0.0024] none (K 0.4146, S 0.4132)
  - S − strong (ice) on sw_score -0.0107 [-0.0201, -0.0013] better
  - K − strong (ice) on sw_score -0.0092 [-0.0181, -0.0004] better
  - S − strong (missforest) on impute_score -0.0329 [-0.0606, -0.0052] better
  - K − strong (missforest) on impute_score -0.0314 [-0.0580, -0.0049] better
- `kc2_20nan` (seen): mean kept epoch K 230 → S 261
  - S − K on sw_score -0.0213 [-0.0491, +0.0066] none (K 0.4209, S 0.3996)
  - S − K on dist_score -0.0263 [-0.0625, +0.0099] none (K 0.2849, S 0.2586)
  - S − K on impute_score -0.0101 [-0.0350, +0.0148] none (K 0.5438, S 0.5336)
  - S − strong (missforest) on sw_score +0.0025 [-0.0487, +0.0538] none
  - K − strong (missforest) on sw_score +0.0238 [-0.0423, +0.0899] none
  - S − strong (softimpute) on impute_score -0.0040 [-0.1051, +0.0971] none
  - K − strong (softimpute) on impute_score +0.0061 [-0.1151, +0.1272] none

Primary, S − K on sw_score over the 26 variants E182's control passes: better on 15, worse on 0 [] -> supported
Secondary, S − K on sw_score, all variants: better 15 / no difference 10 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
Secondary, S − K on dist_score, all variants: better 13 / no difference 12 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
Secondary, S − K on impute_score, all variants: better 1 / no difference 14 / equivalent 5 / worse 6 (fold-pair t, seed agreement)
Secondary, against the best strong baseline on sw_score: S better 13 / no difference 6 / worse 7 (fold-pair t, seed agreement); K better 10 / no difference 5 / worse 11 (fold-pair t, seed agreement); calls that move: ['vehicle_80nan', 'spambase_20nan', 'electricity_40nan', 'electricity_80nan', 'kr-vs-kp_60nan', 'kr-vs-kp_80nan', 'kr-vs-kp_20nan']
Secondary, against the best strong baseline on impute_score: S better 17 / no difference 6 / equivalent 1 / worse 2 (fold-pair t, seed agreement); K better 18 / no difference 6 / worse 2 (fold-pair t, seed agreement); calls that move: ['vehicle_80nan', 'kr-vs-kp_80nan', 'credit-g_80nan']
exit 0
```

- **Primary, supported:** E182's shuffled control passes on all 26 variants, as in ticket 71 (no status flips on the
  three moved variants). Choosing the checkpoint by `sw_score` lowers the output's `sw_score` on 15 of 26 and raises
  it on none (13 were needed, with at most 2 worse). Equivalent on `pendigits_40nan`; no difference on the other 10.
  The largest gains are where the default keeps an early epoch: `credit-g_80nan` −0.2740 (kept epoch 14 → 281),
  `kr-vs-kp_80nan` −0.2255 (19 → 262), `credit-g_60nan` −0.1461 (42 → 274).
- **The cost in point error:** on `impute_score` S is worse on 6 (`credit-g_40nan` +0.0132, `_60nan` +0.0132,
  `_80nan` +0.0424, `kr-vs-kp_80nan` +0.0293, `pendigits_80nan` +0.0058, `letter_20nan` +0.0022), better on 1
  (`biodeg_80nan` −0.0061), equivalent on 5, no difference on 14. `dist_score` moves with `sw_score`: better on 13,
  worse on none.
- **Against the best strong-tier baseline on `sw_score`:** S better 13 / no difference 6 / worse 7, K 10 / 5 / 11.
  Seven calls move: six toward S (`vehicle_80nan`, `kr-vs-kp_80nan` worse to better; `kr-vs-kp_20nan`, `_60nan` no
  difference to better; `spambase_20nan`, `electricity_80nan` worse to no difference) and one away (`electricity_40nan`,
  better to no difference). On `impute_score`: S 17 / 6 / 2 with one equivalent, K 18 / 6 / 2; the calls on
  `vehicle_80nan`, `kr-vs-kp_80nan` and `credit-g_80nan` move. `credit-g_80nan` becomes worse than condexp
  (+0.0490, K no difference); on `kr-vs-kp_80nan` the lead over condexp goes to equivalent.
- **The moved variants:** with gorgona8's `Kc`, K's call against the strong tier on `sw_score` is "better" on
  `electricity_40nan` (−0.0084) where ticket 71 had "no difference"; on `letter_20nan` (−0.1577) and
  `electricity_80nan` (+0.0347) the calls are ticket 71's. That one call is why K reads 10 / 5 / 11 here and 9 / 6 / 11
  there.
- **What it means:** the default's checkpoint is a point-error choice, and at 60 to 80% missing it stops at an early
  epoch whose fills are close to the centre. Choosing by `sw_score` keeps a later epoch with better row distributions,
  for little point error at 20 to 40% and up to +0.04 at 80% on `credit-g`. The selector is a reporting option, not a
  default. No default changes (fixed beforehand). Whether to offer it, or to change the loss (outside this study, and
  needing the user's go), is the user's call.

