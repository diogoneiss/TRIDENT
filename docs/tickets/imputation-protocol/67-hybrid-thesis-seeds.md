# ADR 0031 decision 6's hybrid at the thesis seeds, on the 26 MCAR variants (E174, T54; pre-registered 2026-11-10 16:40 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** the thesis table's primary seed set is 101/202/303 (T21, ADR 0020). The deployable TabPFN-side hybrid
  (`table_tabpfn`, ADR 0031 decision 6) was accepted on 4545/5656/6767 (E90, E91, E95) and read again on 42/7/13 (E167,
  E169), never on the thesis seeds for MCAR variants. This study gives the thesis table a hybrid column on the same
  seeds as its other columns, and a third seed set for decision 6 on all 26 MCAR variants of E164's 38.
- **The user's go:** "Os dois em sequência" (2026-11-10, about 16:30 GMT-3), with E173 (the hybrid on the MIMIC-IV
  natural-gap tables). E174 runs first because it needs no code change; E173's runner needs the masked population.
- **Disclosure:** on 42/7/13 (E167, E169 printouts, already read) `table_tabpfn` is worse than H on none of these 26 and
  worse than TabPFN on none, better than H on 23. On 4545/5656/6767 decision 6 was accepted on a subset of them. Seeds
  101/202/303 have H runs of these variants from the thesis table's studies, but no hybrid cells (the hidden observed
  validation cells are written only by the hybrid's runner), so all runs are new.

## Design

- **Variants (26), `hybrid_fresh_seeds.MCAR`:** E164's 38 less E93's 12 off-MCAR ones: `biodeg_80nan`, `vehicle_80nan`,
  `credit-g_40nan`, `electricity_20nan`, `letter_20nan`, `spambase_20nan`, `electricity_40nan`, `electricity_80nan`,
  `kr-vs-kp_20/40/60/80nan`, `credit-g_20/60/80nan`, `biodeg_20/40nan`, `kc2_20/40/80nan`, `pendigits_20/40/60/80nan`,
  `vehicle_20/40nan`.
- **Cells:** 26 × 101/202/303 = 78 runs of H under `experiment = hybrid-thesis-seeds-2026-11-10`, state
  `/scratch/diogoneiss/trident-2026-11-10-hybrid-thesis-seeds` on gorgona8, runner
  `scripts/experiments/hybrid_fresh_seeds.py --study e174`. Offline per pair as E167: TabPFN refit (H's ledger asserted),
  then missForest refit (logged score asserted), into `cells-mf`.
- **Machine (ADR 0035):** everything on gorgona8 (medusa1's GPUs are down since 2026-11-09 22:40 GMT-3). Its trainings log
  to medusa1's store through the tunnel (ADR 0033); three GPU queues, longest first; two offline lanes, a pair 15 minutes
  after its run is done (the store copy's lag).
- **Cost:** about 11–12 h of training by the runner's per-table minutes, the offline pass overlapping it (`spambase_20nan`
  and `letter_20nan` dominate TabPFN); about 14–20 h in all, ending on 2026-11-11.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean.
- **Primary, `table_tabpfn` (decision 6's form) on the 26:**
  - **supported** if worse than H on none and worse than TabPFN on at most 1;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **Secondary:** better than H on how many, better than TabPFN on how many; `table_missforest` worse than H on which;
  `crossfit` worse than `table_tabpfn` on which (descriptive after E169).
- Three seeds, fixed; no extension.
- **Read** once with all 78 in, on gorgona8: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --study e174 --tally`.
- **What it leads to:** the hybrid column of the thesis table on its primary seeds. No ADR change is proposed by this
  study; a refutation would go to the user as a question on decision 6.

## Amendment, 2026-11-11 02:11 GMT-3: paused while the leader is down (nothing read)

- medusa1, which holds the store, stopped answering at about 00:42 GMT-3 (`ssh: No route to host`; gorgona8's
  `leadership.py tick` logged `peer_unreachable` at 01:12). The three `pendigits_20nan` runs (started 00:38) failed at
  02:04 on `set_terminated` ("Connection refused" on port 47731), the three `pendigits_40nan` runs at 02:10. Only
  `FINISHED` runs are ever read, so their orphans on the leader cannot enter a cell.
- The agent stopped the queues with `$S/STOP` at 02:07. Done: 12 of 78 runs and their 12 offline pairs
  (`letter_20nan`, `electricity_20nan`, `electricity_40nan`, `electricity_80nan`).
- `$S/resume.sh` (detached) polls the tunnel's `/health` every five minutes; after two answers in a row it removes STOP
  and relaunches `queue.sh 1–3` (cells with a selection file are skipped, so the six failed ones rerun) and two lanes of
  `$S/cells_resume.sh`, which is `cells.sh` reading only the training lines after the relaunch marker, so the old FAILED
  lines do not skip their reruns.
- Nothing else changes: machine, variants, seeds, rule and the single read. E173's queues still wait for E174's three
  "queue complete" lines, which the relaunched queues write. Its two offline lanes were released at 02:09 by the stopped
  lanes' "complete" lines; they only poll until E173's own runs exist, then may overlap E174's tail on the GPU.

## Amendment, 2026-11-11 23:25 GMT-3: relaunched at 14:06, leader down again (nothing read)

- The leader answered again at 13:57 GMT-3 and `resume.sh` relaunched at 14:06. By 22:23 the queues were complete with
  no failure; the store copy of 22:58 holds every run they finished.
- Correction to the amendment above: only the three `pendigits_40nan` runs reran. The three `pendigits_20nan` runs had
  written all five selection files before failing (the leader went silent during their folds and only their closing
  MLflow call raised), so the queues skipped them; their runs never reached `FINISHED` and cannot be read. Their
  selections were moved to `$S/selection.leader-down-2026-11-11/` at 23:25 so the queues rerun them.
- medusa1 went silent again at about 23:00. `$S/resume2.sh` (detached) waits for two answers five minutes apart, then
  relaunches `queue.sh 1–3` (only `pendigits_20nan` × 3 left) and E173's runners. The `cells_resume.sh` lanes keep
  running meanwhile on the store copy and take the reruns. 75 of 78 runs are done; nothing else changes.

## Amendment, 2026-11-12 16:46 GMT-3: `pendigits_20nan` rerun, its offline cells taken twice (nothing read)

- medusa1 answered again at about 15:10 GMT-3 on 2026-11-12 and `resume2.sh` relaunched at 15:20. The three
  `pendigits_20nan` runs trained from 15:21 to 16:26 and are `FINISHED` on the leader: 78 of 78.
- Their offline cells failed at 16:41–16:42 before computing anything (`FileNotFoundError: pendigits_20nan s101 fold 1:
  2 ledgers`): the lost runs of 2026-11-11 had left their result directories (`20261111_003846`) beside the reruns', and
  the ledger reader wants exactly one. Those directories were moved to `$S/results.leader-down-2026-11-11/`; the reruns'
  ledgers are untouched. `$S/cells_pendigits20.sh` took the three cells again from 16:43, same command and asserts
  (H's ledger against the `FINISHED` run, missForest's logged score).
- Variants, seeds, rule and the single read unchanged.

## Outcome (read once 2026-11-12 18:05 GMT-3: supported; written 2026-11-12 18:09 GMT-3)

Read on gorgona8 at commit 5a09a53, after all 78 pairs' cells were in. Every FAILED line of the state's progress log
belongs to the two outages and the ledger failure recorded in the amendments above; each of those cells was taken again,
and the H-ledger and missForest asserts held on every pair. The store copy holds 78 `FINISHED` parents under the tag and
three `RUNNING` orphans, the `pendigits_20nan` runs lost on 2026-11-11, which no reader takes.

- **Primary, supported.** `table_tabpfn` (ADR 0031 decision 6's deployable hybrid, TabPFN side) is worse than H on none
  of the 26 and worse than TabPFN on one, `credit-g_80nan` (+0.0068 [+0.0016, +0.0119]); the rule allows one.
- **Secondary.** Better than H on 19 of 26 (no call on `electricity_20nan`, `electricity_40nan`, `kc2_20nan`,
  `kc2_40nan`, `kc2_80nan`, `credit-g_20nan`; equivalent on `spambase_20nan`). Better than TabPFN on 16, equivalent on
  6, no call on 3. `table_missforest` is worse than H on none (better on 13). `crossfit` and `pick` are worse than
  `table_tabpfn` on `pendigits_20nan` only (+0.0006 each), better on 3.
- **Against the disclosure:** on 42/7/13 (E167, E169) `table_tabpfn` was better than H on 23 of these 26 and worse than
  TabPFN on none; here 19 and one. No variant turns worse than H on any of the three seed sets read.
- **What it leads to:** the thesis table has a hybrid column on its primary seeds for the 26 MCAR variants. No ADR change
  is proposed.

Printout, unedited:

```
- `biodeg_80nan` (trails): H 0.9886, tabpfn 0.9189, missforest 0.9255, table_tabpfn 0.9168, table_missforest 0.9122, pick 0.9151, picked_missforest 0.2667, crossfit 0.9141, crossfit_missforest 0.3333
  - table_missforest − H -0.0765 [-0.0876, -0.0654] better
  - table_missforest − missforest -0.0133 [-0.0181, -0.0085] better
  - table_tabpfn − H -0.0719 [-0.0865, -0.0572] better
  - table_tabpfn − tabpfn -0.0021 [-0.0050, +0.0008] equivalent
  - pick − H -0.0736 [-0.0855, -0.0616] better
  - pick − table_tabpfn -0.0017 [-0.0073, +0.0039] equivalent
  - pick − table_missforest +0.0029 [-0.0013, +0.0070] equivalent
  - crossfit − H -0.0745 [-0.0862, -0.0628] better
  - crossfit − table_tabpfn -0.0027 [-0.0085, +0.0032] equivalent
  - crossfit − pick -0.0009 [-0.0030, +0.0011] equivalent
- `vehicle_80nan` (trails): H 0.8154, tabpfn 0.7547, missforest 0.7705, table_tabpfn 0.7539, table_missforest 0.7513, pick 0.7522, picked_missforest 0.3333, crossfit 0.7517, crossfit_missforest 0.2000
  - table_missforest − H -0.0641 [-0.0763, -0.0520] better
  - table_missforest − missforest -0.0192 [-0.0279, -0.0105] better
  - table_tabpfn − H -0.0615 [-0.0706, -0.0525] better
  - table_tabpfn − tabpfn -0.0009 [-0.0031, +0.0014] equivalent
  - pick − H -0.0633 [-0.0741, -0.0524] better
  - pick − table_tabpfn -0.0017 [-0.0050, +0.0016] equivalent
  - pick − table_missforest +0.0009 [-0.0065, +0.0082] none
  - crossfit − H -0.0637 [-0.0742, -0.0532] better
  - crossfit − table_tabpfn -0.0022 [-0.0051, +0.0008] equivalent
  - crossfit − pick -0.0004 [-0.0015, +0.0006] equivalent
- `credit-g_40nan` (trails): H 0.9267, tabpfn 0.9039, missforest 0.9313, table_tabpfn 0.9045, table_missforest 0.9181, pick 0.9075, picked_missforest 0.1333, crossfit 0.9086, crossfit_missforest 0.4667
  - table_missforest − H -0.0085 [-0.0148, -0.0022] better
  - table_missforest − missforest -0.0131 [-0.0236, -0.0026] better
  - table_tabpfn − H -0.0222 [-0.0307, -0.0136] better
  - table_tabpfn − tabpfn +0.0006 [-0.0040, +0.0052] equivalent
  - pick − H -0.0191 [-0.0287, -0.0095] better
  - pick − table_tabpfn +0.0030 [-0.0019, +0.0080] equivalent
  - pick − table_missforest -0.0106 [-0.0163, -0.0049] better
  - crossfit − H -0.0181 [-0.0280, -0.0081] better
  - crossfit − table_tabpfn +0.0041 [-0.0009, +0.0091] none
  - crossfit − pick +0.0011 [-0.0004, +0.0026] equivalent
- `electricity_20nan` (trails): H 0.6154, tabpfn 0.6300, missforest 0.6192, table_tabpfn 0.6102, table_missforest 0.5908, pick 0.5969, picked_missforest 0.8000, crossfit 0.5919, crossfit_missforest 0.9333
  - table_missforest − H -0.0246 [-0.0275, -0.0217] better
  - table_missforest − missforest -0.0284 [-0.0358, -0.0211] better
  - table_tabpfn − H -0.0052 [-0.0114, +0.0009] none
  - table_tabpfn − tabpfn -0.0197 [-0.0272, -0.0123] better
  - pick − H -0.0185 [-0.0276, -0.0094] better
  - pick − table_tabpfn -0.0133 [-0.0179, -0.0086] better
  - pick − table_missforest +0.0061 [-0.0017, +0.0140] none
  - crossfit − H -0.0235 [-0.0279, -0.0192] better
  - crossfit − table_tabpfn -0.0183 [-0.0244, -0.0122] better
  - crossfit − pick -0.0050 [-0.0128, +0.0027] none
- `letter_20nan`: H 0.4213, tabpfn 0.4541, missforest 0.4629, table_tabpfn 0.4073, table_missforest 0.4042, pick 0.4042, picked_missforest 1.0000, crossfit 0.4042, crossfit_missforest 1.0000
  - table_missforest − H -0.0171 [-0.0178, -0.0163] better
  - table_missforest − missforest -0.0587 [-0.0617, -0.0557] better
  - table_tabpfn − H -0.0139 [-0.0146, -0.0133] better
  - table_tabpfn − tabpfn -0.0467 [-0.0496, -0.0438] better
  - pick − H -0.0171 [-0.0178, -0.0163] better
  - pick − table_tabpfn -0.0031 [-0.0037, -0.0025] better
  - pick − table_missforest +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − H -0.0171 [-0.0178, -0.0163] better
  - crossfit − table_tabpfn -0.0031 [-0.0037, -0.0025] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `spambase_20nan`: H 0.8510, tabpfn 0.9372, missforest 0.8582, table_tabpfn 0.8497, table_missforest 0.8395, pick 0.8404, picked_missforest 0.8667, crossfit 0.8420, crossfit_missforest 0.7333
  - table_missforest − H -0.0116 [-0.0129, -0.0103] better
  - table_missforest − missforest -0.0187 [-0.0237, -0.0137] better
  - table_tabpfn − H -0.0013 [-0.0036, +0.0010] equivalent
  - table_tabpfn − tabpfn -0.0874 [-0.1002, -0.0747] better
  - pick − H -0.0107 [-0.0131, -0.0082] better
  - pick − table_tabpfn -0.0094 [-0.0122, -0.0065] better
  - pick − table_missforest +0.0009 [-0.0005, +0.0024] equivalent
  - crossfit − H -0.0091 [-0.0122, -0.0059] better
  - crossfit − table_tabpfn -0.0078 [-0.0110, -0.0045] better
  - crossfit − pick +0.0016 [-0.0008, +0.0040] equivalent
- `electricity_40nan`: H 0.7435, tabpfn 0.7447, missforest 0.7883, table_tabpfn 0.7338, table_missforest 0.7414, pick 0.7331, picked_missforest 0.2000, crossfit 0.7332, crossfit_missforest 0.2667
  - table_missforest − H -0.0021 [-0.0090, +0.0049] none
  - table_missforest − missforest -0.0469 [-0.0648, -0.0289] better
  - table_tabpfn − H -0.0098 [-0.0227, +0.0031] none
  - table_tabpfn − tabpfn -0.0110 [-0.0173, -0.0047] better
  - pick − H -0.0104 [-0.0232, +0.0023] none
  - pick − table_tabpfn -0.0006 [-0.0016, +0.0003] equivalent
  - pick − table_missforest -0.0083 [-0.0214, +0.0048] none
  - crossfit − H -0.0103 [-0.0231, +0.0024] none
  - crossfit − table_tabpfn -0.0006 [-0.0015, +0.0004] equivalent
  - crossfit − pick +0.0001 [-0.0001, +0.0003] equivalent
- `electricity_80nan`: H 0.9036, tabpfn 0.8892, missforest 1.0201, table_tabpfn 0.8882, table_missforest 0.9014, pick 0.8882, picked_missforest 0.0000, crossfit 0.8882, crossfit_missforest 0.0000
  - table_missforest − H -0.0022 [-0.0066, +0.0023] equivalent
  - table_missforest − missforest -0.1187 [-0.1363, -0.1010] better
  - table_tabpfn − H -0.0154 [-0.0279, -0.0029] better
  - table_tabpfn − tabpfn -0.0010 [-0.0016, -0.0003] better
  - pick − H -0.0154 [-0.0279, -0.0029] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0132 [-0.0213, -0.0052] better
  - crossfit − H -0.0154 [-0.0279, -0.0029] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_40nan`: H 0.7262, tabpfn 0.7156, missforest 0.8035, table_tabpfn 0.7182, table_missforest 0.7262, pick 0.7182, picked_missforest 0.0000, crossfit 0.7204, crossfit_missforest 0.5333
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0773 [-0.0864, -0.0681] better
  - table_tabpfn − H -0.0080 [-0.0145, -0.0015] better
  - table_tabpfn − tabpfn +0.0026 [-0.0026, +0.0078] none
  - pick − H -0.0080 [-0.0145, -0.0015] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0080 [-0.0145, -0.0015] better
  - crossfit − H -0.0058 [-0.0121, +0.0005] none
  - crossfit − table_tabpfn +0.0022 [-0.0011, +0.0055] equivalent
  - crossfit − pick +0.0022 [-0.0011, +0.0055] equivalent
- `kr-vs-kp_60nan`: H 0.8716, tabpfn 0.8279, missforest 0.9396, table_tabpfn 0.8279, table_missforest 0.8716, pick 0.8279, picked_missforest 0.0000, crossfit 0.8279, crossfit_missforest 0.0000
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0680 [-0.0819, -0.0541] better
  - table_tabpfn − H -0.0437 [-0.0513, -0.0362] better
  - table_tabpfn − tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − H -0.0437 [-0.0513, -0.0362] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0437 [-0.0513, -0.0362] better
  - crossfit − H -0.0437 [-0.0513, -0.0362] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_80nan`: H 1.0104, tabpfn 0.9412, missforest 1.0113, table_tabpfn 0.9412, table_missforest 0.9989, pick 0.9412, picked_missforest 0.0000, crossfit 0.9412, crossfit_missforest 0.0000
  - table_missforest − H -0.0115 [-0.0260, +0.0030] none
  - table_missforest − missforest -0.0124 [-0.0199, -0.0048] better
  - table_tabpfn − H -0.0692 [-0.0865, -0.0520] better
  - table_tabpfn − tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − H -0.0692 [-0.0865, -0.0520] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0578 [-0.0644, -0.0511] better
  - crossfit − H -0.0692 [-0.0865, -0.0520] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `credit-g_60nan`: H 1.0004, tabpfn 0.9510, missforest 1.0099, table_tabpfn 0.9525, table_missforest 0.9881, pick 0.9525, picked_missforest 0.0000, crossfit 0.9525, crossfit_missforest 0.0667
  - table_missforest − H -0.0123 [-0.0199, -0.0047] better
  - table_missforest − missforest -0.0218 [-0.0321, -0.0115] better
  - table_tabpfn − H -0.0479 [-0.0609, -0.0349] better
  - table_tabpfn − tabpfn +0.0015 [-0.0043, +0.0073] equivalent
  - pick − H -0.0479 [-0.0609, -0.0349] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0356 [-0.0447, -0.0265] better
  - crossfit − H -0.0479 [-0.0609, -0.0349] better
  - crossfit − table_tabpfn -0.0000 [-0.0000, +0.0000] equivalent
  - crossfit − pick -0.0000 [-0.0000, +0.0000] equivalent
- `credit-g_80nan`: H 1.0329, tabpfn 0.9964, missforest 1.0979, table_tabpfn 1.0032, table_missforest 1.0390, pick 1.0129, picked_missforest 0.1333, crossfit 1.0120, crossfit_missforest 0.4000
  - table_missforest − H +0.0061 [-0.0081, +0.0203] none
  - table_missforest − missforest -0.0589 [-0.0752, -0.0425] better
  - table_tabpfn − H -0.0297 [-0.0464, -0.0131] better
  - table_tabpfn − tabpfn +0.0068 [+0.0016, +0.0119] worse
  - pick − H -0.0200 [-0.0392, -0.0008] better
  - pick − table_tabpfn +0.0097 [-0.0045, +0.0240] none
  - pick − table_missforest -0.0261 [-0.0417, -0.0106] better
  - crossfit − H -0.0209 [-0.0373, -0.0045] better
  - crossfit − table_tabpfn +0.0088 [-0.0031, +0.0207] none
  - crossfit − pick -0.0009 [-0.0124, +0.0106] none
- `biodeg_40nan`: H 0.6670, tabpfn 0.7142, missforest 0.6948, table_tabpfn 0.6539, table_missforest 0.6502, pick 0.6516, picked_missforest 0.8667, crossfit 0.6516, crossfit_missforest 0.8667
  - table_missforest − H -0.0168 [-0.0271, -0.0066] better
  - table_missforest − missforest -0.0447 [-0.0589, -0.0305] better
  - table_tabpfn − H -0.0131 [-0.0216, -0.0045] better
  - table_tabpfn − tabpfn -0.0603 [-0.0803, -0.0402] better
  - pick − H -0.0154 [-0.0252, -0.0056] better
  - pick − table_tabpfn -0.0023 [-0.0085, +0.0039] none
  - pick − table_missforest +0.0015 [-0.0009, +0.0038] equivalent
  - crossfit − H -0.0154 [-0.0252, -0.0056] better
  - crossfit − table_tabpfn -0.0023 [-0.0085, +0.0039] none
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kc2_40nan`: H 0.4979, tabpfn 0.4879, missforest 0.5811, table_tabpfn 0.4418, table_missforest 0.4863, pick 0.4506, picked_missforest 0.0667, crossfit 0.4618, crossfit_missforest 0.4000
  - table_missforest − H -0.0116 [-0.0549, +0.0317] none
  - table_missforest − missforest -0.0947 [-0.1366, -0.0529] better
  - table_tabpfn − H -0.0561 [-0.1157, +0.0036] none
  - table_tabpfn − tabpfn -0.0461 [-0.0833, -0.0089] better
  - pick − H -0.0473 [-0.1124, +0.0179] none
  - pick − table_tabpfn +0.0088 [-0.0101, +0.0277] none
  - pick − table_missforest -0.0357 [-0.0645, -0.0068] better
  - crossfit − H -0.0361 [-0.0967, +0.0245] none
  - crossfit − table_tabpfn +0.0200 [-0.0031, +0.0430] none
  - crossfit − pick +0.0112 [-0.0043, +0.0266] none
- `kc2_80nan`: H 0.6866, tabpfn 0.7207, missforest 0.7371, table_tabpfn 0.6812, table_missforest 0.6806, pick 0.6836, picked_missforest 0.4000, crossfit 0.6799, crossfit_missforest 0.5333
  - table_missforest − H -0.0061 [-0.0348, +0.0227] none
  - table_missforest − missforest -0.0566 [-0.0739, -0.0393] better
  - table_tabpfn − H -0.0055 [-0.0371, +0.0261] none
  - table_tabpfn − tabpfn -0.0395 [-0.0570, -0.0220] better
  - pick − H -0.0030 [-0.0339, +0.0278] none
  - pick − table_tabpfn +0.0024 [-0.0031, +0.0080] none
  - pick − table_missforest +0.0030 [-0.0122, +0.0182] none
  - crossfit − H -0.0068 [-0.0343, +0.0208] none
  - crossfit − table_tabpfn -0.0013 [-0.0136, +0.0110] none
  - crossfit − pick -0.0037 [-0.0144, +0.0069] none
- `pendigits_40nan`: H 0.3913, tabpfn 0.4371, missforest 0.4977, table_tabpfn 0.3781, table_missforest 0.3796, pick 0.3782, picked_missforest 0.0667, crossfit 0.3782, crossfit_missforest 0.0667
  - table_missforest − H -0.0117 [-0.0127, -0.0106] better
  - table_missforest − missforest -0.1181 [-0.1234, -0.1128] better
  - table_tabpfn − H -0.0132 [-0.0141, -0.0123] better
  - table_tabpfn − tabpfn -0.0590 [-0.0618, -0.0561] better
  - pick − H -0.0131 [-0.0140, -0.0122] better
  - pick − table_tabpfn +0.0001 [-0.0001, +0.0004] equivalent
  - pick − table_missforest -0.0014 [-0.0022, -0.0006] better
  - crossfit − H -0.0131 [-0.0140, -0.0122] better
  - crossfit − table_tabpfn +0.0001 [-0.0001, +0.0004] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_60nan`: H 0.5751, tabpfn 0.6044, missforest 0.7082, table_tabpfn 0.5557, table_missforest 0.5585, pick 0.5557, picked_missforest 0.0000, crossfit 0.5557, crossfit_missforest 0.0000
  - table_missforest − H -0.0166 [-0.0181, -0.0151] better
  - table_missforest − missforest -0.1497 [-0.1568, -0.1427] better
  - table_tabpfn − H -0.0194 [-0.0212, -0.0176] better
  - table_tabpfn − tabpfn -0.0487 [-0.0510, -0.0463] better
  - pick − H -0.0194 [-0.0212, -0.0176] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0028 [-0.0045, -0.0011] better
  - crossfit − H -0.0194 [-0.0212, -0.0176] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_80nan`: H 0.8340, tabpfn 0.8145, missforest 0.9403, table_tabpfn 0.8102, table_missforest 0.8226, pick 0.8102, picked_missforest 0.0000, crossfit 0.8102, crossfit_missforest 0.0000
  - table_missforest − H -0.0114 [-0.0133, -0.0096] better
  - table_missforest − missforest -0.1177 [-0.1244, -0.1110] better
  - table_tabpfn − H -0.0238 [-0.0263, -0.0212] better
  - table_tabpfn − tabpfn -0.0042 [-0.0052, -0.0032] better
  - pick − H -0.0238 [-0.0263, -0.0212] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0123 [-0.0138, -0.0109] better
  - crossfit − H -0.0238 [-0.0263, -0.0212] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `vehicle_40nan`: H 0.4958, tabpfn 0.4987, missforest 0.5191, table_tabpfn 0.4822, table_missforest 0.4839, pick 0.4830, picked_missforest 0.7333, crossfit 0.4835, crossfit_missforest 0.8000
  - table_missforest − H -0.0119 [-0.0194, -0.0044] better
  - table_missforest − missforest -0.0352 [-0.0551, -0.0153] better
  - table_tabpfn − H -0.0137 [-0.0186, -0.0087] better
  - table_tabpfn − tabpfn -0.0166 [-0.0260, -0.0071] better
  - pick − H -0.0128 [-0.0202, -0.0054] better
  - pick − table_tabpfn +0.0008 [-0.0033, +0.0050] none
  - pick − table_missforest -0.0009 [-0.0028, +0.0009] equivalent
  - crossfit − H -0.0124 [-0.0199, -0.0048] better
  - crossfit − table_tabpfn +0.0013 [-0.0030, +0.0055] none
  - crossfit − pick +0.0005 [-0.0005, +0.0014] equivalent
- `pendigits_20nan`: H 0.3024, tabpfn 0.3080, missforest 0.3341, table_tabpfn 0.2846, table_missforest 0.2862, pick 0.2852, picked_missforest 0.3333, crossfit 0.2852, crossfit_missforest 0.3333
  - table_missforest − H -0.0162 [-0.0172, -0.0152] better
  - table_missforest − missforest -0.0479 [-0.0533, -0.0425] better
  - table_tabpfn − H -0.0177 [-0.0188, -0.0166] better
  - table_tabpfn − tabpfn -0.0234 [-0.0280, -0.0188] better
  - pick − H -0.0172 [-0.0182, -0.0161] better
  - pick − table_tabpfn +0.0006 [+0.0000, +0.0011] worse
  - pick − table_missforest -0.0010 [-0.0017, -0.0002] better
  - crossfit − H -0.0172 [-0.0182, -0.0161] better
  - crossfit − table_tabpfn +0.0006 [+0.0000, +0.0011] worse
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_20nan`: H 0.5947, tabpfn 0.5806, missforest 0.6652, table_tabpfn 0.5845, table_missforest 0.5947, pick 0.5845, picked_missforest 0.0000, crossfit 0.5861, crossfit_missforest 0.2000
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0705 [-0.0778, -0.0632] better
  - table_tabpfn − H -0.0102 [-0.0201, -0.0003] better
  - table_tabpfn − tabpfn +0.0039 [-0.0045, +0.0123] none
  - pick − H -0.0102 [-0.0201, -0.0003] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0102 [-0.0201, -0.0003] better
  - crossfit − H -0.0086 [-0.0185, +0.0013] none
  - crossfit − table_tabpfn +0.0016 [-0.0014, +0.0046] equivalent
  - crossfit − pick +0.0016 [-0.0014, +0.0046] equivalent
- `biodeg_20nan`: H 0.5906, tabpfn 0.6220, missforest 0.6269, table_tabpfn 0.5781, table_missforest 0.5849, pick 0.5845, picked_missforest 0.8667, crossfit 0.5838, crossfit_missforest 0.8667
  - table_missforest − H -0.0057 [-0.0182, +0.0068] none
  - table_missforest − missforest -0.0420 [-0.0566, -0.0274] better
  - table_tabpfn − H -0.0125 [-0.0222, -0.0028] better
  - table_tabpfn − tabpfn -0.0438 [-0.0739, -0.0138] better
  - pick − H -0.0061 [-0.0188, +0.0066] none
  - pick − table_tabpfn +0.0064 [-0.0036, +0.0164] none
  - pick − table_missforest -0.0004 [-0.0012, +0.0004] equivalent
  - crossfit − H -0.0068 [-0.0193, +0.0057] none
  - crossfit − table_tabpfn +0.0057 [-0.0042, +0.0156] none
  - crossfit − pick -0.0007 [-0.0032, +0.0018] equivalent
- `credit-g_20nan`: H 0.8855, tabpfn 0.8763, missforest 0.8907, table_tabpfn 0.8769, table_missforest 0.8826, pick 0.8772, picked_missforest 0.0667, crossfit 0.8772, crossfit_missforest 0.2667
  - table_missforest − H -0.0029 [-0.0122, +0.0065] none
  - table_missforest − missforest -0.0081 [-0.0260, +0.0098] none
  - table_tabpfn − H -0.0085 [-0.0173, +0.0002] none
  - table_tabpfn − tabpfn +0.0007 [-0.0081, +0.0095] none
  - pick − H -0.0083 [-0.0170, +0.0005] none
  - pick − table_tabpfn +0.0003 [-0.0003, +0.0008] equivalent
  - pick − table_missforest -0.0054 [-0.0179, +0.0070] none
  - crossfit − H -0.0083 [-0.0164, -0.0001] better
  - crossfit − table_tabpfn +0.0003 [-0.0034, +0.0039] equivalent
  - crossfit − pick +0.0000 [-0.0037, +0.0037] equivalent
- `vehicle_20nan`: H 0.4133, tabpfn 0.4190, missforest 0.4461, table_tabpfn 0.4024, table_missforest 0.4092, pick 0.4063, picked_missforest 0.5333, crossfit 0.4069, crossfit_missforest 0.6000
  - table_missforest − H -0.0041 [-0.0113, +0.0031] none
  - table_missforest − missforest -0.0369 [-0.0568, -0.0169] better
  - table_tabpfn − H -0.0109 [-0.0144, -0.0074] better
  - table_tabpfn − tabpfn -0.0166 [-0.0264, -0.0069] better
  - pick − H -0.0070 [-0.0145, +0.0006] none
  - pick − table_tabpfn +0.0039 [-0.0017, +0.0095] none
  - pick − table_missforest -0.0029 [-0.0057, -0.0001] better
  - crossfit − H -0.0064 [-0.0137, +0.0010] none
  - crossfit − table_tabpfn +0.0046 [-0.0011, +0.0102] none
  - crossfit − pick +0.0006 [-0.0007, +0.0019] equivalent
- `kc2_20nan`: H 0.5430, tabpfn 0.4930, missforest 0.5718, table_tabpfn 0.4517, table_missforest 0.4942, pick 0.4453, picked_missforest 0.1333, crossfit 0.4445, crossfit_missforest 0.2000
  - table_missforest − H -0.0487 [-0.1309, +0.0334] none
  - table_missforest − missforest -0.0775 [-0.1188, -0.0363] better
  - table_tabpfn − H -0.0913 [-0.1880, +0.0054] none
  - table_tabpfn − tabpfn -0.0413 [-0.0710, -0.0115] better
  - pick − H -0.0977 [-0.1939, -0.0014] better
  - pick − table_tabpfn -0.0064 [-0.0171, +0.0043] none
  - pick − table_missforest -0.0489 [-0.0827, -0.0152] better
  - crossfit − H -0.0985 [-0.1945, -0.0025] better
  - crossfit − table_tabpfn -0.0072 [-0.0179, +0.0035] none
  - crossfit − pick -0.0008 [-0.0026, +0.0010] equivalent

E174 PRIMARY `table_tabpfn` on the 26 MCAR variants: worse than H on none, worse than TabPFN on ['credit-g_80nan'] -> supported
E174 SECONDARY `table_tabpfn` better than H on 19 of 26, better than TabPFN on 16 of 26; `table_missforest` worse than H on none; `crossfit` worse than `table_tabpfn` on ['pendigits_20nan']
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
