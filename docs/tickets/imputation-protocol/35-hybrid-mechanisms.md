# Does the deployable `table` hybrid hold under MNAR and MAR? (E93; pre-registered 2026-10-29)

This ticket was written and committed before the study's queues started.

- **Why:** ADR 0031's decision 6 (accepted 2026-10-29) makes the H + TabPFN hybrid with the `table`
  rule a deployable output. Its scope is MCAR (E90, E91); MAR and MNAR are listed as untested.
- **The user's go:** "sim" (2026-10-29, about 11:00 GMT-3), on the proposal to run it, after
  checking the MLflow sync: the follower's copy equals the leader's (21,647 runs, the same tag and
  metric counts) at 11:04 GMT-3.
- **Code:** no change to `src/`.
  - `hybrid_confirmation.py --study e93` runs E90's pipeline on E84's variants.
  - `run_hybrid_confirmation_queue.sh` now takes the state directory and both deadlines from the
    environment. E90 and E91's defaults are unchanged.
- **Smoke:** kc2_probit_60nan s4545 was trained before this commit (CELL_DONE, five selection
  files). Only its run's headline was printed (H 0.730), and no rule was scored. It counts.

## Design (fixed before the queues)

- **Cells:** E84's 12 shifted variants at 60nan × seeds 4545, 5656 and 6767 (E90's), so 36 cells.
  - **MNAR (self-masking probit):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2 and
    spambase at 0.5.
  - **MAR (`marf`):** biodeg, kc2, vehicle, spambase and credit-g.
  - Each cell is E90's: H with the hidden observed validation cells read in place, then TabPFN v2
    refit on each training fold.
  - The rule is `table`, unchanged.
- **Running:** gorgona8, four training queues and two cells lanes, MLflow tag
  `experiment=hybrid-mechanisms-2026-10-29` through the leader's server; state in
  `/scratch2/diogoneiss/trident-2026-10-29-hybrid-mechanisms/`.
  - About 3–4 h. spambase dominates both the training (about 37 min a cell in E84) and the refits.
  - **Deadlines, by the clock alone:** no training cell starts after 2026-10-29 18:00 GMT-3, and no
    cells job after 19:00 GMT-3. Anything missing then is reported partial, with no call.

- **Amendment 2026-10-29, 14:49 GMT-3, before any E93 result was read:** four cells are retrained.
  - medusa1 rebooted at about 14:30 GMT-3, taking the leader's MLflow server down. Four training
    cells failed between 14:22 and 14:32 on the server's API (`Max retries exceeded`):
    kc2_probit_60nan s5656, vehicle_marf_60nan s5656, credit-g_marf_60nan s4545 and
    vehicle_probit_60nan s6767. Three had finished training and failed when logging; one failed
    at its start.
  - Their outputs were moved to `failed-2026-10-29/` in the state directory, so the queue sees them
    as not done.
  - Each is retrained once with the same seed, by a rescue runner that starts after the four
    queues stop, under the same 18:00 deadline.
  - The failure is infrastructure. The same cells run again, no result was read, and no other cell
    reruns.
  - Their unfinished runs stay in the store (not FINISHED), so `report_common.parents` does not
    count them.

## Measures and rule, fixed now (`hybrid_confirmation.py --study e93 --tally`)

- **Per variant:** `table` − H and `table` − TabPFN on `impute_score`, over 15 fold pairs. The
  project's rule applies, with a SESOI of 1% of the opponent's mean.
- **The call, in E90's form:**
  - **supported** if better than H on 8 or more of 12, worse on none, and worse than TabPFN on 2 or
    fewer;
  - **refuted** if better than H on 5 or fewer, or worse on 2 or more, or worse than TabPFN on 5 or
    more;
  - **mixed** otherwise.
  - The thresholds are E90's (8, 5, 2 and 5), kept as they were rather than rescaled to 12 cells.
- **Descriptive:** the same counts split by MNAR (7) and MAR (5), biodeg's four slopes in order,
  and the other three rules.
- **What follows:** if supported, extending decision 6's scope to MNAR and MAR at 60nan is the
  user's decision. If refuted or mixed, the scope stays MCAR and the Outcome names where it breaks.

## Outcome

**Done 2026-10-29, 16:03 GMT-3: mixed.** All 36 cells were trained and refit on gorgona8 between
11:13 and 16:01 GMT-3:
- 32 in the queues, 4 retrained after the infrastructure failure (amendment), and the smoke;
- one of the 4, credit-g_marf_60nan s4545, finished under the first rescue runner after that
  runner was stopped. It wrote its run and five selection files, and the second runner skipped it.
- The four `FAILED` lines in the log are the infrastructure failures.

The tally, unedited (`hybrid_confirmation.py --study e93 --tally`):

```
- `biodeg_probit0p25_60nan` (H 0.9114, TabPFN 0.9150): column − tabpfn -0.0059 [-0.0089, -0.0030] better; table − model -0.0066 [-0.0111, -0.0021] better; table − tabpfn -0.0102 [-0.0155, -0.0050] better; table − equal +0.0010 [-0.0002, +0.0022] equivalent; shrunk − tabpfn -0.0103 [-0.0143, -0.0063] better; pooled − tabpfn -0.0092 [-0.0129, -0.0054] better
- `biodeg_probit_60nan` (H 0.9123, TabPFN 0.9246): column − tabpfn -0.0111 [-0.0154, -0.0068] better; table − model +0.0017 [-0.0050, +0.0084] equivalent; table − tabpfn -0.0106 [-0.0146, -0.0065] better; table − equal +0.0043 [-0.0007, +0.0094] none; shrunk − tabpfn -0.0144 [-0.0180, -0.0109] better; pooled − tabpfn -0.0162 [-0.0205, -0.0118] better
- `biodeg_probit1p0_60nan` (H 0.9306, TabPFN 0.9379): column − tabpfn -0.0068 [-0.0088, -0.0048] better; table − model -0.0006 [-0.0023, +0.0012] equivalent; table − tabpfn -0.0078 [-0.0101, -0.0056] better; table − equal +0.0012 [+0.0000, +0.0023] equivalent; shrunk − tabpfn -0.0088 [-0.0107, -0.0068] better; pooled − tabpfn -0.0084 [-0.0100, -0.0067] better
- `biodeg_probit2p0_60nan` (H 0.9321, TabPFN 0.9545): column − tabpfn -0.0139 [-0.0177, -0.0100] better; table − model +0.0059 [-0.0003, +0.0121] none; table − tabpfn -0.0165 [-0.0211, -0.0119] better; table − equal -0.0017 [-0.0035, +0.0002] equivalent; shrunk − tabpfn -0.0149 [-0.0194, -0.0104] better; pooled − tabpfn -0.0174 [-0.0230, -0.0118] better
- `vehicle_probit_60nan` (H 0.6308, TabPFN 0.6225): column − tabpfn -0.0063 [-0.0101, -0.0025] better; table − model -0.0169 [-0.0202, -0.0136] better; table − tabpfn -0.0086 [-0.0135, -0.0037] better; table − equal +0.0006 [-0.0008, +0.0020] equivalent; shrunk − tabpfn -0.0096 [-0.0140, -0.0052] better; pooled − tabpfn -0.0086 [-0.0131, -0.0040] better
- `kc2_probit_60nan` (H 0.7253, TabPFN 0.7757): column − tabpfn -0.0363 [-0.0515, -0.0211] better; table − model +0.0063 [-0.0205, +0.0331] none; table − tabpfn -0.0440 [-0.0569, -0.0312] better; table − equal -0.0023 [-0.0111, +0.0065] none; shrunk − tabpfn -0.0424 [-0.0543, -0.0305] better; pooled − tabpfn -0.0368 [-0.0503, -0.0234] better
- `spambase_probit_60nan` (H 0.9597, TabPFN 0.9955): column − tabpfn -0.0266 [-0.0299, -0.0234] better; table − model +0.0090 [+0.0056, +0.0124] worse; table − tabpfn -0.0268 [-0.0310, -0.0226] better; table − equal -0.0032 [-0.0067, +0.0003] equivalent; shrunk − tabpfn -0.0265 [-0.0295, -0.0236] better; pooled − tabpfn -0.0298 [-0.0321, -0.0275] better
- `biodeg_marf_60nan` (H 0.6864, TabPFN 0.7056): column − tabpfn -0.0297 [-0.0506, -0.0089] better; table − model -0.0221 [-0.0328, -0.0115] better; table − tabpfn -0.0413 [-0.0606, -0.0220] better; table − equal +0.0033 [-0.0043, +0.0109] none; shrunk − tabpfn -0.0431 [-0.0583, -0.0279] better; pooled − tabpfn -0.0330 [-0.0529, -0.0131] better
- `kc2_marf_60nan` (H 0.5987, TabPFN 0.6701): column − tabpfn -0.0586 [-0.0892, -0.0281] better; table − model +0.0091 [-0.0031, +0.0214] none; table − tabpfn -0.0623 [-0.1058, -0.0188] better; table − equal -0.0066 [-0.0301, +0.0168] none; shrunk − tabpfn -0.0589 [-0.0838, -0.0340] better; pooled − tabpfn -0.0691 [-0.1027, -0.0355] better
- `vehicle_marf_60nan` (H 0.4510, TabPFN 0.4445): column − tabpfn -0.0090 [-0.0413, +0.0234] none; table − model -0.0154 [-0.0226, -0.0082] better; table − tabpfn -0.0089 [-0.0408, +0.0230] none; table − equal +0.0078 [-0.0018, +0.0173] none; shrunk − tabpfn -0.0166 [-0.0440, +0.0107] none; pooled − tabpfn -0.0079 [-0.0333, +0.0174] none
- `spambase_marf_60nan` (H 0.9058, TabPFN 0.9354): column − tabpfn -0.0383 [-0.0446, -0.0321] better; table − model -0.0071 [-0.0099, -0.0043] better; table − tabpfn -0.0368 [-0.0429, -0.0306] better; table − equal +0.0005 [-0.0024, +0.0033] equivalent; shrunk − tabpfn -0.0401 [-0.0460, -0.0342] better; pooled − tabpfn -0.0425 [-0.0482, -0.0369] better
- `credit-g_marf_60nan` (H 0.9353, TabPFN 0.9054): column − tabpfn +0.0141 [+0.0062, +0.0221] worse; table − model -0.0116 [-0.0183, -0.0048] better; table − tabpfn +0.0184 [+0.0084, +0.0284] worse; table − equal -0.0041 [-0.0104, +0.0022] none; shrunk − tabpfn +0.0184 [+0.0082, +0.0286] worse; pooled − tabpfn +0.0104 [+0.0020, +0.0189] worse

Secondary `column`: better than H on 4 of 12, worse on ['biodeg_probit2p0_60nan', 'spambase_probit_60nan'], worse than TabPFN on 1 (['credit-g_marf_60nan']) -> refuted
Primary `table`: better than H on 6 of 12, worse on ['spambase_probit_60nan'], worse than TabPFN on 1 (['credit-g_marf_60nan']) -> mixed
Secondary `shrunk`: better than H on 6 of 12, worse on ['biodeg_probit2p0_60nan', 'spambase_probit_60nan'], worse than TabPFN on 1 (['credit-g_marf_60nan']) -> refuted
Secondary `pooled`: better than H on 6 of 12, worse on ['spambase_probit_60nan'], worse than TabPFN on 1 (['credit-g_marf_60nan']) -> mixed

(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Primary, `table`: mixed.**
  - Better than H on 6 of 12; the rule needed 8.
  - Worse than H on 1: spambase_probit_60nan, +0.0090.
  - Worse than TabPFN on 1: credit-g_marf_60nan, +0.0184.
- **Why it does less here:** under MNAR and MAR, H already beats TabPFN on 10 of the 12 variants
  (every TabPFN mean but vehicle's and credit-g's is higher than H's). So the hybrid has little to
  take from TabPFN. `table` − TabPFN is better on 10 of 12, while `table` − H is near zero on the
  biodeg slopes and kc2.
- **Where it breaks:**
  - spambase_probit, where the per-table numerical weight drags H's lead toward TabPFN;
  - credit-g_marf, where the categorical pick goes the wrong way. `column`, `shrunk` and `pooled`
    are all worse than TabPFN there too.
- **The secondary rules:** `column` and `shrunk` refuted (each worse than H on 2), `pooled` mixed.
- **What follows, as fixed:** decision 6's scope stays MCAR. Under MNAR and MAR the deployable
  hybrid is never far from H (within 1%) and beats TabPFN on 10 of 12, but it does not meet the
  rule's gain over H.
