# Does the re-masking spread still rank errors under MNAR and MAR? (E84; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0030's spread was tested under MCAR only (E75, E76, E80). Its limits list says "The
  MAR and MNAR families are untested".
- **The user's go:** "de acordo em fazer 1 depois o 2, com 4 em paralelo" (2026-10-27, before
  20:40 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/spread_mechanisms.py` has `train`, `--todo` and `--tally`.
  - `run_spread_mechanisms_queue.sh` runs four GPU queues.
  - Nothing has been read: E79's MCAR ledgers were opened only to check that they load
    (finiteness).

## Design (fixed before the first cell)

- **The 12 shifted variants, all at 60nan:**
  - **MNAR (self-masking probit, E45/E46):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2
    and spambase at slope 0.5.
  - **MAR (E64's new draw, `marf`):** biodeg, kc2, vehicle, spambase and credit-g.
- **Controls:**
  - The MNAR variants' MCAR twins (`<table>_60nan`) are E79's runs at the same seeds. They carry
    the spread at K (E79 had `--score_remask`) and no H.
  - The MAR variants' matched controls (`marfctl_60nan`) run here.
- **Cells:** (7 MNAR + 5 MAR + 5 MAR controls) × seeds 101, 202 and 303 = 51 cells.
  - Today's default, five folds, `--score_checkpoints --score_remask`, as E76 ran.
  - MLflow tag `experiment=spread-mechanisms-2026-10-27`; state in
    `/scratch2/diogoneiss/trident-2026-10-27-spread-mechanisms/`.
  - About 3–3.5 h on four queues. E85's queues start as these end.

## Measures and rule, fixed now (`spread_mechanisms.py --tally`)

- **The measure:** E76's primary, the within-column `num_rank`.
  - Per fold, the Spearman between a numerical cell's spread and its absolute error, within each
    column, averaged over the columns.
  - It is read under K (`induced_remask`) and under H (`induced_remask_loss_checkpoint`), as ADR
    0030 requires.
- **The verdict on each variant (15 folds):**
  - a signal if the 95% t-interval lies above 0 with every seed above 0;
  - against if it lies below 0 with every seed below 0;
  - no signal otherwise.
- **The rule over the 12 shifted variants:**
  - **supported** (ADR 0030's scope extends to these mechanisms): a signal on 10 or more under K
    and under H, with none against under K;
  - **refuted:** a signal on 7 or fewer under K, or any against under K;
  - **mixed:** otherwise.
- **Descriptive:** each shifted variant's `num_rank` minus its control's, fold-paired. That is K
  only for the MNAR twins, and K and H for MAR.

**What it leads to:**
- **Supported:** ADR 0030's limits line is amended, with the user's decision, to "tested under
  MCAR, MAR (one draw) and self-masking MNAR at 60nan".
- **Otherwise:** the limit is stated with the failing mechanisms named.

## Limits fixed now

- 60nan only.
- One MAR draw.
- Self-masking MNAR only; the slope dial is on biodeg alone.

## Amendment (2026-10-27, 20:44 GMT-3, before any reading): cut to end before a store migration

- **Why:** the user needs the GPU and the store quiet by about 23:00 for a database migration ("na
  verdade rode apenas parte do 1, nao tudo, para acabar por volta de 23h. vamos fazer uma migração de
  banco"). No result had been read, and no cell had finished.
- **The scope is now 8 shifted variants:**
  - MNAR: biodeg at slopes 0.5 and 2.0, vehicle and kc2;
  - MAR: biodeg, kc2, vehicle and credit-g, each with its `marfctl` control.
  - spambase (probit and MAR) and biodeg's slopes 0.25 and 1.0 are out.
  - That is 36 cells.
- **The four spambase cells already running** (`marf` at 101/202/303, `marfctl` at 101) are let
  finish, so the store holds no interrupted run before the migration. They are outside the scope
  and are not read.
- **The rule, scaled to 8:**
  - **supported:** a signal on 7 or more under K and under H, with none against under K;
  - **refuted:** a signal on 4 or fewer under K, or any against;
  - **mixed:** otherwise.
- **Running:** the queues were stopped between cells (`STOP`). They relaunch when the spambase cells
  end, with a clock-only deadline: no cell starts after 22:45 GMT-3. Cells not started by then run
  after the migration on the user's go, and the reading waits for all 36.
- **E85 is not run tonight.** Its chained waiters were stopped before any of its cells started. It
  stays pre-registered and waits for the user.

## Outcome (2026-10-27, 22:34 GMT-3): partial, not complete

**Status changed at 22:42 GMT-3, on the user's instruction** ("não quero que vc marque ele
completo a menos que a evidência mostrada seja inegável").
- The amended rule reads supported on the 8 variants that ran, below.
- The study as pre-registered (12 shifted variants) did not run in full: spambase MNAR and MAR, and
  biodeg's slopes 0.25 and 1.0, are missing.
- So the study stays open, and ADR 0030 is not amended from it.
- **To complete it:** the remaining cells, after the migration, on the user's go.
  - spambase's `marf` at 101/202/303 and `marfctl` at 101 already ran, unread.
  - Still to run: spambase `marfctl` at 202/303, spambase probit, and biodeg probit 0.25 and 1.0
    (14 cells).
  - Then the original 12-variant rule is read.


- **Ran:** the amended scope's 36 cells, from 21:22 to 22:33 GMT-3, none failed. The four
  spambase cells let finish (20:35–21:21) are outside the scope and were not read.
- **Primary: supported.** The within-column `num_rank` signals on 8 of 8 shifted variants under K
  and on 8 of 8 under H, with none against.
  - **MNAR:** biodeg at slopes 0.5 and 2.0 (+0.32, +0.26), vehicle (+0.10), kc2 (+0.52).
  - **MAR:** biodeg (+0.43), kc2 (+0.38), vehicle (+0.14), credit-g (+0.18).
- **Shifted minus control (descriptive):**
  - **Weaker** on biodeg under MNAR (−0.05 at slope 0.5, −0.11 at slope 2.0, growing with the
    tilt) and on kc2 under MAR (−0.09 under K; no difference under H).
  - **Stronger** on kc2 under MNAR (+0.10) and on vehicle under MAR (+0.04).
  - **No difference** on vehicle under MNAR and on biodeg and credit-g under MAR.
- **What it would lead to once complete:** ADR 0030's scope extended to MAR and self-masking MNAR,
  for the user's decision. Not before.
  - The descriptive part travels with it: on biodeg the ranking weakens as the self-masking
    tilt grows.
- **Not covered, as amended:** spambase, biodeg's slopes 0.25 and 1.0, and other rates.

The reader's printout (`spread_mechanisms.py --tally`), unedited:

```text
- `biodeg_probit_60nan`: K num_rank +0.322 [+0.290, +0.355] -> signal; minus biodeg_60nan -0.049 [-0.082, -0.017]; H num_rank +0.295 [+0.256, +0.334] -> signal
- `biodeg_probit2p0_60nan`: K num_rank +0.260 [+0.242, +0.279] -> signal; minus biodeg_60nan -0.111 [-0.132, -0.091]; H num_rank +0.239 [+0.209, +0.269] -> signal
- `vehicle_probit_60nan`: K num_rank +0.101 [+0.084, +0.118] -> signal; minus vehicle_60nan -0.010 [-0.028, +0.009]; H num_rank +0.110 [+0.087, +0.134] -> signal
- `kc2_probit_60nan`: K num_rank +0.518 [+0.473, +0.563] -> signal; minus kc2_60nan +0.103 [+0.057, +0.149]; H num_rank +0.467 [+0.431, +0.503] -> signal
- `biodeg_marf_60nan`: K num_rank +0.425 [+0.411, +0.439] -> signal; minus biodeg_marfctl_60nan -0.013 [-0.028, +0.002]; H num_rank +0.413 [+0.399, +0.426] -> signal; minus biodeg_marfctl_60nan -0.021 [-0.042, +0.000]
- `kc2_marf_60nan`: K num_rank +0.378 [+0.321, +0.434] -> signal; minus kc2_marfctl_60nan -0.092 [-0.151, -0.034]; H num_rank +0.414 [+0.376, +0.452] -> signal; minus kc2_marfctl_60nan -0.022 [-0.071, +0.027]
- `vehicle_marf_60nan`: K num_rank +0.143 [+0.121, +0.164] -> signal; minus vehicle_marfctl_60nan +0.040 [+0.010, +0.069]; H num_rank +0.134 [+0.114, +0.155] -> signal; minus vehicle_marfctl_60nan +0.037 [+0.005, +0.068]
- `credit-g_marf_60nan`: K num_rank +0.180 [+0.123, +0.237] -> signal; minus credit-g_marfctl_60nan +0.026 [-0.051, +0.104]; H num_rank +0.195 [+0.140, +0.250] -> signal; minus credit-g_marfctl_60nan +0.008 [-0.078, +0.095]

- K: signal on 8 of 8, against on none
- H: signal on 8 of 8, against on none

Primary: signal on 8 (K) and 8 (H) of 8, against under K on none -> supported (fold-pair t, seed agreement)
```
