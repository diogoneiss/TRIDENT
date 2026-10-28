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

**Status changed at 22:56 GMT-3, on the user's instruction** ("não quero que vc marque ele
completo a menos que a evidência mostrada seja inegável").
- The amended rule reads supported on the 8 variants that ran, below.
- The study as pre-registered (12 shifted variants) did not run in full: spambase MNAR and MAR, and
  biodeg's slopes 0.25 and 1.0, are missing.
- So the study stays open, and ADR 0030 is not amended from it.
- **To complete it:** the remaining cells, after the migration, on the user's go.
  - spambase's `marf` at 101/202/303 and `marfctl` at 101 already ran, unread.
  - Still to run: spambase `marfctl` at 202/303 (2), spambase probit (3), and biodeg probit 0.25
    and 1.0 (6), 11 cells. spambase's MCAR twin is E79's.
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

## Second amendment (2026-10-28, 00:30 GMT-3): the full scope is restored, on torch 2.11.0

- **The user's go:** "voce pode voltar ao que tinha planejado antes: o resto do 1, 2 e 4 em
  paralelo. vá" (2026-10-28, before 00:30 GMT-3). This came after the partial status. The user
  had seen the 8-variant reading.
  - Completing the pre-registered scope is a return to the plan, not a choice that depends on a
    result.
  - The reading is the **original 12-variant rule** (signal on 10 or more under K and H, none
    against). The 8-variant reading above stays as it was written.
- **The 11 remaining cells:** spambase `marfctl` at 202/303, spambase probit at 101/202/303, and
  biodeg probit 0.25 and 1.0 at 101/202/303.
  - They run under **torch 2.11.0+cu128**, which the ADR 0033 preparation installed on 2026-10-27
    at about 23:30 GMT-3 (commits 579e524, 2362d57).
  - The first 40 cells, and E79's MCAR twins, ran under torch 2.5.1+cu121. The new torch draws
    other numbers from the same seed.
- **What this does to the reading:**
  - The primary verdict is per variant, over its own 15 folds. Every variant's three seeds ran
    under one torch, except spambase `marfctl` (101 under 2.5.1; 202 and 303 under 2.11.0).
  - That control enters only the descriptive shifted-minus-control line, which the Outcome will
    flag.
  - The torch of every run is in its `torch_version` tag.
- **Deadline, by the clock only:** no cell starts after 2026-10-28, 08:30 GMT-3. The user stops
  experiments after 09:00 for the migration.

## Outcome on the full scope (2026-10-28, 01:42 GMT-3): supported

- **Ran:** all 51 cells of the pre-registered scope, none failed.
  - 40 ran under torch 2.5.1 (2026-10-27, 20:35–22:33 GMT-3).
  - 11 ran under torch 2.11.0 (2026-10-28, 00:30–01:40 GMT-3).
  - **Which variants ran under 2.11.0:** biodeg at slopes 0.25 and 1.0, spambase probit, and seeds
    202/303 of spambase's `marfctl` control. Every other variant's three seeds ran under 2.5.1.
- **Primary, the original 12-variant rule: supported.** A signal on 12 of 12 under K and 12 of 12
  under H, with none against. The rule needed 10.
  - Every one of the 24 intervals lies above 0 with seed agreement. The lowest lower bound is
    +0.084 (vehicle MNAR, K); under H it is +0.087.
- **Why it is now marked complete** (the user's instruction of 2026-10-27, "não quero que vc marque
  ele completo a menos que a evidência mostrada seja inegável"):
  - the pre-registered scope ran in full;
  - the original rule, not the amended one, is met with a margin of 2 variants under each readout;
  - no interval comes near 0.
- **What travels with it:**
  - **The 8-variant reading was seen before the scope was restored.** The decision to complete was
    the user's and returned to the plan.
  - **Two torch versions:** the primary verdict is within each variant, and each variant's seeds
    share one torch. The only variant that mixes them is spambase's `marfctl` control, which
    enters only the descriptive line below.
- **Shifted minus control (descriptive):**
  - **On biodeg the ranking weakens steadily with the MNAR tilt:** −0.023, −0.049, −0.077 and
    −0.111 at slopes 0.25, 0.5, 1.0 and 2.0. The first interval reaches 0.
  - **Weaker** on spambase MNAR (−0.045) and on kc2 MAR under K (−0.092).
  - **Stronger** on kc2 MNAR (+0.103) and vehicle MAR (+0.040).
  - **No difference** elsewhere. spambase MAR minus its mixed-torch control is −0.009, with an
    interval spanning 0.
- **What it leads to, as fixed:** ADR 0030's limits line can be amended, on the user's decision,
  to "tested under MCAR, one MAR draw and self-masking MNAR, at 60nan". The note that the ranking
  weakens as the self-masking tilt grows goes with it.

The reader's printout on the full scope (`spread_mechanisms.py --tally`), unedited:

```text
- `biodeg_probit0p25_60nan`: K num_rank +0.348 [+0.327, +0.369] -> signal; minus biodeg_60nan -0.023 [-0.051, +0.004]; H num_rank +0.332 [+0.309, +0.354] -> signal
- `biodeg_probit_60nan`: K num_rank +0.322 [+0.290, +0.355] -> signal; minus biodeg_60nan -0.049 [-0.082, -0.017]; H num_rank +0.295 [+0.256, +0.334] -> signal
- `biodeg_probit1p0_60nan`: K num_rank +0.294 [+0.275, +0.314] -> signal; minus biodeg_60nan -0.077 [-0.094, -0.060]; H num_rank +0.304 [+0.289, +0.318] -> signal
- `biodeg_probit2p0_60nan`: K num_rank +0.260 [+0.242, +0.279] -> signal; minus biodeg_60nan -0.111 [-0.132, -0.091]; H num_rank +0.239 [+0.209, +0.269] -> signal
- `vehicle_probit_60nan`: K num_rank +0.101 [+0.084, +0.118] -> signal; minus vehicle_60nan -0.010 [-0.028, +0.009]; H num_rank +0.110 [+0.087, +0.134] -> signal
- `kc2_probit_60nan`: K num_rank +0.518 [+0.473, +0.563] -> signal; minus kc2_60nan +0.103 [+0.057, +0.149]; H num_rank +0.467 [+0.431, +0.503] -> signal
- `spambase_probit_60nan`: K num_rank +0.322 [+0.304, +0.341] -> signal; minus spambase_60nan -0.045 [-0.073, -0.017]; H num_rank +0.287 [+0.245, +0.329] -> signal
- `biodeg_marf_60nan`: K num_rank +0.425 [+0.411, +0.439] -> signal; minus biodeg_marfctl_60nan -0.013 [-0.028, +0.002]; H num_rank +0.413 [+0.399, +0.426] -> signal; minus biodeg_marfctl_60nan -0.021 [-0.042, +0.000]
- `kc2_marf_60nan`: K num_rank +0.378 [+0.321, +0.434] -> signal; minus kc2_marfctl_60nan -0.092 [-0.151, -0.034]; H num_rank +0.414 [+0.376, +0.452] -> signal; minus kc2_marfctl_60nan -0.022 [-0.071, +0.027]
- `vehicle_marf_60nan`: K num_rank +0.143 [+0.121, +0.164] -> signal; minus vehicle_marfctl_60nan +0.040 [+0.010, +0.069]; H num_rank +0.134 [+0.114, +0.155] -> signal; minus vehicle_marfctl_60nan +0.037 [+0.005, +0.068]
- `spambase_marf_60nan`: K num_rank +0.446 [+0.429, +0.463] -> signal; minus spambase_marfctl_60nan -0.009 [-0.028, +0.011]; H num_rank +0.443 [+0.426, +0.459] -> signal; minus spambase_marfctl_60nan -0.014 [-0.039, +0.011]
- `credit-g_marf_60nan`: K num_rank +0.180 [+0.123, +0.237] -> signal; minus credit-g_marfctl_60nan +0.026 [-0.051, +0.104]; H num_rank +0.195 [+0.140, +0.250] -> signal; minus credit-g_marfctl_60nan +0.008 [-0.078, +0.095]

- K: signal on 12 of 12, against on none
- H: signal on 12 of 12, against on none

Primary: signal on 12 (K) and 12 (H) of 12, against under K on none -> supported (fold-pair t, seed agreement)
```
