# Does the re-masking spread still rank errors under MNAR and MAR at 20nan? (E110; pre-registered 2026-11-01 23:59 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** ADR 0030's decision 7 covers MAR and self-masking MNAR at 40 and 60nan (E84, E105). Its
  MCAR scope starts at 20nan, so 20nan under MAR and MNAR is the remaining gap. This is E105 at 20nan.
- **The user's go:** "voce pode colocar mais coisas pra rodar nsa duas maquinas" (2026-11-01, before
  23:59 GMT-3; back at 09:00).
- **Data:** the 20nan MAR family and the probit variants of biodeg, kc2 and vehicle already existed;
  written on medusa1 before this ticket, with the same generators: `spambase_probit_20nan` and
  biodeg at slopes 0.25, 1.0 and 2.0 (`generate_mnar.py --pct 0.2`).
- **Code:** `spread_40nan.py --level 20` and `LEVEL=20 run_spread_40nan_queue.sh`.

## Design, measures and rule (fixed now): E105's at 20nan

- 12 shifted variants (biodeg probit at slopes 0.25/0.5/1/2; vehicle, kc2, spambase probit; `marf`
  for biodeg, kc2, vehicle, spambase, credit-g), the MCAR twins and `marfctl` controls, seeds
  101/202/303, 63 cells, today's default with `--score_checkpoints --score_remask`, five folds,
  MLflow off, medusa1 (four queues).
- **Supported** with a within-column `num_rank` signal on 10 or more of 12 under K and H and none
  against under K; **refuted** with 7 or fewer under K or any against; **mixed** otherwise.
- **The cap:** no cell starts after 2026-11-02 08:30 GMT-3.
- **Amendment (2026-11-02 00:04 GMT-3, before any result):** the cap is removed on the user's word ("nao precisa
  interromper"); the study runs until every item has ended.
- **What follows:** supported → extending decision 7 to 20nan is the user's decision.

## Outcome

**Written 2026-11-02 05:04 GMT-3. Supported: a within-column signal on 12 of 12 shifted variants under K and 12 of 12
under H, none against** (ρ +0.14 to +0.55 under K; lowest lower bound +0.107, vehicle_marf).

- **Complete:** all 63 cells on medusa1, 2026-11-01 23:59 to 2026-11-02 05:03 GMT-3, no failure.
- **Shifted minus control (descriptive):** unlike 40 and 60nan, the ranking does not weaken with the
  self-masking tilt on biodeg at 20nan (+0.002, +0.011, +0.024, −0.004 at slopes 0.25 to 2.0); it is
  a little stronger on spambase and kc2 under MNAR (+0.04).
- **What follows, as fixed above:** extending ADR 0030's decision 7 to 20nan is the user's decision.

### The tally, unedited (`spread_40nan.py --level 20 --tally`, 2026-11-02 05:04 GMT-3)

```
- `biodeg_probit0p25_20nan`: K num_rank +0.465 [+0.447, +0.482] -> signal; minus biodeg_20nan +0.002 [-0.026, +0.031]; H num_rank +0.474 [+0.457, +0.490] -> signal; minus biodeg_20nan +0.015 [-0.008, +0.038]
- `biodeg_probit_20nan`: K num_rank +0.474 [+0.461, +0.487] -> signal; minus biodeg_20nan +0.011 [-0.009, +0.032]; H num_rank +0.473 [+0.455, +0.491] -> signal; minus biodeg_20nan +0.014 [-0.013, +0.042]
- `biodeg_probit1p0_20nan`: K num_rank +0.486 [+0.464, +0.507] -> signal; minus biodeg_20nan +0.024 [+0.002, +0.045]; H num_rank +0.486 [+0.468, +0.504] -> signal; minus biodeg_20nan +0.027 [+0.002, +0.052]
- `biodeg_probit2p0_20nan`: K num_rank +0.458 [+0.440, +0.475] -> signal; minus biodeg_20nan -0.004 [-0.035, +0.026]; H num_rank +0.458 [+0.447, +0.469] -> signal; minus biodeg_20nan -0.000 [-0.027, +0.026]
- `vehicle_probit_20nan`: K num_rank +0.158 [+0.142, +0.174] -> signal; minus vehicle_20nan +0.025 [-0.009, +0.059]; H num_rank +0.157 [+0.139, +0.175] -> signal; minus vehicle_20nan +0.025 [-0.008, +0.057]
- `kc2_probit_20nan`: K num_rank +0.504 [+0.438, +0.570] -> signal; minus kc2_20nan +0.039 [+0.002, +0.076]; H num_rank +0.505 [+0.451, +0.558] -> signal; minus kc2_20nan +0.038 [-0.028, +0.103]
- `spambase_probit_20nan`: K num_rank +0.534 [+0.514, +0.554] -> signal; minus spambase_20nan +0.042 [+0.016, +0.068]; H num_rank +0.526 [+0.512, +0.541] -> signal; minus spambase_20nan +0.044 [+0.023, +0.064]
- `biodeg_marf_20nan`: K num_rank +0.512 [+0.500, +0.523] -> signal; minus biodeg_marfctl_20nan +0.018 [-0.005, +0.040]; H num_rank +0.517 [+0.497, +0.538] -> signal; minus biodeg_marfctl_20nan +0.018 [-0.011, +0.047]
- `kc2_marf_20nan`: K num_rank +0.448 [+0.411, +0.486] -> signal; minus kc2_marfctl_20nan -0.038 [-0.091, +0.016]; H num_rank +0.414 [+0.366, +0.461] -> signal; minus kc2_marfctl_20nan -0.046 [-0.090, -0.002]
- `vehicle_marf_20nan`: K num_rank +0.136 [+0.107, +0.165] -> signal; minus vehicle_marfctl_20nan -0.014 [-0.054, +0.025]; H num_rank +0.138 [+0.113, +0.163] -> signal; minus vehicle_marfctl_20nan -0.017 [-0.061, +0.028]
- `spambase_marf_20nan`: K num_rank +0.550 [+0.538, +0.563] -> signal; minus spambase_marfctl_20nan -0.002 [-0.019, +0.015]; H num_rank +0.551 [+0.537, +0.565] -> signal; minus spambase_marfctl_20nan +0.004 [-0.020, +0.028]
- `credit-g_marf_20nan`: K num_rank +0.275 [+0.218, +0.331] -> signal; minus credit-g_marfctl_20nan -0.005 [-0.081, +0.072]; H num_rank +0.252 [+0.203, +0.301] -> signal; minus credit-g_marfctl_20nan -0.002 [-0.090, +0.086]

- K: signal on 12 of 12, against on none
- H: signal on 12 of 12, against on none

Primary: signal on 12 (K) and 12 (H) of 12, against under K on none -> supported (fold-pair t, seed agreement)
```
