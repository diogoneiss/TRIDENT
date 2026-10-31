# Does the re-masking spread still rank errors under MNAR and MAR at 40nan? (E105; pre-registered 2026-10-31 10:16 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** ADR 0030's decision 7 (2026-10-31) extended the spread's scope to MAR and self-masking
  MNAR at 60nan, from E84 ([32](32-spread-mechanisms.md)). Its MCAR scope runs to 60nan, so 40nan
  under MAR and MNAR is the gap. This is E84 at 40nan.
- **The user's go:** "Isso, rode aa duas em paralelo agora" (2026-10-31, after 09:58 and before
  10:14 GMT-3), on this study and E104 in parallel.
- **New data, no model code:** the 40nan variants, written on medusa1 before this ticket.
  - `datasets/generate_mnar.py --pct 0.4` for biodeg, kc2, vehicle and spambase (slope 0.5), and
    biodeg at slopes 0.25, 1.0 and 2.0: `<base>_probit[slope]_40nan`.
  - `datasets/generate_mar_family.py` with RATES gaining 0.4 (90f1446); each rate draws from its own
    streams, so the 20 and 60nan files are unchanged (the script rewrites a file only if
    identical). `<base>_marf_40nan` and `<base>_marfctl_40nan`; as at 60nan, the drivers stay
    observed, so 0.28 of feature cells are missing.
- **Code:** `scripts/experiments/spread_40nan.py` and `run_spread_40nan_queue.sh`, reusing E84's
  measure (`within_column`) and marks.

## Design (fixed before the first cell)

- **Shifted variants (12), at 40nan:** biodeg probit at slopes 0.25/0.5/1/2; vehicle, kc2 and
  spambase probit; `marf` for biodeg, kc2, vehicle, spambase and credit-g.
- **Controls, trained here:** the MCAR twins (biodeg, kc2, vehicle, spambase at 40nan) and the five
  `marfctl` controls.
- **Cells:** 21 variants × seeds 101/202/303 = 63, today's default with `--score_checkpoints
  --score_remask`, five folds, MLflow off; all on medusa1, four queues, beside E104.

## Measures and rule, fixed now (`spread_40nan.py --tally`), E84's

- **Within-column `num_rank`** at K and H: a signal where the 95% interval over 15 folds lies above 0
  with seed agreement.
- **Supported** with a signal on 10 or more of 12 under K and H and none against under K;
  **refuted** with 7 or fewer under K or any against; **mixed** otherwise. Shifted minus control is
  descriptive.
- **What follows:** supported → an amendment of ADR 0030's decision 7 to 40nan, the user's decision.

## Outcome

**Written 2026-10-31 12:11 GMT-3. Supported: the within-column ranking signals on 12 of 12 shifted variants under
K and 12 of 12 under H, none against** (ρ +0.11 to +0.54 under K, +0.10 to +0.53 under H; lowest lower bound +0.073, H on vehicle_marf).

- **Complete:** all 63 cells on medusa1, 10:16–12:09 GMT-3, no failure.
- **Shifted minus control (descriptive):** as at 60nan, the ranking weakens on biodeg as the
  self-masking tilt grows: +0.002, +0.001, −0.021 and −0.080 at slopes 0.25 to 2.0 (K). It is
  stronger on kc2 under MNAR (+0.07). The other differences are within ±0.05, most with intervals
  spanning zero.
- **What follows, as fixed above:** extending ADR 0030's decision 7 to 40nan is the user's decision.

### The tally, unedited (`spread_40nan.py --tally`, 2026-10-31 12:11 GMT-3)

```
- `biodeg_probit0p25_40nan`: K num_rank +0.444 [+0.431, +0.456] -> signal; minus biodeg_40nan +0.002 [-0.017, +0.021]; H num_rank +0.443 [+0.433, +0.454] -> signal; minus biodeg_40nan +0.006 [-0.012, +0.024]
- `biodeg_probit_40nan`: K num_rank +0.443 [+0.431, +0.455] -> signal; minus biodeg_40nan +0.001 [-0.011, +0.013]; H num_rank +0.447 [+0.435, +0.459] -> signal; minus biodeg_40nan +0.010 [-0.004, +0.024]
- `biodeg_probit1p0_40nan`: K num_rank +0.421 [+0.403, +0.438] -> signal; minus biodeg_40nan -0.021 [-0.044, +0.002]; H num_rank +0.412 [+0.399, +0.426] -> signal; minus biodeg_40nan -0.025 [-0.042, -0.007]
- `biodeg_probit2p0_40nan`: K num_rank +0.361 [+0.350, +0.373] -> signal; minus biodeg_40nan -0.080 [-0.093, -0.067]; H num_rank +0.361 [+0.338, +0.385] -> signal; minus biodeg_40nan -0.076 [-0.098, -0.054]
- `vehicle_probit_40nan`: K num_rank +0.130 [+0.106, +0.155] -> signal; minus vehicle_40nan -0.014 [-0.039, +0.012]; H num_rank +0.119 [+0.096, +0.143] -> signal; minus vehicle_40nan -0.027 [-0.050, -0.003]
- `kc2_probit_40nan`: K num_rank +0.536 [+0.503, +0.569] -> signal; minus kc2_40nan +0.072 [+0.029, +0.116]; H num_rank +0.533 [+0.509, +0.556] -> signal; minus kc2_40nan +0.082 [+0.042, +0.121]
- `spambase_probit_40nan`: K num_rank +0.471 [+0.459, +0.484] -> signal; minus spambase_40nan -0.023 [-0.046, +0.000]; H num_rank +0.467 [+0.443, +0.491] -> signal; minus spambase_40nan -0.006 [-0.035, +0.023]
- `biodeg_marf_40nan`: K num_rank +0.469 [+0.452, +0.486] -> signal; minus biodeg_marfctl_40nan -0.006 [-0.030, +0.018]; H num_rank +0.476 [+0.462, +0.490] -> signal; minus biodeg_marfctl_40nan +0.003 [-0.014, +0.020]
- `kc2_marf_40nan`: K num_rank +0.448 [+0.408, +0.488] -> signal; minus kc2_marfctl_40nan -0.035 [-0.084, +0.013]; H num_rank +0.418 [+0.369, +0.467] -> signal; minus kc2_marfctl_40nan -0.068 [-0.109, -0.027]
- `vehicle_marf_40nan`: K num_rank +0.109 [+0.088, +0.131] -> signal; minus vehicle_marfctl_40nan -0.020 [-0.053, +0.012]; H num_rank +0.100 [+0.073, +0.127] -> signal; minus vehicle_marfctl_40nan -0.029 [-0.060, +0.001]
- `spambase_marf_40nan`: K num_rank +0.521 [+0.501, +0.541] -> signal; minus spambase_marfctl_40nan -0.012 [-0.044, +0.021]; H num_rank +0.523 [+0.515, +0.532] -> signal; minus spambase_marfctl_40nan +0.000 [-0.021, +0.021]
- `credit-g_marf_40nan`: K num_rank +0.267 [+0.230, +0.305] -> signal; minus credit-g_marfctl_40nan +0.050 [-0.007, +0.106]; H num_rank +0.237 [+0.204, +0.269] -> signal; minus credit-g_marfctl_40nan +0.041 [-0.025, +0.107]

- K: signal on 12 of 12, against on none
- H: signal on 12 of 12, against on none

Primary: signal on 12 (K) and 12 (H) of 12, against under K on none -> supported (fold-pair t, seed agreement)
```
