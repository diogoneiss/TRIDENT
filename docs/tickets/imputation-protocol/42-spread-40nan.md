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

(open)
