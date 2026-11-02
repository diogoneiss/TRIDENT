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
- **What follows:** supported → extending decision 7 to 20nan is the user's decision.

## Outcome

(open)
