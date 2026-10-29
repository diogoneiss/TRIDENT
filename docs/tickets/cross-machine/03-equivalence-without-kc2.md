# Is the imputation shift inside the SESOI once kc2 is left out? (E96; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** E94 ([02](02-equivalence.md)) found classification equivalent across the servers, and
  imputation without a verdict: +0.54% [−0.06%, +1.13%] over 350 fold pairs.
  - The tail came from kc2_20nan, with folds at +0.51, +0.44 and +0.39.
  - Read post hoc, the other five variants give −0.02% [−0.30%, +0.26%] over 250 pairs. That read
    chose its variants after seeing the data, so it changes no rule.
  - This study repeats those five variants on fresh seeds, with the rule fixed in advance.
- **The exclusion is declared, and motivated by E94:** kc2 is left out because E94 showed its folds
  to be heavy-tailed. A positive result covers the tables tested here, not kc2.
- **The user's go:** "rode o mais barato então!" (2026-10-29, about 18:15 GMT-3), choosing this
  version over one with spambase and kr-vs-kp (about 5–6 h).
- **Code:** no change to `src/`.
  - `cross_machine.py --study e96` runs E86's code.
  - `run_cross_machine_e94_queue.sh` takes `STUDY=e96`; E94's default is unchanged.

## Design (fixed before the first cell)

- **MLflow off on both machines,** as E86 and E94.
- **Cells:** imputation only, today's default with five folds. The variants are credit-g at 20 and
  60nan, vehicle at 40 and 80nan, and biodeg_40nan, × seeds 11 to 20: 50 cells and 250 fold pairs a
  machine.
- **Running:** four queues on each machine.
  - medusa1 uses GPU1, while E95 keeps GPU0.
  - About 1 h on gorgona8, by these variants' times in E94.
  - No cell starts after 2026-10-30 02:00 GMT-3.

## Measures and rule, fixed now (`cross_machine.py --study e96 --tally`)

- **The measure:** E86's, per fold, medusa − gorgona8, relative on the default output's induced
  `impute_score`. The SESOI is 1%.
- **Equivalent** (the 95% interval inside ±1%): imputation studies whose tables are among
  credit-g, vehicle and biodeg may split their cells between the machines by seed.
  - Both arms of a pair stay on one machine, and the `host` tag records which.
  - Other tables keep one machine a study until they are checked. That holds above all for kc2.
- **Anything else:** imputation keeps one machine a study.

## Outcome

(open)
