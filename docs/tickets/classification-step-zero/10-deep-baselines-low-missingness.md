# The deep baselines at 00 and 20nan, descriptive (E107; pre-registered 2026-11-01 09:22 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E99 ([08](08-deep-baselines.md)) supported TRIDENT leading MLP, SAINT and TabNet at
  40–80nan. Its scope left out 00 and 20nan. Does TRIDENT also lead with few gaps? Descriptive: no
  claim depends on it.
- **The user's go:** "3 e 4 agora então" (2026-11-01, after 05:58 and before 06:10 GMT-3); this is
  item 4.
- **No new code:** `deep_baselines.py --study e107` is E99's driver, recipe and spaces at
  `LEVELS = ("00", "20")`.

## Design (fixed before the first cell)

- **Units:** MLP, SAINT and TabNet × the 18 variants at 00 and 20nan = 54, each E99's (20 TPE trials
  on seed 42's validation macro-F1, evaluations at seeds 42/7/13, ten folds). TRIDENT's folds are
  E98's at those levels.
- **Running:** `STUDY=e107 run_deep_baselines_queue.sh`, the units dealt by cost (41 on medusa1, 13 on
  gorgona8), beside E106. Outputs in `trident-2026-11-01-deep-e107/`; MLflow off.
- **The cap:** no unit starts after 2026-11-02 08:00 GMT-3 (a STOP file on both machines).

## Measures (`deep_baselines.py --study e107 --tally`)

- **Per level and baseline:** TRIDENT − baseline, test macro-F1, 270 fold pairs, the project's rule:
  "TRIDENT higher", "baseline higher" or no verdict. Printed; no primary call.

## Outcome

(open)
