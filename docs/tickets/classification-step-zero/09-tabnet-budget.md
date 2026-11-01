# Was E99's TabNet undertrained? Its tuned configuration with a bigger budget (E106; pre-registered 2026-11-01 11:26 GMT-3)

This ticket was written and committed before the study's first cell. A cost probe came first (below).

- **Why:** in E99 ([08](08-deep-baselines.md)) TabNet was by far the weakest baseline (mean macro-F1
  0.47–0.64, ~0.1 under MLP), within a recipe of at most 200 epochs, patience 20 and batch 1024.
  The obvious objection to E99's TabNet margin is that it was undertrained.
- **The user's go:** "3 e 4 agora então" (2026-11-01, after 05:58 and before 09:22 GMT-3), item 3;
  then, on the probe's cost, "Retreinar sem re-tunar" (before 10:55 GMT-3).
- **The cost probe** (gorgona8, sharing its GPU with E107; seconds only, no score read): one
  mid-space configuration on seed 42's ten folds at 40nan, with the bigger budget. vehicle 81 s,
  letter 3133 s, electricity 5247 s, against E99's probe 21, 717 and 683 s. Kept epochs on letter
  100–167 (E99's TabNet median 31). A full re-tune would cost about 190 GPU-hours.
- **Code:** `deep_baselines.py --study e106`: TabNet only, batch 256, at most 400 epochs, patience
  50, and each variant's configuration read from E99's study (`e99_json/`, copied unread); no
  search.

## Design (fixed before the first cell)

- **Units:** TabNet × the 27 variants at 40, 60 and 80nan. Each trains E99's best TabNet
  configuration for that variant with the bigger budget at seeds 42, 7 and 13 on their ten folds
  (`build_folds(frame, 'class', 10, seed)`), E99's recipe otherwise.
- **Running:** `STUDY=e106 run_deep_baselines_queue.sh` beside E107, dealt by cost (20 units on
  medusa1, 7 on gorgona8). Outputs in `trident-2026-11-01-deep-e106/`; MLflow off. No unit starts
  after 2026-11-02 08:00 GMT-3.

## Measures and rule, fixed now (`deep_baselines.py --study e106 --tally`)

- **Per level:** TRIDENT − TabNet-B, test macro-F1, 270 fold pairs, the project's rule.
- **E99's TabNet reading holds** if TRIDENT leads TabNet-B by E99's rule (higher at 2 or more of the
  3 levels and lower at none); otherwise it does not hold, and E99's TabNet margin is reported as a
  budget artefact.
- **Descriptive:** TabNet-B − TabNet (E99), same folds, and kept epochs.

## Outcome

(open)
