# E99 on fresh seeds: TRIDENT against MLP and SAINT at 40–80nan (E111; pre-registered 2026-11-02 00:01 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E99 ([08](08-deep-baselines.md)) found TRIDENT leading MLP, SAINT and TabNet at 40–80nan
  on seeds 42/7/13; the MLP margin is small (+0.013, +0.008, a tie at 80nan). A replication on seeds
  E99 never used, without re-tuning either side, tests whether the reading depends on those seeds.
  TabNet is left out: its margin is large (+0.09 to +0.11) and E106 already checked it.
- **The user's go:** "voce pode colocar mais coisas pra rodar nsa duas maquinas" (2026-11-01, before
  23:59 GMT-3; back at 09:00).
- **Code:** `stage4_trident.py eval <variant> <seed> <experiment>` (an optional experiment tag) and
  `run_e111_trident_queue.sh`; `deep_baselines.py --study e111` (MLP and SAINT, E99's configurations
  from `e99_json/`, copied unread, E99's recipe, no search).

## Design (fixed before the first cell)

- **Variants:** the 27 at 40, 60 and 80nan. **Seeds:** 5151, 6262, 7373 (never used), ten folds each
  (`build_folds(frame, 'class', 10, seed)`).
- **TRIDENT:** each variant's E98 best configuration, evaluated at the three seeds, MLflow tag
  `experiment=step-zero-e111-2026-11-02`; on the machine holding its E98 study (electricity_40nan on
  gorgona8, the rest on medusa1).
- **MLP and SAINT:** E99's best configuration per variant, evaluated at the three seeds; 54 units
  dealt by cost between the machines (gorgona8's start when E108 ends).
- **The cap:** no item starts after 2026-11-02 08:30 GMT-3 (STOP files on both machines). Missing items
  are reported missing and their levels incomplete.

## Measures and rule, fixed now (`deep_baselines.py --study e111 --tally`)

- E99's: per level and baseline, TRIDENT − baseline test macro-F1 over 270 fold pairs, the project's
  rule. TRIDENT leads a baseline if higher at 2 or more of the 3 levels and lower at none.
- **Replicated** if TRIDENT leads both MLP and SAINT; **not replicated** otherwise.

## Outcome

(open)
