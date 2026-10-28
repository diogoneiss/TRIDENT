# Does the occlusion map hold under MNAR and MAR? (E88; pre-registered 2026-10-28)

This ticket was written and committed before the study's first cell.

- **Why:** the map's claim is "faithful from 20 to 60nan, MCAR, K" (E78, E79, E85). Every one of
  those cells was MCAR. E84 showed that ADR 0030's spread survives MNAR and MAR at 60nan; whether
  the map survives them too is untested.
- **The user's go:** "aprovo! rode os 2 em paralelo" (2026-10-28, about 16:50 GMT-3), for this study
  on gorgona8 and E87 on medusa1.
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e88` runs and reads this study with E78's code. The older studies'
    paths are unchanged: E85's tally printed the same two Primary lines after the edit.
  - `run_occlusion_queue.sh` takes `STUDY=e88`, and it now exports `MLFLOW_TRACKING_URI` (the
    leader's server, ADR 0033) when the shell has none.

## Design (fixed before the first cell)

- **Cells:** E84's 12 shifted variants at 60nan × seeds 101, 202, 303, so 36 cells.
  - **MNAR (self-masking probit):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2 and
    spambase at 0.5.
  - **MAR (`marf`):** biodeg, kc2, vehicle, spambase and credit-g.
  - Today's default, five folds, `--score_remask`, read in place exactly as E78.
- **No synthetic table:** the truth measure (C) needs `synthblock`, which is MCAR by construction,
  so C is not read here. The tally skips its gate when no synthetic variant is registered.
- **Running:** four GPU queues on gorgona8 (`STUDY=e88`), MLflow tag
  `experiment=occlusion-mechanisms-2026-10-28`, logged through the leader's server; state in
  `/scratch2/diogoneiss/trident-2026-10-28-occlusion-mechanisms/`. About 3–4 h. Each cell runs on
  gorgona8, as E79's MCAR twins did (E86: a study stays on one machine).
- **Smoke:** one cell runs first, and the queues start only after it lands in the leader's store
  with `host=gorgona8`. It counts.

## Measures and rule, fixed now (`occlusion_map.py --study e88 --tally`)

- **Primary, B (deletion faithfulness):** E78's D on top 3. Per variant, over 15 folds:
  - faithful if the 95% interval lies above the SESOI (0.01) with every seed above 0;
  - not faithful if it lies below the SESOI;
  - inconclusive otherwise.
- **The call:** E79's 7 and 4 of 9, scaled to 12.
  - **supported** if faithful on 10 or more of 12 and not faithful on none;
  - **refuted** if faithful on 5 or fewer;
  - **mixed** otherwise.
- **Descriptive:**
  - D top 1, the seed stability of the map, attention against occlusion, |Spearman| against
    occlusion, and the top sources, as E78 prints them;
  - D on each shifted variant against its MCAR twin `<table>_60nan` at the same seeds (E79's
    cells, also on gorgona8);
  - K against E84's run of the same cell. E84's first 40 cells ran under torch 2.5.1 with
    `--score_checkpoints`, so a difference there is expected and not read.
- **Stale reads:** `--todo` reads the follower's copy of the store, up to 10 minutes old, so a
  relaunch within that window can deal a just-finished cell again. That is harmless: the second
  run is a duplicate parent, and the tally takes one.

## Outcome

(open)
