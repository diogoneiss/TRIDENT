# Step zero proper: does TRIDENT's classification lead survive a fair rerun? (E98, H1c; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell. A cost probe ran first on
medusa1 GPU1, with MLflow off and the scores unread: one 10-fold cell a table at 40nan, at 40/40
epochs on the old sweep's other hyperparameters. It took about 15 min for the nine tables
(electricity 412 s, kc2 10 s).

- **Why:** H1c, the thesis's claim that TRIDENT's lead from 40% missingness survives a fair
  rerun: TRIDENT tuned on validation only, against the best tuned GBDT, with the same tuning budget
  and paired differences ([plan](../../plans/2026-10-23-classification-step-zero.md), "Why" and
  Stage 4).
  - E97 (Stage 1) found untuned native HGB ahead of the untuned old-sweep TRIDENT by 1.5–3.3 points
    at 20–80nan.
- **The user's go:** "Ok, prepare para executar a partir de agora" (2026-10-29, about 22:20 GMT-3).
  That answered the proposal: the paper-budget arm tuned, the fork-budget arm as an untuned
  reference, and a compute cap of two nights (decision 6).
- **Settled decisions applied** (the plan's "Settled" section): `--cv_folds 10`, 20 trials,
  `cosine_legacy`, ADR 0029's validation ranking, no XGBoost, no deep baselines, today's eager path,
  and a split between the servers by seed (ADR 0035).
- **Code:** no change to `src/` beyond ADR 0029 (629aef4).
  - `scripts/experiments/stage4_trident.py` (`study`, `eval`, `--todo`) and
    `run_stage4_queue.sh` run the TRIDENT arm.
  - The tuned-GBDT script `scripts/experiments/stage4_gbdt.py` is written after this commit, to the
    specification below, and only it runs those arms.

## Design (fixed before the first cell)

- **Variants:** all 45 (9 tables × 00, 20, 40, 60 and 80nan, MCAR).
  - Every variant is tuned. At the measured cost this fits the cap, so no covering subset and no
    borrowed configurations are needed.
- **Folds:** `build_folds(frame, 'class', 10, seed)`, 80/10/10, the folds E87 and E92 used.
  - Tuning uses seed 42's ten folds.
  - Evaluation uses seeds 42, 7 and 13. Seed 42's test folds were never read in tuning, which ranks
    on validation only.
- **The TRIDENT arm (tuned, the paper's budget):**
  - an ADR 0029 study per variant: 20 TPE trials (sampler seed 42) of the `full` profile, whose
    epoch ranges are 20–60 for pre-training and fine-tuning, the paper's 40/40 budget;
  - each trial is scored as the mean `val_f1_macro` at the kept epoch over the ten folds, with
    `cosine_legacy`;
  - the study's retrain on seed 42 is the seed-42 evaluation, and seeds 7 and 13 rerun the best
    configuration (`eval`).
  - Each variant's study and its evaluations run on one machine.
- **The fork-budget reference (untuned):** E87's arm A, at seeds 42, 7 and 13 and the same ten folds
  (300/150 epochs, the old sweep's hyperparameters). It is not tuned: tuning it would cost about
  100 GPU-hours, more than the cap.
- **The GBDT bar (tuned):** native-NaN HistGradientBoosting and LightGBM, as E97's native arms:
  training-split vocabulary codes, NaN for gaps and unseen values, `class_weight='balanced'`.
  - 20 TPE trials each (sampler seed 42), on the same objective as TRIDENT: the mean validation
    macro-F1 over seed 42's ten folds. Each fold's validation split drives early stopping, with
    patience 20 and at most 500 rounds.
  - **HGB space:** `learning_rate` log 0.01–0.3, `max_leaf_nodes` int log 8–128, `min_samples_leaf`
    int log 5–100, `l2_regularization` log 1e-6–10, `max_features` 0.5–1.0.
  - **LightGBM space:** `learning_rate` log 0.01–0.3, `num_leaves` int log 8–128,
    `min_child_samples` int log 5–100, `reg_lambda` log 1e-6–10, `colsample_bytree` 0.5–1.0,
    `subsample` 0.5–1.0 (with `subsample_freq` 1).
  - **The bar a variant uses** is whichever of the two tuned GBDTs has the higher mean validation
    macro-F1 at seed 42, chosen before any test number is read.
  - **Descriptive:** missForest + indicators + HGB (E97's strongest impute-then-learn arm), tuned
    the same way with the imputer fit once a fold.
- **The strong tier, read beside the bar** (ADR 0032): E92's TabPFN, plain and balanced, at seeds
  42/7/13. It covers 8 tables; letter is out.
- **Running:**
  - TRIDENT: six queues (`run_stage4_queue.sh <n> 6`). Queues 1–3 run on gorgona8, 4–6 on medusa1
    GPU1; another user holds most of medusa1's GPU0. MLflow logs through the leader's server.
  - GBDTs: CPU on gorgona8.
  - State: `trident-2026-10-30-step-zero/` on each machine.
  - **The cap:** no item starts after 2026-10-31 08:00 GMT-3. A variant missing then is reported
    missing, and the levels it belongs to are partial.

## Measures and rule, fixed now

- **The measure:** test macro-F1, paired by fold and seed. Per level, the pairs pool over the 9
  tables, one array per seed: 270 pairs. The project's rule applies (95% t-interval, seed
  agreement).
- **H1c, at 40, 60 and 80nan:** tuned TRIDENT − the tuned GBDT bar.
  - **Refuted** if at any of the three rates the bar is higher (interval below zero, every seed
    below).
  - **Supported** if TRIDENT is higher (interval above zero, every seed above) at all three.
  - **Unconfirmed, not refuted,** otherwise: an interval spans zero somewhere, and no rate favours
    the bar.
- **Descriptive:**
  - the same difference at 00 and 20nan;
  - tuned TRIDENT − the fork-budget reference (did tuning help?);
  - tuned TRIDENT − TabPFN, plain and balanced (the strong tier);
  - the impute-then-learn arm;
  - per-table means and the kept epochs, with collapsed folds (a kept epoch below 5) counted.

## Outcome

(open)
