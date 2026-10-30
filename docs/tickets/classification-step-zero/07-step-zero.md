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

- **Amendment 2026-10-29, 23:45 GMT-3, before any result was read:** helper queues on medusa1
  GPU0.
  - The other user's job on GPU0 ended, so three helper queues (`run_stage4_helper.sh`) now take
    medusa1's items from the end of its lists.
  - `stage4_trident.py` claims each item atomically before it runs. The six items already running
    were claimed by hand, so no item runs twice. A study that is finished is skipped.
  - Each variant's study and evaluations still run on medusa1 or gorgona8 alone, never across both.
    The design and the cap are unchanged.

- **Amendment 2026-10-30, 06:52 GMT-3, before any result was read:** gorgona8's unstarted variants
  move to medusa1.
  - **Why:** after 8 h, gorgona8's three electricity studies are at 16–17 of 20 trials (about 30 min a
    trial on the 3090 Ti, shared three ways). Its 21 other variants would not fit the cap. medusa1 has
    finished most of its own and has an idle GPU.
  - **What moves:** each of those 21 variants, whole (study and both evaluations). They are claimed on
    gorgona8 so its queues skip them, and run on medusa1 by `run_stage4_variants.sh`. That runner also
    picks up medusa1's evaluations left behind when a helper ran the study. Each variant still runs on
    one machine.
  - **kc2_00nan** ended up with a study on both machines. The smoke ran on gorgona8, but the deal gave
    the variant to medusa1, whose queue ran it again. **medusa1's study and its two evaluations are the
    ones read;** the gorgona8 smoke is set aside, unread, since its evaluations would have had to cross
    machines.
  - Electricity at 00, 20 and 40nan stays on gorgona8. The design, the rule and the cap are unchanged.

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

**Written 2026-10-30 15:59 GMT-3. Refuted: the tuned GBDT bar is higher than tuned TRIDENT at 40, 60 and 80nan**
(interval below zero and every seed below at each), and at 00 and 20nan too. Tuned on validation
alone with the same 20-trial budget, TRIDENT trails the bar by 0.024–0.032 macro-F1 from 40nan up.

- **Complete:** all 45 variants tuned and evaluated at seeds 42/7/13, inside the cap (the last cell
  ended 2026-10-30 15:05 GMT-3). Every level has 270 fold pairs against the bar and 240 against
  TabPFN (8 tables). No training cell crashed.
  - The `FAILED` lines in the queue logs (42 on gorgona8, 4 on medusa1) are all evaluations that
    exited at once with "no finished study yet": on gorgona8, evaluations of the variants that
    moved to medusa1; on medusa1, evaluations of pendigits_20nan and letter_80nan tried before
    their studies ended. Each of those ran later on its own machine.
  - **Read:** gorgona8 for electricity 00/20/40nan; medusa1 for the other 42 variants, kc2_00nan
    included (the tally prefers medusa1's copy; the gorgona8 smoke stays unread).
  - Pairing assumes fold k is the same split in every arm. All come from
    `build_folds(frame, 'class', 10, seed)` at the same seeds, as in E87 and E92.
- **Process notes, none touching the design or the rule:**
  - letter_80nan's two evaluations ran at 14:56 on two sweep lanes on medusa1 GPU0, launched by
    hand before the end chain would have (the sweep of the amendment; claims kept them single).
  - The end chain never left its first wait: `grep -c` exits 1 at zero matches, so `|| echo 1`
    read "busy" forever. It was stopped, and the copy and the tally ran by hand at 15:57.
  - The tally ran twice. After the first run, its per-table means (pre-registered as
    descriptive) were added to the script; the second run printed the same level means, the same
    differences and the same Primary line. The tally below is the second run, unedited.
- **Descriptive:**
  - **Did tuning help?** Tuned TRIDENT − the untuned fork budget: no verdict at 20–80nan
    (−0.001 to +0.001), slightly below at 00nan (−0.004 [−0.008, −0.000]). The gap to the GBDT is
    not a budget artifact: E97's untuned gap (+0.015 to +0.033) stays at +0.017 to +0.032 with both
    sides tuned.
  - **The strong tier:** TRIDENT trails TabPFN at every level, by 0.010–0.020 plain and
    0.020–0.035 balanced (as E83 and E92).
  - **missForest + indicators + HGB** is above the bar at 20 and 40nan and below it at 00, 60 and
    80nan; TRIDENT trails it at every level.
  - **Per table:** TRIDENT's mean is above the bar's only on kc2 at 00/20nan, credit-g at 40nan and
    kr-vs-kp at 60nan. The largest gap is electricity at 00nan (0.833 against 0.942).
  - **Kept epochs:** median 29 over 1350 folds; 15 collapsed (kept epoch below 5).
- **What follows is the user's to decide:** what H1c becomes in the thesis, and the deep-baseline
  amendment agreed on 2026-10-29. No ADR, plan or report was changed.
- **The user's decision (2026-10-30 16:37 GMT-3):** the GBDT bar was an internal sanity check. It stays in these
  records and is not reported in the thesis or anywhere outside them ("isso foi apenas um sanity
  check, é sabido que ele performaria pior").

### The tally, unedited (`stage4_tally.py`, 2026-10-30 15:59 GMT-3)

```
## 00nan · TRIDENT mean 0.8594, bar mean 0.8764 · bar per table: biodeg hgb, credit-g hgb, electricity hgb, kc2 lgbm, kr-vs-kp lgbm, letter lgbm, pendigits lgbm, spambase lgbm, vehicle hgb
- per table, TRIDENT/bar mean over seeds: biodeg 0.838/0.849, credit-g 0.684/0.695, electricity 0.833/0.942, kc2 0.738/0.719, kr-vs-kp 0.994/0.996, letter 0.964/0.972, pendigits 0.993/0.994, spambase 0.936/0.955, vehicle 0.755/0.766
- TRIDENT − tuned GBDT bar: -0.0170 [-0.0227, -0.0113] seeds -0.0192, -0.0177, -0.0140, 270 fold pairs -> the other higher
- TRIDENT − missForest+ind+HGB: -0.0156 [-0.0216, -0.0096] seeds -0.0170, -0.0180, -0.0118, 270 fold pairs -> the other higher
- TRIDENT − fork budget (E87 A): -0.0041 [-0.0080, -0.0001] seeds -0.0035, -0.0067, -0.0020, 270 fold pairs -> the other higher
- TRIDENT − TabPFN: -0.0198 [-0.0269, -0.0127] seeds -0.0220, -0.0210, -0.0165, 240 fold pairs -> the other higher
- TRIDENT − TabPFN balanced: -0.0224 [-0.0280, -0.0167] seeds -0.0255, -0.0218, -0.0198, 240 fold pairs -> the other higher

## 20nan · TRIDENT mean 0.8038, bar mean 0.8285 · bar per table: biodeg lgbm, credit-g lgbm, electricity hgb, kc2 lgbm, kr-vs-kp hgb, letter lgbm, pendigits lgbm, spambase lgbm, vehicle lgbm
- per table, TRIDENT/bar mean over seeds: biodeg 0.813/0.831, credit-g 0.645/0.666, electricity 0.776/0.846, kc2 0.721/0.706, kr-vs-kp 0.884/0.902, letter 0.845/0.881, pendigits 0.969/0.975, spambase 0.912/0.932, vehicle 0.670/0.717
- TRIDENT − tuned GBDT bar: -0.0247 [-0.0293, -0.0201] seeds -0.0230, -0.0267, -0.0244, 270 fold pairs -> the other higher
- TRIDENT − missForest+ind+HGB: -0.0297 [-0.0349, -0.0244] seeds -0.0254, -0.0315, -0.0320, 270 fold pairs -> the other higher
- TRIDENT − fork budget (E87 A): +0.0004 [-0.0036, +0.0043] seeds +0.0022, +0.0054, -0.0065, 270 fold pairs -> no verdict
- TRIDENT − TabPFN: -0.0100 [-0.0152, -0.0047] seeds -0.0092, -0.0105, -0.0102, 240 fold pairs -> the other higher
- TRIDENT − TabPFN balanced: -0.0214 [-0.0255, -0.0173] seeds -0.0173, -0.0226, -0.0244, 240 fold pairs -> the other higher

## 40nan · TRIDENT mean 0.7509, bar mean 0.7750 · bar per table: biodeg lgbm, credit-g hgb, electricity lgbm, kc2 lgbm, kr-vs-kp lgbm, letter lgbm, pendigits lgbm, spambase hgb, vehicle lgbm
- per table, TRIDENT/bar mean over seeds: biodeg 0.772/0.796, credit-g 0.638/0.620, electricity 0.728/0.775, kc2 0.708/0.717, kr-vs-kp 0.805/0.819, letter 0.683/0.737, pendigits 0.912/0.932, spambase 0.895/0.914, vehicle 0.618/0.665
- TRIDENT − tuned GBDT bar: -0.0241 [-0.0290, -0.0192] seeds -0.0238, -0.0272, -0.0213, 270 fold pairs -> the other higher
- TRIDENT − missForest+ind+HGB: -0.0291 [-0.0346, -0.0236] seeds -0.0285, -0.0308, -0.0279, 270 fold pairs -> the other higher
- TRIDENT − fork budget (E87 A): -0.0008 [-0.0045, +0.0028] seeds -0.0003, +0.0018, -0.0040, 270 fold pairs -> no verdict
- TRIDENT − TabPFN: -0.0107 [-0.0176, -0.0037] seeds -0.0092, -0.0116, -0.0112, 240 fold pairs -> the other higher
- TRIDENT − TabPFN balanced: -0.0196 [-0.0242, -0.0150] seeds -0.0202, -0.0190, -0.0196, 240 fold pairs -> the other higher

## 60nan · TRIDENT mean 0.6714, bar mean 0.7037 · bar per table: biodeg lgbm, credit-g hgb, electricity lgbm, kc2 lgbm, kr-vs-kp hgb, letter lgbm, pendigits lgbm, spambase hgb, vehicle lgbm
- per table, TRIDENT/bar mean over seeds: biodeg 0.748/0.756, credit-g 0.578/0.608, electricity 0.683/0.707, kc2 0.671/0.733, kr-vs-kp 0.710/0.705, letter 0.494/0.537, pendigits 0.793/0.820, spambase 0.873/0.883, vehicle 0.491/0.584
- TRIDENT − tuned GBDT bar: -0.0323 [-0.0386, -0.0260] seeds -0.0263, -0.0359, -0.0347, 270 fold pairs -> the other higher
- TRIDENT − missForest+ind+HGB: -0.0306 [-0.0376, -0.0236] seeds -0.0252, -0.0333, -0.0333, 270 fold pairs -> the other higher
- TRIDENT − fork budget (E87 A): +0.0011 [-0.0036, +0.0059] seeds +0.0050, -0.0026, +0.0011, 270 fold pairs -> no verdict
- TRIDENT − TabPFN: -0.0135 [-0.0226, -0.0044] seeds -0.0077, -0.0178, -0.0150, 240 fold pairs -> the other higher
- TRIDENT − TabPFN balanced: -0.0285 [-0.0352, -0.0218] seeds -0.0242, -0.0311, -0.0302, 240 fold pairs -> the other higher

## 80nan · TRIDENT mean 0.5599, bar mean 0.5893 · bar per table: biodeg hgb, credit-g hgb, electricity hgb, kc2 lgbm, kr-vs-kp lgbm, letter lgbm, pendigits lgbm, spambase lgbm, vehicle lgbm
- per table, TRIDENT/bar mean over seeds: biodeg 0.659/0.708, credit-g 0.510/0.547, electricity 0.604/0.623, kc2 0.671/0.730, kr-vs-kp 0.627/0.629, letter 0.248/0.279, pendigits 0.536/0.560, spambase 0.792/0.802, vehicle 0.392/0.427
- TRIDENT − tuned GBDT bar: -0.0295 [-0.0345, -0.0245] seeds -0.0260, -0.0282, -0.0342, 270 fold pairs -> the other higher
- TRIDENT − missForest+ind+HGB: -0.0218 [-0.0278, -0.0158] seeds -0.0135, -0.0258, -0.0261, 270 fold pairs -> the other higher
- TRIDENT − fork budget (E87 A): +0.0004 [-0.0052, +0.0061] seeds +0.0083, +0.0015, -0.0085, 270 fold pairs -> no verdict
- TRIDENT − TabPFN: -0.0102 [-0.0177, -0.0026] seeds -0.0024, -0.0127, -0.0154, 240 fold pairs -> the other higher
- TRIDENT − TabPFN balanced: -0.0345 [-0.0409, -0.0282] seeds -0.0279, -0.0377, -0.0379, 240 fold pairs -> the other higher

- kept epochs: 1350 folds, median 29, collapsed (below 5): 15

Primary (H1c, 40/60/80nan): 40nan the other higher, 60nan the other higher, 80nan the other higher -> refuted: the bar is higher at 40nan, 60nan, 80nan (fold-pair t, seed agreement)
```
