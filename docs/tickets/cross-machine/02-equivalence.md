# Is the machine's shift inside the SESOI? (E94; pre-registered 2026-10-29)

This ticket was written and committed before the study's first cell.

- **Why:** E86 ([01](01-gorgona8-medusa1.md)) found no verdict. The mean shift between the
  machines was small (+0.59% on imputation, +0.0039 on classification), but 60 and 36 fold pairs
  could not put it inside the SESOI. By its tie-break, each study runs whole on one machine. The
  user wants to split work between the two machines, "principalmente coisas altamente
  paralelizaveis" (2026-10-29, about 14:45 GMT-3). This check has the power to show equivalence,
  if it holds.
- **The user's go:** "sim, vamos" (2026-10-29, about 15:00 GMT-3).
- **Code:** no change to `src/`. `cross_machine.py --study e94` runs E86's code on more cells;
  `run_cross_machine_e94_queue.sh` runs four queues per machine.

## Design (fixed before the first cell)

- **MLflow off on both machines,** as in E86. Nothing reaches the store, and nothing is mirrored.
- **Imputation,** today's default with five folds: credit-g at 20 and 60nan, kc2 at 20 and 60nan,
  vehicle at 40 and 80nan, and biodeg_40nan, × seeds 1 to 10. That is 70 cells and 350 fold pairs.
- **Classification,** E71's arm A with three folds: vehicle at 20 and 40nan, kc2_40nan,
  credit-g_20nan and biodeg_60nan, × seeds 1 to 10. That is 50 cells and 150 fold pairs.
- **Sizing:** E86's per-fold sd (about 0.046 relative on imputation, about 0.03 on classification)
  puts the 95% half-width near 0.005 on imputation and 0.005 on classification. So a shift of
  E86's size would sit inside ±0.01.
- **Running:** 120 cells on each machine, four queues each.
  - medusa1 runs on GPU1, beside E95 on GPU0.
  - gorgona8 starts when E93's refits end.
  - About 1.5–2.5 h a machine.
  - No cell starts after 2026-10-30 02:00 GMT-3. A study cut there is partial.

## Measures and rule, fixed now (`cross_machine.py --study e94 --tally`)

- **The measure is E86's:** per fold, medusa − gorgona8.
  - Imputation: relative, on the default output's induced `impute_score`, SESOI 1%.
  - Classification: macro-F1, SESOI 0.01.
- **The rule, per task:**
  - **equivalent** (the 95% interval inside ±SESOI): later studies may split their cells between
    the machines by seed. Both arms of a pair stay on one machine, and the `host` tag (ADR 0034)
    records which.
  - **better or worse** (a systematic shift): each study runs whole on one machine.
  - **none:** each study runs whole on one machine. E86's tie-break by per-fold size is not
    reused: a fold can move as much as a seed change, and that is the seed noise every study
    already carries.
- **A split is allowed for a task only if that task is equivalent.** The two tasks are read
  separately.
- **Descriptive:** the largest per-fold difference, and the CPU baselines (hgb, knn5).

## Outcome

(open)
