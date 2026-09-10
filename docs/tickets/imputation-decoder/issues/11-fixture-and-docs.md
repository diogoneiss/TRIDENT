# 11. Regression fixture and documentation

Status: done (this commit) on `feat/imputation-task`
Blocked by: 08, 10
Plan task: 11. ADR 0004 decision 14. Wayfinder ticket 14.

## Goal

Pin the imputation task's behaviour on `credit-g_20nan` in the existing fixture's exact
pattern, and bring the documentation in line with ADR 0004.

## Seams under test

- `tests/integration/test_credit_g_imputation_regression.py` (`@pytest.mark.integration`),
  same shape as the vehicle test, reading
  `tests/fixtures/credit-g_20nan_imputation_regression.json`.

## Acceptance criteria

- [x] Fixture generated from the ADR decision 14 configuration (`DIM` 16, `HIDDEN_DIM` 8,
      `HEADS` 4, `LAYERS` 1, `DIM_FEED` 16, `DROPOUT` 0.1, `EPOCHS_PRE` 2, `EPOCHS_DECODE` 2,
      `BATCH` 64, `LAMBDA_NUM` 1.0, `EVAL_MASK_RATE` 0.2, `cv_folds` 2, seed 42, tracking
      disabled), run three times, `observed_max_deviation` recorded, `tolerance` set well
      above it, environment block recorded, regeneration command in the test docstring.
- [x] Pins `impute_score`, `rmse_num_z`, `acc_cat` for both populations, per fold and mean.
- [x] README: `--task`, `--score_null_path`, the six new keys with the `PROB_MASCARA` /
      `EVAL_MASK_RATE` pairing, `task` and `is_optuna` filters in the MLflow comparison
      section, the artifact inventory, the `--lr_scheduler cosine` recommendation.
- [x] `docs/ARCHITECTURE.md`: a `TridentDecoder` section with theory and implementation
      diagrams; the decode stage in the orchestration diagram.
- [x] `AGENTS.md` names both regression fixtures. `docs/BACKLOG.md` marks B3 fixed.
- [x] Full verification: `pytest -m "not integration"`, `pytest -m integration`,
      `graphify update .`.
- [x] ADR 0004's fourteen decisions each map to a test or a documented manual check.

## Constraints

- The degenerate compositions (all-numerical, all-categorical, zero baseline) are covered
  by ticket 03's unit tests, not by extra integration runs.
- Do not tighten the tolerance below the existing fixture's 0.01.

## Comments

- 2026-09-10 (from ticket 06): gather the rest of ticket 12's pre-registered comparison.
  One fold of `credit-g_20nan` put the `[MASK]` path at 0.965 and the `[NULL]` path at
  1.042; the criterion needs a majority of folds on at least two datasets at both 20% and
  60%. Record the verdict in ADR 0004 decision 11.

- 2026-09-10, ticket complete.

  **Fixture.** `tests/fixtures/credit-g_20nan_imputation_regression.json`, generated from
  three runs of the ADR's configuration. `observed_max_deviation` is **0.0**, so the
  imputation path is bit-reproducible exactly as the classification one is; `tolerance`
  stays at the existing fixture's 0.01 so the test survives a different device or torch
  build. It pins `impute_score`, `rmse_num_z` and `acc_cat` for both populations, per fold
  and mean. The scores it records are poor (around 1.4, worse than mean-and-mode) because
  two decode epochs cannot beat a naive imputer; the fixture pins determinism, not quality.

  Regenerate with the block in this ticket's history, or re-derive it from
  `tests/integration/test_credit_g_imputation_regression.py`'s configuration.

  **Ticket 12's pre-registered comparison, measured.** The criterion is **not met**, so
  ticket 03's decision to substitute `[MASK]` stands and now rests on evidence:

  | dataset | folds | `[MASK]` | `[NULL]` |
  |---|---|---|---|
  | `credit-g_20nan` | 2 | 1.008, 1.004 | 1.155, 1.180 |
  | `credit-g_60nan` | 2 | 1.022, 1.121 | 1.227, 1.422 |
  | `kr-vs-kp_20nan` | 2 | 0.984, 0.969 | 1.920, 1.533 |
  | `kr-vs-kp_60nan` | 2 | 1.012, 0.980 | 1.986, 1.640 |

  `[MASK]` won **8 of 8** folds. The margin is widest on all-categorical `kr-vs-kp`, where
  the `[NULL]` path is about twice as bad as filling the mode -- a head never trained at a
  null position has nothing to transfer from. Recorded in ADR 0004 decision 11.

  **Documentation.** README gains a task-selection section, an "Imputation Task" section
  (what is scored, the ranking score, the artifact table), the six new config keys with the
  nominal-versus-realised warning on `EVAL_MASK_RATE`, and the `task` / `is_optuna` filters
  in the MLflow comparison section. `docs/ARCHITECTURE.md` gains a `TridentDecoder` section
  in both theory and implementation flavours plus the task branch in the orchestration
  diagram. `AGENTS.md` names both fixtures and the two tasks. `docs/BACKLOG.md` marks B3
  fixed. Every relative link in the touched documents resolves.

  Unit suite 128 green; **both** integration regressions pass, the vehicle one unedited.
