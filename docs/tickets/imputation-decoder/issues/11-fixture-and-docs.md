# 11. Regression fixture and documentation

Status: ready-for-agent
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

- [ ] Fixture generated from the ADR decision 14 configuration (`DIM` 16, `HIDDEN_DIM` 8,
      `HEADS` 4, `LAYERS` 1, `DIM_FEED` 16, `DROPOUT` 0.1, `EPOCHS_PRE` 2, `EPOCHS_DECODE` 2,
      `BATCH` 64, `LAMBDA_NUM` 1.0, `EVAL_MASK_RATE` 0.2, `cv_folds` 2, seed 42, tracking
      disabled), run three times, `observed_max_deviation` recorded, `tolerance` set well
      above it, environment block recorded, regeneration command in the test docstring.
- [ ] Pins `impute_score`, `rmse_num_z`, `acc_cat` for both populations, per fold and mean.
- [ ] README: `--task`, `--score_null_path`, the six new keys with the `PROB_MASCARA` /
      `EVAL_MASK_RATE` pairing, `task` and `is_optuna` filters in the MLflow comparison
      section, the artifact inventory, the `--lr_scheduler cosine` recommendation.
- [ ] `docs/ARCHITECTURE.md`: a `TridentDecoder` section with theory and implementation
      diagrams; the decode stage in the orchestration diagram.
- [ ] `AGENTS.md` names both regression fixtures. `docs/BACKLOG.md` marks B3 fixed.
- [ ] Full verification: `pytest -m "not integration"`, `pytest -m integration`,
      `graphify update .`.
- [ ] ADR 0004's fourteen decisions each map to a test or a documented manual check.

## Constraints

- The degenerate compositions (all-numerical, all-categorical, zero baseline) are covered
  by ticket 03's unit tests, not by extra integration runs.
- Do not tighten the tolerance below the existing fixture's 0.01.

## Comments
