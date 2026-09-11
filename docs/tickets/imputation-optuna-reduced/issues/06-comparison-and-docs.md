# 06. The comparison and the documentation

Status: ready-for-agent
Blocked by: 05
Plan task: 6. ADR 0005 decision 6 (second half). Wayfinder ticket 06.

## Goal

Twelve 5-fold runs, one promoted and one default per (dataset, variant) pair, compared on
the induced `impute_score` mean and its interval; the table recorded in ADR 0005's status;
the README updated for everything this effort added.

## What runs

`.\imputation_studies.ps1 -Compare`: per pair, `main.py --dataset_name <base>_<variant>
--task imputation --cv_folds 5 --lr_scheduler cosine --seed 42` once with the promoted
file in place and once with it moved aside, restored afterwards; `params.config_source`
tells the two apart in the store.

## Acceptance criteria

- [ ] Twelve parents with `is_optuna = false`, `task = imputation`, `lr_scheduler = cosine`,
      `evaluation_mode = cross_validation`, six with `config_source` naming the promoted
      file and six with `defaults`.
- [ ] A table in ADR 0005's status section, dated: per pair,
      `cv/test/impute/induced/impute_score/mean` with `ci95_lower` / `ci95_upper` and
      `cv/test/impute/masked/impute_score/mean`, promoted against default, and whether the
      promoted interval excludes the default mean.
- [ ] `README.md`: `--search_space`, `--promote_best`, the lookup order, `search_space` and
      `config_source` in "MLflow Cross-Validation Comparisons", and a note that a study's
      `optuna/best_objective_value` is a validation-split score.
- [ ] `docs/BACKLOG.md`: I2 in "Fixed already" (if ticket 03 did not already move it).
- [ ] `graphify update .` run; ADR 0005's seven decisions each have a test or a recorded
      run; commit.

## Constraints

- The protocol is the glossary's *paired single-seed CV protocol*: fold-level comparisons
  are paired; conclusions do not quantify seed variability.
- About two hours of GPU time.

## Comments

- 2026-09-11 (while ticket 05's studies run): the documentation slice landed early in
  `df440b3`, since it does not depend on the results: `--search_space` in the README flag
  list; `search_space`, `optuna/importance/*`, the validation-split objective and
  `config_source` in "MLflow Cross-Validation Comparisons"; a "Tuning it with Optuna"
  subsection under "The Imputation Task". `--promote_best` and the lookup order were
  documented by ticket 03 (`a24dcec`), and I2 moved to "Fixed already" there. Left for
  after the studies: the twelve comparison runs, the table in ADR 0005's status section,
  the specification-coverage walk and `graphify update .`. Note for the comparison: the
  user's own schedule sweep writes shared `<dataset>.json` files under
  `datasets/hiperparams/` while it runs; a "defaults" run that starts while one exists
  for its dataset reads it, and `config_source` on the run says so, so check that param
  before building the table.
