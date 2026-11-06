# Observed cells hidden on tables with natural gaps (E148; pre-registered 2026-11-05 as E150)

Written and committed before either table exists; blocked on [01](01-external-datasets-root.md) and
[02](02-tables.md). The first of three ways to study tables whose gaps have no ground truth (the user's agreement,
2026-11-05); the others are [04](04-pattern-transplant.md) and [05](05-downstream-mortality.md).

- **Why:** every table in the project so far is complete, with gaps cut by a mechanism the project wrote. MIMIC-IV's
  gaps are a clinician's decision not to measure, and no complete sibling exists. The pipeline already has a
  population that needs none: observed cells hidden for scoring (`impute/masked/*`), with the row's natural gaps
  left as `[NULL]` inputs. The masks are drawn on observed cells only (`draw_dynamic_mask`).
- **Hypothesis:** on a clinical table with natural gaps, the decoder's fills of hidden observed cells are better
  than the best baseline of the bar's.

## Design and rule (fixed before the first cell)

- **Tables:** `mimic-cardio_natural` and `mimic-sofa_natural`, 11,082 rows each.
- **Arm:** the imputation defaults with `--decode_checkpoint masked`. The default, `induced` (ADR 0015), needs a
  complete sibling and falls back to the epoch of the lowest loss without one (`decoding.py`, `kept_criterion`);
  `masked` keeps the epoch whose validation mask scores best, the same selection on the population this study
  scores. The baselines of the bar are scored on the same cells by the run itself. The cells also carry
  `--export_gap_fills` (ticket 01): the fills of the natural gaps, which [05](05-downstream-mortality.md) reads.
  It changes no score.
- **Cells:** 2 tables × seeds 42/7/13, five folds each: 6 runs.
- **Primary:** per table, the masked population's `impute_score` minus the best baseline's on the same cells, paired
  per (seed, fold), 15 replicates, plain 95% t interval; Holm over the two tables. **Better** with the upper bound
  below 0, **worse** with the lower bound above 0, otherwise no different.
- **Beside it, descriptive:** the per-column scores of the sparse columns (lactate, troponin and the arterial line;
  bilirubin, PaO2/FiO2 and urine output); the score by a row's count of natural gaps; TabPFN (ADR 0028's tier) if
  the strong-tier script scores this population.
- **What this cannot say:** the hidden cells are a uniform draw among values that were measured. An unmeasured
  lactate is not a measured one (deaths 7.9% against 19.9%), so nothing here says how well the true gaps are filled.
- **Cost:** estimated from the store on 2026-11-06, before the first cell. A default imputation run of the same
  configuration took about 40 minutes on gorgona8 on `pendigits` (10,992 rows, 16 columns; means of 38 to 43
  minutes by level, 75 runs) and about 60 on `letter` (20,000 rows). These cells share the GPU with another
  project's trainers (see Run), so one to two hours a cell was expected. Measured on the first two cells
  (2026-11-06): 33 minutes on the SOFA table and 36 on the cardio one, two cells at a time beside four of the
  other project's trainers.
- **What follows:** no default change and no ADR either way. A first reading on clinical data; a claim needs fresh
  seeds.
- **Read only with all 6 runs done.**

## Run

- **Started 2026-11-06 07:20 GMT-3**, on the user's instruction ("Comece a rodar em paralelo com o kg impute"): the
  cells share gorgona8's GPU with kg-impute's training queue instead of waiting for a free one, as ticket 01's
  item (c) had asked. Nothing in the design changes; only how long a cell takes.
- Runner `scripts/experiments/mimic_run.py` through `c15_queue.sh`, two queues (one table each); state directory
  `/scratch2/diogoneiss/trident-2026-11-06-mimic-e148`. Tracked in MLflow on the leader (medusa1): tag
  `experiment = mimic-natural-gaps-2026-11-06`, arm `M`.
- Read by `scripts/experiments/mimic_report.py e148`, from the state directory's fold tables.
