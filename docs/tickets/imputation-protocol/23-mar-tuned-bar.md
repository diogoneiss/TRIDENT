# The tuned bar on the MAR folds: D4 under MAR (E68; pre-registered 2026-10-23)

This ticket was written and committed before the first pair.

- **Why:** this is T34, accepted by the user on 2026-10-22.
  - ADR 0022 decision 5: every study that claims against the bar runs E42's driver on its own
    folds, and D4 counts a win only against both bars.
  - E63 and E64 read the untuned bar only. E64 found (H − bar), MAR minus control, worse on biodeg,
    kc2 and vehicle at 60nan, so "the standing against the bar under MAR" has no D4 reading yet.
- **The user's go:** "vamos na parte de imputacao agora entao" (2026-10-23, about 17:50 GMT-3).
- **The code, with nothing in `src/`:**
  - E42's driver `scripts/experiments/baseline_budget_run.py`, unchanged: the weak tier's families
    knn, hgb and lgbm, threads capped at 4;
  - the new reader `scripts/experiments/mar_tuned_bar_report.py`.

## Design (fixed before the first pair)

- **Pairs:**
  - E64's 24 variants (6 tables × 20 and 60nan × MAR and control) at seeds 404/505/606;
  - E63's 8 variants (4 tables at 60nan × MAR and control) at seeds 42/7/13.
  - That makes 96 (variant, seed) pairs. E64's slope dial is left out.
- **What the driver does:**
  - It rebuilds each run's five folds and evaluation mask.
  - It chooses each family's setting on the validation rows' gaps against the complete sibling.
  - It scores the chosen settings on the test split, and writes `tuned_bar` per fold.
  - Nothing goes to the store.
- **Running:**
  - four CPU lanes, beside E67's GPU queues;
  - about 3–5 h (E44's 45 pairs took about 4 h on 4 lanes);
  - state in `/scratch2/diogoneiss/trident-2026-10-23-mar-tuned-bar/`.

## Measures and reading, fixed now (`mar_tuned_bar_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per mechanism, with the 1% SESOI:** H − bar, H − tuned bar, K − bar and K − tuned bar. D4 on H
  and on K counts a win where both calls are better.
- **The mechanism effect, fold-paired:** (H − tuned bar) and (K − tuned bar), MAR minus control.
- **Primary,** over E64's 12 table-rates, (H − tuned bar) MAR minus control:
  - **supported** (MAR costs the model ground against the tuned bar): worse on 4 or more, and better
    on at most 1;
  - **refuted:** worse on at most 1;
  - **mixed:** otherwise.
  - E63's four are secondary.
- **The denominator, as in E63 and E64:** under MAR the mean/mode denominator comes from observed
  values the mechanism biases. So scores are compared across mechanisms only as differences of
  within-variant gaps.
- **What each reading leads to (no default changes):**
  - **refuted:** the thesis's MAR statement holds against both bars.
  - **supported:** the thesis states that under MAR the tuned bar gains on the model, naming where.
  - **mixed:** reported, and the user chooses.
