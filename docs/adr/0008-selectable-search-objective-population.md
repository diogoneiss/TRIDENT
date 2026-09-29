# Let an imputation study rank its trials by the validation split's induced gaps

## Status

Accepted (2026-09-28). Amends [ADR 0005](0005-reduced-optuna-search-for-imputation.md)
decision 3, which fixed the imputation search objective to the masked validation cells,
by making the population a choice; the masked population stays the default, so every
study launched as before ranks as before.

## Context

ADR 0005 scores an imputation search on the validation split so the test split never
chooses hyperparameters, and on the masked population, the only one every variant has.
The headline of a `_20nan`..`_80nan` variant is the induced population, the gaps the
generator cut, scored against the complete `_00nan` sibling. The two are not the same
question, and the review addendum of 2026-09-15 (§8.7) had already named "score the
induced population on validation" as the next protocol step.

The overnight full-profile studies of 2026-09-28
([pre-registration and outcome](../tickets/imputation-optuna-full/01-overnight-full-profile-studies.md))
showed the cost of the mismatch: on `credit-g_80nan` the configuration the masked
objective chose was worse than the defaults on the induced test cells on every fold
(+0.137), and lost to filling the column mean or mode with disjoint intervals.

## Decision

1. **`--search_objective {masked,induced}`**, imputation only, default `masked`. The
   flag is refused at parse time without `--task imputation`, and `induced` is refused
   before a study opens (no MLflow run, no trial) on a variant with no complete sibling,
   where every trial would fail on a missing key. The objective's direction is the
   ranking metric's, minimise.

2. **The decode stage scores the validation rows' own gaps.** When a search asks for its
   validation score and the variant has a complete sibling, the decode stage also scores
   the validation split's gaps against the sibling, through the same code as the induced
   test cells (`_score_induced_missing`, which now takes the rows explicitly), and logs
   `validation/impute/induced/<metric>` beside `validation/impute/masked/<metric>`. It does
   so for every trial whichever objective ranks it: nothing is drawn at random, and a
   study then carries both objectives on every trial for later comparison. No baseline is
   scored on the validation family (ADR 0007 decision 4).

3. **Tag `search_objective`** on the study run and every trial, and the same as a param:
   the metric key the study read (`validation/impute/masked/impute_score`,
   `validation/impute/induced/impute_score`, or `f1_macro` for classification), which is
   what `optuna/objective_value` holds. Dense on Optuna runs.
   `scripts/backfill_search_objective_tag.py` stamps the task's default on every Optuna
   run recorded before this ADR, marks it `search_objective_backfilled=true`, and
   re-mirrors the stamped trees (ADR 0006). **Applied 2026-09-28:** 1030 Optuna runs in 27
   study trees stamped (every one an imputation run, so the masked key), after a copy of
   the store; a second pass finds nothing, and no source or mirror is left untagged.

## Verification

Two trials of the 2026-09-28 `credit-g_80nan` full-profile study (seed 42, cosine) were
rerun on the new code against a scratch store. Under the default
objective, `optuna/objective_value` reproduced the stored study to the last digit
(1.0276916250490287 and 1.0224268590354355): scoring the validation gaps changed nothing
the masked objective reads. Under `--search_objective induced`, the same two trials were
ranked by their induced values (1.0295617249426714 and 1.0350308992851245) and both run
kinds carried the induced key. Both regression fixtures pass unedited.

## Considered options

- **Make `induced` the default.** Rejected: every stored study ranked by the masked cells,
  and a default that changes silently would mix two objectives under one command line.
- **Rank by the induced test cells.** Never: the test split must not choose
  hyperparameters (ADR 0005).
- **A blend of both populations.** Deferred until the two single objectives have been
  compared on the same trials, which decision 2 makes possible from now on.

## Consequences

- An imputation trial on a variant with gaps logs seven more `validation/impute/induced/*`
  keys and pays one more forward pass over the validation gaps.
- Studies recorded before this ADR carry only the masked validation score and cannot be
  re-ranked by the induced one.
- `induced` is unavailable on `_00nan` variants by construction.
