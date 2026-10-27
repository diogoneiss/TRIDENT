# ADR 0031: An opt-in offline hybrid output: KC and TabPFN, per column

- **Status:** **proposed on 2026-10-27, 12:44 GMT-3** for the user's decision. The user asked for it to be
  recorded as T39 ("Gosto da ideia de propor uma ADR", 2026-10-27) and started it ("vamos começar o
  t39", 2026-10-27).
- **Builds on:**
  - ADR 0027, the calibrated output (KC);
  - ADR 0028, TabPFN v2 as an offline member of the strong tier;
  - ADR 0024, H as the deployable headline.
  - It amends none of them.

## Context

- **The method:** per column, the hybrid blends today's default fill (KC) with TabPFN v2's, refitted
  on the same training fold.
  - **Numerical column:** the model's weight w is chosen from {0, 0.25, 0.5, 0.75, 1}, giving
    w·KC + (1 − w)·TabPFN.
  - **Categorical column:** it picks either model's category.
  - The choice for fold k is made on the other four folds' induced cells (least squared error;
    more right). Ties go to the equal blend.
  - It is offline: `hybrid_blend.py blend` over a cross-validation run's fold ledgers, as E60, E69
    and E74 ran it. No `src/` code takes part.
- **E69** ([imputation-protocol/24](../tickets/imputation-protocol/24-blend-weights.md), mixed,
  descriptive): with K as the model side, it beats both K and TabPFN on 13 of 21 variants (the
  equal blend: 10). On the 10 hard variants it is better than TabPFN on 3 (the numerical 80nan
  ties) and worse on 1 (kr-vs-kp_60nan, +0.0016).
- **E74** ([27](../tickets/imputation-protocol/27-hybrid-output.md), pre-registered, **mixed**): with
  KC as the model side, on fresh seeds 1212/2323/3434 and the same 10 hard variants.
  - **Against KC:** better on 7 of 10 and never worse. Its rule needed 8.
  - **Against TabPFN:** worse on 2, kr-vs-kp at 40 and 60nan (+0.005 and +0.004). The rule allowed
    1. Equivalent on kr-vs-kp_80nan and credit-g at all three rates.
  - **Against both:** it beats both on the same three numerical 80nan variants as E69:
    electricity, vehicle and biodeg at 80nan, by 0.002–0.013 over TabPFN.
  - **The weights:** TabPFN's mean weight is 0.65–0.68 on those three. A column's choice holds on
    54–100% of its folds.
- **The caveat this proposal must carry: most of the hybrid's gain over KC is TabPFN's own.**
  - On E74's 10 variants, TabPFN alone is better than KC on 8 and no different on 2.
  - Its own gain over TabPFN shows only on the three numerical 80nan variants.
  - **Why not TabPFN alone?** Across the broader sets TabPFN is not uniformly better.
    - E57: KC − TabPFN on the 15 confirmation variants is better 6, no difference 4, equivalent 1,
      worse 4.
    - E58 adds every other seed set, MNAR included, with no uniform winner.
    - The hybrid hedges per column between two imputers that win on different tables. It does not
      claim to be a better imputer than TabPFN.

## Decision (proposed)

1. **The hybrid is an opt-in, offline output,** in exactly the shape E74 measured.
   - It is `hybrid_blend.py blend` over a cross-validation run's ledgers, with E69's grid, E69's
     selection rule and E74's model side (KC).
   - It needs the `tabpfn` extra and one TabPFN refit per column a fold, as ADR 0028.
   - No flag, no `src/` change, no tag and no backfill. Its fills and weights are files beside the
     run, as E74 wrote them, never logged into the store.
2. **The headline stays KC** (ADR 0027), and nothing changes by default.
3. **Its information class is KC's.** Its weights read induced truth from other folds through the
   complete table, as KC's calibration and K's checkpoint (ADR 0015) read the validation rows'.
   - So it is a *benchmark* output, exactly as deployable as KC: not deployable without a complete
     table.
   - A report that uses it says so, beside ADR 0024's H.
4. **What a report may claim:**
   - It is never worse than KC on the 10 hard variants (E74).
   - It is the best of the three on numerical cells at 80nan (E69 and E74, the same three
     variants).
   - It reaches TabPFN's level on the categorical tables, a little behind on kr-vs-kp at 40/60nan.
   - It states that E74's pre-registered rule was not met, and that TabPFN alone carries most of
     the gain.
5. **Scope:** MCAR only. E69's 21 variants at 707/808/909 and 404/505/606 (K side). E74's 10 hard
   variants at 1212/2323/3434 (KC side). MAR and MNAR are untested.

## Considered options

- **Keep it descriptive (no ADR).** This is what E74's mixed reading allows by default. Proposed
  against it: the hybrid is never worse than KC on any measured variant, and it is the only output
  that beats TabPFN where TabPFN beats the model.
- **An in-run flag, with weights from the fold's own validation gaps.** These are the cells ADR 0027
  fits KC on, so it would run inside every training run.
  - E74 never tested that population. Proposing it needs a pre-registered check, E82: do weights
    chosen on validation gaps reproduce E74's reading on E74's cells?
  - That needs a retraining with an in-place reader (about 4 h of GPU) plus the TabPFN refits.
  - It also adds a `src/` change behind a flag.
  - Deferred to the user.
- **A deployable hybrid, with weights from observed cells (the masked population).** It would pair
  with H, or with E47's masked calibration, not with KC.
  - Deferred as its own study. The fold ledger holds few masked cells (228 a fold on biodeg_80nan
    over 41 columns, about 5 a column), and its masked fill is the raw checkpoint, not KC.
- **TabPFN alone as the output.** Rejected. It is an offline baseline (ADR 0028), and it loses to
  KC on 6 of 15 confirmation variants (E57).
- **On by default.** Rejected. It needs the `tabpfn` extra and refits, and E74's rule was not met.

## Consequences

- **The thesis can say:** an offline per-column hedge with TabPFN is never worse than the default
  on the hard variants measured, and beats both on numerical cells at 80nan. It is a benchmark
  output, like KC.
- **It cannot say:** that the hybrid is a better imputer than TabPFN in general, or that it is
  deployable.
- **Follow-ups, each its own pre-registered study if the user wants it:**
  - E82, the in-run validation-gap weights;
  - the deployable masked-weight hybrid with H;
  - MAR and MNAR.
