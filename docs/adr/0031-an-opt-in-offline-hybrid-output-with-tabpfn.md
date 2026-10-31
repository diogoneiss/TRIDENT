# ADR 0031: An opt-in offline hybrid output: KC and TabPFN, per column

- **Status:** **accepted on 2026-10-27, 12:51 GMT-3** by the user ("eu apoio a 031"). Proposed 2026-10-27, 12:44 GMT-3.
  Amended 2026-10-27, 19:20 GMT-3 with E82's deployable route, on the user's decision ("show, registre essa
  decisão"). Amended 2026-10-28, 23:10 GMT-3 with E89's evidence, on the user's request ("sim"); no
  decision changes. **Decision 6 (a deployable form) proposed 2026-10-29, 10:27 GMT-3**, on the user's
  request ("sim, proponha"), from E90 and E91, and **accepted 2026-10-29, 10:42 GMT-3** ("aprovo"). **Decision 6's scope extended to 20nan
  2026-10-31 09:57 GMT-3 with E95**, on the user's decision ("apoio fazer o 3.2 e 3.1, nessa ordem", 2026-10-31). The user asked for it to be
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

## Decision

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
6. **Accepted (2026-10-29): a second, deployable form of the hybrid,** opt-in and offline like the
   first, from E89, E90 and E91
   ([imputation-protocol/33](../tickets/imputation-protocol/33-hybrid-selection-rules.md),
   [34](../tickets/imputation-protocol/34-hybrid-confirmation.md)).
   - **What it is:** H (ADR 0024) and TabPFN v2 refit on each training fold, blended with E89's
     `table` rule:
     - one numerical weight per table, from E69's grid, with each column's residuals scaled by its
       training split's sd;
     - one categorical pick per table, model or TabPFN.
     - Both are chosen on the validation split's *observed* cells, hidden as `[MASK]` in 5
       groups and filled by the fold's own H, as E82 reads them.
   - **Its information class is H's.** Nothing in it reads the complete table: the selection
     truths are observed values. So it is deployable, unlike decision 1's form.
   - **Shape today:** the scripts that measured it.
     - `hybrid_confirmation.py train`: E82's in-place selection read, a wrapper around the
       runner, with the random states restored;
     - then `cells` (the TabPFN refits) and the rule.
     - It needs the `tabpfn` extra and no `src/` change. Its fills and weights are files beside
       the run, never logged into the store.
     - A built-in form (a flag that writes the selection cells during training) would be a
       `src/` change and its own decision.
   - **What a report may claim, on fresh seeds 4545/5656/6767 (E90, E91):**
     - never worse than H or TabPFN on any of 17 variants (E82's 10 hard ones, and 7 where the
       equal blend already beat TabPFN);
     - better than H on 14 of 17, and better than TabPFN on 7;
     - on the 10 hard variants: better than TabPFN on 1, equivalent on 4, no verdict on 5.
   - **What it must state:**
     - its rule was chosen on E82's seeds (E89) and confirmed on fresh ones (E90), with
       `shrunk`, E89's primary, not replicating;
     - it is a little worse than the equal blend on 3 variants at 40nan (electricity, kc2,
       pendigits), where one weight per table is coarse.
   - **The headlines do not change:** H stays the deployable headline (ADR 0024) and KC the
     benchmark default (ADR 0027).
   - **Scope:** MCAR, the 17 variants above and the nine tables at 20nan (below), H as the model
     side. MAR and MNAR are untested here; E93 found the hybrid mixed off MCAR, so the scope stays
     MCAR.
   - **At 20nan (amended 2026-10-31, from E95,
     [imputation-protocol/36](../tickets/imputation-protocol/36-hybrid-20nan.md), supported):** the
     nine tables × seeds 4545/5656/6767, E91's rule.
     - never worse than H or TabPFN on any of the 9;
     - better than H on 6 and better than TabPFN on 6 (electricity, letter, pendigits, spambase,
       biodeg, credit-g); no verdict against either on kr-vs-kp, vehicle and kc2;
     - a little worse than the equal blend on letter (+0.002) and pendigits (+0.010), as at 40nan
       where one weight per table is coarse. A report states it.

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
  with H, or with E47's masked calibration, not with KC. *Since 2026-10-29: decision 6 (accepted),
  after E82 (per-column weights, refuted), E89 (rules, development) and E90/E91 (`table`, fresh
  seeds, supported).*
  - Deferred as its own study. The fold ledger holds few masked cells (228 a fold on biodeg_80nan
    over 41 columns, about 5 a column), and its masked fill is the raw checkpoint, not KC.
- **TabPFN alone as the output.** Rejected. It is an offline baseline (ADR 0028), and it loses to
  KC on 6 of 15 confirmation variants (E57).
- **On by default.** Rejected. It needs the `tabpfn` extra and refits, and E74's rule was not met.

## Consequences

- **The thesis can say:** an offline per-column hedge with TabPFN is never worse than the default
  on the hard variants measured, and beats both on numerical cells at 80nan. It is a benchmark
  output, like KC.
- **It cannot say:** that the hybrid is a better imputer than TabPFN in general, or that decision
  1's form is deployable.
- **With decision 6,** the thesis can also say that a deployable hedge (H, TabPFN and one
  weight a table chosen on observed cells) is never worse than either on 17 MCAR variants on fresh
  seeds. It still cannot say the hybrid beats TabPFN in general: on the hard variants it mostly
  matches it.
- **What it is for (the user, on acceptance):** a measure of the method's *viability*: what a
  per-column hedge with TabPFN can reach when the weights are known well.
  - In a real deployment the weights would be found by trying combinations on the data at hand.
    That is the deployable route under Considered options, and the follow-up that answers it.
- **The deployable route was tested and did not reach TabPFN (E82,
  [imputation-protocol/31](../tickets/imputation-protocol/31-deployable-hybrid.md), refuted,
  2026-10-27).**
  - **What it was:** H, which reads no complete table, plus TabPFN, with E69's per-column weights
    chosen on the validation split's *observed* cells hidden in 5 groups. This is "trying weight
    combinations on the data at hand".
  - **Against H:** better on 10 of 10.
  - **Against KC:** better on 7 of 10 and worse on none, without the complete table.
  - **Against TabPFN:** worse on 5 (vehicle_80nan, biodeg_80nan, kr-vs-kp_60/80nan,
    credit-g_60nan), by 0.5–2.3%. Better on 2 (electricity).
  - **Against the oracle** (the same rule with complete-table weights): worse on 6. With the
    oracle's weights, the H-side hybrid is at TabPFN's level (post hoc).
  - **So the selection, not H, is the loss:** it breaks on columns with few selection cells
    (about 20 a column a fold at 80nan on the small tables) and on categorical picks.
- **The decision this records:** the hybrid stays a benchmark output, as decided above. Its
  deployable form is viable against the TRIDENT readouts (H, KC) and not yet against TabPFN.
  - On the 10 hard variants the order is TabPFN ≳ deployable hybrid > KC > H.
  - On the broader sets TabPFN is not uniformly better (E57: KC better on 6 of 15, worse on 4).
- **Follow-ups, each its own pre-registered study if the user wants it:**
  - a deployable selection with more cells, or with fewer weights (one per table, or shrinkage
    toward the equal blend where a column has few cells), which is offline on E82's files;
  - the in-run validation-gap weights (benchmark-class);
  - MAR and MNAR.
- **Since (2026-10-28), evidence only: E89 tested the fewer-weights follow-up**
  ([imputation-protocol/33](../tickets/imputation-protocol/33-hybrid-selection-rules.md),
  supported as a development read).
  - It used E82's own cells and seeds, with TabPFN refit and no training. Four ways of choosing the
    weights were scored offline, none of them reading the complete table:
    - `column`: E82's rule, which reproduces E82;
    - `table`: one numerical weight and one categorical pick per table;
    - `shrunk`: per-column weights pulled toward the equal blend, more where a column has fewer
      than about 50 selection cells;
    - `pooled`: per-column weights chosen on all five folds' validation cells.
  - **`table`:** better than H on 10 of 10, and worse than TabPFN on none. It is better than TabPFN
    on electricity at 40 and 80nan, and equivalent or without verdict on the other 8.
  - **`shrunk`:** better than H on 10 of 10, and worse than TabPFN on 2 (kr-vs-kp at 60 and
    80nan).
  - **`pooled`:** mixed (worse than TabPFN on 3).
  - **What this changes: nothing yet.** The rules were written after reading E82 and scored on its
    seeds, so a deployable form waits for the fresh-seed confirmation with `table` fixed in
    advance (E90, [imputation-protocol/34](../tickets/imputation-protocol/34-hybrid-confirmation.md)).
    E91 checks its cost on variants where the hybrid is not needed.
  - **If E90 holds,** the order on the 10 hard variants becomes deployable hybrid (`table`) ≈
    TabPFN > KC > H, without the complete table. Adding that output is the user's decision.
