# ADR 0030: An opt-in per-cell uncertainty signal from re-masking passes

- **Status:** **accepted on 2026-10-26, 22:30 GMT-3** by the user ("ok, aceito"). Proposed 2026-10-25. Amended the same day with E77 (refuted: no
  better intervals) before any decision. Updated 2026-10-26 with E76 (supported: the
  within-column reading replicates, pre-registered). Amended 2026-10-27 with E80's limit, on the
  user's decision ("sim, o adr 30 deve registrar isso"). **Amended 2026-10-31 09:56 GMT-3 with E84: the scope extends
  to MAR and self-masking MNAR at 60% missing (decision 7), on the user's decision ("apoio fazer
  o 3.2 e 3.1, nessa ordem", 2026-10-31).** **Amended 2026-10-31 16:56 GMT-3 with E105: decision 7 covers 40nan
  too, on the user's decision ("aprovo, Extenda", 2026-10-31).** **Amended 2026-11-03 09:43 GMT-3 with E110: decision 7 covers
  20nan too, on the user's decision ("sim, de acordo", 2026-11-03).** **Amended 2026-11-06 16:55 GMT-3 with E151:
  an interval for the fill, calibrated per column and per tertile of the row's gaps, without the spread (decision 8),
  on the user's decision ("faça a emenda da 030", 2026-11-06).** The user asked for the proposal
  ("Podemos propor essa ADR", 2026-10-25) on E75's Outcome
  ([imputation-protocol/28](../tickets/imputation-protocol/28-remask-ensemble.md)), T36.
- **Builds on:** ADR 0024 (a claim that must transfer is read on H), ADR 0025 (the offline panel)
  and ADR 0027 (the calibrated output). It amends none of them.

## Context

- **The imputer gives one fill per cell and no uncertainty.** The thesis's clinical framing and
  multiple imputation both need one (T36).
- **`--score_remask`** (5dc83bd) is a diagnostic flag, off by default.
  - It fills each gap from 20 passes that also hide a fresh draw of the row's observed cells, at
    the decode stage's own rate, and writes each cell's `spread` to the fold ledger.
  - The spread is the passes' standard deviation of a number, or 1 − the vote share of a category.
  - It changes no training: with the flag on, every metric and ledger row of a run is identical.
- **E75 pre-registered a bar for proposing this ADR:** the numerical spread must rank the absolute
  errors (Spearman, 95% interval above 0) on 12 or more of 15 variants, under K and under H.
  - The study ran on E44's 15 confirmation variants at seeds 707/808/909, 45 cells.
  - **It met the bar exactly:** 12 of 15 under K and 12 of 15 under H.
  - Pooled ρ was 0.65–0.66 on biodeg_40nan and kc2_40nan, and 0.14–0.37 on pendigits, letter and
    electricity at 40–60nan.
  - It found no signal at 80nan on pendigits, letter and vehicle.
- **A post-hoc check (not pre-registered):** does the spread rank cells, or only columns?
  - It ran within each numerical column (`remask_ensemble.py --within-column`; printout in the
    ticket).
  - **Cells:** it ranks them on 11 of 15 under K and under H.
    - At 40–60nan it holds on all 9 variants, with ρ 0.06–0.48. That is weaker than pooled:
      electricity_40nan drops from 0.36 to 0.21, biodeg_40nan from 0.65 to 0.42.
    - At 80nan it holds only on the small tables: biodeg (0.22) and kc2 (0.26).
  - **Columns:** they carry the rest. On electricity the column-level relation is negative.
- **It is a ranking, not an interval.** At a nominal 90% (fill ± 1.645 spreads), coverage is
  17–66%. The spread is too narrow, more so as missingness grows.
- **E76 replicated the within-column reading, pre-registered**
  ([29](../tickets/imputation-protocol/29-remask-development.md), supported).
  - It ran on the 21 development variants at 101/202/303.
  - It found a signal on 14 of 15 variants up to 60nan, under K and under H, and none against.
    The exception is credit-g_60nan.
  - At 80nan, spambase signals and credit-g does not.
  - On categorical gaps, the single pass's confidence again beats the vote share, on 9 of 9.
- **It does not beat a constant width (E77, [30](../tickets/imputation-protocol/30-remask-conformal.md),
  refuted).** Scaled to 90% coverage with cross-fitting, the spread interval is wider than a
  per-column constant one on 15 of 15 variants. At 40nan it is 9–144% wider. The ranking is real but
  too weak to adapt a width.
- **Categories:** the single pass's own 1 − confidence ranks the misses better than the vote share,
  on all 3 variants with categorical gaps.
- **The averaged fill is not an output.**
  - Its accuracy is mixed: better 4 and worse 4 under K.
  - Its 80nan gains are shrinkage. Dispersion was lower on 15 of 15, and once calibrated it was
    worse than KC on 8.

## Decision

1. **The uncertainty signal is opt-in, through `--score_remask`.** It stays off by default.
   - A run that asks for it carries `spread` on its ledger's `induced_remask` rows, at the run's
     checkpoint.
   - With `--score_checkpoints`, it also carries `spread` on the `induced_remask_loss_checkpoint`
     rows (H).
2. **The fill stays the run's headline:** KC under ADR 0027, K below 40nan. The averaged fill and
   `impute/induced/*remask*` stay diagnostics and are never the output.
3. **For numerical cells, the signal is `spread`, read as a rank.** A higher spread means a larger
   error is more likely.
   - It is never presented as an interval, a standard error or a probability.
   - A claim using it reads it on H as well as K (ADR 0024). E75 met that.
4. **For categorical cells, the signal is the single pass's `1 − confidence`,** which every run's
   ledger already holds. The vote share is not used.
5. **Its scope is up to 60% missing.**
   - At 80nan it is unsupported, unless a study shows it on that table. E75 showed it only on
     biodeg and kc2.
   - A report that uses it states the scope.
6. **No backfill and no tag,** as for the other `--score_*` diagnostics. The ledger columns and
   metric keys mark a run that has it. On acceptance, the flag's help text says it is the
   uncertainty output; no other code changes.
7. **Its scope includes MAR and self-masking MNAR at 20, 40 and 60% missing** (amended 2026-10-31, from
   E84, [imputation-protocol/32](../tickets/imputation-protocol/32-spread-mechanisms.md), supported
   on its full pre-registered scope, at 60nan; extended to 40nan the same day from E105,
   [imputation-protocol/42](../tickets/imputation-protocol/42-spread-40nan.md), supported; and to 20nan on
   2026-11-03 from E110, [imputation-protocol/45](../tickets/imputation-protocol/45-spread-20nan.md),
   supported).
   - E84 ran 12 shifted variants at 60nan, seeds 101/202/303: self-masking MNAR (probit) on biodeg
     at slopes 0.25/0.5/1/2, vehicle, kc2 and spambase; one MAR draw (`marf`) on biodeg, kc2,
     vehicle, spambase and credit-g.
   - The within-column ranking signals on 12 of 12 under K and under H, none against; ρ +0.10 to
     +0.52, lowest lower bound +0.084.
   - **It weakens as the self-masking tilt grows:** on biodeg, shifted minus MCAR is −0.023,
     −0.049, −0.077 and −0.111 at slopes 0.25 to 2.0. A report that uses it under MNAR states this.
   - **At 40nan (E105):** the same 12 variants at 40nan (new data files from the same generators),
     seeds 101/202/303, the MCAR twins and controls trained beside them. A signal on 12 of 12 under
     K and under H, none against; ρ +0.10 to +0.54. On biodeg the ranking weakens with the tilt as
     at 60nan: shifted minus MCAR +0.002, +0.001, −0.021, −0.080 at slopes 0.25 to 2.0.
   - **At 20nan (E110):** the same 12 variants at 20nan, seeds 101/202/303; a signal on 12 of 12 under K
     and H, none against; ρ +0.14 to +0.55. Unlike 40 and 60nan, the ranking does not weaken with the tilt
     on biodeg at 20nan.
   - **Not covered:** other MAR draws and MNAR mechanisms other than self-masking. Decision 5's 60%
     ceiling stands.
8. **An interval for a numerical fill is a split-conformal one, per column and per tertile of the row's gap count,
   and it does not use the spread** (amended 2026-11-06, from E151,
   [imputation-protocol/54](../tickets/imputation-protocol/54-conformal-by-row-gaps.md), Q1 supported, Q2 refuted).
   - **The interval:** fill ± w, w the ⌈(n + 1) × 0.9⌉-th smallest absolute error among the n calibration cells of the
     same column whose row's gap count falls in the same tertile; the tertile edges are cut on the calibration cells.
     A column-and-tertile group with fewer than 30 cells uses its column's width. Reader:
     `scripts/experiments/conformal_row_gaps.py` (arm MC).
   - **Why per tertile:** a single width per column, E77's, keeps 90% overall but under-covers the rows that miss the
     most: 0.81 to 0.87 in the top tertile on pendigits, letter, electricity and vehicle (electricity_60nan 0.812),
     0.92 to 0.94 in the bottom one. Per tertile, the top tertile is at 0.900 to 0.905 on every one of them. E151's
     primary set: better on 11 of 15, worse on none, under K and H alike.
   - **What it costs:** nothing on the large tables ((MC − C) / C −0.022 to +0.001); 1 to 12% wider on the small
     ones (vehicle_40nan +0.027, biodeg_40nan +0.053, kc2_40nan +0.116), post hoc.
   - **Where it is not needed:** biodeg, kc2, spambase and credit-g under MCAR, where the column's width already
     covers every tertile (0.88 to 0.91). The development set read Q1 mixed (7 of 17 better), worse only on
     credit-g_80nan (+0.007 of under-coverage). Under MAR (E84, descriptive) the column's width already covers.
   - **Under self-masking MNAR** (E84 at 60nan, descriptive): better on all 7 variants; kc2_probit_60nan's top tertile
     goes from 0.776 to 0.907.
   - **Decision 3 stands:** the spread is still a rank. Inside a tertile, a spread-scaled interval is wider than the
     constant on 15 of 15 (Q2 refuted), so E77's conclusion extends to binned intervals.
   - **It is an oracle calibration:** the calibration cells are the other test folds' gaps, whose truth a deployed
     model would not have. A report that uses the interval says so. The deployable form needs observed cells hidden
     on purpose in the training or validation rows, which no run writes yet; it is a follow-up with new runs.
   - **No code in the pipeline changes:** the interval is computed offline from a run's ledger, as E151 does.

## Considered options

- **Keep it descriptive (no ADR).** Rejected in this proposal. It met the bar fixed before the run,
  under both readouts, and the within-column check holds at every rate up to 60nan.
- **On by default.** Rejected.
  - As an interval it is not calibrated.
  - Every run would carry four more ledger populations and the passes' cost: +0.3 s a fold on
    vehicle_40nan, 20 forward passes per checkpoint, more on the large tables.
- **The averaged fill as the output.** Rejected on E75 (a): it is worse on pendigits and letter at
  40–60nan, and calibration already takes its 80nan gain.
- **Scaled into intervals.** Rejected on E77: a cross-fitted spread interval is wider than a
  constant one at the same coverage on 15 of 15. A per-column constant interval needs no passes at
  all. Rejected again inside a tertile of the row's gaps on E151 (Q2).
- **One width per column, E77's, as the interval** (2026-11-06). Rejected on E151: it under-covers the rows with the
  most gaps on the large tables and under self-masking MNAR, which decision 8's tertiles fix.

## Consequences

- **What the thesis can claim:** the decoder gives, without retraining, a per-cell signal that ranks
  its numerical errors up to 60% missing. The claim cites E75's ρ and the coverage shortfall beside
  it.
- **What it cannot claim:** a better interval. E77 found a constant per-column width narrower at the
  same coverage, so the signal's use is to flag cells (which fills to check first), not to size them.
- **What the thesis can claim about intervals (decision 8):** a 90% interval for the fill, calibrated per column
  and per tertile of the row's gaps, holds its coverage in the rows that miss the most, where one width per column
  falls to 0.81 to 0.87; stated with its oracle calibration.
- **Limits that travel with the claim:**
  - Under MCAR: E75 on the 15 confirmation variants (707/808/909), E76 on the 21 development
    variants (101/202/303). credit-g_60nan shows no signal.
  - Under MAR and self-masking MNAR: E84 at 60nan, E105 at 40nan and E110 at 20nan (decision 7),
    weakening with the tilt at 40 and 60nan.
  - **It runs the wrong way along a row's missingness** (E80,
    [imputation-interpretability/03](../tickets/imputation-interpretability/03-spread-and-missing-sources.md)).
    - The spread *falls* as the cell's row has more gaps: Spearman −0.04 to −0.47 on all 16 real
      variants at 40 and 60nan.
    - The error *rises* when the cell's top sources (E78's occlusion map) are missing: +0.02 to
      +0.19 on all 16.
    - So a cell whose informative columns are all missing can show a low spread and a large error.
      The passes agree most where the model knows least. A likely cause, not tested: with fewer
      observed cells, the passes have less to hide and vary.
    - **How to use it:** read the spread as a rank *within a column, among rows with similar
      missingness*. A report that flags cells by spread also flags rows whose top sources are
      missing, from E78's map, since the spread does not.
- **Follow-ups, each its own pre-registered study if the user wants it:**
  - the MNAR family, where an honest signal should grow with the mechanism's tilt. **Done by E84
    (decision 7):** it holds, but it weakens, not grows, as the tilt grows.
