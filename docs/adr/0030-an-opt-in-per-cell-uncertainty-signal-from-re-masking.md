# ADR 0030: An opt-in per-cell uncertainty signal from re-masking passes

- **Status:** **accepted on 2026-10-26, 22:30 GMT-3** by the user ("ok, aceito"). Proposed 2026-10-25. Amended the same day with E77 (refuted: no
  better intervals) before any decision. Updated 2026-10-26 with E76 (supported: the
  within-column reading replicates, pre-registered). Amended 2026-10-27 with E80's limit, on the
  user's decision ("sim, o adr 30 deve registrar isso"). The user asked for the proposal
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
  all.

## Consequences

- **What the thesis can claim:** the decoder gives, without retraining, a per-cell signal that ranks
  its numerical errors up to 60% missing. The claim cites E75's ρ and the coverage shortfall beside
  it.
- **What it cannot claim:** a better interval. E77 found a constant per-column width narrower at the
  same coverage, so the signal's use is to flag cells (which fills to check first), not to size them.
- **Limits that travel with the claim:**
  - It was tested under MCAR only: E75 on the 15 confirmation variants (707/808/909), E76 on the 21
    development variants (101/202/303). credit-g_60nan shows no signal.
  - The MAR and MNAR families are untested.
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
  - the MNAR family, where an honest signal should grow with the mechanism's tilt.
