# C15: `[NULL]` tokens against imputation + indicator under informative missingness (plan, 2026-11-03)

The protocol is the wiki's `projects/trident/protocol-c15-mnar-prediction.md` (draft of 2026-10-23, owner
decisions of 2026-10-23 and 2026-10-24). This plan fixes what the fork runs; each stage gets its own ticket
under [tickets/c15-mnar-prediction/](../tickets/c15-mnar-prediction/) and an experimentation-log entry before its
first cell. The user chose C15 over C1 on 2026-11-03 ("prefiro começar pelo c15").

## The claim

`[NULL]` tokens predict as well as imputation + indicator under informative missingness (H2, non-inferiority
of A1 against A2 on macro-F1 and log loss, pooled over the primary set P), and the predictor uses the gap only
where it is informative (H2.2). Secondaries: H2-c2 (A1 beats MICE without an indicator), H2-dial, H2b.

## Decisions

From the protocol (owner, 2026-10-23 and 24), unchanged:
- **Repetitions:** short mode `--cv_folds 5` × seed 42; decision mode `--cv_folds 5` × seeds 42/7/13 (70/10/20,
  E45/E46's folds); reserve `--cv_folds 10` × 3 seeds only if confirmation lacks power.
- **H2's rule:** pooled over P's variants with the tables fixed, A1 inferior on no variant as a guard.
- **Readouts:** per-row validation and test probabilities written in-run plus a reloadable checkpoint, with the
  shipped checkpoint rule (lowest class-weighted validation cross-entropy; ADR 0029 decision 5 keeps it).
- **A4:** scikit-learn's `IterativeImputer(sample_posterior=True)`, m = 5, one full run per imputation.
- **Fresh draws:** a new seedable module with per-column streams and an overwrite guard
  (`datasets/mnar_family.py`); `generate_mnar.py`'s files stay as E45/E46 made them.

Taken by the user on 2026-11-03:
- **The pre-training objective:** every C15 arm passes `--pretrain_objective embedding_normalized` explicitly.
  Classification's default does not change, so no ADR and no fixture move. (No classification run has used
  this objective; E112/E114 tested the EMA target, `embedding_ema`.)
- **Scope:** the full core, Stages 0–3, including Stage 2 (A4 on 20 variants, H2-c2, H2.2, H2b). Stage 4 stays
  optional and unapproved.
- **GBDT:** internal only. G2's GBDT is a gate; R1, the native-NaN anchor, if run, never enters a report, the
  meeting page or the thesis.
- **Margins fixed now:** −0.01 macro-F1 and +2% relative log loss, frozen; the variance-only gate is dropped.

## Stages

| Stage | Work | GPU | Decides | Ticket |
|---|---|---|---|---|
| 0 | the F3 and F5 generators; the driving-column table; gate G2 over 19 candidates × 15 cells | no | P (or stop) | [01](../tickets/c15-mnar-prediction/01-stage0.md), E122 |
| 1a | {A1, A2} and A4 on F5 (gate G1), seed 42 of 1b | yes | G1 (or stop) | after Stage 0 |
| 1b | {A1, A2} × (F0 7 + F1 7 + F2 3 + F3 6), decision mode | yes | development estimates | |
| 2 | A4 on F0, F1, F3; A1 on F7 and F6; H2.2's contrasts; the H2b grid | yes | H2-c2, H2.2, H2b (development) | |
| 3 | confirmation, read once: {A1, A2} on fresh draws of P and its MCAR siblings | yes | H2, H2.2 | |

## Fork changes, in the order the stages need them

1. Stage 0: `datasets/mnar_family.py` (F3 outcome-dependent, F5 noise control, seeded probit) and
   `scripts/experiments/c15_stage0.py` (driving columns, G2). No `src/` change.
2. Stage 1: behind flags, off by default, the classification fork's per-row probabilities and log loss, a
   reloadable checkpoint, and A2's input (the value MLP takes the fill and the indicator, 2 → 16 → 128, and
   mask-aware pre-training). `vehicle_00nan`'s fixture must not move; new tags are backfilled as ADR 0034 did.
3. Stage 2: the MICE route for A4.

## Constraints

- Seed counts are fixed in each stage's ticket before its first cell; no optional stopping (G2's 15 cells too).
- Confirmation is read once, on fresh draws from `mnar_family.py` on a never-used seed set.
- Data files are gitignored; the generators and their seeds are the record, and both machines' md5 sums are
  compared before a cell runs there.
