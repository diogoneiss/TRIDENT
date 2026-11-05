# TabPFN v2 under informative missingness: native and explicit-mask arms beside A1 (E137, T44; pre-registered 2026-11-04)

This ticket was written and committed before the study's first fit (22:33 GMT-3).

- **Why (T44, proposed 2026-10-29 from the wiki's claims ledger; the user's go "vamos no 1 e 2 paralelos",
  2026-11-04):** untuned TabPFN v2 leads TRIDENT's classification under MCAR by about 2 macro-F1 points (E83, E92;
  ADR 0032), but its prior has only MCAR gaps and no published study tests its native missing-value handling under
  a controlled MNAR mechanism. C15's family (E124's A1 cells, 23 files × 3 seeds × 5 folds) is that test bed.
- **Code:** `scripts/experiments/tabpfn_mnar.py` (`run`, `--todo`, `--tally`); no change to `src/`, nothing trained,
  the store untouched. Folds are `build_folds(frame, label, 5, seed)`, each fold's test indices asserted equal to
  A1's readout; the matrix is E83's (`tabpfn_classification._matrix`); v2's checkpoints, 8 estimators,
  `random_state` 0, the GPU, the training split as context (E83's convention; A1 used validation for selection
  only), capped at 10,000 rows (no file here reaches it). Per fold the test probabilities over the label ids are
  saved, so the tally reads macro-F1 and log loss paired with A1 through `c15_report.scores`.
- **Smoke, outside the study:** kc2_probit_60nan and kc2_60nan, seed 42: 4 s a pair, both arms' rows sum to one,
  fold indices match A1's. Deleted.

## Design (fixed before the first fit)

- **Arms, one fit a fold each:** **native** (raw NaNs, a missing categorical as NaN); **mask** (the same matrix with
  one 0/1 missingness column appended per feature, so the in-context learner can use the pattern as a predictor;
  its built-in indicator was trained where gaps carry no signal). From each fit two predictions:
  `balance_probabilities` on (E92's outcome: the fairer strong tier against a class-weighted TRIDENT) as the
  **primary**, plain **beside**. Log loss under balancing is on the rebalanced probabilities.
- **Files:** C15's development family, F0 (MCAR controls: biodeg/kc2/vehicle at 20 and 60nan, spambase 60nan), F1
  (their probit MNAR siblings), F2 (biodeg probit scales 0.25/1/2) and F3 (outcome-dependent MNAR, six files);
  seeds 42/7/13; 69 pairs, 345 folds, 690 fits. About five minutes a machine.
- **Machines (ADR 0035):** each pair on the machine that holds its A1 cell (`c15_run.machine`), as E92 did with E87.

## Rule, fixed now

- **Primary, two tests as one family:** the interaction (arm − A1) on a probit file minus (arm − A1) on its MCAR
  sibling, in test macro-F1, paired per (table, rate, seed, fold), pooled over the seven F1/F0 pairs per replicate
  (15 replicates), plain 95% t interval, Nadeau–Bengio beside; Holm over the two arms, balanced probabilities.
  **Lead shrinks under MNAR** if the interval is below 0, **grows** if above 0, **holds** if inside ±0.01, otherwise
  **no verdict**.
- **Beside, descriptive:** arm − A1 pooled on F0 and on F1 (the plain leads), per file on F2 and F3; mask − native
  on the 16 MNAR files pooled (what the MCAR prior leaves unused); the plain-probability readings of all of it.
- **What follows:** this settles whether TabPFN's lead is robust to informative missingness (no published study
  shows it) and whether TRIDENT's per-column `[NULL]` gains ground there (H2). ADR 0032's strong tier is not changed
  here; whether to adopt the balanced arm there stays the user's decision from E92. GBDT is not an arm.
- **Threats stated now:** TabPFN untuned (as in E83/E92); one `random_state`; its context sees no validation rows
  while A1 selected its epoch on them; the mask doubles the feature count (spambase 57 → 114, within v2's 500).
- **Read only with all 69 pairs done.**

## Outcome (2026-11-04, 23:22 GMT-3; 69 of 69 pairs, gorgona8 10 done 23:09 GMT-3, medusa1 59 done before 23:15 GMT-3)

Read with `tabpfn_mnar.py --tally` (medusa1's pairs copied beside gorgona8's). Unedited tally, balanced probabilities
(primary), arm − A1 in test macro-F1:

| arm | F0 MCAR controls (7 files) | F1 probit MNAR (7) | F2 probit scales (3) | F3 outcome-dependent (6) |
|---|---|---|---|---|
| native | +0.0284 [+0.0236, +0.0332] | +0.0360 [+0.0277, +0.0443] | +0.0484 [+0.0371, +0.0597] | −0.0031 [−0.0087, +0.0025] |
| mask | +0.0295 [+0.0243, +0.0347] | +0.0333 [+0.0260, +0.0406] | +0.0425 [+0.0321, +0.0530] | +0.0038 [−0.0010, +0.0086] |

| primary: interaction (arm − A1) MNAR − MCAR, 7 sibling pairs | pooled | NB | p | Holm | verdict |
|---|---|---|---|---|---|
| native | +0.0076 [−0.0014, +0.0166] | [−0.0131, +0.0283] | 0.091 | 0.182 | **no verdict** |
| mask | +0.0038 [−0.0034, +0.0110] | [−0.0127, +0.0204] | 0.274 | 0.274 | **no verdict** |

- **Mask − native on the 16 MNAR files:** +0.0003 [−0.0020, +0.0026]: the explicit mask adds nothing TabPFN's own
  handling does not already use.
- **Plain probabilities (beside):** the same picture, leads about 0.003 smaller; interaction native +0.0051
  [−0.0044, +0.0145] no verdict, mask +0.0011 [−0.0063, +0.0086] *lead holds* (inside ±0.01).
- **Per file, F1:** TabPFN ahead on 6 of 7 (kc2_probit_20nan the exception, +0.002 n.s.); largest on
  biodeg_probit_60nan +0.061 and vehicle_probit_60nan +0.050. **F3 (outcome-dependent MNAR, draw s2060):** A1 ahead
  on kc2 at both strengths (−0.019 [−0.037, −0.000], −0.016 [−0.031, −0.001]), TabPFN ahead on spambase 1.5
  (+0.013) and biodeg 3.0 (+0.008), the rest within ±0.01; log loss favours TabPFN on 5 of 6.
- **Reading, by the rule:** the primary is *no verdict* for both arms: TabPFN's lead does not shrink under
  value-dependent (probit) MNAR; if anything it leans larger (+0.008, interval reaching +0.017), and the lead grows
  with the probit scale (F2). The lead vanishes only under *outcome-dependent* missingness (F3, pooled −0.003 [−0.009,
  +0.003]), where a gap carries the label directly and TRIDENT's per-column `[NULL]` can read it; that family is
  descriptive here (no MCAR sibling by draw) and is the place where H2 could be tested with a sibling design.
  ADR 0032's strong tier is unchanged; adopting the balanced arm there stays the user's decision from E92. T44 done.
