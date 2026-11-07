# H2 behind an external gate: A1 against A2 on fresh seeds, on P's fresh draws and on MIMIC-IV's natural tables (E155, T51; pre-registered 2026-11-06)

This ticket was written and committed before the study's first cell.

## A deviation, named

- **What it overrides:** E128's ticket ([07](07-g1-a3-decision-mode.md)) bound C15 to stop for good if G1 failed
  again, and it did (T45 closed 2026-11-04). [E152](../mimic/06-sofa-positive-control.md)'s leads-to said RQ2 stops at
  the gate on natural gaps too. This study runs H2's test anyway, on a new gate.
- **The user's go:** "vamos no 1" (2026-11-06), on my written proposal: RQ2 redesigned, the gap's information shown
  outside the pipeline, then H2's non-inferiority test on fresh seeds on the SOFA table, the outcome-dependent family
  and the cardio table as a negative control.
- **Why the gate moves:** G1 asked two things of the pipeline: that A1 reads the gap with the right sign, and that
  A2's indicator beats A3's plain fill on log loss. The first held every time (E124 3 of 3, E128 3 of 3, E152's
  bilirubin −0.651 logit). The second failed every time but on spambase (E124, E128, E152). It tests the comparator's
  indicator inside TRIDENT, not whether the gaps carry information, and as written H2 can never run. **The new gate is
  the data's, outside TRIDENT:**
  - **P's synthetic files:** E123's G2 ([02](02-g2-out-of-fold.md), valid, passed by its own positive controls): the
    indicators drop the out-of-fold log loss of an external model by 2% or more on the five variants of P.
  - **`mimic-sofa_natural`:** E150 ([mimic/05](../mimic/05-downstream-mortality.md)): a logistic regression with one
    indicator per column beats the one without them, +0.0087 [+0.0047, +0.0126] macro-F1, Holm p 0.0012.
  - **`mimic-cardio_natural`:** the same test did not hold, +0.0014 [−0.0005, +0.0034], Holm p 0.35, so it is the
    negative control.
- **Task:** a new one, T51; T45 stays closed and notes that this supersedes its stop for H2 only.

## Disclosure (what is known before the first cell)

- **The development set leans against A1** (E124's Stage 1b, [03](03-stage1.md), seeds 42/7/13): A1 is inferior to A2
  on 5 of 23 variants (vehicle at 60nan MCAR and under both self-masking levels, biodeg_60nan, biodeg's dial at 0.25),
  non-inferior on none, inconclusive on 18; its macro-F1 is lower on 17 of 23, under MCAR too. **E152** on the SOFA
  table: A1 0.6568, A2 0.6684 macro-F1 (−0.012, past the −0.01 margin by point estimate). So H2 is likely to read
  *inferior* or *inconclusive*; it is run because the thesis's primary question needs its confirmatory answer either
  way.
- **The relative log loss explodes on the odds-3 outcome files** (Stage 1b: +1.27, +2.52, −0.55), where the indicators
  nearly separate the classes and A2's log loss nears 0. The absolute difference is printed beside; the rule stays the
  ratio, as the margins the user accepted are relative.
- **E139/E142's interaction** (TabPFN − A1 on an outcome file minus on its MCAR sibling, −0.045 and −0.039) is known;
  A2 never ran on the fresh draws. E142's A1 cells on the s4848 files at 1515/2626/3737 are medusa1's and read, so
  **nothing is reused**: every cell here is new, at seeds that appear in no log, ticket or script.

## Design (fixed now)

- **Arms (C15's, unchanged):** A1, `[NULL]`; A2, missForest's fill + a gap indicator read by each column's value MLP.
  E124's configuration (`embedding_normalized`, the table's resolved classification defaults, the shipped checkpoint,
  five folds, MLflow off).
- **Seeds:** **8282/8383/9292**, in no log entry, ticket or script before this one.
- **Synthetic front (E126's design, [05](05-stage3.md), new seeds):** P's fresh draws (data seed 4848, on disk since
  2026-11-04): `biodeg_probit0p5_s4848_60nan`, `vehicle_probit0p5_s4848_60nan`,
  `{biodeg,kc2,spambase}_outcome3p0_s4848_60nan`; their MCAR siblings `{biodeg,vehicle,kc2,spambase}_mcar_s4848_60nan`.
  9 files × 2 arms × 3 seeds: 54 cells. Runner: `C15_STAGE=5 c15_run.py`.
- **Natural front:** `mimic-sofa-num_natural` (E152's numerical copy, unchanged; rewritten identical today) and
  `mimic-cardio-num_natural`, written today by `datasets/make_mimic_numeric.py` (now taking a base) into E152's data
  root: 11,082 rows, 30 features (`gender` 0/1, `admission_type` one-hot over 9 values), 16.5% deaths, 9.8% of feature
  cells missing, 88.3% of rows with a gap; troponin 7,292 gaps, the arterial line 6,361–6,686, lactate 3,156, the
  non-invasive pressures about 615. 2 tables × 2 arms × 3 seeds: 12 cells. Runner: `scripts/experiments/h2_natural.py`.
- **Machine (ADR 0035):** all 66 cells on gorgona8 (medusa1's GPUs are held by another user's job), three queues,
  every arm of a (file, seed) pair in one queue.
- **Cost, estimated before the first cell:** from E124's gorgona8 cells (kc2 A1 1.3 min, A2 2.6 min; spambase A2
  30 min on four queues) and E152's (SOFA A1 32 min, A2 36 min on three queues): about 16 GPU-hours, 5 to 6 hours on
  three queues, more beside the other user's trainers on the same GPU.

## Rule, fixed now

The margins are C15's: −0.01 macro-F1 and +2% relative log loss. Intervals Nadeau–Bengio corrected over 15 cells
(variance × (1/15 + 0.286), 14 df), as E126.

**Two primaries, Holm over them:** each is first read at the 97.5% level; if one reads its favourable call there,
the other is read at 95%.

1. **H2 on the synthetic front (E126's rule, `c15_report.h2`):** per variant of P, d = A1 − A2 in test macro-F1 and
   (LL_A1 − LL_A2) / LL_A2; non-inferior if the F1 lower bound > −0.01 and the LL upper bound < +0.02 with every seed
   agreeing; inferior if the F1 upper bound < −0.01 or the LL lower bound > +0.02; inconclusive otherwise. Pooled: the
   mean over P's 5 variants per cell. **Supported** if the pooled F1 lower bound > −0.01 and the pooled LL upper bound
   < +0.02 and A1 is inferior on no variant of P; **reversed** if the pooled F1 upper bound < −0.01 or the pooled LL
   lower bound > +0.02; **inconclusive** otherwise.
2. **H2 on `mimic-sofa-num_natural`:** the same per-variant call on its 15 cells: **non-inferior**, **inferior** or
   **inconclusive**.

**Secondary, one question, decides nothing above:**
3. **Does A1 gain ground on A2 where the gap is informative?** E139's interaction: (A1 − A2) on a file of P minus
   (A1 − A2) on its MCAR sibling (biodeg_probit0p5 and biodeg_outcome3p0 with biodeg_mcar, vehicle_probit0p5 with
   vehicle_mcar, kc2 and spambase outcome3p0 with their mcar), test macro-F1, paired per (seed, fold), pooled over the
   five pairs per cell, plain 95% t interval, Nadeau–Bengio beside. **Gains** if the interval is above 0, **loses** if
   below, **holds** if inside ±0.01, otherwise **no verdict**. On the natural front, SOFA − cardio in the same
   difference is printed beside, descriptive (different tables, not a matched sibling).

**Printed beside, decide nothing:** the absolute log-loss difference per variant; per-seed means; the count rule and
per-table means (E126's sensitivity reads); the cardio table's per-variant call (the negative control: A1 against A2
where the gap is not shown to carry information); plain intervals.

**Read once, only with all 66 cells done.**

**What each reading leads to:**
- **Both primaries favourable (supported, non-inferior):** H2 holds behind the external gate; the thesis sentence and
  any ADR are the user's call.
- **Either reversed or inferior:** H2 fails on that front: `[NULL]` tokens predict worse than missForest + indicator
  under informative missingness in the same encoder; recorded in T51 for the thesis.
- **Inconclusive:** H2 has no confirmatory answer at this power; no extension of seeds (no optional stopping).
- **The secondary** is reported with the primaries; *gains* with an inferior primary reads as "A1 trails A2 but less
  where the gap carries information".

**Not here:** H2.2 (E126's direct-contrast secondary), A3 and A4, TabPFN arms, H2b's shift grid.

## Launch note (2026-11-06 18:21 GMT-3, before any cell ended)

- **Started 18:21:36 GMT-3** by `scripts/experiments/run_e155_chain.sh`, run from a copy in
  `/scratch2/diogoneiss/trident-2026-11-06-h2-natural` (its `progress.log` and `logs/`): three queues of 22 cells each,
  spambase's pairs first. Synthetic cells write under C15's own state dir (`trident-2026-11-04-c15-stage1/A1`, `/A2`,
  at the new seeds); natural cells under the new one.
- **The GPU is shared:** another user's three trainers (`10_cv.py`, about 0.5 GB each) held gorgona8's 3090 Ti at 97%
  when the chain started. medusa1's GPUs were still held by another user's job.
- **No smoke:** A1 and A2 are C15's arms unchanged; A2 never ran on the s4848 files nor on the cardio copy, so the first
  cell of each is watched for a failure at its start.

## Outcome (2026-11-06 21:48 GMT-3): inconclusive on both primaries; the secondary reads *gains*

The chain ended at 21:45 GMT-3 with all 66 cells done and no FAILED line in `progress.log`. The reader ran once, at
21:46 GMT-3; its completeness check passed. Neither primary read its favourable call at 97.5%, so Holm's second step
did not open.

**By the rule:**
- **Primary 1 (H2 pooled over P): inconclusive**, and it leans against A1. The pooled macro-F1 difference is −0.0269,
  2.7 times the margin; its corrected 97.5% upper bound, −0.0076, misses *reversed* by 0.0024. The plain interval
  printed beside, [−0.0341, −0.0197], lies wholly below −0.01 (descriptive, decides nothing). The guard also fails:
  A1 is inferior on `vehicle_probit0p5_s4848_60nan` (F1 −0.098, LL +22%). The pooled log-loss ratio is noise, as the
  disclosure foresaw: spambase's odds-3 file gives +20.8 (one seed +57.7) while every absolute log-loss difference is
  under 0.2.
- **Primary 2 (`mimic-sofa-num_natural`): inconclusive**, F1 +0.0021 [−0.0285, +0.0326], LL −0.7% [−9.3%, +7.9%].
  E152 gave −0.012 at seeds 42/7/13; this is +0.002 at 8282/8383/9292. The 0.014 swing between seed triples lies
  inside either interval; it is not a correction of E152.
- **Secondary: gains**, +0.0155 [+0.0018, +0.0291] on the plain 95% t the rule named. Two facts go with it:
  - The Nadeau–Bengio interval printed beside, [−0.0160, +0.0469], crosses 0; under the primaries' correction it
    would read *no verdict*.
  - The pairs split by mechanism. The three outcome-dependent files are positive (+0.041, +0.061, +0.011): there A1 is
    at parity with A2 (F1 +0.000, +0.020, −0.000), while on their MCAR siblings A1 trails by 0.012 to 0.042. The two
    probit files are negative (−0.015, −0.021): there A1 trails more than under MCAR. (Sibling differences are the
    printout's per-file means minus the per-pair interactions.)
- **Cardio (negative control, beside):** inconclusive, F1 +0.0083 [−0.0124, +0.0290].

**Leads to, as fixed:** H2 has no confirmatory answer at this power; no extension of seeds. The secondary is reported
with the primaries: A1 trails A2 under MCAR on all four siblings, and the gap closes where missingness depends on the
outcome, not where it depends on the value (probit). The thesis sentence is the user's call; no ADR changes.

**The printout** (`scripts/experiments/h2_natural.py --read`, unedited):

```text
## E155: two primaries, Holm: each read at 97.5% first; a favourable call there lets the other be read at 95%
## H2 (E155, confirmation, read once): A1 − A2 on P's fresh draws, 15 cells (seeds 8282/8383/9292 x 5 folds), Nadeau–Bengio corrected 97.5% t; margins −0.01 macro-F1, +0.02 relative log loss
- biodeg_probit0p5_s4848_60nan: F1 -0.0559 [-0.1169, +0.0052]; LL +0.1590 [-0.0924, +0.4103]; seeds F1 -0.0509, -0.0597, -0.0570; LL +0.1358, +0.2003, +0.1408; absolute LL +0.0715 -> inconclusive
- vehicle_probit0p5_s4848_60nan: F1 -0.0982 [-0.1682, -0.0282]; LL +0.2194 [+0.0740, +0.3649]; seeds F1 -0.0870, -0.0908, -0.1168; LL +0.1668, +0.2263, +0.2651; absolute LL +0.1805 -> inferior
- biodeg_outcome3p0_s4848_60nan: F1 +0.0000 [-0.0112, +0.0113]; LL +0.0893 [-1.3899, +1.5685]; seeds F1 -0.0011, +0.0032, -0.0021; LL -0.0141, -0.2260, +0.5081; absolute LL -0.0041 -> inconclusive
- kc2_outcome3p0_s4848_60nan: F1 +0.0197 [-0.0191, +0.0586]; LL +0.7412 [-2.6056, +4.0879]; seeds F1 +0.0083, +0.0291, +0.0217; LL +1.4058, +0.0838, +0.7339; absolute LL -0.0160 -> inconclusive
- spambase_outcome3p0_s4848_60nan: F1 -0.0002 [-0.0038, +0.0033]; LL +20.7835 [-75.5251, +117.0922]; seeds F1 +0.0002, -0.0002, -0.0007; LL -0.2307, +4.8401, +57.7413; absolute LL +0.0019 -> inconclusive

Pooled over P (5 variants): F1 -0.0269 [-0.0463, -0.0076] (plain [-0.0341, -0.0197]); LL +4.3985 [-14.8592, +23.6562] (plain [-2.7603, +11.5572])
Primary (H2): inconclusive (guard: inferior on vehicle_probit0p5_s4848_60nan)
- sensitivity, count rule: non-inferior on 0 of 5, inferior on 1 -> inconclusive
- sensitivity, per table: biodeg F1 -0.0279, LL +0.1242; vehicle F1 -0.0982, LL +0.2194; kc2 F1 +0.0197, LL +0.7412; spambase F1 -0.0002, LL +20.7835

- mimic-sofa-num_natural (97.5%): F1 +0.0021 [-0.0285, +0.0326]; LL -0.0066 [-0.0925, +0.0793]; seeds F1 -0.0000, -0.0020, +0.0082; LL +0.0076, -0.0034, -0.0240; absolute LL -0.0043 -> inconclusive

Primary 1 (H2, synthetic, pooled over P): inconclusive
Primary 2 (H2, mimic-sofa-num_natural): inconclusive

## Secondary: does A1 gain ground on A2 where the gap is informative? (A1 − A2) on P's file minus on its MCAR sibling, macro-F1, plain 95% t over 15 cells (Nadeau–Bengio beside)
- biodeg_probit0p5_s4848_60nan − biodeg_mcar_s4848_60nan: -0.0154
- vehicle_probit0p5_s4848_60nan − vehicle_mcar_s4848_60nan: -0.0205
- biodeg_outcome3p0_s4848_60nan − biodeg_mcar_s4848_60nan: +0.0405
- kc2_outcome3p0_s4848_60nan − kc2_mcar_s4848_60nan: +0.0613
- spambase_outcome3p0_s4848_60nan − spambase_mcar_s4848_60nan: +0.0114
- pooled (5 pairs): +0.0155 [+0.0018, +0.0291] (NB [-0.0160, +0.0469]) -> gains

## Beside, decides nothing
- mimic-cardio-num_natural (negative control, 95%): F1 +0.0083 [-0.0124, +0.0290]; LL -0.0152 [-0.0772, +0.0469]; seeds F1 +0.0131, +0.0131, -0.0013; LL -0.0348, -0.0118, +0.0011; absolute LL -0.0081 -> inconclusive
- (A1 − A2) SOFA minus cardio, macro-F1 per cell: -0.0062 (different tables, descriptive)
```
