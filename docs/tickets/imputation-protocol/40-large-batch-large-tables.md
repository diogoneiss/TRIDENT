# The large batch and longer decode on letter and electricity (E103; pre-registered 2026-10-31 00:17 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E101 ([38](38-large-batch.md)) found P600 (BATCH 1024, LR_PRE and LR_DECODE x4,
  EPOCHS_DECODE 600) better than the default on pendigits 20/40/60nan; E102 ([39](39-large-batch-confirmation.md))
  checks it on fresh seeds. letter and electricity are the only other tables where batch 1024
  still leaves an epoch enough steps (14 and 31 at five folds; spambase 3, kr-vs-kp 2). Does the
  gain hold there? If so, an ADR would be a rule for the large tables rather than a pendigits case.
- **The user's go:** "sim, agende" (2026-10-31, between 00:15 and 00:17 GMT-3), on the proposal to schedule it for
  the night without intervention, with a cap.
- **No code change:** `large_batch.py --study e103` runs M and P600 over each variant's resolved
  configuration (letter_20nan keeps ADR 0018's encoder).

## Design (fixed before the first cell)

- **Cells:** letter and electricity at 20, 40, 60 and 80nan × seeds 42, 7, 13 × arms M and P600,
  five folds, MLflow off. 48 cells.
- **Running:** `STUDY=e103 run_large_batch_queue.sh`, started on each machine when its E102 queues
  end: seed 42 on gorgona8 (two queues), seeds 7 and 13 on medusa1 (four), both arms of a variant
  and seed in one queue, letter's pairs before electricity's. Outputs in
  `trident-2026-10-30-large-batch-e103/`.
- **The cap:** at 2026-10-31 09:00 GMT-3 a STOP file is written on both machines, so no cell
  starts after it; cells running then finish. A variant missing a seed is reported missing and its
  table's reading is partial.

## Measures and rule, fixed now (`large_batch.py --study e103 --tally`)

- **Per variant:** P600 − M on the induced `impute_score`, relative to M, 15 fold pairs, the
  project's rule, 1% SESOI.
- **Primary, per table (letter, electricity) over its 4 variants:** supported if better on 2 or
  more and worse on none; refuted if better on none; mixed otherwise.
- **What follows:** no default changes from this study; with E101 and E102, the user decides on an
  ADR.

## Outcome

**Written 2026-10-31 07:36 GMT-3. letter mixed, electricity refuted.**

- **letter: mixed** (better on 2, equivalent on 1, worse on 1): −2.9% at 20nan and −1.5% at 40nan,
  equivalent at 60nan (−0.1%), worse at 80nan (+0.2%, under the SESOI, every seed above zero).
- **electricity: refuted** (better on none): no verdict at 20 and 40nan (+0.1%, +0.8%, seeds split),
  equivalent at 60 and 80nan.
- **Complete:** all 48 cells, 00:54–07:35 GMT-3, no failure, inside the cap; the STOP file was
  written after the queues ended.
- **Read with E101 and E102 (descriptive):** on pendigits and letter the large batch helps at 20 and
  40nan and fades by 60–80nan, where it costs a fraction of a percent; electricity, with the most
  steps an epoch (31 at batch 1024), gains nothing. No pre-registered rule here supports a default
  change. What to do with it (nothing, or a rule limited to some tables and rates as a new
  question) is the user's call.

### The tally, unedited (`large_batch.py --study e103 --tally`, 2026-10-31 07:36 GMT-3)

```
## P600 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- letter_20nan: -2.90% [-3.42%, -2.38%] seeds -2.84%, -2.91%, -2.94% -> better; masked -2.59%
- letter_40nan: -1.46% [-1.63%, -1.29%] seeds -1.21%, -1.65%, -1.51% -> better; masked -1.04%
- letter_60nan: -0.13% [-0.31%, +0.05%] seeds +0.02%, -0.25%, -0.15% -> equivalent; masked +0.01%
- letter_80nan: +0.21% [+0.09%, +0.32%] seeds +0.22%, +0.22%, +0.18% -> worse; masked +0.19%
- electricity_20nan: +0.09% [-1.15%, +1.33%] seeds +1.19%, +0.34%, -1.26% -> none; masked +1.71%
- electricity_40nan: +0.80% [-2.49%, +4.09%] seeds +1.28%, +3.83%, -2.71% -> none; masked +0.27%
- electricity_60nan: -0.09% [-0.96%, +0.77%] seeds +0.25%, +0.47%, -1.00% -> equivalent; masked +0.01%
- electricity_80nan: -0.02% [-0.27%, +0.23%] seeds -0.03%, -0.16%, +0.12% -> equivalent; masked -0.06%

Primary (P600 − M), electricity: better 0 / no difference 2 / equivalent 2 / worse 0 (fold-pair t, seed agreement) of 4 -> refuted

Primary (P600 − M), letter: better 2 / no difference 0 / equivalent 1 / worse 1 (fold-pair t, seed agreement) of 4 -> mixed
```
