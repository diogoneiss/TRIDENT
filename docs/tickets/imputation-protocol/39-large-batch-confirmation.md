# pendigits' large batch and longer decode on fresh seeds (E102; pre-registered 2026-10-30 23:14 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E101 ([38](38-large-batch.md)) was supported: P600 (BATCH 1024, LR_PRE and LR_DECODE
  x4, EPOCHS_DECODE 600) better than today's default on pendigits 20/40/60nan, equivalent at 80nan.
  Its ticket fixed a fresh-seed confirmation as the next step, before any ADR.
- **The user's go:** E101's ("sim, rode", 2026-10-30), whose ticket names this confirmation.
- **No code change:** `large_batch.py --study e102` runs E101's M and P600 only.

## Design (fixed before the first cell)

- **Cells:** pendigits at 20, 40, 60 and 80nan × fresh seeds 7878, 8989, 9090 (never used before) ×
  arms M and P600, five folds, MLflow off. 24 cells.
- **Running:** `STUDY=e102 run_large_batch_queue.sh`; seed 7878 on gorgona8, 8989 and 9090 on medusa1
  (ADR 0035), both arms of a variant and seed in one queue. Outputs in
  `trident-2026-10-30-large-batch-e102/`.

## Measures and rule, fixed now (`large_batch.py --study e102 --tally`)

- **Per variant:** P600 − M on the induced `impute_score`, relative to M, 15 fold pairs, the
  project's rule, 1% SESOI.
- **Replicated** if P600 is better on 2 or more of the 4 and worse on none; **not replicated** if
  better on none or worse on any; **partial** otherwise.
- **What follows:** replicated → an ADR proposal for the user (a per-table batch for pendigits). No
  default changes without the user.

## Outcome

**Written 2026-10-31 01:09 GMT-3. Not replicated, by the rule fixed above: P600 is better on 3 of 4 and worse on
1.** The rule asked for worse on none, and pendigits_80nan is worse (+0.42% [+0.19%, +0.65%], every
seed above zero). That is below the 1% SESOI, but the rule reads the sign and the seeds, not the
SESOI, for "worse".

- **What did replicate:** the gains at 20, 40 and 60nan, at nearly E101's sizes: −5.8% (E101 −5.4%),
  −4.2% (−4.3%), −1.2% (−1.2%), every seed below zero.
- **What changed:** 80nan, equivalent in E101 (+0.16%), is worse here (+0.42%). Read with E101's P
  arm (+0.37% at 80nan, worse), the large batch costs a little at 80nan and helps below it.
- **Complete:** all 24 cells, 23:14–01:08 GMT-3, no failure.
- **What follows:** no ADR proposal from this rule. A pendigits rule limited to 20–60nan would be a
  new question, the user's to ask; E103 (letter, electricity) runs on, unchanged.

### The tally, unedited (`large_batch.py --study e102 --tally`, 2026-10-31 01:09 GMT-3)

```
## P600 − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_20nan: -5.84% [-6.36%, -5.32%] seeds -5.66%, -5.88%, -5.98% -> better; masked -5.10%
- pendigits_40nan: -4.21% [-4.65%, -3.77%] seeds -4.21%, -3.98%, -4.45% -> better; masked -3.90%
- pendigits_60nan: -1.18% [-1.53%, -0.82%] seeds -1.01%, -1.13%, -1.40% -> better; masked -0.79%
- pendigits_80nan: +0.42% [+0.19%, +0.65%] seeds +0.54%, +0.22%, +0.50% -> worse; masked +0.22%

Primary (P600 − M), pendigits: better 3 / no difference 0 / worse 1 (fold-pair t, seed agreement) of 4 -> not replicated
```
