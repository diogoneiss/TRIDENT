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

(open)
