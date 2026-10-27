# Does the re-masking spread still rank errors under MNAR and MAR? (E84; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0030's spread was tested under MCAR only (E75, E76, E80). Its limits list says "The
  MAR and MNAR families are untested".
- **The user's go:** "de acordo em fazer 1 depois o 2, com 4 em paralelo" (2026-10-27, before
  20:40 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/spread_mechanisms.py` has `train`, `--todo` and `--tally`.
  - `run_spread_mechanisms_queue.sh` runs four GPU queues.
  - Nothing has been read: E79's MCAR ledgers were opened only to check that they load
    (finiteness).

## Design (fixed before the first cell)

- **The 12 shifted variants, all at 60nan:**
  - **MNAR (self-masking probit, E45/E46):** biodeg at slopes 0.25, 0.5, 1.0 and 2.0; vehicle, kc2
    and spambase at slope 0.5.
  - **MAR (E64's new draw, `marf`):** biodeg, kc2, vehicle, spambase and credit-g.
- **Controls:**
  - The MNAR variants' MCAR twins (`<table>_60nan`) are E79's runs at the same seeds. They carry
    the spread at K (E79 had `--score_remask`) and no H.
  - The MAR variants' matched controls (`marfctl_60nan`) run here.
- **Cells:** (7 MNAR + 5 MAR + 5 MAR controls) × seeds 101, 202 and 303 = 51 cells.
  - Today's default, five folds, `--score_checkpoints --score_remask`, as E76 ran.
  - MLflow tag `experiment=spread-mechanisms-2026-10-27`; state in
    `/scratch2/diogoneiss/trident-2026-10-27-spread-mechanisms/`.
  - About 3–3.5 h on four queues. E85's queues start as these end.

## Measures and rule, fixed now (`spread_mechanisms.py --tally`)

- **The measure:** E76's primary, the within-column `num_rank`.
  - Per fold, the Spearman between a numerical cell's spread and its absolute error, within each
    column, averaged over the columns.
  - It is read under K (`induced_remask`) and under H (`induced_remask_loss_checkpoint`), as ADR
    0030 requires.
- **The verdict on each variant (15 folds):**
  - a signal if the 95% t-interval lies above 0 with every seed above 0;
  - against if it lies below 0 with every seed below 0;
  - no signal otherwise.
- **The rule over the 12 shifted variants:**
  - **supported** (ADR 0030's scope extends to these mechanisms): a signal on 10 or more under K
    and under H, with none against under K;
  - **refuted:** a signal on 7 or fewer under K, or any against under K;
  - **mixed:** otherwise.
- **Descriptive:** each shifted variant's `num_rank` minus its control's, fold-paired. That is K
  only for the MNAR twins, and K and H for MAR.

**What it leads to:**
- **Supported:** ADR 0030's limits line is amended, with the user's decision, to "tested under
  MCAR, MAR (one draw) and self-masking MNAR at 60nan".
- **Otherwise:** the limit is stated with the failing mechanisms named.

## Limits fixed now

- 60nan only.
- One MAR draw.
- Self-masking MNAR only; the slope dial is on biodeg alone.
