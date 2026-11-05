# The quarter-label channel on fresh seeds, with the validation split scarce too (E138; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (23:37 GMT-3).

- **Why:** E136 ([08](08-label-scarce.md)) found pre-training's first positive channel under full fine-tuning on
  classification here: S25 − B25 +0.016 [+0.007, +0.024], Holm 0.003, four files better, none worse. It was read on
  seeds 42/7/13 with the whole validation split selecting the kept epoch, an oracle shared by the arms. The user
  agreed to confirm it ("concordo com voce", 2026-11-04, about 23:30 GMT-3) on fresh seeds and with a scarce validation
  split, so the reading is deployable; the 21 variants only if it holds.
- **Code:** `FINETUNE_SCARCE_VALIDATION` (`src/training/types.py`, `finetuning.py`): "on" applies
  `FINETUNE_LABEL_FRACTION`'s fixed stratified draw to the validation rows too; "off" (every run before it) keeps
  them all. The readout records `validation_rows` beside `labelled_rows`. Unit test: the flag defaults off and
  validates. Arms **S25V** / **B25V** and cut `9` in `c1_run.py`; reader `c1_report.py --confirm`.
- **Smoke, outside the study (biodeg_probit_60nan, seed 42, deterministic, B25V):** 184 labelled rows, 26
  validation rows (a quarter of 105) on 5 of 5 folds; probabilities and labels agree in length. Deleted.

## Design (fixed before the first cell)

- **Arms, on the eight C1 files, decision mode:** S25V (pre-trained, fraction 0.25, scarce validation) against B25V
  (no pre-training, the same rows), seeds **1515 / 2626 / 3737** (the project's confirmation seeds, C15 ticket 05) ×
  5 folds. 48 cells; otherwise E136's configuration.
- **Validation rows at a quarter (seed 42, fold 1):** biodeg 26, kc2 13, vehicle 21, spambase 115; the kept-epoch rule
  (ADR 0029 decision 5) runs on these for both arms.
- **Machines (ADR 0035):** `c15_run.machine` sends every fresh-seed pair to medusa1 (4 queues).

## Rule, fixed now

- **Primary, one test:** S25V − B25V in test macro-F1, 15 paired replicates a file, pooled over the eight files per
  replicate, plain 95% t interval (14 df), Nadeau–Bengio beside. **Holds** if the plain interval is above 0; otherwise
  **does not hold**. Per-file lines, the MNAR / MCAR halves and log loss are flags. The audit prints any (file, rows)
  entry where the two arms' labelled or validation counts differ (none expected).
- **What follows:** *holds* → stage 2, a proposal to the user with the cost: the 21 classification-usable variants of
  the confirmatory set (E30's list: credit-g / kr-vs-kp / spambase at 20–80nan, vehicle / biodeg / kc2 at 20 and
  60nan, pendigits / letter / electricity at 20nan), S25V against B25V, seeds 42/7/13, 630 cells, of which
  electricity, letter and pendigits are the bulk (pre-training 300 epochs on 10–45 k rows); launched only on the
  user's go. *Does not hold* → the E136 reading stays an oracle-selector result, recorded as such, no claim.
- **Threats stated now:** kc2's 13 validation rows (about 3 positives) make its kept epoch noisy for both arms; one
  fraction (0.25); 60nan files only; the fresh seeds change folds and initialisation but not the missingness draw.
- **Read only with all 48 cells done.**

**Amendment (23:45 GMT-3):** at 23:42 GMT-3 another user's job took both of medusa1's GPUs (31 GB each) and every cut-9 cell
there failed with CUDA out of memory before training (48 fast failures, nothing read). The 48 cells moved whole to
gorgona8 (`c1_run.py` cut 9 routes to gorgona8): the fresh-seed pairs had no arm on any machine, so ADR 0035 holds.
The design and the rule are unchanged.
