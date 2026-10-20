# T28's D2: collapse and linear probes on the pre-training stage, and the encoder's drift (E53; pre-registered 2026-10-20)

This ticket was written and committed before the first study cell.

- **Why:** T28's report ([2026-10-19](../../reports/2026-10-19-t28-why-pretraining-adds-little.md),
  §3 H1, H2, H4 and H5, §5 step 2) asks the question that ADR 0011 and T13 both assume and nobody
  has measured under today's normalised objective (N). Does pre-training collapse onto the
  column-constant part of its target, and does the encoder end up knowing anything about the
  hidden value?
  - It also takes the drift half of D3: how far today's decode stage moves the encoder, from a
    pre-trained start and from a random one.
- **The user's go:** 2026-10-20, in the afternoon, with D3 (E52, [10](10-frozen-encoder.md)).
- **Offline:** `scripts/experiments/pretrain_probe.py` runs `train_pretrainer` itself and only
  watches it. Nothing is written to the store or to `src/`.
  - **How it watches:** a tracker computes the diagnostics after each epoch's own validation
    pass, in eval mode, with every random stream saved and restored.
  - **Checked before launch:** `fidelity` on credit-g_20nan, seed 101, on the CPU. The watched
    stage's 300 validation losses equal a plain call's exactly (largest difference 0), with
    diagnostics at every epoch.
  - **Smoked on credit-g_20nan at seed 7,** not a study cell.
    - The first logistic probe, at C = 1 on one mask, read worse than the mode (1.09). The probes
      were changed before any study cell: four training masks, and C chosen per column by 3-fold
      CV on the training cells.
    - The drift mode ran end to end.

## Design (fixed before the first cell)

**What one cell is:** the first of the five folds of the variant at one seed, under today's
configuration as a default run reads it, plus the setting's change.

- **probe cells** (on the CPU, 24 in all):
  - **N** (none) and **A** (`PRETRAIN_OBJECTIVE embedding`, the shipped objective), on
    credit-g_20nan, kr-vs-kp_40nan and pendigits_20nan;
  - **N_lr4** (`LR_PRE` ÷ 4) and **N_b64** (`BATCH 64`), on credit-g_20nan only (H5);
  - seeds 101, 202 and 303.
  - **Each cell runs the stage twice from the same seed:**
    - with `EPOCHS_PRE 0`: the Xavier start, the same as E52's FR;
    - and in full: diagnostics every 10 epochs and at the last.
- **drift cells** (on the GPU, after E52's queues end, 8 in all):
  - **N** and **R** (the Xavier start), then today's full decode stage, on E52's four variants;
  - seed 101.

**Measured on a fixed mask of the validation split, at its hidden cells,** overall (`all`) and by
kind:

- **E04's diagnostics, in the loss's own space.** Under N, that is the prediction after the
  predictor and the layer-normalised target:
  - NMSE against the column-mean target;
  - the centred cosine and its standard error 1/sqrt(d·n);
  - the within-column share of the target's second moment;
  - the same share for the raw embedder target;
  - the effective rank of each column's centred targets and predictions.
- **Linear probes,** at the start and at the end:
  - **What they read:** the transformer's output at the hidden cells, the input the decode stage
    reads. Each column gets a `BayesianRidge` for numbers or a logistic regression for categories.
  - **Training cells:** four fixed masks of the training split.
  - **Score:** `impute_score`'s shape, the pooled z-space RMSE over the mean fill's and the error
    over the mode's, weighted by cells. 1.0 means no information.
- **Drift:**
  - the linear CKA between the encoder's outputs on the fixed masked validation input, before and
    after the decode stage (after = the checkpoint it keeps);
  - `‖ΔW‖/‖W‖` per module.

## Reading, fixed now

All readings use setting N, kind `all`, and the three seeds.

- **Collapse on a table:** on all three seeds, three things hold at the last epoch:
  - the within-column share is below the start's;
  - |centred cosine| < 2 standard errors;
  - NMSE ≥ 1.
- **N collapsing:** collapse on 2 or more of the 3 tables.
- **Value information on a table:**
  - the final probe score is below the start's on all three seeds;
  - and the mean gain is over 1% of the start's mean score.
- **Value information under N:** present on 2 or more of the 3 tables. Otherwise, "no value
  information under N".
- **H5, on credit-g:** a diagnostic moves if N_lr4 or N_b64 changes the collapse call, or moves
  the final probe score by more than 1% of N's, all three seeds in one direction.
- **Reported, without a rule:**
  - A's readings, against E04;
  - the split by kind (H4);
  - the drift.

**What it leads to:** E52's decision table ([10](10-frozen-encoder.md), "Reading, fixed now"),
which takes D3's reading and this one together. No default changes; each "propose" is a
pre-registered study and an ADR for the user.

## Launch

- **The probe cells started at 19:34 GMT-3 on 2026-10-20,** on 0bf1ebf. They run in six CPU
  lanes at 4 threads each, in `/scratch2/diogoneiss/trident-2026-10-20-collapse-probes/`.
  - The pendigits cells take about 80 minutes each on the CPU (16 s an epoch with the bigger
    encoder).
  - E52's trainers slowed by about 10% under the load (1.2 s against 1.1 s a pre-training epoch).
- **The drift cells** start on the GPU once E52's four queues have ended
  (`drift_after_e52.sh`, two lanes).
- **The reading** is `scripts/experiments/collapse_probes_report.py`, written before any study cell
  finished.
