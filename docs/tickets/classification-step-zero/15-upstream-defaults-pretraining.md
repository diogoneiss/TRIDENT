# Does pre-training help classification under the upstream code's defaults? (E118; pre-registered 2026-11-03)

This ticket was written and committed before the study's first cell (11:01 GMT-3).

- **Why (T41):** E87 ([04](04-tenfold-pretraining.md)) refuted the paper's pre-training gain under the
  fork's old sweep (`EPOCHS_PRE 300`, `EPOCH_FINE 150`): B − A inside ±0.01 at every level. Table 4
  of the paper reports about +1.7 macro-F1 points, with no pre-training ahead at 0% (0.8533 against
  0.8426) and behind at 20–80%. This study is T41's other half, as close to the paper as the repository
  allows.
- **What this is not:** the paper tuned each model–dataset pair with Optuna (20 trials of 50 epochs, on
  validation F1) and trained the winner with early stopping. Those tuned values are not in the
  repository, nor in the upstream commit `14d226c`, so they stay untested. This study runs the
  **upstream code's built-in defaults** instead. A refuted call here does not say that the paper fails to
  replicate; it says the gain does not appear under the untuned upstream defaults.
- **The user's go:** "rode o t41 agora nas duas" (2026-11-03).
- **Code:** no change to `src/`. `cls_tenfold.py --study e118` and `STUDY=e118` for
  `run_cls_tenfold_queue.sh`.

## Design (fixed before the first cell)

- **The defaults:** upstream `14d226c`'s `train.py` and HEAD's classification defaults agree on every
  key (DIM 128, HIDDEN_DIM 16, HEADS 16, LAYERS 2, DIM_FEED 32, DROPOUT 0.2, BATCH 256, LR_PRE 0.00034,
  WEIGHT_DECAY_PRE 0.005, PROB_MASCARA 0.5, LR_FINE 0.001, WEIGHT_DECAY_FINE 0.0019) except the epochs:
  upstream 40/40, HEAD 300/150. No classification file exists under `datasets/hiperparams/`, so the
  runs read the defaults with the override below and no sweep file.
- **Arms**, as override, with `cosine_legacy` (upstream's per-batch `CosineAnnealingLR(T_max=epochs)`):
  - **A:** `EPOCHS_PRE 40`, `EPOCH_FINE 40`;
  - **B:** `EPOCHS_PRE 0`, `EPOCH_FINE 40`, no pre-training (Table 4's arm).
  - Fine-tuning keeps the best-validation-loss checkpoint, HEAD's stand-in for the paper's early stopping.
- **Cells:** E87's 9 tables (biodeg, credit-g, electricity, kc2, kr-vs-kp, letter, pendigits, spambase,
  vehicle) × 5 levels (00 to 80nan) × 2 arms × seeds 42, 7, 13, 101, 202, `--cv_folds 10`: 450 cells.
  - **Split:** ten folds, with a validation slice of 0.1/0.9 of each training part, so about 80/10/10
    per fold, the paper's ratio, repeated so the result has an interval (the paper used one split).
- **Code version:** HEAD. E70 ([01](01-determinism.md)) found HEAD does not reproduce the old sweep fold
  by fold, so this is the fork's implementation of the upstream defaults, not the upstream code itself.
- **Running (ADR 0035, split by seed, each A/B pair on one machine):**
  - medusa1: seeds 42, 7, 13, 101, eight queues (four per RTX 5090; another user's job holds 5–10 GB on
    each GPU);
  - gorgona8: seed 202, two queues.
  - MLflow tag `experiment=cls-upstream-defaults-2026-11-03`, logged through the leader's server; state
    in `trident-2026-11-03-cls-upstream/` under `/scratch` (medusa1) and `/scratch2` (gorgona8).
    The tally runs on medusa1 after gorgona8's results are copied there.
  - **Cost:** E87 took 10.4 h on medusa1 at 300/150 epochs. These epochs are about a fifth of that, so
    about 2 h on medusa1; gorgona8 is about 3× slower a cell with fewer queues, so about 4–5 h.
- **Smoke:** one cell on each machine first (kc2_00nan B), and the queues start only after both land in
  the leader's store. They count.

## Measures and rule, fixed now (`cls_tenfold.py --study e118 --tally`)

E87's, verbatim, so the two halves read the same:

- **Per level:** B − A on test macro-F1, per fold, pooled over the 9 tables with one array per seed: 450
  fold pairs a level. The project's rule (95% t-interval, seed agreement), SESOI 0.01:
  - **A higher** (interval below 0, every seed below 0): pre-training helps at that level;
  - **B higher**: pre-training hurts; **equivalent** (inside ±0.01): no gain of the paper's size;
    **no verdict** otherwise.
- **The call:** **supported** (the gain appears) if A is higher at 3 or more of 5 levels and B at none;
  **refuted** if A is higher at 1 level or none; **mixed** otherwise.
- **Read only with every cell done.** The tally refuses a partial read; a cut stays partial, with no call.
- **Descriptive:** each table's mean B − A per level; A's mean macro-F1 against Table 4's TRIDENT column
  (0.8426, 0.8128, 0.7564, 0.7107, 0.5806) and E87's.
