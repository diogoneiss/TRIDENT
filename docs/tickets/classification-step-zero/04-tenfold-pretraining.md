# Does pre-training help classification with ten folds and five seeds? (E87; pre-registered 2026-10-28)

This ticket was written and committed before the study's first cell.

- **Why (T41):** E71 found no detectable pre-training gain on classification (B − A from −0.005 to
  +0.004 macro-F1, no verdict at any level). The paper's Table 4 reports about +1.7 points. E71 ran
  three folds and two complete seeds, so it could not tell "no gain" from "too little power".
  This study is T41's old-sweep half. The paper's-parameters half still waits on the paper's
  hyperparameters.
- **The user's go:** "aprovo! rode os 2 em paralelo" (2026-10-28, about 16:50 GMT-3), and on scope
  "9 tabelas de uma vez", one read at the end.
- **Code:** no change to `src/`. `scripts/experiments/cls_tenfold.py` has `train`, `--todo` and
  `--tally`, and reuses E71's `sweep_run`, `cell_dir` and `fold_metrics`.
  `run_cls_tenfold_queue.sh` runs eight GPU queues.

## Design (fixed before the first cell)

- **Arms**, as E71's A and B. Both take the old sweep's `hyperparameters.json` of the variant as
  `hyperparams_override`, with `cosine_legacy`:
  - **A:** as is (`EPOCHS_PRE 300`, `EPOCH_FINE 150`);
  - **B:** `EPOCHS_PRE 0`, no pre-training (the paper's Table 4 arm).
- **Cells:** E71's 9 tables (biodeg, credit-g, electricity, kc2, kr-vs-kp, letter, pendigits,
  spambase, vehicle) × 5 levels (00 to 80nan) × 2 arms × seeds 42, 7, 13, 101, 202, with
  `--cv_folds 10`. That is 450 cells.
  - The cells are dealt seed by seed, the largest tables first.
- **Running:** all on medusa1, so every A/B pair shares a machine (E86). Eight queues, four per
  RTX 5090; another user's job also holds 7–9 GB on each GPU.
  - MLflow tag `experiment=cls-tenfold-2026-10-28`, logged through the leader's server; state in
    `/scratch/diogoneiss/trident-2026-10-28-cls-tenfold/`.
  - **Cost:** about 180 GPU-hours at gorgona8's speed (E71's times × 4.5 for ten folds and the
    larger training splits). On medusa1 that is about 16 h of wall time if the eight queues scale
    cleanly, and more likely 20–24 h.
- **Smoke:** one cell runs first (kc2_00nan s42 B), and the queues start only after it lands in
  the leader's store with `host=medusa`. It counts.

## Measures and rule, fixed now (`cls_tenfold.py --tally`)

- **Per level:** B − A on test macro-F1, per fold. The pairs are pooled over the 9 tables, with one
  array per seed: 450 fold pairs a level. The project's rule applies (95% t-interval, seed
  agreement), with SESOI 0.01:
  - **A higher** (interval below 0, every seed below 0): pre-training helps at that level;
  - **B higher** (above 0): pre-training hurts;
  - **equivalent** (inside ±0.01): no gain of the paper's size;
  - **no verdict** otherwise.
- **The call:**
  - **supported** (the gain replicates) if A is higher at 3 or more of the 5 levels and B is
    higher at none;
  - **refuted** if A is higher at 1 level or none;
  - **mixed** otherwise.
- **Read only with every cell done.** The tally refuses a partial read. A cut stays partial, with no
  call.
- **Descriptive:** each table's mean B − A per level; A's mean macro-F1; the comparison with E71's
  three-fold numbers (run on gorgona8, so descriptive only, by E86).

## Outcome

(open)
