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

## Outcome

**Written 2026-11-03 14:26 GMT-3. Refuted: no pre-training gain under the upstream defaults either.** A is higher at no
level and B at no level; at every level the 95% interval of B − A lies inside ±0.01 (equivalent), from −0.0009
(60nan) to +0.0023 (00nan).

- **Complete:** all 450 cells, 2026-11-03 11:02–13:36 GMT-3, none failed; seeds 42/7/13/101 on medusa1 (360),
  202 on gorgona8 (90), smoke cells included.
- **Amendment (2026-11-03, about 12:00–14:24 GMT-3):** the account's home in the department's directory moved to
  `/home/pos/diogoneiss`, which broke ssh between the machines. Training was unaffected (the MLflow tunnel
  stayed up); the end chain waited, copied gorgona8's results to medusa1 at 14:26 and its tally failed on a
  missing `uv` in the new home's PATH. The tally was rerun by hand on medusa1, at commit 7feea1c, with the old
  home's PATH. No cell or rule changed.
- **A's mean against Table 4's TRIDENT and E87's A** (descriptive; other protocol, other split):

| level | E118 A | Table 4 TRIDENT | E87 A |
|---|---|---|---|
| 00nan | 0.8516 | 0.8426 | 0.8625 |
| 20nan | 0.7965 | 0.8128 | 0.8043 |
| 40nan | 0.7444 | 0.7564 | 0.7515 |
| 60nan | 0.6683 | 0.7107 | 0.6720 |
| 80nan | 0.5555 | 0.5806 | 0.5592 |

- **What it says:** neither the old sweep (E87) nor the untuned upstream defaults (E118) show the paper's
  +1.7 points; the gain, if it exists, depends on the paper's per-pair tuned values, which stay untested. The
  thesis can say pre-training does not help classification under these two configurations, not that the paper
  fails to replicate.

### The tally, unedited (`cls_tenfold.py --study e118 --tally`, medusa1, 2026-11-03 14:26 GMT-3)

```
- 00nan (A's mean 0.8516): B − A +0.0023 [-0.0000, +0.0046] seeds +0.0048, +0.0022, +0.0002, +0.0031, +0.0011, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0042; credit-g +0.0085; electricity +0.0088; kc2 -0.0060; kr-vs-kp -0.0005; letter +0.0099; pendigits +0.0000; spambase +0.0028; vehicle +0.0014
- 20nan (A's mean 0.7965): B − A +0.0013 [-0.0014, +0.0041] seeds +0.0046, +0.0016, +0.0004, +0.0007, -0.0007, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0037; credit-g -0.0010; electricity +0.0039; kc2 -0.0000; kr-vs-kp +0.0022; letter +0.0113; pendigits +0.0003; spambase +0.0005; vehicle -0.0016
- 40nan (A's mean 0.7444): B − A +0.0005 [-0.0028, +0.0037] seeds +0.0043, -0.0018, -0.0066, +0.0031, +0.0032, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0020; credit-g +0.0070; electricity +0.0030; kc2 -0.0015; kr-vs-kp -0.0024; letter +0.0083; pendigits -0.0007; spambase -0.0013; vehicle -0.0062
- 60nan (A's mean 0.6683): B − A -0.0009 [-0.0045, +0.0027] seeds -0.0065, +0.0074, +0.0007, -0.0028, -0.0033, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg +0.0006; credit-g +0.0008; electricity +0.0011; kc2 -0.0049; kr-vs-kp +0.0038; letter +0.0037; pendigits +0.0007; spambase -0.0012; vehicle -0.0127
- 80nan (A's mean 0.5555): B − A +0.0012 [-0.0036, +0.0060] seeds +0.0053, -0.0089, +0.0055, +0.0030, +0.0012, 450 fold pairs -> equivalent
  - by table (mean over seeds): biodeg -0.0017; credit-g +0.0002; electricity +0.0020; kc2 +0.0072; kr-vs-kp -0.0014; letter +0.0007; pendigits -0.0007; spambase -0.0050; vehicle +0.0096

Primary: A higher at no level, B higher at no level of 5 -> refuted: no pre-training gain (fold-pair t, seed agreement, SESOI 0.01)
```
