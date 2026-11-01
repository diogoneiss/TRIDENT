# Deep baselines for classification: does TRIDENT generally beat MLP, SAINT and TabNet? (E99; pre-registered 2026-10-31 21:34 GMT-3)

This ticket was written and committed before the study's first cell. A draft (2026-10-30), a cost
probe and a memory check came first, and they are disclosed under "Before the ticket". No score of
any in-scope cell has been read.

- **Why:** the user's claim to test: on classification with gaps, TRIDENT "em geral" beats similar
  algorithms, meaning deep tabular models trained per dataset. The user chose MLP, SAINT and TabNet
  (2026-10-30). TabPFN, a pre-trained foundation model, stays in the strong tier (ADR 0032) and is
  not one of them. GBDTs were an internal sanity check (E98) and stay out of the thesis.
- **The user's decisions (2026-10-31, "vamos pro e99", then three answers):** the majority rule
  below; only 40, 60 and 80nan; a cap of one night.
- **The TRIDENT arm is E98's,** unchanged: an ADR 0029 study per variant (20 trials, validation
  macro-F1 at seed 42's ten folds), evaluated at seeds 42/7/13 on the same ten folds
  (`build_folds(frame, 'class', 10, seed)`). Read with `stage4_tally.trident`, which reads medusa1's
  copy first (E98's amendments).

## Design (fixed before the first cell)

- **Variants:** the 27 at 40, 60 and 80nan (9 tables, MCAR).
- **Baselines** (`scripts/experiments/deep_baselines.py`, pytorch-widedeep 1.5.0, the
  `deep-baselines` extra): `TabMlp`, `SAINT` (with its row attention) and `TabNet`.
- **Each (model, variant) unit:** 20 TPE trials (sampler seed 42) on the mean validation macro-F1
  over seed 42's ten folds, then the best configuration evaluated at seeds 42, 7 and 13 on each
  seed's ten folds. TRIDENT's recipe for every fit:
  - cross-entropy weighted by the training split's balanced class weights; AdamW;
  - at most 200 epochs, early stopping on the validation loss with patience 20, best weights
    restored (the epoch of the lowest validation loss is kept, as TRIDENT keeps it);
  - inputs fit on the training split: a numerical gap filled with the training mean plus a 0/1
    indicator per numerical column with a training gap; a categorical gap as its own level
    `[NULL]`; numerical columns standardised.
- **Search spaces and batches** (in the script, `_space` and `BATCH`):
  - MLP: hidden 128-64 / 256-128 / 256-256-128 / 512-256, dropout 0–0.5, lr log 1e-4–1e-2, batch 256;
  - SAINT: input_dim 16/32, blocks 1–3, heads 2/4/8, attention and feed-forward dropout 0–0.3, lr
    log 1e-4–3e-3, batch 256;
  - TabNet: steps 3–7, step_dim (= attn_dim) 8/16/32/64, gamma 1–2, lambda_sparse log 1e-6–1e-3, lr
    log 5e-3–5e-2, batch 1024;
  - all: weight decay log 1e-6–1e-3.
- **Running:** 81 units dealt between the machines by the probe's cost, dearest first
  (`--deal`; about 61 to medusa1, 20 to gorgona8). Each unit runs whole on one machine. Lanes
  (`run_deep_baselines_queue.sh`) take units by lock: six on medusa1 (three a GPU), three on
  gorgona8. MLflow off; outputs in `trident-2026-10-31-deep-baselines/` on each machine.
- **The cap:** at 2026-11-01 08:00 GMT-3 a STOP file is written on both machines; no unit starts
  after it, units running then finish. A unit missing then is reported missing, and its level for
  that baseline is incomplete.

## Measures and rule, fixed now (`deep_baselines.py --tally`)

- **Per level and baseline:** TRIDENT − baseline, test macro-F1, paired by fold and seed and pooled
  over the 9 tables, one array per seed: 270 pairs. The project's rule (95% t-interval, seed
  agreement): "TRIDENT higher", "baseline higher" or no verdict.
- **TRIDENT leads a baseline** if it is higher at 2 or more of the 3 levels and lower at none (the
  user's majority rule, on three levels).
- **Supported:** TRIDENT leads all three. **Refuted:** any baseline is higher than TRIDENT at 2 or
  more levels. **Mixed:** otherwise.
- **What it allows:** supported → the thesis may say TRIDENT generally beats these deep tabular
  baselines on classification at 40–80% missing. Refuted or mixed → it may not.

## Before the ticket (disclosed)

- **The draft** (2026-10-30, 6e155e8) and its fixes: all-numerical tables,
  in-process data loading (widedeep's default of one worker per core cost 38 s a kc2 fold).
- **A memory leak, found and fixed before this ticket** (afc948d): widedeep's trainer sits in
  reference cycles, so each fit's model stayed on the GPU; SAINT reached 23 GiB by the eighth
  biodeg fold. A collection after each fit holds 0.02 GiB; a single fit peaks at 3.8 GiB at batch
  256, flat over 3–60 epochs. That was the probe's "out of memory", not SAINT's size.
- **A smoke on kc2_00nan** (outside the scope), all three models, to run the whole path once; its
  scores are not part of the study.

## Cost probe (medusa1, 2026-10-30 17:15–17:49 GMT-3; seconds only, no score read)

One mid-space configuration per model (`MIDDLE` in the script), seed 42's ten folds, each table at
40nan, `deep_baselines.py probe`; log in `/scratch/diogoneiss/trident-2026-10-30-deep-probe/`. MLP
and TabNet shared GPU0, SAINT ran on GPU1; from 17:31 E100's queues shared both.

| table | MLP | SAINT | TabNet |
|---|---|---|---|
| kc2 | 4 s | 14 s | 18 s |
| credit-g | 7 s | 12 s | 19 s |
| vehicle | 6 s | 19 s | 21 s |
| biodeg | 6 s | out of memory | 20 s |
| spambase | 12 s | out of memory | 110 s |
| kr-vs-kp | 20 s | 28 s | 70 s |
| pendigits | 63 s | 117 s | 317 s |
| letter | 211 s | 140 s | 717 s |
| electricity | 223 s | 286 s | 683 s |
| **sum** | **9 min** | **10 min + 2** | **33 min** |

- **The whole study** is about 23 ten-fold fits a variant (20 trials and three evaluations) over 5
  levels, so roughly 18, 20+ and 63 GPU-hours: about 100 in all, before parallelism.
- **SAINT ran out of memory** (21 GB) on biodeg and spambase, the tables with the most columns
  once the gap indicators double them. To settle: a smaller SAINT batch, or no indicators for it.
- **Fixed during the probe** (the commit after 6e155e8): the all-numerical tables (kc2) and widedeep's default
  of one loader worker per core, which cost 38 s a kc2 fold against 0.4 s in-process.
- Written 2026-10-30 18:36 GMT-3.

## Outcome

**Written 2026-11-01 05:57 GMT-3. Supported: TRIDENT leads MLP, SAINT and TabNet** by the rule fixed above, each
higher than TRIDENT at no level and lower at 2 or more of 40, 60 and 80nan.

| TRIDENT − baseline, test macro-F1 (270 fold pairs) | 40nan | 60nan | 80nan |
|---|---|---|---|
| MLP | +0.013 [+0.009, +0.017] | +0.008 [+0.003, +0.014] | −0.001 [−0.007, +0.004], no verdict |
| SAINT | +0.036 | +0.036 | +0.029 |
| TabNet | +0.111 | +0.106 | +0.091 |

- **Complete:** all 81 units (61 on medusa1, 20 on gorgona8), 2026-10-31 21:35 to 2026-11-01 05:56
  GMT-3, no failure, inside the cap; the STOP file was written after every lane had ended.
- **What the claim can and cannot carry (descriptive):**
  - **The margin over MLP is small:** about one point at 40nan, under one at 60nan, none at 80nan.
    "Generally beats" holds by the majority rule, not at every rate.
  - **TabNet is weak here:** mean macro-F1 0.640 / 0.565 / 0.469 against MLP's 0.738 / 0.663 /
    0.561, lowest on vehicle (0.33) and letter (0.37). It is not collapsing (median kept epoch 31, 1%
    of folds at 1–2), so it trained, but within this recipe (200 epochs, patience 20, batch 1024, 20
    trials) it is the weakest baseline by far. A report should not lean on the TabNet margin; the
    claim stands on MLP and SAINT alone as well.
  - Kept epochs: MLP median 28 (7% of folds at 1–2), SAINT 11 (2%), TabNet 31 (1%).
- **What it allows (as fixed above):** the thesis may say that TRIDENT generally beats these deep
  tabular baselines on classification at 40–80% missing, with the MLP margin and the 80nan tie
  stated beside it. Nothing about GBDTs or TabPFN follows (the user, 2026-10-30).

### The tally, unedited (`deep_baselines.py --tally`, 2026-11-01 05:57 GMT-3)

```
## 40nan
- TRIDENT − mlp: +0.0128 [+0.0085, +0.0171] seeds +0.0119, +0.0153, +0.0111 -> TRIDENT higher
- TRIDENT − saint: +0.0358 [+0.0294, +0.0422] seeds +0.0375, +0.0344, +0.0355 -> TRIDENT higher
- TRIDENT − tabnet: +0.1108 [+0.1013, +0.1204] seeds +0.1113, +0.1159, +0.1053 -> TRIDENT higher

## 60nan
- TRIDENT − mlp: +0.0083 [+0.0031, +0.0135] seeds +0.0162, +0.0036, +0.0050 -> TRIDENT higher
- TRIDENT − saint: +0.0364 [+0.0277, +0.0450] seeds +0.0420, +0.0344, +0.0327 -> TRIDENT higher
- TRIDENT − tabnet: +0.1061 [+0.0976, +0.1146] seeds +0.1105, +0.1028, +0.1049 -> TRIDENT higher

## 80nan
- TRIDENT − mlp: -0.0014 [-0.0073, +0.0044] seeds +0.0017, +0.0025, -0.0085 -> no verdict
- TRIDENT − saint: +0.0294 [+0.0216, +0.0371] seeds +0.0323, +0.0318, +0.0240 -> TRIDENT higher
- TRIDENT − tabnet: +0.0913 [+0.0818, +0.1008] seeds +0.0981, +0.0919, +0.0841 -> TRIDENT higher

- leads: mlp yes, saint yes, tabnet yes

Primary (E99, 40/60/80nan): supported (fold-pair t, seed agreement)
```
