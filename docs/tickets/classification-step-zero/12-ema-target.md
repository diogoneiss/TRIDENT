# Does an EMA target make classification pre-training help? (E112, T13; pre-registered 2026-11-02 10:17 GMT-3)

This ticket was written and committed before the study's first cell. A unit test and a smoke came
first (below); no in-scope cell has run.

- **Why:** T13. E71 and E87 found no pre-training gain on classification (E87: B − A equivalent within
  ±0.01 at every level). Classification pre-trains with the legacy objective, regressing the clean
  cell's own input embedding, whose falling loss E04 traced to norm-matching. An EMA teacher with a
  predictor (data2vec, BYOL; SMART's decay 0.996) is the standard guard against that. T28's diagnosis
  for imputation (no collapse, washout) is why imputation is left out.
- **The user's go:** "vamos no t13, dividindo sua execução entre máquinas", then "Concordo, so para
  classificação" (2026-11-02, before this ticket).
- **The code** (behind the existing `--pretrain_objective` flag; every other objective unchanged):
  - `PRETRAIN_OBJECTIVE embedding_ema` → `EmaTargetPretrainer` (`src/models.py`): the target of a
    masked cell is a teacher copy of the embedder and transformer, run in eval mode on the clean row,
    layer-normalised; the student keeps `NormalizedEmbeddingPretrainer`'s predictor. The teacher is
    set to the initialised student once and moved by EMA (0.996) after every optimiser step; it never
    takes a gradient. The fine-tuning stage receives the student's embedder and transformer.
  - The tag `pretrain_objective` already records the objective on every run.
  - **Checks:** a new unit test (the teacher starts equal, takes no gradient, moves by exactly the
    EMA rule); the unit suite (460, with the typing gate) and both regression fixtures pass; a
    2-fold, 5-epoch smoke on kc2_00nan at seed 999, MLflow off, unread.

## Design (fixed before the first cell)

- **E87's protocol:** the old sweep's `hyperparameters.json` per variant, `cosine_legacy`, ten folds,
  the 45 variants (00–80nan), seeds 42/7/13/101/202 (`scripts/experiments/cls_ema.py`,
  `run_cls_ema_queue.sh`, MLflow tag `experiment=cls-ema-v2-2026-11-02` since the amendment below).
- **Arm E:** that configuration with `PRETRAIN_OBJECTIVE embedding_ema`. **Arm B:** `EPOCHS_PRE 0`
  (no pre-training). **Arm A:** E87's, the legacy objective.
- **Running (each pair on one machine, ADR 0035):** medusa1 runs E at seeds 42/7/13/101 and pairs it
  with E87's B and A, which ran there; gorgona8 runs E and B at seed 202. No cap.

## Measures and rule, fixed now (`cls_ema.py --tally`)

- **Primary:** E − B per level, test macro-F1, paired by fold and seed, pooled over the 9 tables, one
  array per seed (450 pairs), the project's rule, SESOI 0.01.
  - **Supported** if E is higher at 3 or more of the 5 levels and lower at none; **refuted** if
    higher at 1 or fewer; **mixed** otherwise.
- **Descriptive:** E − A on seeds 42/7/13/101 (the EMA against the legacy objective).
- **What follows:** supported → the EMA objective as classification's pre-training is an ADR for
  the user. Refuted → T13 closes for classification.

## Amendment (2026-11-02 11:28 GMT-3): a NaN bug, fixed, and a relaunch

- **What happened:** the first launch (10:17 GMT-3, tag `cls-ema-2026-11-02`) failed its first letter
  cells (20 and 40nan, seed 42) with a non-finite final loss; their fold metrics showed a constant
  prediction (accuracy 1/26), the only scores seen, and they are not part of any reading.
- **The cause:** the teacher was fed the raw encoding of the clean row, which holds NaN at a numerical
  gap; through attention the NaN reached every position, so every loss was NaN from the first epoch on
  any table with gaps. The legacy objectives read their target only at the masked cells and never meet
  it. Reproduced on letter_40nan's first fold (12 epochs, all NaN).
- **The fix:** the teacher reads the clean row as the student reads its rows, every gap as `[NULL]`
  (`EpochMasker.apply` with nothing hidden). The same fold now trains (train loss 1.12 → 0.099 in 12
  epochs). A regression test trains the objective on a table with gaps; the unit suite (461) and both
  fixtures pass.
- **The relaunch:** every queue was stopped, the first launch's outputs moved to
  `trident-2026-11-02-cls-ema-failed/` on both machines, and the study reruns from scratch under the tag
  `cls-ema-v2-2026-11-02`, the design and the rule unchanged. The first tag's runs stay in the store,
  unread.

## Second amendment (2026-11-02 14:31 GMT-3, before any reading): seed 202 moves to medusa1

- **Why:** an E cell took 85 minutes on gorgona8 (three queues on one GPU) against 29 on medusa1; at
  that rate gorgona8's seed 202 would have taken over a day, against about ten hours for medusa1's four
  seeds.
- **What changes:** gorgona8's queues are stopped and its cells (4 E, none read) set aside; medusa1
  runs E at seed 202 too once its first four seeds end, against E87's B and A there, as for the other
  seeds. Every pair stays on one machine; the design and the rule are unchanged. E − A, descriptive,
  now covers all five seeds.

## Outcome

**Written 2026-11-02 22:34 GMT-3. Mixed: E is higher than B at 40 and 60nan and equivalent at 00, 20 and 80nan.**
The rule asked for 3 levels; there are 2, and none where B is higher.

| E − B, test macro-F1 (450 fold pairs) | 00nan | 20nan | 40nan | 60nan | 80nan |
|---|---|---|---|---|---|
| E (EMA pre-training) − B (none) | +0.001 equiv. | +0.001 equiv. | **+0.005 E higher** | **+0.009 E higher** | +0.004 equiv. |
| E − A (legacy pre-training), descriptive | +0.002 equiv. | +0.004 equiv. | +0.004 E higher | +0.005 equiv. | +0.002 equiv. |

- **Complete:** all 225 E cells on medusa1 (2026-11-02 11:28–22:34 GMT-3), no failure, read against E87's
  A and B there.
- **Reading:** for the first time in this project, pre-training moves classification at all: with the
  EMA target it is ahead of no pre-training at 40 and 60nan, where E87's legacy objective was not.
  But the gains are half a point and under a point of macro-F1, below the 0.01 SESOI, and every
  other level is equivalent to no pre-training. It does not reach the rule's three levels.
- **What follows, as fixed above:** mixed → reported, and the user chooses (an ADR is not indicated by
  the rule; a fresh-seed check at 40–60nan, or closing T13, are the options).

### The tally, unedited (`cls_ema.py --tally`, 2026-11-02 22:34 GMT-3)

```
- 00nan: E − B +0.0012 [-0.0011, +0.0035] seeds +0.0019, +0.0003, +0.0039, -0.0031, +0.0029 -> equivalent
- 20nan: E − B +0.0014 [-0.0017, +0.0044] seeds -0.0018, +0.0003, +0.0019, +0.0041, +0.0023 -> equivalent
- 40nan: E − B +0.0050 [+0.0021, +0.0080] seeds +0.0048, +0.0025, +0.0090, +0.0038, +0.0051 -> E higher
- 60nan: E − B +0.0085 [+0.0051, +0.0120] seeds +0.0079, +0.0022, +0.0127, +0.0106, +0.0093 -> E higher
- 80nan: E − B +0.0036 [-0.0005, +0.0078] seeds +0.0032, +0.0088, -0.0023, +0.0076, +0.0010 -> equivalent
  - 00nan, descriptive (all seeds): E − A +0.0017 [-0.0008, +0.0043] -> equivalent
  - 20nan, descriptive (all seeds): E − A +0.0035 [+0.0006, +0.0064] -> equivalent
  - 40nan, descriptive (all seeds): E − A +0.0041 [+0.0013, +0.0068] -> E higher
  - 60nan, descriptive (all seeds): E − A +0.0050 [+0.0018, +0.0083] -> equivalent
  - 80nan, descriptive (all seeds): E − A +0.0023 [-0.0020, +0.0066] -> equivalent

Primary (E112, E − B): E higher at ['40', '60'], B higher at no level of 5 -> mixed (fold-pair t, seed agreement, SESOI 0.01)
```
