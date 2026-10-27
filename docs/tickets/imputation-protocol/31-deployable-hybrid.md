# Can the hybrid's weights be chosen without the complete table? (E82; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 (accepted) makes the KC + TabPFN hybrid a *benchmark* output, because its
  weights read the induced truth through the complete table, as KC does.
  - The user wants the method's viability: "numa implementação real experimentaríamos uma
    combinação de pesos para ver".
  - The deployable route is to try the weight grid on cells that were observed and are hidden on
    purpose, whose truth is known without a complete table. The model side is then a fill that
    reads no complete table either: H (ADR 0024).
- **The user's go:** "vamos no 1 agora então" (2026-10-27, before 13:30 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/deployable_hybrid.py` has `train`, `blend` and `--tally`.
  - `run_deployable_hybrid_queue.sh` runs four training queues, then two GPU blend lanes, chained,
    as E74 ran.
- **Smoke, outside the study:** credit-g_40nan at seed 1, `--no-mlflow`.
  - `train` ran (5 min 28 s) with `DECODE_CHECKPOINT loss` and `DECODE_CALIBRATION none`, and wrote
    five selection files.
  - `blend` ran (94 s) and asserted the ledger equals the run's H on all 5 folds.
  - Only shapes were printed: 25 score rows over the five methods; 49–60 selection cells per column
    and fold. No score was read.

## Design (fixed before the first cell)

- **Cells:** E74's 10 hard variants (`blend_weights.HARD`) at E74's seeds 1212, 2323 and 3434,
  30 cells.
  - These are electricity at 40 and 80nan; vehicle and biodeg at 80nan; kr-vs-kp and credit-g at
    40, 60 and 80nan.
  - The seeds are E74's on purpose. `build_folds` then gives the same test cells, so E74's KC and
    KC hybrid read beside these, fold by fold.
  - No seed set has seen this selection rule, or H as the model side.
- **Training:** today's default with `--decode_checkpoint loss --decode_calibration none`.
  - The run keeps H, raw. Nothing in training reads the complete table: the checkpoint is the
    loss's, and there is no calibration. The table only scores.
  - Five folds; MLflow tag `experiment=deployable-hybrid-2026-10-27`.
- **The selection cells (read in place after each fold, random states restored):**
  - The validation split's *observed* cells are dealt into 5 groups at random (stream
    `[seed, fold, 82]`).
  - Each group in turn is hidden as `[MASK]`, beside the real gaps, and H fills it. So every
    observed validation cell gets an out-of-sample fill, and its truth is the observed value.
  - TabPFN, refitted on the training fold as in E74, fills the same cells group by group.
- **The five fills of each fold's test induced cells (`blend`):**
  - **H:** the deployable model alone;
  - **TabPFN;**
  - **equal:** the 0.5 blend;
  - **deployable:** E69's grid and rule, with weights chosen on that fold's selection cells. No
    complete table is used;
  - **oracle:** the same rule with weights chosen on the other four folds' induced cells, as in
    E69/E74 but with H as the model side. It is the ceiling for this model side.
- **Running:** four GPU training queues, then two GPU blend lanes, chained; state in
  `/scratch2/diogoneiss/trident-2026-10-27-deployable-hybrid/`. About 5–6 h; E74's identical
  training took 4 h 40 min, and its blend 24 min.

## Measures and reading, fixed now (`deployable_hybrid.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Primary, viability, over the 10 variants:**
  - **supported:** the deployable hybrid is better than H on 8 or more, worse than H on none, and
    worse than TabPFN on at most 2 (E74's own KC hybrid was worse than TabPFN on 2);
  - **refuted:** better than H on 5 or fewer, or worse than H on 2 or more, or worse than TabPFN on
    5 or more;
  - **mixed:** otherwise.
- **Secondary, the price of deployability:**
  - deployable − oracle per variant, as counts;
  - how often the deployable weight equals the oracle's;
  - the number of variants where the deployable hybrid beats both H and TabPFN (E74's KC hybrid
    beat both on 3).
- **Descriptive:**
  - oracle − H;
  - deployable − KC and deployable − E74's KC hybrid, from E74's blend files on the same folds;
  - the selection cells per column.

**What each reading leads to:**
- **Supported:** a weight chosen on observed cells is enough. A deployable hybrid with H is viable,
  and its cost against the benchmark hybrid is the secondary's numbers.
  - Making it an output would be an amendment to ADR 0031, for the user.
- **Mixed or refuted:** ADR 0031 stays a benchmark output. The study names where selecting without
  the complete table fails.

## Limits fixed now

- MCAR only, on the 10 hard variants.
- The model side is H raw. E47's masked calibration is not tried.
- The selection cells come from the validation split only, about 10% of the rows.
