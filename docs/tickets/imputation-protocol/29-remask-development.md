# Does the re-masking spread rank cells on the development variants? (E76; pre-registered 2026-10-25)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0030 (proposed) rests on E75 ([28](28-remask-ensemble.md)).
  - E75's numerical spread met its bar exactly, 12 of 15.
  - The reading that sets the ADR's scope, that the spread ranks cells within a column up to 60nan,
    was post-hoc.
  - E75 ran one seed set on the confirmation variants. It had no categorical-heavy table and no
    20nan.
- **The user's go:** "De acordo com sua sugestão, porém lance o item 5 imediatamente depois,
  minimizando espera!" (2026-10-25, about 09:45 GMT-3).
- **Code:** no change to `src/`. `remask_ensemble.py --study e76` runs and reads this study with
  E75's code. The reader now resolves the single pass through `readout_key`, since the 20nan runs
  are not calibrated. E75's `--tally` printout is unchanged by this.

## Design (fixed before the first cell)

- **Cells:** the 21 development variants (`thesis_table.DEVELOPMENT`) at seeds 101, 202 and 303,
  63 cells.
  - The variants: kr-vs-kp, spambase and credit-g at 20/40/60/80nan; vehicle, biodeg and kc2 at 20
    and 60nan; pendigits, letter and electricity at 20nan.
  - Each cell is today's default configuration, five folds, with `--score_checkpoints
    --score_remask`, exactly as in E75.
  - ADR 0018's bigger encoder applies on pendigits_20nan and letter_20nan.
- **Running:** four GPU queues (`run_remask_queue.sh` with `STUDY=e76`), MLflow tag
  `experiment=remask-development-2026-10-25`, state in
  `/scratch2/diogoneiss/trident-2026-10-25-remask-dev/`. About 5–8 h, with no cutoff.

## Measures and reading, fixed now (`remask_ensemble.py --study e76 --tally`)

**Primary: the within-column `num_rank`.**
- **The measure:** Spearman between a numerical cell's spread and its absolute error, within each
  column, averaged over the columns per fold. It is E75's post-hoc measure, now fixed in advance.
- **The verdict on each variant:**
  - a signal if the 95% t-interval over 15 folds lies above 0;
  - against if it lies below 0;
  - no signal otherwise.
- **Scope:** the 15 variants up to 60nan that hold numerical gaps. That is all 21 except
  kr-vs-kp's four, which have no numerical columns, and spambase_80nan and credit-g_80nan.
- **The rule:**
  - **supported** (ADR 0030's scope replicates): a signal on 13 or more of the 15, under K and
    under H, with none against under K;
  - **refuted:** a signal on 10 or fewer under K, or any variant against under K;
  - **mixed:** otherwise.

**Descriptive, printed by the same reader:**
- E75's pooled `num_rank`, `cat_rank` (beside the single pass's 1 − confidence) and `coverage90`;
- the column-level relation;
- remask − single on `impute_score` under K, H and, where calibrated, KC;
- the dispersion;
- the two 80nan variants with numerical gaps;
- K against the thesis table's runs of the same cells (`development_source`).

**What each reading leads to:**
- **supported:** ADR 0030's scope stands, now on a pre-registered reading, for the user's decision.
- **mixed or refuted:** ADR 0030's scope is narrowed or withdrawn before the user decides.

## Amendment (2026-10-26, 16:51 GMT-3): the machine went down mid-run

- **What happened:** launched 2026-10-25, 09:55 GMT-3, on 10b4452. gorgona8 went down at about
  13:09 GMT-3 that day. Four running cells' logs stopped in the same second, and so did the store
  replica's cron. It came back on 2026-10-26 at 15:02 GMT-3.
  - 5 of 63 cells had finished: electricity_20nan at all three seeds, and letter_20nan at 101
    and 202.
  - The four running cells left source runs in `RUNNING` in the store: pendigits_20nan at all three
    seeds, and letter_20nan at 303. The reader counts finished runs only.
  - Those runs are left as they are; marking them failed is a store write, for the user.
- **What was done:**
  - The four cells' partial output and logs were moved to `interrupted/` in the state directory,
    so each cell has one ledger.
  - The four queues were relaunched; they skip finished cells, so the four interrupted cells
    rerun from scratch.
  - Nothing in the design or the reading changes, and nothing had been read.
