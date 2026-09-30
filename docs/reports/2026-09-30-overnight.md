# Overnight on gorgona8, 2026-09-29/30

What ran on the server between 21:24 GMT-3 on 2026-09-29 and 05:22 GMT-3 on 2026-09-30, what
it found, what changed in the code, and the decisions left for the morning. Every experiment
here has its pre-registration and outcome in `docs/tickets/imputation-pretraining/` and an
entry in the [experimentation log](../EXPERIMENTATION_LOG.md) (E24–E26).

## In five lines

1. **Pre-training ablation, done (E24).** Pre-training as it is never helps against none; on
   pendigits it hurts, and giving its epochs to the decoder (arm C) beats the best baseline
   there for the first time in this effort.
2. **A better pre-training target exists (E25, ADR 0011).** The normalised embedding target
   (layer-normalised, read through a predictor head) is never worse than the current one,
   better on two variants, and better than no pre-training on three of four. Reconstructing
   values instead helps only on the all-numerical table.
3. **Normalised pre-training with a long decode stage (E26):** normalised pre-training followed by a 450-epoch decode stage is
   never worse than either half: it beats C on kr-vs-kp, matches C on pendigits (both clear
   `knn10`), and ties elsewhere. It is the strongest candidate for an imputation default.
4. **Speed.** The store's move to the SSD made a cell 1.6x to 2.0x faster than on the hard
   disk (2.2x to 2.5x faster than Windows). Two exact (bit-identical) decoder fixes remove its
   per-column host synchronisations; a CPU thread cap stops concurrent runs stalling.
5. **Nothing is running now,** and seven decisions wait for you (end of this report).

## 1. Pre-training ablation (E24)

Outcome: [01 § Outcome](../tickets/imputation-pretraining/01-pretraining-ablation.md). The ten
server cells ran after two dated amendments (move to the server; move to the SSD store).

| variant | B − A (no pre-training − current) | C − A (epochs to decode − current) |
|---|---|---|
| credit-g_20nan | no detectable difference | no detectable difference |
| credit-g_80nan | no detectable difference | no detectable difference |
| kr-vs-kp_40nan | no detectable difference | A better than C (rests on the one cross-machine pair) |
| pendigits_20nan | **pre-training hurts** (−0.022) | **C better** (−0.066); C 0.335 beats `knn10` 0.350 |

Estimated against real minutes per cell after the SSD move (single trainer):

| cell | estimated | real | vs the hard-disk store | vs Windows |
|---|---|---|---|---|
| pendigits s42 C | 22.2 | 21.3 | 1.6x | 2.3x |
| pendigits s42 A | 15.2 | 13.9 | 2.0x | 2.4x |
| pendigits s7 B | 8.2 | 7.8 | 1.8x | 2.2x |
| pendigits s7 C | 22.2 | 21.9 | 1.6x | 2.3x |
| pendigits s7 A | 15.2 | 13.5 | 2.0x | 2.5x |
| pendigits s13 B | 8.2 | 7.9 | 1.8x | 2.2x |
| pendigits s13 C | 22.2 | 21.7 | 1.6x | 2.3x |
| pendigits s13 A | 15.2 | 13.6 | 2.0x | 2.4x |

The hard-disk figures for A and C are modelled (training time measured here plus the write
cost fitted on the two cells that ran on the hard disk); B's is measured.

## 2. What changed in the code

All on `feat/imputation-task`, pushed; unit suite (248, with the typing gate) and both
integration fixtures pass.

- `d42f07c` **`--pretrain_objective {embedding,value,embedding_normalized}`** (ADR 0011),
  default `embedding`, so every run launched as before trains as before. Tag
  `pretrain_objective` on every parent and trial run. Backfill script
  `scripts/backfill_pretrain_objective_tag.py`: applied at 05:02–05:22 GMT-3 after a
  replica sync and a backup (`/scratch2/diogoneiss/mlflow.db.bak-2026-09-30-pretrain-objective`,
  4163 runs): 1456 parent and trial runs stamped `embedding` and re-mirrored; a second dry run
  finds nothing. Re-mirroring also created a `FAILED` mirror of the interrupted pendigits
  s42 C run in `TRIDENT/mirror/imputation`. That is ADR 0006 working as decided (decision 5:
  "A `FAILED` source is mirrored with its status"; only the trainer's live path waits for a
  normal finish); an earlier draft of this report said otherwise. Deleted at the user's
  request later that morning; `scripts/mirror_runs.py --apply` would recreate it.
- `71c6a9b`, `6dd07a5` **decoder: no per-column host synchronisation.** The same cells are
  selected in the same order; a three-fold CPU run gives all 56 fold metrics identical to the
  old code, and the ablation's GPU check cell gives induced 0.9164988203285273 before and after.
  Decode loop on pendigits: 1.30–1.42 s per epoch before, 0.42–0.48 s after, with two
  trainers sharing the GPU (about 3x; smaller alone). kr-vs-kp is unchanged (its step is
  GPU-bound).
- `scripts/experiments/`: runners, reports and queue launchers for studies 02 and 03.

## 3. Pre-training objectives (E25)

Outcome: [02 § Outcome](../tickets/imputation-pretraining/02-pretraining-objectives.md).

| variant | V − A | N − A | N − B | N − C |
|---|---|---|---|---|
| credit-g_20nan | none | none | none | none |
| credit-g_80nan | **A better** | **N better** | **N better** | none |
| kr-vs-kp_40nan | **A better** | none | **N better** | **N better** |
| pendigits_20nan | **V better** | **N better** | **N better** | **C better** |

## 4. Normalised pre-training with a long decode stage (E26)

Outcome: [03 § Outcome](../tickets/imputation-pretraining/03-normalised-pretraining-with-a-long-decode.md).

| variant | M − C (primary) | M − N | M against the best baseline |
|---|---|---|---|
| credit-g_20nan | none | none | overlaps `hgb` |
| credit-g_80nan | none | none | loses to `mean_mode` (as every arm) |
| kr-vs-kp_40nan | **M better** (−0.021) | none | overlaps `hgb` |
| pendigits_20nan | none (−0.001) | **M better** (−0.031) | **beats `knn10`** (0.334 against 0.350) |

M spends 750 epochs against C's 450, a confound stated in advance; N − C on kr-vs-kp in E25
(N at 450 epochs in total) points the same way.

## 5. Where the CPU time goes

- **The hard-disk store** was most of a run's wall time (fixed by ADR 0010 and ticket 0005).
- **Concurrent runs stalled on thread oversubscription.** Each run fits the baseline imputers
  with 32 OpenMP/BLAS threads; three at once on 32 cores took a credit-g fold's scoring from
  seconds to about ten minutes (cells at 5x their estimate). The fills are identical to the
  last value at 1, 4 and 32 threads (credit-g, pendigits), so the study launchers now cap the
  threads at 4; cells then ran at 0.9x–1.2x their single-trainer estimate with two trainers.
- **The decoder synchronised with the device per column per batch** (fixed, exact, above).
- **What is left is kernel-launch bound:** a pendigits step is about 22 ms, mostly the
  backward pass and the per-column heads. The fixes that would remove it change the numerics
  (batched per-column heads, a larger batch, `torch.compile` or CUDA graphs), so each needs a
  flag and a decision.
- **Baselines are recomputed by every arm**, although they depend only on variant, seed and
  fold: about 8 s per fold on kr-vs-kp. Caching them would give identical values.
- **Not the bottleneck:** the pandas masking and encoding (1–8% of a decode epoch), so nobody
  needs to chase it.
- The night's runs inherited nice 5 from the background session; harmless on an idle box.

### Measured the morning after (for decisions 3–5)

Scripts and raw outputs: `/scratch2/diogoneiss/trident-night-2026-09-30/analysis/`.

- **Where the best decode epoch falls** (`decode/val_loss`, per-fold curves exist only on the
  best and worst fold children, so this is a biased sample of 6 curves per cell set). With 450
  decode epochs, the median best epoch is 9–87 on credit-g and kr-vs-kp, but about 390 on
  pendigits, whose 150-epoch runs are still improving at epochs 137–143. Early stopping with
  patience 50 would save 73–86% of the decode epochs on credit-g and kr-vs-kp while keeping
  every 150-epoch run's checkpoint. On pendigits it saves 3–18%, and patience 20 or less
  loses the long decode's gain (kept checkpoint 14–17% worse in validation loss). Caveat: the
  cosine schedule spans the full epoch count, so a run re-budgeted to stop early would follow
  another path.
- **Baseline imputers per fold** (fit plus masked and induced scoring, 4 threads): credit-g_20nan
  9.1 s, credit-g_80nan 3.6 s, kr-vs-kp_40nan 8.9 s, pendigits_20nan 6.5 s. As a share of a
  cell's fold time: credit-g_20nan 18–38%, credit-g_80nan 5–19%, kr-vs-kp 4–18%, pendigits
  1.5–7%. The `hgb` fit dominates everywhere but pendigits (82–96%); on pendigits the two KNN
  scorings are half of it.
- **Batch size, GPU batch loop only, idle GPU:** pendigits 384 / 264 / 187 ms per epoch at
  256 / 512 / 1024 (2.1x at 1024), kr-vs-kp 151 / 125 / 97 ms (1.55x). The batch loop is about
  70% of a pendigits decode epoch, so 1024 would cut a pendigits C cell by about a third. It
  also means 2–4x fewer updates per epoch, so the learning rate would need retuning.

## 6. Decisions for you

1. **Was the 23:35 message the authorisation I took it for?** I read "study improving the
   pre-training architecture ... align it with values ... work on the CPU bottleneck" as
   covering ADR 0011's flag and the two exact decoder commits. All are behind defaults or
   bit-identical; revert is `git revert d42f07c 71c6a9b 6dd07a5`.
2. **Make `embedding_normalized` (and `EPOCHS_DECODE` 450) the imputation default?** E25 says it is never worse and often
   better; E26 says it keeps its gains with a 450-epoch decode stage, which pendigits needs. It would follow the usual path: flag default change for imputation
   only, tag, ADR 0011 amendment.
3. **Decode budget.** C (450 decode epochs) wins on pendigits and loses on kr-vs-kp. Tune
   `EPOCHS_DECODE` per variant, or leave it at 150?
4. **Behaviour-changing speed-ups** (batched heads, larger batch, `torch.compile`): worth a
   flag and an exactness-free comparison study?
5. **Cache the baselines per (variant, seed, fold)** across arms: identical values, but a new
   artifact to manage.
6. **Carry-overs from the log:** remove `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`
   (E17: worse than the defaults, paired p ≈ 0.008); fix the `[NULL]`-path defect D-1 (E01).
7. **Mirrors of `FAILED` runs** (answered 2026-09-30, then corrected): the mirror was deleted as
   asked, but ADR 0006 mirrors failed runs by design, so no script was changed. Keep that
   (the deleted mirror comes back at the next `mirror_runs.py --apply`), or amend ADR 0006
   to stop mirroring failed runs in the script and the backfills?

## 7. State

- Code and docs: `feat/imputation-task` at 57ec1e0 or later, pushed to GitHub. On Windows, `git pull`
  before any `sync_remote.py push` or `code`.
- Store: on the SSD, the only copy written (ADR 0010); the replica is refreshed every 10 min.
- The queue launchers stop at the night's cutoff (06:30 GMT-3 on 2026-09-30); to relaunch one
  later, set `CUTOFF_UTC`, e.g. `CUTOFF_UTC="2026-10-01 09:30:00"`.
- Removable when you are satisfied: the two store backups on `/scratch2`
  (`mlflow.db.bak-2026-09-30-zombies`, `mlflow.db.bak-2026-09-30-pretrain-objective`, about
  0.7 and 1 GB) and the merged worktree below.
- The ablation's analysis script still lives outside the repo
  (`~/trident-handoff-2026-09-29/kit/`); copy it into `scripts/experiments/`?
- Night folder (logs, per-cell results, the profiling and exactness scripts):
  `/scratch2/diogoneiss/trident-night-2026-09-30/`. The worktree
  `/scratch2/diogoneiss/TRIDENT-night` (branch `feat/pretrain-objective`, merged) can be removed.

## Afternoon follow-up (2026-09-30)

Decisions the user took after reading this report, and what was done:

- Decision 3 (early stopping), then studied: `--decode_patience` (`521df3d`, the summary fix for
  folds that stop at different epochs `a1d54a9`). **E27:** a patience of 50 keeps the scores on
  credit-g_20nan, credit-g_80nan and kr-vs-kp with 61–107 of 450 decode epochs (a kr-vs-kp cell
  18.5 → 4.9 min) and costs 0.006 on pendigits.
- Decision 4 (larger batch), studied: **E28:** batch 1024 with scaled learning rates helps only
  pendigits (x4: −0.011, 0.323, its best score yet) and hurts credit-g_20nan (x4) and kr-vs-kp
  (x2); the small tables keep one and three steps an epoch at that batch.
- Decision 5 (baseline cache), done: `8818f6d` (ADR 0007 decision 10), exact; a credit-g_20nan
  B cell 96.5 → 49.0 s.
- Decision 6, done: the promoted `credit-g_20nan` imputation file removed (`8954217`); the
  `[NULL]`-path defect D-1 fixed (`918bcd6`) and the ADR 0004 gate re-measured, `[MASK]` 8 of 8.
- Decision 7, done: `FAILED` runs are no longer mirrored (`a2a6a26`, ADR 0006 amended); the
  eight existing `FAILED` mirrors deleted.
- Both tag backfills applied: `pretrain_objective` (1456 runs) and `decode_patience` (1435 runs,
  30 minutes of re-mirroring); a second dry run of each finds nothing.
- **New exactness reference.** With `credit-g_20nan` back on the defaults, the check cell
  (`pretrain_ablation_run.py credit-g_20nan 42 B ... --no-mlflow`) gives induced
  0.8865289951105849 and masked 0.9198916682895757; the 0.9164988203285273 of the night belonged
  to the removed file.

Still open: whether the imputation default becomes `embedding_normalized`, and with which decode
setting (150, 450, 450 with patience 50, or per variant).
