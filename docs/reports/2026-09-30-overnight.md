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
  finds nothing. Side effect: re-mirroring the tree of the interrupted pendigits s42 C run
  (`FAILED`, ADR 0006 mirrors only finished runs) created a `FAILED` mirror of it in
  `TRIDENT/mirror/imputation`; every report ignores it (FINISHED, non-mirror runs only).
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
7. **Delete the `FAILED` mirror** of `impute_pendigits_20nan_20260930_005934` in the UI, and
   should the backfill scripts' re-mirroring skip trees whose root did not finish (the
   `search_objective` backfill shares the mechanism)?

## 7. State

- Code and docs: `feat/imputation-task` at 57ec1e0 or later, pushed to GitHub. On Windows, `git pull`
  before any `sync_remote.py push` or `code`.
- Store: on the SSD, the only copy written (ADR 0010); the replica is refreshed every 10 min.
- Night folder (logs, per-cell results, the profiling and exactness scripts):
  `/scratch2/diogoneiss/trident-night-2026-09-30/`. The worktree
  `/scratch2/diogoneiss/TRIDENT-night` (branch `feat/pretrain-objective`, merged) can be removed.
