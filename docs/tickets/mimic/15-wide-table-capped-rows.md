# TRIDENT on 10,000 rows of `mimic-wide`: does the lead over TabPFN v2 need the other rows? (E177, T57; pre-registered 2026-11-13 08:28 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E176 ([mimic/14](14-wide-table.md)) found the decoder better than TabPFN v2 on `mimic-wide`'s natural gaps
  and transplanted patterns, but had no arm separating "TRIDENT trains on every row, TabPFN on 10,000" from "the
  model is better on these patterns": under MCAR, with the same cap on TabPFN, the two were level. This study trains
  TRIDENT on 10,000 rows of each fold and reads it against the same TabPFN files.
- **The user's go:** "2 e 1, vá" (2026-11-13), answering "Posso soltar o híbrido, e você autoriza a flag em `src/`
  para o braço de 10.000 linhas?"; option 2 was this arm, and the answer authorises the flag.
- **Disclosure:**
  - E176's scores were read: decoder − TabPFN −0.0991 on the natural table, −0.0577 on `tmatch`, −0.0044 on `mcar`.
  - TabPFN is not refit: its nine files are E176's, on the same folds and cells.
  - The cap is not symmetric, and it leans against TRIDENT. TabPFN's context is 10,000 *observed* rows per column;
    TRIDENT's cap is 10,000 rows with their gaps, so on `tmatch` it sees about 4,200 observed PTT values against
    TabPFN's 10,000.
  - The epochs are the full-table ones, so a capped run takes 3.3 times fewer optimizer steps on the donor table and
    4.6 times fewer on the natural one. A capped arm that loses may be losing to steps and not to rows; no arm with
    matched steps is run here.
  - One timing cell ran on `mimic-wide-donor_tmatch_20nan` at seed 999, a seed outside the study, with MLflow off,
    stopped after its first fold; only its clock and its row counts were read, no score (below).

## The flag (commit 9e780d3)

`--train_rows N` (`TRAIN_ROWS`, tag `train_rows` on imputation runs; default 0, every row, every run before it):
the pre-training and decode models train on a fixed-seed sample of N of the fold's training rows, in table order
(`src/training/schedulers.py::training_rows`). The sample comes from its own generator, so with the flag off no
random stream moves and both integration fixtures pass unchanged. The scaler is fit on the whole table, as in
every run; the naive fills the score divides by and the baseline imputers keep the fold's whole training split;
the validation rows that choose the checkpoint and the scored test rows do not change. So a capped run's scores are
in E176's units, on E176's cells. No backfill: as with `min_steps_per_epoch`, a
run without the tag trained on every row.

## Design

- **Cells:** E176's three, with the cap: `mimic-wide_natural` under arm `M10k` (`--decode_checkpoint masked
  --train_rows 10000`, masked population), `mimic-wide-donor_tmatch_20nan` and `mimic-wide-donor_mcar_20nan` under
  `D10k` (the defaults with `--train_rows 10000`, readout K, induced cells). Seeds 42, 7, 13; 9 runs, five folds
  each. Runner `scripts/experiments/mimic_wide_run.py`, tag `mimic-wide-2026-11-12` with arm `M10k`/`D10k`, in
  E176's state directory `/scratch/diogoneiss/trident-2026-11-12-mimic-wide` (its `results/` are keyed by arm).
- **Opponents, already there:** E176's nine TabPFN files and its nine full-table runs.
- **Machine (ADR 0035, the split of studies between servers):** medusa1, where E176's runs and TabPFN files were
  made; one queue on GPU 0.
- **Cost:** the timing cell's first fold took 3.2 minutes on an RTX 5090 (67 s of pre-training, 115 s of decoding, the baselines read from E176's cache; its training rows printed as 32,550 -> 10,000), against 16 minutes uncapped: about 16 minutes a donor run and somewhat more on the natural table, so about 3 h for the 9 runs in one queue, more beside E178's queue on the same card.
- **Not in this study:** a matched-steps arm; other caps; `trand`; the hybrid on this table (E178, mimic/16).

## Measures and rule (fixed now)

`impute_score` per fold, lower is better. Per variant, the 15 (seed, fold) differences capped decoder − TabPFN, by
the project's fold-pair rule (95% t interval, every seed agreeing), equivalence margin 1% of TabPFN's mean score:
E176's rule, unchanged.

- **Primaries:** `mimic-wide_natural` and `mimic-wide-donor_tmatch_20nan`.
- **Supported** (the lead does not need the other rows) if the capped decoder is worse than TabPFN on neither primary
  and better on at least one; **refuted** if worse on either; **no verdict** otherwise.
- **Secondary:** capped decoder − E176's decoder per variant, what the rows past 10,000 buy; the contrast with TabPFN
  on `mcar`; capped decoder − the bar (fit on the whole fold).
- Three seeds, fixed; no extension.
- **Read** once with the 9 capped runs in, on medusa1:
  `QUEUE_STATE=/scratch/diogoneiss/trident-2026-11-12-mimic-wide TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_report.py e177`
- **What it leads to:** the wording of E176's thesis sentence (with or without "past TabPFN's context" as the
  reason), the user's call. No ADR.

## Outcome (2026-11-13 11:29 GMT-3): supported

The 9 capped runs finished between 08:48 and 11:27 GMT-3 on 2026-11-13, about 20 minutes each beside E178's queue on
the same card, none failed. Read once, on medusa1. The printout, unedited below its first two lines (the commit and
the clock, added here):

```
1085c6e
read 2026-11-13 11:28 GMT-3
mimic-wide_natural (masked cells): decoder 0.6811, TabPFN 0.7644 (fit 19 s a fold), bar 0.6887
  decoder - TabPFN: -0.0833 [-0.0843, -0.0823], by seed -0.0839, -0.0825, -0.0835 -> better
  decoder - bar: -0.0076 [-0.0086, -0.0066], by seed -0.0079, -0.0074, -0.0074 -> better
  decoder - E176's decoder: +0.0158 [+0.0153, +0.0163], by seed +0.0155, +0.0156, +0.0164 -> worse
mimic-wide-donor_tmatch_20nan (induced cells): decoder 0.9188, TabPFN 0.9662 (fit 25 s a fold), bar 0.9352
  decoder - TabPFN: -0.0474 [-0.0486, -0.0462], by seed -0.0474, -0.0478, -0.0470 -> better
  decoder - bar: -0.0164 [-0.0178, -0.0149], by seed -0.0164, -0.0156, -0.0171 -> better
  decoder - E176's decoder: +0.0103 [+0.0093, +0.0112], by seed +0.0108, +0.0098, +0.0102 -> worse
mimic-wide-donor_mcar_20nan (induced cells): decoder 0.7303, TabPFN 0.7216 (fit 26 s a fold), bar 0.7303
  decoder - TabPFN: +0.0087 [+0.0073, +0.0101], by seed +0.0090, +0.0079, +0.0092 -> worse
  decoder - bar: -0.0000 [-0.0010, +0.0009], by seed +0.0004, -0.0005, +0.0000 -> equivalent
  decoder - E176's decoder: +0.0132 [+0.0120, +0.0143], by seed +0.0137, +0.0125, +0.0133 -> worse
PRIMARIES (natural, tmatch) against TabPFN: ['better', 'better'] -> supported (fold-pair t, seed agreement, margin 1% of TabPFN's mean)
```

- **Primaries:** trained on 10,000 rows a fold, the decoder is still better than TabPFN v2 on both, by −0.0833 on the
  natural table and −0.0474 under the transplanted patterns, every seed agreeing. Supported: the lead on the real
  patterns does not need the rows past TabPFN's context. It keeps 84% of E176's lead on the natural table (−0.0991)
  and 82% on `tmatch` (−0.0577).
- **What the other rows buy:** the capped decoder is worse than E176's on all three variants, by +0.0158, +0.0103
  and +0.0132, every seed agreeing. This mixes rows and optimizer steps (3.3 to 4.6 times fewer here); no arm
  separates them.
- **Under MCAR:** the capped decoder is worse than TabPFN, by +0.0087, past the 1% margin (0.0072), and level with
  the bar (0.7303 both). E176's parity there (−0.0044) did come from the extra rows or steps.
- **The bar:** the capped decoder is still better than the bar on the natural table (−0.0076) and on `tmatch`
  (−0.0164), with the bar fit on the whole fold.
- **Scope:** TabPFN v2 as ADR 0028 pins it, its files not refit; one table, seeds 42/7/13; a cap that leans against
  TRIDENT (rows with their gaps against observed rows a column) and full-table epochs.
- **What it leads to:** E176's thesis sentence can say the lead holds at equal row counts on real gap patterns and
  that under MCAR the decoder needs the whole table to stay level; the wording is the user's call. No ADR changes.
