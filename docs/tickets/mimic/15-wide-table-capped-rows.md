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
