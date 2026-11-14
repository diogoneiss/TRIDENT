# TRIDENT on 10,000 rows of `mimic-wide` at the full table's optimizer steps: is the capped arm's loss rows or steps? (E180, T57; pre-registered 2026-11-14 19:39 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E177 ([mimic/15](15-wide-table-capped-rows.md)) trained TRIDENT on 10,000 rows of each fold of
  `mimic-wide` and found it worse than E176's full-table decoder on all three variants (+0.0158, +0.0103, +0.0132).
  It kept the full table's epochs, so the capped runs also took 3.2 to 4.5 times fewer optimizer steps, and the
  ticket said the two were not separated. This study adds the missing arm: 10,000 rows, the full table's steps.
- **The user's go:** "Sim, pré registre e vá" (2026-11-14), answering "o braço de passos equiparados do E177 (TRIDENT
  com 10.000 linhas da `mimic-wide`), para separar o efeito das linhas do efeito dos passos de otimização. Posso
  pré-registrar e soltar?".
- **Disclosure:**
  - **Three arms are trained, not one, and this is my choice, not the user's.** The go was for the matched-steps
    arm. medusa1, where E176's and E177's runs were trained, has had no CUDA since its GPU 0 failed on 2026-11-13,
    so the new arm can only train on gorgona8. ADR 0035 (a study may split between the servers by seed) keeps both
    arms of a pair on one machine, so the capped and the full arms are retrained on gorgona8 beside the new one.
    Reading the new arm against medusa1's files instead would be a departure from that ADR, the user's call; it is
    not taken here. It costs 18 more runs.
  - E176's and E177's scores on medusa1 were read (above). E178's two H runs on gorgona8 (`mimic-wide_natural` and
    `mimic-wide-donor_mcar_20nan`, seed 13) trained the full-table configuration on this machine and were read in
    E178's tally; they are not reused.
  - **What differs between the matched-steps arm and the full arm besides the rows:**
    - its checkpoint is chosen among 1440 or 2014 decode epochs against 450, so among more validation readings;
    - the cosine schedule steps once an epoch, so it traces the same curve over the same steps in finer steps;
    - each of its rows is seen 3.2 or 4.5 times more often.
    Pre-training keeps its last epoch in every arm. The batch is 256 in every arm.
  - On the natural table the scaled epochs are rounded up: 53,720 and 80,560 steps against 53,700 and 80,550.
  - One timing cell ran: arm `D10kS` on `mimic-wide-donor_tmatch_20nan` at seed 999, a seed outside the study, with
    MLflow off, stopped by hand after 55 pre-training epochs. Only its clock and its epoch count (960) were read.
  - The matched-steps runs log `config_source = override`: their epochs reach the run through
    `hyperparams_override`, over the same task defaults the other arms resolve to. Checked before launch: against
    the full arm, the capped arm's resolved configuration differs in `train_rows` only, and the matched-steps arm's
    in `train_rows`, `pretraining_epochs` and `decode_epochs` only.

## Design

- **Arms, per variant:**
  - **F**, the full table (`M` on the natural table, `D` on the donor variants): E176's configuration.
  - **C**, capped (`M10k`, `D10k`): `--train_rows 10000`, the full table's 300 pre-training and 450 decode epochs.
    E177's configuration.
  - **S**, capped at the full table's steps (`M10kS`, `D10kS`): `--train_rows 10000` with the epochs scaled so the
    optimizer takes the full arm's steps. At batch 256 an epoch is 40 steps on 10,000 rows, 128 on the donor
    table's 32,550 training rows and 179 on the natural table's 45,755:

    | table | pre-training epochs | decode epochs |
    |---|---|---|
    | donor (`tmatch`, `mcar`) | 960 | 1440 |
    | natural | 1343 | 2014 |

- **Cells:** E176's three variants: `mimic-wide_natural` (`--decode_checkpoint masked`, masked population),
  `mimic-wide-donor_tmatch_20nan` and `mimic-wide-donor_mcar_20nan` (defaults, readout K, induced cells). Seeds 42,
  7, 13. 27 runs, five folds each. As in E177, the naive fills, the baselines, the validation rows and the scored
  rows keep the whole fold in every arm.
- **Code:** runner `scripts/experiments/mimic_wide_steps_run.py` (no `src/` change, no new flag), tag
  `mimic-wide-steps-2026-11-14` with the arm's name; state `/scratch/diogoneiss/trident-2026-11-14-mimic-wide-steps` on gorgona8.
- **Machine:** gorgona8, every run, two queues on its one card; runs log to the leader's store on medusa1 through
  the tunnel and carry `host = gorgona8`. If medusa1's CUDA returns, whole seeds may move there (every arm of
  every variant of a seed), by a dated amendment.
- **Cost:** the timing cell took 1.43 s a pre-training epoch of 40 steps while another account's 15 processes held
  the card at 100%: a step about three times as long as E178 measured on this card alone. The 27 runs are about
  3.6 million steps: in two queues, between a day and a half if the card frees and five days if it stays as
  contended as at launch.
- **Not in this study:** TabPFN (not refit); other caps; matching by a smaller batch (`--min_steps_per_epoch`),
  which would change the batch with the steps.

## Measures and rule (fixed now)

`impute_score` per fold, lower is better. Per variant, three contrasts on the 15 (seed, fold) pairs, by the
project's fold-pair rule (95% t interval, every seed agreeing), equivalence margin 1% of F's mean score:

- **C − F**, the premise: E177's contrast, on this machine.
- **S − C**, the steps, at 10,000 rows.
- **S − F**, the rows, at the full table's steps.

A variant's reading, the capped arm's loss belongs to:

| C − F | S − C | S − F | reading |
|---|---|---|---|
| not worse | any | any | no loss to split |
| worse | better | equivalent or better | steps |
| worse | better | worse | both |
| worse | equivalent or worse | worse | rows |
| worse | anything else | | unclear |

- **Primaries:** `mimic-wide_natural` and `mimic-wide-donor_tmatch_20nan`, as in E176 and E177.
- **Hypothesis:** part of the loss is steps: S is better than C on both primaries. **Supported** if both primaries
  read "steps" or "both"; **refuted** if both read "rows"; **mixed** otherwise. The study's line prints the
  primaries' common reading, or "mixed".
- **Secondary, no rule:** `mcar`; the share of C − F that the steps recover, (C − S) / (C − F) on the means; S's
  lead over TabPFN v2 against the TabPFN means E178 refit on gorgona8 at these seeds and folds (0.7644, 0.9662,
  0.7216, in [mimic/16](16-wide-table-hybrid.md)'s printout).
- Three seeds, fixed; no extension.
- **Read** once with the 27 runs in, on gorgona8:
  `QUEUE_STATE=/scratch/diogoneiss/trident-2026-11-14-mimic-wide-steps TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_report.py e180`
- **What it leads to:** the wording of the thesis sentence on E176 and E177 (what the rows past 10,000 buy, as
  rows or as training length), the user's call. No ADR.
