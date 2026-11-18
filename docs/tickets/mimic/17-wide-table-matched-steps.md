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

## Amendment (2026-11-15 21:20 GMT-3): the leader left the network, four cells failed, the queues wait for it; nothing read

medusa1, whose MLflow server every run of this study logs to through the tunnel on port 47731, stopped answering
between 20:12 and 20:22 GMT-3 on 2026-11-15: the follower's last copy of the store is of 20:12:02,
`sync/leadership.log` records `peer_unreachable` at 20:43 with `answers_ssh: false`, and at 21:00 ssh returns "No
route to host". It is the machine's fourth absence from the network this month. With the port refused a training can
neither open nor close its run, and four cells of seed 7 failed (`progress.log`):

| cell | arm | ran (GMT-3, 2026-11-15) | failed |
|---|---|---|---|
| `M mimic-wide_natural 7` | F | 11:37 to 20:54 | closing its MLflow run, the five folds trained |
| `D10k mimic-wide-donor_tmatch_20nan 7` | C | 17:53 to 21:19 | closing its MLflow run, the five folds trained |
| `D10kS mimic-wide-donor_tmatch_20nan 7` | S | 20:54 to 21:00 | opening its run, nothing trained |
| `D10k mimic-wide-donor_mcar_20nan 7` | C | 21:00 to 21:04 | opening its run, nothing trained |

None of the four wrote a `cv_summary.json`, so the runner does not count them and the reader cannot see them. I
stopped both queues with the `STOP` file at 21:01 GMT-3; they ended at 21:04 and 21:19. **11 of 27 runs are in**: the
nine of seed 42, and `M10k` and `M10kS` on the natural table at seed 7. No score of any cell was read, of the
finished ones or of the failed ones.

What follows changes nothing in the design, the arms, the seeds or the rule:

- The four cells are retrained whole at seed 7, as new runs, by relaunching the two queues as they were launched
  (`c15_queue.sh` skips a cell that has a `cv_summary.json`). Two of them repeat a training that ended; their first
  attempts' files stay in `results/` without a summary and are not read (`mimic_report.py` takes a cell's one run
  directory with a summary and exits on a cell with two).
- The failed attempts' runs stay in the leader's store as its server left them; I do not edit them.
- A detached script in the state directory, `resume_when_leader_answers.sh` (started 21:20 GMT-3), removes `STOP`
  and relaunches the queues once the server answers three checks a minute apart, and writes a line to
  `progress.log` when it does.
- **Not taken, the user's call:** promoting gorgona8 by hand (ADR 0033, the two servers as leader and follower,
  decision 6), and training with MLflow off. Without either, the same decision 6 has the follower promote itself
  after 24 hours of silence, at about 20:12 GMT-3 on 2026-11-16, if medusa1 still does not answer ssh; the script
  above would then relaunch the queues against gorgona8's own server, and the remaining runs would land in the fork
  that ADR 0033 reconciles afterwards.
- **Cost:** about 12 h 40 of trainings on the shared card are repeated (9 h 17 and 3 h 26), and the card is idle on
  this study's side until the leader returns.

## Amendment (2026-11-18 11:27 GMT-3): the resume script died with gorgona8's reboots; restarted, still 11 of 27, nothing read

gorgona8 rebooted at 18:37 GMT-3 on 2026-11-16, at 13:55 on 2026-11-17 and at 10:15 on 2026-11-18 (`last -x reboot`).
The first reboot killed `resume_when_leader_answers.sh`, so when medusa1 answered again (contact regained 13:58 GMT-3
on 2026-11-17, `sync/leadership.log`) the queues were not relaunched; the leader's GPUs went to E181 meanwhile. The
follower did not promote itself in the 42 hours of silence (no `promoted` line; why is not checked here), so no
fork exists. medusa1 stopped answering again before 11:26 GMT-3 on 2026-11-18 (last copy of the store 11:22).

The study is where the first amendment left it: 11 of 27 runs, `STOP` in place, the four failed cells of seed 7 to
retrain whole. I restarted the script at 11:27 GMT-3 on 2026-11-18, unchanged; a reboot kills it again, so it is
checked after each one. Design, seeds and rule unchanged; promoting gorgona8 by hand stays the user's call.

## Amendment (2026-11-18 18:47 GMT-3): medusa1 fell again after three cells; 14 of 27, queues ended, nothing relaunched

The script relaunched both queues at 11:31 GMT-3 on 2026-11-18. Three cells finished at seed 7 (`D10k` and `D` on
`tmatch`, `M` on the natural table). medusa1 stopped answering at about 14:31 GMT-3 (last copy of the store 14:30;
`peer_unreachable` at 15:01). `D10kS` on `mcar` at seed 7 trained from 14:33 to 16:40 and failed closing its run;
the other twelve cells the queues reached failed opening theirs (`ConnectionRefusedError` on port 47731), and both
queues ran out of cells at 15:03 and 17:01. The resume script had already exited after its relaunch, so nothing of
this study runs now.

**14 of 27 runs are in**, all of seed 42 and five of seed 7; to train are four of seed 7 (`D10kS` on `tmatch` and
`mcar`, `D10k` and `D` on `mcar`) and all nine of seed 13. No score read. On the user's instruction of 2026-11-18
("nao relance coisas na medusa1") nothing is relaunched; since every run of this study logs to medusa1's server,
the study stays paused until the user says how it continues. Design, seeds and rule unchanged.

## Amendment (2026-11-18 19:38 GMT-3): gorgona8 leads, on the user's call; the third amendment's cause corrected; 13 cells relaunched

**Correction to the third amendment.** medusa1 did not fall at 14:31: it was up from 11:26 GMT-3 on, and its MLflow
server ran from 11:28. What broke was gorgona8's side. The tick of `scripts/leadership.py` started at 15:01 GMT-3
took the fork-base backup of ADR 0033 (decision 6), and its `quick_check` of the 25 GB copy on the hard disk was still
reading at about 0.4 MB/s four hours later, in uninterruptible disk wait. While it held the tick lock no tick ran, so
the ssh tunnel to medusa1's server on port 47731 was not reopened, and the thirteen cells failed against a closed
port. Why the tunnel and the 14:31 copy dropped first is not checked.

**The user's go:** "promova a gorgona8 a lider! apenas lembre de fazer backup do mlflow em caso de algo dar errado"
(2026-11-18), then "o relancar era para coisas na gpu, comandos e etcs voce pode fazer sim" for commands run on medusa1.

- **Backup first:** on medusa1, `leadership.py backup --kind pre-switchover` at 19:17 GMT-3:
  `/scratch/diogoneiss/TRIDENT/sync/backups/2026-11-18T2217-pre-switchover.db`, 25,374,081,024 bytes, 34,214 runs,
  `quick_check` ok. The stuck tick (pid 394683) was then killed by the user; its unverified copy of the 14:30 store
  stays at `sync/backups/2026-11-18T1801-fork-base.db.staged` on gorgona8.
- **Switchover, not promotion:** medusa1 answered, so `promote` would have refused. `leadership.py switchover` at
  19:34 GMT-3 had medusa1 yield, pulled its last pages and made gorgona8 the leader at epoch 3, with no fork. Checked
  from the files: both `role.json` at epoch 3 (gorgona8 `leader`, medusa1 `follower`, its store mode 444, no server
  there), gorgona8's server answering `/health`, and 34,214 runs in gorgona8's store, as in the backup.
- **Relaunch:** the two queues at 19:35 GMT-3 as launched before, writing to gorgona8's store; the thirteen cells
  of the third amendment are retrained whole. Runs from here carry the store's epoch-3 history; nothing in the design,
  seeds, rule or reader changes, and nothing has been read.
