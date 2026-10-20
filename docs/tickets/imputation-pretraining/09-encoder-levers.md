# The decode stage's levers against washing out the encoder (E50; pre-registered 2026-10-19)

This ticket was written and committed before the first cell.

- **Why:** E49 (T28's D1, [08](08-decode-curves.md)) found washout on pendigits_20nan.
  - Normalised pre-training gives the decode stage a +58% head start in its validation loss.
  - The 450-epoch decode stage erases it by about epoch 185.
  - On kr-vs-kp_40nan the gain lasts (+2.5% at the minima).
  - T28's report (§5) names the levers to try before T13: hold the encoder fixed for the first
    decode epochs (SMART's unfreeze epoch), or give it a lower learning rate. The user agreed on
    2026-10-19.
- **Code, behind the usual gate** (flag, tag and backfill; the old behaviour is the default):
  - `--decode_freeze_epochs N` (`DECODE_FREEZE_EPOCHS`, default 0): the first N decode epochs train
    the heads only, with the embedder and transformer held fixed.
  - `--decode_encoder_lr_scale s` (`DECODE_ENCODER_LR_SCALE`, default 1.0): the embedder and
    transformer get s times the decode learning rate, in a parameter group of their own.
  - Tags `decode_freeze_epochs` and `decode_encoder_lr_scale` on every imputation run. Older runs
    are stamped 0 and 1.0 by `scripts/backfill_decode_encoder_tags.py` (dry run by default; its
    `--apply` waits for the user).
  - **Tests:**
    - unit tests show a frozen encoder coming out unchanged, and training again after the freeze;
    - the optimizer is split into heads and encoder only when the scale is not 1.0;
    - both regression fixtures pass unmodified, so the default path did not move.

## Design (fixed before the first cell)

**Arms.** Each is today's default as a run reads it (the variant's promoted file over the task
defaults), with `--score_checkpoints`, plus one change. Runner:
`scripts/experiments/encoder_levers_run.py`; tag `experiment=encoder-levers-2026-10-19` plus
`arm`.

| arm | change |
|---|---|
| G | none |
| F100 | `DECODE_FREEZE_EPOCHS 100`: heads only for the first 100 of 450 decode epochs |
| L10 | `DECODE_ENCODER_LR_SCALE 0.1`: the encoder at a tenth of the decode learning rate throughout |
| C0 | `EPOCHS_PRE 0`: no pre-training (the reference E49 compared against) |

- **The lever settings:** 100 epochs sits inside E49's head start, which lasts to about epoch 185
  on pendigits. The 0.1 scale undoes the decode stage's roughly 3x rate over pre-training (T28,
  H3).

**Cells.**

- **Variants:**
  - pendigits_20nan, where E49 found washout. It is on today's bigger encoder (ADR 0018).
  - kr-vs-kp_40nan, where E49 found real transfer. It is the control that a lever must not harm.
- **Seeds:** 101, 202 and 303. 24 cells in all.
  - **Why these seeds:** G must equal the development runs of today's default on them, fold by fold
    (`--integrity`). Those are E38's ALL arm on pendigits_20nan and E39's D arm on kr-vs-kp_40nan,
    read through `report_common.development_source`.
  - These seeds have decided earlier studies, so E50 is a development reading (ADR 0020).
- **Running it:** four queues (`scripts/experiments/run_encoder_levers_queue.sh`).
  - About 73 minutes per pendigits cell and 19 per kr-vs-kp cell, from those runs.
  - About 4.5 h of wall time; no cutoff.

## Measures

All on `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI, through
`scripts/experiments/encoder_levers_report.py --tally`:

- **Primary:** F100 − G and L10 − G on K (the default checkpoint, read uncalibrated per ADR 0027),
  induced population.
- **Secondary:**
  - the same on H and on the masked population;
  - C0 − G: does pre-training help under today's default;
  - each arm's minimum of the cross-fold decode validation loss.

## Reading, fixed now

- **supported:** a lever is better than G on pendigits_20nan and not worse than G on
  kr-vs-kp_40nan;
- **refuted:** neither lever is better than G on pendigits_20nan;
- **mixed:** otherwise, a lever that helps pendigits but hurts kr-vs-kp.

**What each reading leads to (never a default change from this study):**

- **supported:** the lever goes to a pre-registered screening on the 21 variants. Only that
  screening could lead to an ADR.
- **refuted:** the washout levers are closed for now. T28 moves to D2 (collapse and probes) and D3
  (frozen encoder against random). The decode-loss head start E49 saw then does not reach
  `impute_score`.
- **mixed:** reported. A follow-up would need the user's choice.

**If the integrity check fails** (a difference above 1e-6 on G's K), nothing is read until it is
explained.

## Launch

To be filled in at launch.
