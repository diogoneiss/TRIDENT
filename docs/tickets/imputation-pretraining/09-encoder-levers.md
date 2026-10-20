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

- **Checked before launch, with `--no-mlflow`, passed (2026-10-19, about 21:10 GMT-3):**
  - **G on kr-vs-kp_40nan at seed 101** equals E39's D cell fold by fold on K, both populations
    (largest difference 0). It now also logs ADR 0027's calibrated headline, read apart.
  - **F100 on kc2_60nan at seed 7** (not a study cell) ran end to end. Its scores were not read.
- **Launched at 21:11 GMT-3** on f7541b4: four queues in
  `/scratch2/diogoneiss/trident-2026-10-19-encoder-levers/`.

## Outcome (2026-10-20, 01:55 GMT-3)

All 24 cells finished, none failed. The last queue ended at 01:53 GMT-3 on 10-20.

**Integrity: passed.** G equals the development runs of today's default fold by fold on K, both
populations: E38's ALL arm on pendigits_20nan and E39's D arm on kr-vs-kp_40nan, largest
difference 0 on all six (variant, seed) pairs.

**The printout** (`scripts/experiments/encoder_levers_report.py --tally`, unedited). Each line is
a mean difference over the 15 fold pairs of `impute_score` (lower is better), with the 95%
interval and the call.

```text
=== pendigits_20nan
  G: K 0.3203; decode validation loss at its minimum 0.1248
  F100 − G, K +0.0109 [+0.0090, +0.0128] worse; F100 − G, H +0.0117 [+0.0093, +0.0141] worse; F100 − G, masked K +0.0113 [+0.0093, +0.0133] worse; decode validation loss at its minimum 0.1343
  L10 − G, K +0.0371 [+0.0341, +0.0400] worse; L10 − G, H +0.0366 [+0.0337, +0.0395] worse; L10 − G, masked K +0.0398 [+0.0375, +0.0421] worse; decode validation loss at its minimum 0.1531
  C0 − G, K +0.0038 [+0.0015, +0.0060] worse; C0 − G, H +0.0037 [+0.0017, +0.0057] worse; C0 − G, masked K +0.0015 [-0.0005, +0.0036] none; decode validation loss at its minimum 0.1253
=== kr-vs-kp_40nan
  G: K 0.7258; decode validation loss at its minimum 0.2876
  F100 − G, K +0.0016 [-0.0095, +0.0128] none; F100 − G, H +0.0058 [-0.0035, +0.0151] none; F100 − G, masked K -0.0059 [-0.0197, +0.0079] none; decode validation loss at its minimum 0.2880
  L10 − G, K +0.0322 [+0.0224, +0.0420] worse; L10 − G, H +0.0286 [+0.0209, +0.0364] worse; L10 − G, masked K +0.0267 [+0.0126, +0.0408] worse; decode validation loss at its minimum 0.3002
  C0 − G, K +0.0000 [-0.0068, +0.0069] equivalent; C0 − G, H +0.0035 [-0.0045, +0.0115] none; C0 − G, masked K +0.0081 [-0.0083, +0.0245] none; decode validation loss at its minimum 0.2882

Primary: levers better than G on pendigits_20nan (K): none; of those, not worse on kr-vs-kp_40nan: none -> refuted (fold-pair t, seed agreement)
```

**Reading, by the rule fixed in advance: refuted.** Neither lever is better than G on
pendigits_20nan; both are worse.

- **F100** (heads only for the first 100 decode epochs):
  - pendigits: +3.4% of G's K (+0.0109), worse on K, H and masked;
  - kr-vs-kp: no difference (+0.2%).
- **L10** (encoder at a tenth of the decode learning rate):
  - pendigits: +11.6% (+0.0371);
  - kr-vs-kp: +4.4% (+0.0322), worse on every readout.
- **The decode loss agrees.** Each lever's minimum of the cross-fold decode validation loss is
  above G's:
  - pendigits: 0.1343 and 0.1531 against 0.1248;
  - kr-vs-kp: 0.2880 and 0.3002 against 0.2876.
- **C0 − G (secondary): does pre-training help under today's default?**
  - **pendigits_20nan:** worse without it, +1.2% (+0.0038) on K and on H; no difference on masked.
    Under today's bigger encoder, pre-training gives pendigits a small gain that lasts to K.
  - **kr-vs-kp_40nan:** equivalent on K (+0.0000, inside the 1% SESOI); no difference on H and
    masked. Today's D configuration gets nothing from pre-training here, though E49 read the
    decode loss as real transfer (+2.5% at the minima). E49 read E24–E26's arms, which are not
    today's configuration. Today's decode-loss minima differ by 0.2% (0.2882 against 0.2876).

**What it means:**

- **The decode stage needs the encoder to move, at full rate, from the start.** Holding it fixed
  or slowing it costs more than any washout it prevents.
- **E49's washout reading does not reach `impute_score`.** The +58% head start in decode loss
  disappears at today's 450 epochs. Even so, pre-training still leaves pendigits 1.2% better at K
  than no pre-training.

**What it leads to (fixed in advance):**

- The washout levers are closed for now. The two flags stay, with their defaults of 0 and 1.0.
- T28 moves to D2 (collapse and linear probes under the normalised objective) and D3 (a frozen
  pre-trained encoder against a random one).
- The tag backfill (`scripts/backfill_decode_encoder_tags.py`) has the user's go. Its dry run at
  23:41 GMT-3 on 10-19 found 2,716 runs in 1,312 trees. The apply waits for the quiet gap after
  E51, behind a backup.
- **Limit:** the seeds are development seeds (101/202/303), so this is a development reading
  (ADR 0020).

