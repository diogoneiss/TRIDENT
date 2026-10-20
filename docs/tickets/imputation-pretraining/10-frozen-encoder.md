# T28's D3: a frozen encoder, pre-trained against random (E52; pre-registered 2026-10-20)

This ticket was written and committed before the first cell.

- **Why:** T28's report ([2026-10-19](../../reports/2026-10-19-t28-why-pretraining-adds-little.md),
  §3 H1 and H3, §5 step 3) asks whether pre-training leaves a usable representation at all.
  - **The test:** decode with the encoder held fixed throughout, from a pre-trained start and from
    a random one.
  - **What came before:** E49 (D1) saw the head start wash out on pendigits. E50 then found that
    neither holding the encoder for 100 epochs nor slowing it helps.
- **The user's go:** 2026-10-20, about 08:40 GMT-3, for T28's D2 and D3 before T29 and TabPFN.
- **No new code in `src/`.** E50's `--decode_freeze_epochs` set to the variant's own decode budget
  freezes the embedder and transformer for every decode epoch.
  - `EPOCHS_PRE 0` still enters `train_pretrainer`, so the random encoder is the Xavier sweep the
    stage starts from: the same start as E50's C0 arm.
  - **Checked before launch:** FR on kc2_20nan at seed 7 (`--no-mlflow --check-frozen`, not a
    study cell). All 154 tensors of the embedder and transformer leave the decode stage
    bit-identical, on all five folds.

## Design (fixed before the first cell)

**Arms.** Each is today's default as a run reads it (the variant's promoted file over the task
defaults), with `--score_checkpoints`, plus one change. Runner:
`scripts/experiments/frozen_encoder_run.py`; tag `experiment=frozen-encoder-2026-10-20` plus `arm`.

| arm | change |
|---|---|
| FN | `DECODE_FREEZE_EPOCHS` = the decode budget (450 on all four variants): today's normalised pre-training, then heads only |
| FA | the same, after the shipped `embedding` objective (`PRETRAIN_OBJECTIVE embedding`) |
| FR | the same, with `EPOCHS_PRE 0`: the never-trained Xavier encoder |
| C0 | `EPOCHS_PRE 0` alone: a full decode without pre-training. On credit-g only |

**Read from earlier runs on the same seeds and folds:**

- **G**, today's full decode:
  - E50's G arm on pendigits_20nan and kr-vs-kp_40nan. It reproduced E38 ALL and E39 D exactly.
  - The development run on credit-g, through `report_common.development_source`.
- **C0** on pendigits_20nan and kr-vs-kp_40nan: E50's C0 arm.

**Cells.**

- **Variants:** E26's four, the ones D1 read: pendigits_20nan, kr-vs-kp_40nan, credit-g_20nan and
  credit-g_80nan.
- **Seeds:** 101, 202 and 303. These are development seeds, so this is a development reading (ADR
  0020).
- **Size:** 42 cells, 3 arms on each variant plus C0 on the two credit-g ones.
- **Running it:** four queues (`scripts/experiments/run_frozen_encoder_queue.sh`), longest first.
  - About 2.5–3 h, a guess: a frozen encoder skips the backward pass through it, but the
    pre-training stage still runs on FN and FA.
  - No cutoff.

## Measures

All on `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI, through
`scripts/experiments/frozen_encoder_report.py --tally`:

- **Primary:** FN − FR on K (uncalibrated, ADR 0027), induced population, per variant.
- **Secondary:**
  - FA − FR (does the shipped objective leave anything);
  - FN − FA;
  - FN − G (what training the encoder in the decode stage adds);
  - FR − C0 (the same from a random start);
  - C0 − G (what pre-training adds to a full decode; E50's for pendigits and kr-vs-kp);
  - each on H and on the masked population too.

## Reading, fixed now

- **Per variant:** a usable pre-trained representation if FN is better than FR; nothing usable if
  they are equivalent or show no difference.
- **Over the four:**
  - **usable:** FN better than FR on 3 or more;
  - **nothing usable:** on none;
  - **mixed:** otherwise.

**What it leads to, with D2 (E53, pre-registered separately before it runs).** E50 has closed the
report's branch "usable but washing out, try the levers". Nothing here changes a default; each
"propose" is a pre-registered study and an ADR for the user.

| D3 (this study) | D2 (E53) | leads to |
|---|---|---|
| nothing usable or mixed | no value information under N | propose dropping pre-training for imputation: a 21-variant `EPOCHS_PRE 0` study under today's defaults |
| usable | N collapsing | T13 in plan §2h's screening shape (an EMA embedder, an EMA encoder with a contextual target, the centred target) |
| usable | no collapse | the representation is real and the full decode overtakes it (FN − G). Propose the same 21-variant `EPOCHS_PRE 0` study, to price what pre-training is worth today; T13 has nothing to fix |
| any | a diagnostic moves with `LR_PRE` or the batch | T12 first, in E26's shape |

- **Any other combination:** reported, and the user chooses.
