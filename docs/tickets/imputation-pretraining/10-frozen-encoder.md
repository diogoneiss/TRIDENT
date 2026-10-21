# T28's D3: a frozen encoder, pre-trained against random (E52; pre-registered 2026-10-20)

This ticket was written and committed before the first cell.

- **Why:** T28's report ([2026-10-19](../../reports/2026-10-19-t28-why-pretraining-adds-little.md),
  §3 H1 and H3, §5 step 3) asks whether pre-training leaves a usable representation at all.
  - **The test:** decode with the encoder held fixed throughout, from a pre-trained start and from
    a random one.
  - **What came before:** E49 (D1) saw the head start wash out on pendigits. E50 then found that
    neither holding the encoder for 100 epochs nor slowing it helps.
- **The user's go:** 2026-10-20, in the afternoon (after 14:50 GMT-3), for T28's D2 and D3
  before T29 and TabPFN.
- **No new code in `src/`.** E50's `--decode_freeze_epochs` set to the variant's own decode budget
  freezes the embedder and transformer for every decode epoch.
  - `EPOCHS_PRE 0` still enters `train_pretrainer`, so the random encoder is the Xavier sweep the
    stage starts from: the same start as E50's C0 arm.
  - **Checked before launch** (about 19:19 GMT-3): FR on kc2_20nan at seed 7
    (`--no-mlflow --check-frozen`, not a study cell). All 154 tensors of the embedder and transformer leave the decode stage
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

## Launch

- **Launched at 19:21 GMT-3 on 2026-10-20** on 68e0baf: four queues in
  `/scratch2/diogoneiss/trident-2026-10-20-frozen-encoder/`.
- **Expected end:** about 22:00–22:30 GMT-3.

## Outcome (2026-10-20, 22:21 GMT-3)

All 42 cells finished, none failed, from 19:21 to 22:19 GMT-3.

**The printout** (`scripts/experiments/frozen_encoder_report.py --tally`, unedited). Each line is
a mean difference over 15 fold pairs of the induced `impute_score` (lower is better), with its
percent of the second arm's K, the 95% interval and the call.

```text
=== pendigits_20nan
  FN: K 0.3870
  FA: K 0.4787
  FR: K 0.8263
  C0: K 0.3241
  G: K 0.3203
  FN − FR, K -0.4392 (-53.2%) [-0.4482, -0.4303] better; H -0.4392 (-53.2%) [-0.4479, -0.4305] better; masked K -0.4351 (-52.7%) [-0.4411, -0.4291] better
  FA − FR, K -0.3475 (-42.1%) [-0.3563, -0.3388] better; H -0.3476 (-42.1%) [-0.3561, -0.3391] better; masked K -0.3443 (-41.7%) [-0.3498, -0.3387] better
  FN − FA, K -0.0917 (-19.2%) [-0.0944, -0.0890] better; H -0.0916 (-19.1%) [-0.0945, -0.0887] better; masked K -0.0909 (-19.0%) [-0.0949, -0.0869] better
  FN − G, K +0.0667 (+20.8%) [+0.0630, +0.0705] worse; H +0.0661 (+20.6%) [+0.0623, +0.0699] worse; masked K +0.0688 (+21.5%) [+0.0665, +0.0710] worse
  FR − C0, K +0.5022 (+155.0%) [+0.4943, +0.5101] worse; H +0.5016 (+154.8%) [+0.4938, +0.5094] worse; masked K +0.5024 (+155.0%) [+0.4962, +0.5086] worse
  C0 − G, K +0.0038 (+1.2%) [+0.0015, +0.0060] worse; H +0.0037 (+1.2%) [+0.0017, +0.0057] worse; masked K +0.0015 (+0.5%) [-0.0005, +0.0036] none
=== kr-vs-kp_40nan
  FN: K 0.8022
  FA: K 0.9987
  FR: K 0.9596
  C0: K 0.7258
  G: K 0.7258
  FN − FR, K -0.1574 (-16.4%) [-0.1666, -0.1481] better; H -0.1716 (-17.9%) [-0.1819, -0.1613] better; masked K -0.1428 (-14.9%) [-0.1592, -0.1263] better
  FA − FR, K +0.0392 (+4.1%) [+0.0312, +0.0471] worse; H +0.0248 (+2.6%) [+0.0189, +0.0307] worse; masked K +0.0297 (+3.1%) [+0.0181, +0.0412] worse
  FN − FA, K -0.1965 (-19.7%) [-0.2021, -0.1909] better; H -0.1964 (-19.7%) [-0.2033, -0.1895] better; masked K -0.1724 (-17.3%) [-0.1912, -0.1536] better
  FN − G, K +0.0764 (+10.5%) [+0.0669, +0.0858] worse; H +0.0763 (+10.5%) [+0.0670, +0.0855] worse; masked K +0.0734 (+10.1%) [+0.0544, +0.0924] worse
  FR − C0, K +0.2337 (+32.2%) [+0.2240, +0.2435] worse; H +0.2444 (+33.7%) [+0.2337, +0.2551] worse; masked K +0.2080 (+28.7%) [+0.1868, +0.2293] worse
  C0 − G, K +0.0000 (+0.0%) [-0.0068, +0.0069] equivalent; H +0.0035 (+0.5%) [-0.0045, +0.0115] none; masked K +0.0081 (+1.1%) [-0.0083, +0.0245] none
=== credit-g_20nan
  FN: K 0.9643
  FA: K 0.9978
  FR: K 0.9720
  C0: K 0.8904
  G: K 0.8751
  FN − FR, K -0.0077 (-0.8%) [-0.0229, +0.0075] none; H -0.0038 (-0.4%) [-0.0142, +0.0066] none; masked K -0.0107 (-1.1%) [-0.0252, +0.0038] none
  FA − FR, K +0.0257 (+2.6%) [+0.0122, +0.0392] worse; H +0.0228 (+2.3%) [+0.0127, +0.0329] worse; masked K +0.0191 (+2.0%) [+0.0041, +0.0341] worse
  FN − FA, K -0.0334 (-3.3%) [-0.0427, -0.0242] better; H -0.0266 (-2.7%) [-0.0368, -0.0165] better; masked K -0.0298 (-3.0%) [-0.0385, -0.0211] better
  FN − G, K +0.0892 (+10.2%) [+0.0735, +0.1050] worse; H +0.0839 (+9.6%) [+0.0694, +0.0983] worse; masked K +0.0805 (+9.2%) [+0.0626, +0.0984] worse
  FR − C0, K +0.0816 (+9.2%) [+0.0598, +0.1035] worse; H +0.0864 (+9.7%) [+0.0705, +0.1023] worse; masked K +0.0817 (+9.2%) [+0.0580, +0.1055] worse
  C0 − G, K +0.0153 (+1.7%) [+0.0034, +0.0272] worse; H +0.0013 (+0.1%) [-0.0108, +0.0134] none; masked K +0.0095 (+1.1%) [-0.0057, +0.0246] none
=== credit-g_80nan
  FN: K 1.0051
  FA: K 1.0060
  FR: K 1.0074
  C0: K 1.0167
  G: K 1.0105
  FN − FR, K -0.0023 (-0.2%) [-0.0070, +0.0024] equivalent; H -0.0039 (-0.4%) [-0.0096, +0.0018] equivalent; masked K +0.0001 (+0.0%) [-0.0148, +0.0150] none
  FA − FR, K -0.0014 (-0.1%) [-0.0055, +0.0028] equivalent; H -0.0004 (-0.0%) [-0.0061, +0.0054] equivalent; masked K -0.0139 (-1.4%) [-0.0318, +0.0040] none
  FN − FA, K -0.0009 (-0.1%) [-0.0059, +0.0041] equivalent; H -0.0036 (-0.4%) [-0.0078, +0.0007] equivalent; masked K +0.0140 (+1.4%) [-0.0013, +0.0293] none
  FN − G, K -0.0054 (-0.5%) [-0.0123, +0.0015] none; H -0.0241 (-2.4%) [-0.0403, -0.0079] better; masked K +0.0005 (+0.1%) [-0.0120, +0.0131] none
  FR − C0, K -0.0094 (-0.9%) [-0.0152, -0.0035] better; H -0.0135 (-1.3%) [-0.0205, -0.0065] better; masked K +0.0008 (+0.1%) [-0.0117, +0.0134] none
  C0 − G, K +0.0063 (+0.6%) [+0.0003, +0.0122] worse; H -0.0067 (-0.7%) [-0.0243, +0.0109] none; masked K -0.0004 (-0.0%) [-0.0188, +0.0181] none

Primary, FN − FR on K: a usable pre-trained representation (FN better) on ['pendigits_20nan', 'kr-vs-kp_40nan']; FN worse on none; better 2 / no difference 1 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
Reading: mixed (2 of 4; usable needs 3 or more, nothing usable needs none)
```

**Reading, by the rule fixed in advance: mixed.** FN is better than FR on 2 of the 4 variants
(usable needs 3).

- **pendigits_20nan: usable.** Heads alone on today's pre-trained encoder reach 0.387, against
  0.826 on a random one (−53%).
- **kr-vs-kp_40nan: usable.** 0.802 against 0.960 (−16%).
- **credit-g_20nan: no difference** (−0.8%).
- **credit-g_80nan: equivalent** (−0.2%). Every arm sits at mean/mode parity there (K ≈ 1.00 to
  1.02).

**Secondary:**

- **The shipped objective (FA) is the worst start of the three.**
  - It is below today's N everywhere it differs: −19% on pendigits, −20% on kr-vs-kp, −3% on
    credit-g_20nan.
  - On kr-vs-kp and credit-g_20nan, it is worse than a random encoder (+4.1% and +2.6%).
- **The full decode stage overtakes the frozen pre-trained encoder.** FN − G is +21% on pendigits
  and +10% on kr-vs-kp and credit-g_20nan.
- **From a random start, the full decode reaches nearly as far.** C0 − G is +1.2% on pendigits
  (as in E50), equivalent on kr-vs-kp, and +1.7% on credit-g_20nan and +0.6% on credit-g_80nan.
- **At 80nan the frozen arms do slightly better than a full decode.** FR − C0 is −0.9% on credit-g
  (better), and FN − G is −2.4% on H. That fits regularisation, at mean/mode parity.

**What it means:** on the two tables where pre-training matters, the stage leaves a real,
linearly usable representation. The decode stage then learns one as good from scratch, so the
pre-trained start buys at most 1–2% at K. On credit-g it leaves nothing the heads can use.

**With D2 (E53), through the decision table fixed above:**

- **E53's probe reading,** final before its drift cells ran:
  - **N collapsing: no** (0 of 3 tables);
  - **value information under N: present** (2 of 3: kr-vs-kp +7.5%, pendigits +35.8%; credit-g
    absent);
  - **H5:** `BATCH 64` moves credit-g's final probe (−0.016, all seeds lower); `LR_PRE` ÷ 4 does not.
- **D3 mixed with value information present** matches none of the first three rows. The fourth row
  applies whatever D3 says: **a diagnostic moves with the batch, so T12 comes first, in E26's
  shape.**
  - **Two caveats for the user's choice:**
    - what moved is the batch, not `LR_PRE`, and only on credit-g, where the probes sit near no
      information (about 1.0);
    - T12 as written is a lower `LR_PRE`, so its screening would need a `BATCH 64` arm.
  - Nothing changes a default; T12's screening is a pre-registered study for the user to decide.
- **Limit:** the seeds are development seeds (101/202/303), so this is a development reading.

