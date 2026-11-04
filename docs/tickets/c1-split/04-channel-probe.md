# C1's channel probe: a frozen pre-trained encoder against a frozen random one (E132; pre-registered 2026-11-04)

This ticket was written and committed before the probe's first cell (18:16 GMT-3).

- **Why:** E131 ([03](03-primary.md)) was Inconclusive with G2 failing: S ≈ B, so pre-training reaches the head by no
  measurable amount and the split has no channel. The protocol's Inconclusive row names "a classification pre-training
  fix" as a next step; the user chose that direction ("vamos no 2", 2026-11-04, afternoon) and, asked, the diagnosis
  alone first ("Só diagnóstico: FP e FR"). Before any fix, this probe says on which side the problem sits.
- **The question:** does pre-training leave label-relevant structure in the encoder that full fine-tuning then throws
  away (washout, the imputation side's T28 diagnosis), or does it leave nothing the head can use?
- **Code:** `FINETUNE_FREEZE_ENCODER all` (`src/training/types.py`, `finetuning.py`; default `off`, every run before it
  unchanged): the embedder and transformer stay as pre-training left them, in eval mode (no dropout), out of the
  optimizer; the shipped head (128 → 64 → ReLU → Dropout 0.3 → classes; a head-only probe, not a linear one) trains
  alone with the shipped budget (150 epochs, LR 0.001, `cosine_legacy`). Also new: the whole encoder is dumped after
  pre-training whenever the readouts are on (`c1_encoder_pretrained_fold_<k>.pt`). Unit test in
  `tests/unit/test_c1_tied.py`; a 3-epoch cell on kc2 kept the encoder bit-equal to the dump while the head moved.
  Arms `FP`, `FR`, cut `4` in `c1_run.py`; reader `c1_report.py --channel`.

## Design (fixed before the first cell)

- **Arms:** **FP** S's pre-training, then the frozen fine-tuning above; **FR** `EPOCHS_PRE 0` plus the same freeze (a
  random frozen encoder with a trained head). S and B are E124's and E131's cells, not rerun.
- **Files, mode:** the four MNAR files `{biodeg,kc2,vehicle,spambase}_probit_60nan` and their MCAR siblings, decision
  mode (seeds 42/7/13 × 5 folds); E131's configuration otherwise. 48 new cells; machines as E131 (ADR 0035: gorgona8
  seed 42 of kc2 and spambase, 8 cells; medusa1 40).

## Rule, fixed now

- **Primary:** FP − FR in test macro-F1, 15 paired replicates per file, pooled over the four MNAR files per replicate,
  plain 95% t interval (14 df), Nadeau–Bengio beside; per-file lines are flags.
  - **Content present** if the pooled interval lies above 0: pre-training leaves structure the head can read, so with
    S ≈ B (E131) the diagnosis is washout, and a fix ticket on the fine-tuning side follows (encoder learning rate
    scaled down, or frozen for the first epochs; E50 found both made pendigits worse on imputation).
  - **Content absent** if the interval lies inside ±0.010: a random frozen encoder serves the head as well, so
    fine-tuning-side fixes are pointless and the lever is pre-training itself (objective, epochs, mask rate); the
    objective lever has been pulled three times with little effect (the anchor, E112, E114).
  - Otherwise **no verdict**.
- **MCAR files:** the same reading as a second step; descriptive.
- **Beside, descriptive:** S − FP (what full fine-tuning adds over frozen pre-trained features) and B − FR; log loss.
- **Threats stated now:** the head's budget was set for full fine-tuning, untuned here (zero tuning stands); FR's
  "random encoder" is the initialisation at the cell's seed, one draw per cell; the probe reads [CLS] through the
  shipped head, so a representation readable only from other positions would not show.
- **Read only with all 48 cells done.** Nothing here changes C1's status (E131 stays Inconclusive).

## Outcome (2026-11-04 18:34 GMT-3): content present, but small; full fine-tuning adds ten times more than pre-training leaves

All 48 cells done (18:22–18:32 GMT-3, none failed). The unedited `c1_report.py --channel` at 9af3d5d:

```
## Channel probe (E132): FP − FR macro-F1 on the four MNAR files, then the MCAR files; 15 paired replicates, plain 95% t intervals, Nadeau–Bengio beside; S − FP and B − FR descriptive

### MNAR
- biodeg_probit_60nan: FP − FR F1  +0.0105 [-0.0113, +0.0323] (NB [-0.0397, +0.0607]; s42 +0.0102, s7 +0.0119, s13 +0.0094) -> no difference; LL(FR) − LL(FP) -0.0087; S − FP +0.0618; B − FR +0.0758
- kc2_probit_60nan: FP − FR F1  +0.0250 [-0.0002, +0.0502] (NB [-0.0329, +0.0829]; s42 +0.0163, s7 +0.0198, s13 +0.0389) -> no difference; LL(FR) − LL(FP) +0.0192; S − FP +0.0047; B − FR +0.0392
- vehicle_probit_60nan: FP − FR F1  +0.0059 [-0.0177, +0.0295] (NB [-0.0484, +0.0602]; s42 +0.0055, s7 +0.0037, s13 +0.0085) -> no difference; LL(FR) − LL(FP) +0.0281; S − FP +0.1713; B − FR +0.1603
- spambase_probit_60nan: FP − FR F1  -0.0047 [-0.0228, +0.0135] (NB [-0.0464, +0.0371]; s42 -0.0189, s7 +0.0136, s13 -0.0087) -> no difference; LL(FR) − LL(FP) -0.0314; S − FP +0.1537; B − FR +0.1370

Pooled FP − FR (MNAR): +0.0092 [+0.0013, +0.0171] (NB [-0.0091, +0.0274]) -> content present (pre-training leaves label-relevant structure)
Pooled LL(FR) − LL(FP) (MNAR):  +0.0018 [-0.0108, +0.0144] (NB [-0.0272, +0.0308]; s42 -0.0059, s7 +0.0208, s13 -0.0095) -> no difference
Pooled S − FP (MNAR): +0.0979; B − FR: +0.1031

### MCAR
- biodeg_60nan: FP − FR F1  -0.0041 [-0.0334, +0.0251] (NB [-0.0715, +0.0632]; s42 -0.0113, s7 -0.0004, s13 -0.0008) -> no difference; LL(FR) − LL(FP) -0.0172; S − FP +0.1339; B − FR +0.1400
- kc2_60nan: FP − FR F1  +0.0468 [+0.0228, +0.0707] (NB [-0.0083, +0.1018]; s42 +0.0428, s7 +0.0644, s13 +0.0331) -> better; LL(FR) − LL(FP) +0.0218; S − FP -0.0090; B − FR +0.0420
- vehicle_60nan: FP − FR F1  -0.0191 [-0.0432, +0.0049] (NB [-0.0744, +0.0362]; s42 -0.0234, s7 -0.0245, s13 -0.0094) -> no difference; LL(FR) − LL(FP) +0.0225; S − FP +0.1783; B − FR +0.1386
- spambase_60nan: FP − FR F1  +0.0405 [+0.0192, +0.0618] (NB [-0.0084, +0.0894]; s42 +0.0549, s7 +0.0514, s13 +0.0151) -> better; LL(FR) − LL(FP) +0.0471; S − FP +0.1403; B − FR +0.1733

Pooled FP − FR (MCAR): +0.0160 [+0.0043, +0.0277] (NB [-0.0109, +0.0429]) -> content present (pre-training leaves label-relevant structure)
Pooled LL(FR) − LL(FP) (MCAR):  +0.0185 [+0.0054, +0.0316] (NB [-0.0116, +0.0487]; s42 +0.0177, s7 +0.0265, s13 +0.0113) -> better
Pooled S − FP (MCAR): +0.1109; B − FR: +0.1235
```

- **By the rule, content present:** pooled FP − FR on the MNAR files +0.0092 [+0.0013, +0.0171] macro-F1, the plain
  interval above 0 (the Nadeau–Bengio one reaches −0.009). A frozen pre-trained encoder gives the head about one
  macro-F1 point over a frozen random one; the seed means agree in sign on three files. On the MCAR files +0.0160
  [+0.0043, +0.0277], *better* on kc2 and spambase.
- **But the channel is narrow:** S − FP is +0.098 and B − FR +0.103 pooled. Full fine-tuning rebuilds the encoder and
  gains ten times what pre-training left, from a random start as well as from a pre-trained one. The structure
  pre-training leaves is real and small, below the +0.010 label on the MNAR files; this is the washout diagnosis the
  imputation side reached (T28), with a smaller frozen advantage than the decode stage's.
- **What it bounds:** a fine-tuning-side fix (encoder learning rate scaled down, or frozen for the first epochs) can
  recover at most about what FP holds over FR, around +0.01, if it keeps that structure without losing what full
  fine-tuning adds; E50 found both fixes made pendigits worse on imputation. The paper's +1.7 points (Table 4) sits at
  the top of that bound. Whether a fix ticket is worth its cells is the user's decision; by this ticket's rule it is
  the branch that follows.
