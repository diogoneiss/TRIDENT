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
