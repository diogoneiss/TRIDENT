# C1's fine-tuning fix: a gentler fine-tuning against S, and the probe with a converging head (E133; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (18:55 GMT-3).

- **Why:** E132 ([04](04-channel-probe.md)) read washout: a frozen pre-trained encoder beats a frozen random one by
  about +0.009 macro-F1, and full fine-tuning adds +0.10 from either start. Its loss curves (the addendum) show
  fine-tuning overfitting after about epoch 20 at LR 0.001 on every parameter, with the kept epoch at 13–24, and the
  frozen heads unconverged at 150 epochs. The user chose the fix ticket ("1!", 2026-11-04), after the curves were
  read. **Expected ceiling, stated now:** about +0.01, what FP holds over FR; E50 found both fixes made pendigits worse
  on imputation.
- **Code:** `FINETUNE_ENCODER_LR_SCALE`, `FINETUNE_FREEZE_EPOCHS`, `FINETUNE_HEAD_LR_SCALE` (`src/training/types.py`,
  `finetuning.py`; defaults 1.0 / 0 / 1.0 keep one optimizer group at one rate, every run before them unchanged;
  `_finetune_parameters` builds the groups; a frozen encoder runs in eval mode, out of the optimizer). Unit-tested;
  2-epoch smokes of FRZ and LRS ran end to end with the encoder moving. Arms and cut `5` in `c1_run.py`; reader
  `c1_report.py --fix`.

## Design (fixed before the first cell)

- **Fix arms, on the four MNAR files and their MCAR siblings, decision mode (seeds 42/7/13 × 5 folds):**
  **LRS** the encoder at a tenth of the head's rate (0.0001 against 0.001); **FRZ** the head alone for the first 50
  epochs (encoder frozen, eval mode), then everything at the base rate. Otherwise E131's configuration; S and B are
  E124's and E131's cells. 48 cells.
- **The probe with a converging head, on the four MNAR files:** **FP10** and **FR10**, E132's FP and FR with the head
  at ten times the rate (0.01). 24 cells. Descriptive: it checks whether E132's +0.009 was a floor set by an
  unconverged head.
- **Machines (ADR 0035):** as E131: gorgona8 seed 42 of kc2 and spambase (12 cells), medusa1 the other 60.

## Rule, fixed now

- **Primary, two tests as one family:** X − S in test macro-F1 for X ∈ {LRS, FRZ}, 15 paired replicates a file, pooled
  over the four MNAR files per replicate, plain 95% t interval (14 df), Nadeau–Bengio beside; p from the paired t test,
  Holm-adjusted over the two arms. An arm **helps** if its Holm p < 0.05 with a positive mean, **hurts** if negative;
  otherwise **no verdict**. Per-file lines are flags.
- **MCAR files:** the same reading as a second step; descriptive.
- **Beside:** X − B (what the arm keeps of pre-training against no pre-training), log loss, FP10 − FR10 against
  E132's FP − FR.
- **What follows:** if an arm helps, a new ticket reruns C1's cut 3 (S, T, B under that arm) before any status
  changes; E131 stays as read. If neither helps, the fine-tuning side is closed and C1's channel question rests on
  pre-training itself.
- **Threats stated now:** the two arms' values (0.1, 50) are the first and only ones tried, untuned; FRZ's head trains
  50 epochs at a rate E132 found slow to converge; the kept-epoch rule may land inside the freeze.
- **Read only with all 72 cells done.**
