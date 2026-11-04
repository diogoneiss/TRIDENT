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

## Outcome (2026-11-04 19:31 GMT-3): neither arm helps; the fine-tuning side is closed

All 72 cells done (19:01–19:29 GMT-3; gorgona8 12, medusa1 60; none failed). The unedited `c1_report.py --fix` at 2fdcaf7:

```
## Fix (E133): X − S macro-F1 for X in LRS (encoder LR x0.1) and FRZ (head alone for 50 epochs), 15 paired replicates a file, pooled over the four MNAR files; plain 95% t intervals, Holm over the two arms; MCAR second

### MNAR
- LRS − S biodeg_probit_60nan: F1  +0.0036 [-0.0082, +0.0155] (NB [-0.0236, +0.0309]; s42 +0.0059, s7 +0.0076, s13 -0.0026) -> no difference; LL(S) − LL(LRS) +0.0064; LRS − B +0.0001
- LRS − S kc2_probit_60nan: F1  +0.0094 [-0.0163, +0.0352] (NB [-0.0497, +0.0686]; s42 +0.0283, s7 -0.0133, s13 +0.0132) -> no difference; LL(S) − LL(LRS) +0.0808; LRS − B -0.0000
- LRS − S vehicle_probit_60nan: F1  -0.0155 [-0.0251, -0.0059] (NB [-0.0375, +0.0066]; s42 -0.0196, s7 -0.0185, s13 -0.0084) -> worse; LL(S) − LL(LRS) +0.0354; LRS − B +0.0015
- LRS − S spambase_probit_60nan: F1  -0.0029 [-0.0068, +0.0009] (NB [-0.0118, +0.0059]; s42 +0.0022, s7 -0.0033, s13 -0.0077) -> equivalent; LL(S) − LL(LRS) +0.0061; LRS − B +0.0091
- FRZ − S biodeg_probit_60nan: F1  +0.0057 [-0.0111, +0.0225] (NB [-0.0329, +0.0443]; s42 +0.0167, s7 +0.0113, s13 -0.0109) -> no difference; LL(S) − LL(FRZ) +0.0114; FRZ − B +0.0021
- FRZ − S kc2_probit_60nan: F1  -0.0038 [-0.0333, +0.0256] (NB [-0.0715, +0.0639]; s42 -0.0150, s7 -0.0034, s13 +0.0069) -> no difference; LL(S) − LL(FRZ) +0.0532; FRZ − B -0.0133
- FRZ − S vehicle_probit_60nan: F1  -0.0138 [-0.0372, +0.0096] (NB [-0.0676, +0.0400]; s42 -0.0204, s7 -0.0328, s13 +0.0118) -> no difference; LL(S) − LL(FRZ) +0.0138; FRZ − B +0.0031
- FRZ − S spambase_probit_60nan: F1  +0.0012 [-0.0049, +0.0074] (NB [-0.0129, +0.0154]; s42 +0.0018, s7 +0.0008, s13 +0.0011) -> equivalent; LL(S) − LL(FRZ) +0.0025; FRZ − B +0.0133

Pooled LRS − S (MNAR): -0.0013 [-0.0067, +0.0041] (NB [-0.0138, +0.0111]); p 0.6067, Holm 1.0000 -> no verdict

Pooled FRZ − S (MNAR): -0.0027 [-0.0142, +0.0088] (NB [-0.0291, +0.0238]); p 0.6261, Holm 1.0000 -> no verdict

### MCAR
- LRS − S biodeg_60nan: F1  +0.0078 [-0.0052, +0.0209] (NB [-0.0221, +0.0378]; s42 +0.0108, s7 -0.0093, s13 +0.0220) -> no difference; LL(S) − LL(LRS) +0.0209; LRS − B -0.0024
- LRS − S kc2_60nan: F1  +0.0286 [+0.0052, +0.0519] (NB [-0.0251, +0.0823]; s42 +0.0027, s7 +0.0628, s13 +0.0203) -> better; LL(S) − LL(LRS) +0.0356; LRS − B +0.0244
- LRS − S vehicle_60nan: F1  -0.0250 [-0.0462, -0.0038] (NB [-0.0737, +0.0237]; s42 -0.0206, s7 -0.0476, s13 -0.0067) -> worse; LL(S) − LL(LRS) +0.0107; LRS − B -0.0043
- LRS − S spambase_60nan: F1  +0.0005 [-0.0025, +0.0036] (NB [-0.0065, +0.0076]; s42 +0.0025, s7 -0.0009, s13 -0.0000) -> equivalent; LL(S) − LL(LRS) +0.0030; LRS − B +0.0079
- FRZ − S biodeg_60nan: F1  +0.0032 [-0.0051, +0.0116] (NB [-0.0160, +0.0224]; s42 +0.0032, s7 +0.0006, s13 +0.0058) -> no difference; LL(S) − LL(FRZ) -0.0020; FRZ − B -0.0070
- FRZ − S kc2_60nan: F1  +0.0161 [-0.0076, +0.0399] (NB [-0.0384, +0.0707]; s42 +0.0028, s7 +0.0299, s13 +0.0158) -> no difference; LL(S) − LL(FRZ) -0.0024; FRZ − B +0.0119
- FRZ − S vehicle_60nan: F1  -0.0205 [-0.0466, +0.0055] (NB [-0.0805, +0.0394]; s42 +0.0130, s7 -0.0501, s13 -0.0245) -> no difference; LL(S) − LL(FRZ) -0.0139; FRZ − B +0.0001
- FRZ − S spambase_60nan: F1  +0.0015 [-0.0033, +0.0062] (NB [-0.0095, +0.0124]; s42 +0.0048, s7 +0.0008, s13 -0.0012) -> equivalent; LL(S) − LL(FRZ) -0.0044; FRZ − B +0.0088

Pooled LRS − S (MCAR): +0.0030 [-0.0040, +0.0100] (NB [-0.0132, +0.0192]); p 0.3774, Holm 0.7548 -> no verdict

Pooled FRZ − S (MCAR): +0.0001 [-0.0069, +0.0071] (NB [-0.0160, +0.0161]); p 0.9834, Holm 0.9834 -> no verdict

### The probe with a converging head (descriptive): FP10 − FR10 on the MNAR files, against E132's FP − FR
- biodeg_probit_60nan: FP10 − FR10  -0.0778 [-0.1777, +0.0220] (NB [-0.3074, +0.1517]; s42 -0.0332, s7 -0.0708, s13 -0.1295) -> no difference; FP − FR was +0.0105
- kc2_probit_60nan: FP10 − FR10  +0.0579 [+0.0102, +0.1056] (NB [-0.0517, +0.1675]; s42 +0.1095, s7 +0.0295, s13 +0.0347) -> better; FP − FR was +0.0250
- vehicle_probit_60nan: FP10 − FR10  -0.0373 [-0.0878, +0.0131] (NB [-0.1533, +0.0786]; s42 -0.1114, s7 +0.0100, s13 -0.0106) -> no difference; FP − FR was +0.0059
- spambase_probit_60nan: FP10 − FR10  -0.2306 [-0.3566, -0.1046] (NB [-0.5203, +0.0591]; s42 -0.2015, s7 -0.3293, s13 -0.1609) -> worse; FP − FR was -0.0047
Pooled FP10 − FR10:  -0.0720 [-0.1193, -0.0246] (NB [-0.1808, +0.0369]; s42 -0.0592, s7 -0.0902, s13 -0.0666) -> worse; FP − FR was +0.0092
```

- **Primary:** LRS − S −0.0013 [−0.0067, +0.0041] and FRZ − S −0.0027 [−0.0142, +0.0088] pooled over the MNAR files;
  Holm p 1.0 for both: **no verdict**, and the means are negative. Flags: LRS *worse* on vehicle (−0.016); nothing
  *better*. X − B is 0 to +0.013 on every file: the gentler fine-tunes land where S and B land.
- **MCAR:** the same (+0.003 and +0.000 pooled); LRS *better* on kc2 (+0.029) and *worse* on vehicle (−0.025), mixed.
- **What the kept epochs say** (seed-pooled medians, MNAR): S 14–25; LRS 52–137; FRZ 74–109. A slower encoder or an
  initial freeze delays the validation minimum by 50–100 epochs and lands at the same test macro-F1 and log loss
  (LRS test log loss 0.37–0.97 against S's 0.37–0.98). Where the minimum falls is a schedule artefact; its height
  is not.
- **FP10 / FR10, a failed probe, not a reading:** at ten times the head's rate the frozen heads diverge on spambase
  (FP10 macro-F1 0.34, log loss 0.693 = ln 2, a constant output; FR10 0.58 against FR's 0.69) and degrade on biodeg
  and vehicle; kept epochs 13–45 on spambase. FP10 − FR10 = −0.072 says only that the rate was too high; E132's FP −
  FR (+0.009 over heads converging at 0.001) stands as read, as a floor.
- **What follows, as fixed above:** neither arm helps, so the fine-tuning side is closed; C1's channel question rests
  on pre-training itself. E131 stays Inconclusive. The fine-tuning budget (150 epochs with the cosine annealed well
  past the useful window, the user's question of 19:0x GMT-3) is a separate question, for its own ticket: it bears
  on every classification default, not on C1's contrast, which these arms show is insensitive to where the kept
  epoch falls.
