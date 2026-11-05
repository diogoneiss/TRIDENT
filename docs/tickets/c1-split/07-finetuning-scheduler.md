# Fine-tuning's schedule: single annealings against `cosine_legacy`'s sawtooth (E135; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (19:50 GMT-3).

- **Why:** the user asked for the best schedule to be studied apart from the budget ("alem de estudar o melhor
  metodo em outro estudo (inves de cosine_legacy)", 2026-11-04). ADR 0003 made the schedule selectable and kept
  `cosine_legacy` as the default for comparability, "a measured decision, not a silent one"; the measure was never
  taken on classification. E132's curves show the validation minimum by epoch 13–24 at a rate `cosine_legacy` never
  anneals to zero before the sawtooth restarts (period 50 / 75 / 50 / 11.5 epochs on biodeg / kc2 / vehicle /
  spambase at 150 epochs).
- **Code:** `FINETUNE_LR_SCHEDULER` (`src/training/types.py`, `finetuning.py`): fine-tuning's own schedule name, ""
  (every run before it) meaning the shared `LR_SCHEDULER`; pre-training keeps `cosine_legacy`, whose curves were
  clean. Unit-tested; each name ran a 3-epoch smoke. Arms and cut `7` in `c1_run.py`; reader `--scheduler`.

## Design (fixed before the first cell)

- **Arms, S's pre-training with fine-tuning's schedule changed, on the eight files, decision mode:** **COS150** and
  **COS50** (`cosine`: one half-cosine per epoch from the base rate to zero over 150 or 50 epochs); **WUC150** and
  **WUC50** (`warmup_cosine`: linear warmup over the first 10% of steps, then a half-cosine to zero); **PLAT150**
  (`plateau`: halves the rate after 10 epochs without a new best validation loss); **CONST150** (`constant`, the
  FT-Transformer-style baseline). 144 cells. S at 150 under `cosine_legacy` is E124's.
- **Machines (ADR 0035):** gorgona8 seed 42 of kc2 and spambase (24 cells), medusa1 the other 120.

## Rule, fixed now

- **Primary, six tests as one family:** X − S in test macro-F1, 15 paired replicates a file, pooled over the eight
  files per replicate, plain 95% t interval (14 df), Nadeau–Bengio beside; Holm over the six arms. **Better** /
  **worse** if Holm p < 0.05 by the sign of the mean; otherwise **no verdict**. Per-file lines and the MNAR / MCAR
  halves are flags.
- **Beside, descriptive:** log loss; the kept epochs per arm (from the fold metrics).
- **What follows:** an arm *better* goes to the user with a proposal for an ADR (fine-tuning's default schedule) and a
  confirmation on the 21 variants; if several are *better*, the user picks among them on the budget and the kept
  epochs, in a new ticket; none *better* keeps `cosine_legacy` and closes the question at this budget.
- **Threats stated now:** one base rate (0.001) for every schedule, untuned; `plateau`'s factor and patience and the
  warmup fraction are ADR 0003's constants; the kept-epoch rule stays; pre-training's schedule is not varied.
- **Read only with all 144 cells done.**

## Outcome (2026-11-04, 21:21 GMT-3; medusa1 done 20:50 GMT-3, gorgona8 21:16 GMT-3, 144 of 144 cells, 0 failed)

Read with `c1_report.py --scheduler` on all 144 cells. Unedited tally, X − S(`cosine_legacy`, 150) in macro-F1,
pooled over the eight files (15 replicates), Holm over the six:

| arm | pooled | NB | p | Holm | verdict | files better / worse |
|---|---|---|---|---|---|---|
| COS150 | −0.0023 [−0.0054, +0.0009] | [−0.0095, +0.0050] | 0.143 | 0.847 | **no verdict** | none / vehicle_probit (−0.012) |
| COS50 | −0.0017 [−0.0064, +0.0030] | [−0.0125, +0.0091] | 0.452 | 1.000 | **no verdict** | none / none |
| WUC150 | −0.0007 [−0.0078, +0.0064] | [−0.0170, +0.0156] | 0.839 | 1.000 | **no verdict** | none / none |
| WUC50 | +0.0014 [−0.0043, +0.0072] | [−0.0118, +0.0146] | 0.606 | 1.000 | **no verdict** | none / none |
| PLAT150 | −0.0025 [−0.0059, +0.0009] | [−0.0103, +0.0053] | 0.141 | 0.847 | **no verdict** | none / spambase_60nan (−0.004) |
| CONST150 | −0.0026 [−0.0062, +0.0011] | [−0.0110, +0.0058] | 0.150 | 0.847 | **no verdict** | none / spambase_60nan (−0.004) |

- **Halves (flags):** every MNAR and MCAR half spans zero; the widest is WUC50 MNAR +0.003 [−0.006, +0.013].
- **Kept epochs (`c1_report.py --kept`; median [min, max] over 15 replicates; biodeg/kc2/vehicle/spambase probit,
  then MCAR):** S 11 / 13 /
  32 / 25 / 13 / 11 / 31 / 20; COS150 11 / 12 / 21 / 12 / 13 / 11 / 25 / 9; COS50 11 / 14 / 27 / 11 / 14 / 12 / 27 /
  11; WUC150 17 / 18 / 31 / 16 / 18 / 20 / 29 / 15; WUC50 12 / 12 / 33 / 16 / 14 / 13 / 28 / 11; PLAT150 12 / 12 / 23 /
  11 / 13 / 11 / 25 / 8; CONST150 12 / 12 / 23 / 11 / 13 / 11 / 25 / 8. The single annealings keep an earlier epoch
  than the sawtooth (medians 8–25 against 11–32), and COS50 keeps about the same epochs as COS150, so the validation
  minimum is set by overfitting, not by where the schedule ends.
- **PLAT150 equals CONST150 to the digit on five of eight files.** Under the kept-epoch rule the kept state precedes
  the first halving (`plateau` halves 10 epochs after the last best), so `plateau` is `constant` unless a new best
  follows a halving; that happened only on the two kc2 files and vehicle_probit.
- **Reading, by the rule:** no arm is *better*; the three tightest intervals (COS150, PLAT150, CONST150) exclude a gain
  above +0.001 against `cosine_legacy`, and two arms are *worse* on one file each. `cosine_legacy` stays
  fine-tuning's default; the schedule question is closed at this base rate and budget. No ADR proposal follows.
  `FINETUNE_LR_SCHEDULER` stays in the code, default "" (unchanged behaviour), for a future tuned comparison.
