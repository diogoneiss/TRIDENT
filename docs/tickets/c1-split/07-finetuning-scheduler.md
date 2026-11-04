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
