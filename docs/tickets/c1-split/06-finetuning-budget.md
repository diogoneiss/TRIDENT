# The fine-tuning budget: 30 and 50 epochs against 150 under `cosine_legacy` (E134; pre-registered 2026-11-04)

This ticket was written and committed before the cut's first cell (19:50 GMT-3).

- **Why:** E132's loss curves ([04](04-channel-probe.md), addendum) show fine-tuning's validation loss at its minimum
  by epoch 13–24 of 150 and then rising two to three fold; the kept-epoch rule protects the test score, but the
  learning rate is never annealed inside the useful window, and 150 epochs cost about three times what the kept model
  needs. E133 ([05](05-finetuning-fix.md)) moved the minimum 50–100 epochs later at the same scores. The user asked
  that the overfitting be addressed ("TEMOS overfitting no finetuning, nao? isso nao deveria ser corrigido?", "rode o
  134 entao", 2026-11-04). This bears on every classification default, not on C1's contrast; a default change needs
  an ADR and a confirmation on the 21 variants, which this ticket does not do.
- **A fact fixed here, from ADR 0003:** `cosine_legacy` steps the cosine per batch with a period of `EPOCH_FINE`
  optimizer steps, so it is a sawtooth, not an annealing: at 150 epochs the period is 50 epochs on biodeg and vehicle
  (3 batches an epoch), 75 on kc2 (2) and 11.5 on spambase (13). Shortening `EPOCH_FINE` under it shortens the period
  too (30 epochs: 10, 15, 10 and 2.3 epochs). A single annealing is E135's question ([07](07-finetuning-scheduler.md)).
- **Code:** none new in `src/` (`EPOCH_FINE` exists). Arms and cut `6` in `c1_run.py`; reader `c1_report.py --budget`.

## Design (fixed before the first cell)

- **Arms, on the four MNAR files and their MCAR siblings, decision mode (seeds 42/7/13 × 5 folds):** **S30** and **S50**
  (S with `EPOCH_FINE` 30 and 50); **B30** and **B50** (no pre-training at the same budgets, the channel beside). 96
  cells. Otherwise E131's configuration; S and B at 150 are E124's and E131's cells.
- **Machines (ADR 0035):** gorgona8 seed 42 of kc2 and spambase (16 cells), medusa1 the other 80.

## Rule, fixed now

- **Primary, two tests as one family:** S(k) − S(150) in test macro-F1 for k ∈ {30, 50}, 15 paired replicates a file,
  pooled over the eight files per replicate, plain 95% t interval (14 df), Nadeau–Bengio beside; p from the paired t
  test, Holm over the two. **Better** / **worse** if Holm p < 0.05 by the sign of the mean; otherwise **no verdict**.
  Per-file lines and the MNAR / MCAR halves are flags.
- **Beside, descriptive:** log loss; S(k) − B(k), the pre-training channel at the shorter budget.
- **What follows:** a budget that is *better* goes to the user with a proposal for an ADR and a confirmation on the
  21 variants; one with *no verdict* and no file *worse* is a cost argument only (about 3–5× less fine-tuning time),
  also the user's call; *worse* closes the budget question under `cosine_legacy`.
- **Threats stated now:** 30 and 50 are the first values tried; under the sawtooth the kept epoch may sit at any
  phase; the kept-epoch rule (ADR 0029 decision 5) stays, so a shorter budget removes epochs the rule already
  discards.
- **Read only with all 96 cells done.**
