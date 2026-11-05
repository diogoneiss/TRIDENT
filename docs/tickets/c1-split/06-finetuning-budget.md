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

## Outcome (2026-11-04, 21:21 GMT-3; both machines done 20:16 GMT-3, 96 of 96 cells, 0 failed)

Read with `c1_report.py --budget` on all 96 cells (medusa1's copied under `<state>/medusa`). Unedited tally:

| arm − S(150), macro-F1 | pooled over 8 files (15 replicates) | NB | p | Holm | verdict | files better / worse |
|---|---|---|---|---|---|---|
| S30 | −0.0081 [−0.0153, −0.0010] | [−0.0247, +0.0084] | 0.029 | 0.058 | **no verdict** | kc2 ×2 better (+0.021, +0.019) / vehicle ×2 worse (−0.041, −0.047) |
| S50 | −0.0009 [−0.0049, +0.0031] | [−0.0100, +0.0083] | 0.644 | 0.644 | **no verdict** | none / none (spambase ×2 equivalent) |

- **Halves (flags):** S30 MNAR −0.008 [−0.019, +0.002], MCAR −0.008 [−0.015, −0.001]; S50 MNAR −0.001 [−0.009,
  +0.006], MCAR −0.001 [−0.006, +0.005].
- **Channel beside, S(k) − B(k) per file (probit then MCAR, biodeg/kc2/vehicle/spambase):** S30: −0.020, +0.009,
  +0.005, +0.005, −0.002, +0.013, +0.019, +0.007; S50: −0.006, +0.012, −0.000, +0.007, +0.010, −0.000, +0.019, +0.008.
  As at 150 (E131, E132): small and of both signs; no budget opens a pre-training channel.
- **Kept epochs (median [min, max] over 15 replicates; files in the order above):** S(150) 11 / 13 / 32 / 25 / 13 /
  11 / 31 / 20; S30 20 / 23 / 27 / 21 / 21 / 21 / 29 / 15, with vehicle's maxima at the cap (29); S50 27 / 13 / 36 / 22
  / 28 / 12 / 35 / 18. Vehicle's validation minimum sits at epoch 30–35, so a 30-epoch budget truncates it, which is
  where S30's two *worse* files come from; the other files keep their minimum well inside 30.
- **Reading, by the rule:** neither budget is *better*; S30 is *no verdict* with two files *worse* (fails the Holm
  threshold by 0.008 and is worse on vehicle), so it is not a cost argument; S50 is *no verdict* with no file worse or
  better, which the rule calls a cost argument only: about a third of fine-tuning's time at a score this study cannot
  tell from 150's, within ±0.005 pooled. No ADR change follows from this ticket; whether to spend a 21-variant
  confirmation on a budget of 50 is the user's call. The budget question under `cosine_legacy` is otherwise closed:
  the sawtooth's shorter period at 30 did not help.
