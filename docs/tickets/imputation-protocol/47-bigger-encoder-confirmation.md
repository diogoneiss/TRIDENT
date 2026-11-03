# The bigger encoder on pendigits and letter at 40–60nan, on fresh seeds (E116; pre-registered 2026-11-03 01:45 GMT-3)

This ticket was written and committed before the study's first cell. No cell has run.

- **Why:** E113 ([46](46-bigger-encoder-large-tables.md)) was supported, better on 4 of 4 (−6.0%, −3.1%,
  −5.5%, −1.1%). As for ADR 0036, a fresh-seed confirmation comes before an ADR proposal.
- **The user's go:** "gosto da ideia de rodar o 3, com a outra maquina rodar 1 ou 2. vá" (2026-11-03,
  after 00:38 GMT-3), this being item 2.
- **No code change** beyond `large_batch.py --study e116`.

## Design (fixed before the first cell)

- **Cells:** pendigits and letter at 40 and 60nan × fresh seeds 6161, 7171, 8181 (never used) × arms M
  (today's default) and BIG (ADR 0018's encoder over it), five folds, MLflow off. 24 cells.
- **Running (ADR 0035):** seed 6161 on gorgona8, 7171 and 8181 on medusa1; both arms of a variant and seed
  in one queue on one machine.

## Measures and rule, fixed now (`large_batch.py --study e116 --tally`)

- BIG − M on the induced `impute_score`, relative, 15 fold pairs, the project's rule, 1% SESOI.
- **Replicated** if better on 3 or more of 4 and worse on none; **not replicated** if better on 1 or
  fewer or worse on any; **partial** otherwise.
- **What follows:** replicated → an ADR proposal for the user (ADR 0018's encoder on these variants).

## Outcome

**Written 2026-11-03 04:37 GMT-3. Replicated: BIG is better on 4 of 4, every seed below zero**, at E113's sizes:
pendigits_40nan −6.3% (E113 −6.0%), pendigits_60nan −3.1% (−3.1%), letter_40nan −5.5% (−5.5%), letter_60nan
−1.2% (−1.1%).

- **Complete:** all 24 cells, 01:45–04:36 GMT-3, no failure; seed 6161 on gorgona8, 7171/8181 on medusa1.
  The end chain's early rsync errors (medusa1's BIG directory did not exist yet) are harmless.
- **What follows, as fixed above:** an ADR proposal for the user (ADR 0037).

### The tally, unedited (`large_batch.py --study e116 --tally`, 2026-11-03 04:37 GMT-3)

```
## BIG − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_40nan: -6.26% [-6.78%, -5.75%] seeds -6.31%, -6.54%, -5.95% -> better; masked -4.19%
- pendigits_60nan: -3.09% [-3.62%, -2.55%] seeds -2.89%, -3.51%, -2.85% -> better; masked -1.28%
- letter_40nan: -5.45% [-5.79%, -5.11%] seeds -5.25%, -5.23%, -5.86% -> better; masked -3.75%
- letter_60nan: -1.23% [-1.40%, -1.06%] seeds -1.27%, -1.16%, -1.26% -> better; masked +0.03%

Primary (E116, BIG − M, fresh seeds): better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 4 -> replicated
```
