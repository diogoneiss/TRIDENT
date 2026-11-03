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

(open)
