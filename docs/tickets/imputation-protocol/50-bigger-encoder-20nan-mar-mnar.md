# Does ADR 0018's bigger encoder help off MCAR at 20nan? (E121; pre-registered 2026-11-03)

This ticket was written and committed before the study's first cell (20:55 GMT-3).

- **Why:** ADR 0018 adopted the bigger encoder (LAYERS 4, DIM_FEED 256, ENCODER_FINAL_NORM layer) on
  pendigits_20nan and letter_20nan from MCAR evidence (E38). E119 ([49](49-bigger-encoder-mar-mnar.md))
  found it helps at 40–60nan under MAR and MNAR too, and ADR 0037 now covers all three mechanisms there
  (decision 3). This asks the same at 20nan, the one level left out, so the thesis can state the scope of the
  bigger encoder on the large tables from 20 to 60nan.
- **The user's go:** "faça os dois em paralelo (1 e 2)" (2026-11-03), on my proposal.
- **Code:** `large_batch.py --study e121` (`STUDY=e121` for `run_large_batch_queue.sh`). No change to `src/`.

## Data (generated before the first cell; the data directory is gitignored, the scripts are the record)

- **MAR:** `<base>_marf_20nan`, already written by `generate_mar_family.py` for E119 (seed 4060, slope 1.5),
  unused until now; 0.137 (pendigits) and 0.138 (letter) of feature cells missing.
- **MNAR:** `generate_mnar.py --bases pendigits letter --pct 0.2` (seed 1060, slope 0.5):
  `<base>_probit_20nan`, 0.200 of feature cells missing.
- md5 sums equal on both machines for all four files (2026-11-03 20:52 GMT-3).

## Design (fixed before the first cell)

- **Variants (4):** {pendigits, letter} × {marf, probit} × 20nan.
- **Arms**, both as override over the MCAR sibling's resolved configuration (`<base>_20nan`, ADR 0018's
  encoder and ADR 0036's P600), as E119:
  - **M:** the sibling's configuration with the encoder undone (LAYERS 2, DIM_FEED 32, ENCODER_FINAL_NORM none);
  - **BIG:** the sibling's configuration as is.
- **Seeds:** 42 on gorgona8, 7 and 13 on medusa1 (ADR 0035), five folds, MLflow off. 24 cells.
- **Cost:** about 2–3 h on gorgona8 (two queues), less on medusa1 (four queues).

## Measures and rule, fixed now (`large_batch.py --study e121 --tally`)

- Per variant, BIG − M on the induced `impute_score`, relative to M, 15 fold pairs, the project's rule,
  1% SESOI (E119's).
- **Per mechanism (2 variants each):** **holds** if BIG is better on both and worse on none; **does not hold**
  if better on neither, or worse on any; **partial** otherwise.
- **What follows:** the result goes to the user, who decides whether ADR 0018 states its scope as all three
  mechanisms. Nothing changes on its own.
- **Read only with every cell done**; a cut stays partial.
