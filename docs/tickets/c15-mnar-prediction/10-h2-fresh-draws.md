# H2's fresh-draw confirmation: odds 3 at seed 4848 against its MCAR siblings, confirmation seeds (E142; pre-registered 2026-11-05)

This ticket was written and committed before any cell (00:30 GMT-3). **Conditional:** it runs only if E139
([09](09-h2-outcome-siblings.md)) reads *lead shrinks* for at least one TabPFN arm; otherwise it is void and nothing runs.

- **Files (exist on both machines since C15's stage 3):** `{biodeg,kc2,spambase}_outcome3p0_s4848_60nan` and
  `{biodeg,kc2,spambase}_mcar_s4848_60nan`.
- **Cells:** A1 on the six files × seeds 1515/2626/3737 = 18 (`c15_run.py` stage `h2c`, all on medusa1 by
  `machine`); TabPFN native and mask arms offline on the 18 pairs (`tabpfn_mnar.py run` picks them up once the A1
  cell exists); reader `--tally-h2c`.
- **Rule:** E139's, on the three odds-3 files and their siblings, pooled per replicate (15), Holm over the two arms:
  shrinks / grows / holds (±0.01) / no verdict. *Shrinks* again → the H2 reading is confirmed on a fresh draw and fresh
  seeds, to the user; otherwise E139 stands alone as a first reading.
