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

## Amendment 2026-11-03 22:03 GMT-3 (the user's decision, "sim")

As in E119: medusa1 finished its 16 cells first, so gorgona8's one unstarted pair, `letter_probit_20nan` seed
42, runs on medusa1's GPU 1, both arms there (ADR 0035's pairing holds; seed 42 is split between machines on
this one variant). gorgona8 stops on its STOP file between cells (touched 22:02:57); since queue 1 stops there
too before its pair's last arm, `BIG_pendigits_probit_20nan_s42` runs on gorgona8 by a one-cell runner, so that
pair stays on gorgona8. `--tally` reads the moved pair from medusa1's directory (`MOVED`). Same cells, seeds and
rule.

## Outcome (2026-11-03 22:47 GMT-3): holds under MAR and under MNAR

All 24 cells done, none failed (the moved pair on medusa1 and gorgona8's one-cell runner as amended). The
unedited `large_batch.py --study e121 --tally`:

```
## BIG − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_marf_20nan: -4.59% [-5.56%, -3.62%] seeds -5.58%, -4.15%, -4.05% -> better; masked -5.96%
- letter_marf_20nan: -5.81% [-6.25%, -5.37%] seeds -5.51%, -5.85%, -6.06% -> better; masked -6.37%
- pendigits_probit_20nan: -5.00% [-5.61%, -4.39%] seeds -4.97%, -4.70%, -5.33% -> better; masked -5.71%
- letter_probit_20nan: -6.50% [-6.98%, -6.02%] seeds -6.70%, -6.11%, -6.69% -> better; masked -5.83%

Primary (E121, BIG − M), MAR (marf): better 2 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 2 -> holds

Primary (E121, BIG − M), MNAR (probit): better 2 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 2 -> holds
```

- **MAR (marf): holds**, BIG better on 2 of 2 (−4.6%, −5.8%), worse on none.
- **MNAR (probit): holds**, BIG better on 2 of 2 (−5.0%, −6.5%), worse on none.
- All 12 seed means negative; masked better on 4 of 4.
- **What follows:** the user decides whether ADR 0018 states its scope as all three mechanisms; until then it
  stays as written (MCAR evidence).

