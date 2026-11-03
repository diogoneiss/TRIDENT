# Does ADR 0037's bigger encoder help off MCAR? (E119; pre-registered 2026-11-03)

This ticket was written and committed before the study's first cell (15:20 GMT-3).

- **Why:** ADR 0037 adopted ADR 0018's encoder (LAYERS 4, DIM_FEED 256, ENCODER_FINAL_NORM layer) on
  pendigits and letter at 40–60nan from MCAR evidence alone (E113, E116). The thesis also reports results
  under MAR and MNAR (ADR 0030 decision 7, E84, E115). This asks whether the gain holds when the gaps are not
  random, so ADR 0037's scope can be stated.
- **The user's go:** "1 e 2 agora" (2026-11-03), on my proposal.
- **Code:** `large_batch.py --study e119` (`STUDY=e119` for `run_large_batch_queue.sh`); pendigits and letter
  added to `generate_mar_family.py`'s bases. No change to `src/`.

## Data (generated before the first cell; the data directory is gitignored, the scripts are the record)

- **MAR:** `generate_mar_family.py` (seed 4060, slope 1.5): `<base>_marf_{40,60}nan`, 5 of 16 columns as
  fully observed drivers, every other column at the MCAR count. Feature cells missing: 0.275 at 40nan, 0.412
  at 60nan. The manipulation check passed (driver-score shift +0.72 to +0.94; controls within ±0.01). The new
  bases also wrote 20nan and the uniform controls, unused here; existing files were unchanged.
- **MNAR:** `generate_mnar.py --bases pendigits letter --pct 0.4|0.6` (seed 1060, slope 0.5, self-masking
  probit weights): `<base>_probit_{40,60}nan`, every feature column at the MCAR count.
- Both machines generate the same files; their md5 sums are compared before the queues start.

## Design (fixed before the first cell)

- **Variants (8):** {pendigits, letter} × {marf, probit} × {40, 60}nan.
- **Arms**, both as override over the MCAR sibling's resolved configuration (`<base>_<rate>nan`, with ADR
  0036's P600 where it applies), since the new variants have no promoted file:
  - **M:** the sibling's configuration with ADR 0037's encoder undone (LAYERS 2, DIM_FEED 32,
    ENCODER_FINAL_NORM none);
  - **BIG:** the sibling's configuration as is (ADR 0037's encoder).
- **Seeds:** 42 on gorgona8, 7 and 13 on medusa1 (ADR 0035: split by seed, both arms of a pair on one
  machine); five folds, MLflow off, as E113. 48 cells.
- **Cost:** E113's times (pendigits 19–33, letter 32–63 minutes a cell on gorgona8): about 5 h on gorgona8
  (two queues) and 4–5 h on medusa1 (four queues, beside E120 and another user's job on GPU 0).

## Measures and rule, fixed now (`large_batch.py --study e119 --tally`)

- Per variant, BIG − M on the induced `impute_score`, relative to M, 15 fold pairs, the project's rule,
  1% SESOI (E113's).
- **Per mechanism (4 variants each):** **holds** if BIG is better on 3 or more and worse on none; **does not
  hold** if better on 1 or fewer, or worse on any; **partial** otherwise.
- **What follows:** the result goes to the user, who decides whether ADR 0037 states its scope as MCAR only
  or as all three mechanisms. Nothing changes on its own.
- **Read only with every cell done**; a cut stays partial.

## Amendment 2026-11-03 18:56 GMT-3 (the user's decision)

gorgona8 ran slower than planned (20–40 minutes a cell), and medusa1's GPU 1 fell idle at 18:35. To finish
sooner, gorgona8's one unstarted pair, `letter_probit_60nan` seed 42, runs on medusa1's GPU 1 instead, both
arms there (ADR 0035's pairing holds; seed 42 is now split between machines on this one variant). gorgona8 stops
between cells once queue 1 starts its last cell, so it never starts that pair. `--tally` reads the pair from
medusa1's directory (`MOVED` in `large_batch.py`). Nothing else changes: same cells, seeds, rule.

## Outcome (2026-11-03 19:54 GMT-3): holds under MAR and under MNAR

All 48 cells done, none failed (the moved pair ran on medusa1 as amended; gorgona8 stopped on the STOP file
after its last pair). The unedited `large_batch.py --study e119 --tally`:

```
## BIG − M on impute/induced/impute_score, relative to M; 15 fold pairs a variant (fold-pair t, seed agreement, SESOI 0.01)
- pendigits_marf_40nan: -3.61% [-4.41%, -2.81%] seeds -3.81%, -2.97%, -4.05% -> better; masked -4.06%
- pendigits_marf_60nan: -4.73% [-5.25%, -4.21%] seeds -4.96%, -4.59%, -4.65% -> better; masked -3.04%
- letter_marf_40nan: -5.94% [-6.26%, -5.63%] seeds -6.02%, -5.92%, -5.89% -> better; masked -5.35%
- letter_marf_60nan: -5.06% [-5.41%, -4.70%] seeds -4.86%, -4.98%, -5.33% -> better; masked -2.32%
- pendigits_probit_40nan: -6.89% [-7.34%, -6.43%] seeds -7.17%, -6.22%, -7.27% -> better; masked -4.37%
- pendigits_probit_60nan: -3.55% [-3.93%, -3.18%] seeds -3.49%, -3.32%, -3.85% -> better; masked -0.87%
- letter_probit_40nan: -5.30% [-5.64%, -4.96%] seeds -5.65%, -5.14%, -5.11% -> better; masked -3.77%
- letter_probit_60nan: -1.79% [-1.97%, -1.61%] seeds -1.88%, -1.56%, -1.92% -> better; masked +0.20%

Primary (E119, BIG − M), MAR (marf): better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 4 -> holds

Primary (E119, BIG − M), MNAR (probit): better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) of 4 -> holds
```

- **MAR (marf): holds**, BIG better on 4 of 4 (−3.6% to −5.9%), worse on none.
- **MNAR (probit): holds**, BIG better on 4 of 4 (−1.8% to −6.9%), worse on none.
- All 24 seed means are negative (BIG better). Secondary, masked: better on 7 of 8; `letter_probit_60nan`
  +0.20% (not scored by the rule).
- **What follows:** the user decides whether ADR 0037 states its scope as all three mechanisms; until then it
  stays as written (MCAR evidence).

