# Does the forest lose on a filled table because it cannot tell a filled cell from an observed one? (E189, T66; pre-registered 2026-11-24 02:03 GMT-3)

## Why

On `kr-vs-kp` 40–80nan and `credit-g` 60–80nan a forest fitted on the table filled by the decoder (K, the imputation
default) scores below the same forest fitted on the table with its gaps left missing (E186, E187, E188: three seed
sets). E188's core read ruled the calibration of the output out as the main cause. The mean/mode fill loses there
too, so the loss is not special to the decoder's values. The leading alternative: on a filled table the forest
treats a filled value as an observed one, while on the table with gaps it can treat the gaps apart.

The user's go: 2026-11-24, "dou o ok", to my proposal of 2026-11-23 23:50 GMT-3 (the fill with gap indicators
against the fill alone, CPU only, on E186's and E187's ledgers).

This is not a claim that the gap pattern predicts the label: the variants are MCAR, so it does not. The hypothesis
is that the forest does better when it can discount filled values.

## What was seen before the first cell

- The losses that are this study's denominator, on all three seed sets (tickets 75, 76, 77). E188's core, seeds
  2718/6464/7272: F1(K) − F1(missing) −0.0229 [−0.0307, −0.0150]; mean/mode − missing −0.0142. E187 per variant:
  `kr-vs-kp` −0.0192, −0.0380, −0.0231; `credit-g` +0.0005 (no loss at 40nan), −0.0258, −0.0543.
- Logistic regression's `missing` table is, by E186's construction, the mean with an indicator (numbers) or a
  category of its own: a flagged fill. It beats mean/mode and K in every read (E188 core: 0.6487 against 0.6289 and
  0.6328). So logistic regression already suggests flags matter; the forest is the informative test.
- `check` on three cells (`credit-g_60nan` 8484, `kr-vs-kp_40nan` 2718, `credit-g_80nan` 202): this study's code
  reproduces the stored F1 of the inherited fills within 2e-16. No flagged table was fitted before this commit; the
  flagged code path has therefore not run, and a crash in it would be fixed by a dated amendment.

**Stated before the first cell:** I expect the gate to pass (mean/mode with flags close to the gaps left missing),
K with flags to recover part of the loss on `kr-vs-kp`, and K with flags to be about level with mean/mode with
flags. I am unsure of the second.

## Disclosures

- **Seeds beyond the go.** The proposal named E186's and E187's ledgers. The core also uses E188's core runs (seeds
  2718/6464/7272, calibrated rows), whose downstream F1 was read at 2026-11-23 23:42 GMT-3. No ledger of E188's
  second batch is opened: its read is still pending.
- **Which K.** Seeds 101/202/303: E184's and E185's arm K, as E186 read them (not E76's runs). Seeds
  1717/8484/9595: E187's arm K. Seeds 2718/6464/7272: E188's arm KR, the `induced` rows.
- **Nothing is fresh for K and missing:** their F1 on every cell here was read before. The flagged tables are new on
  every cell.
- **Intervals.** The plain 95% t interval over (seed, fold) replicates gives the verdict, as in E186 to E188. The
  corrected interval adds a constant to 1/n, so it barely tightens with nine seeds; where it changes a verdict the
  line is marked fragile. Beside, for the core's pooled lines: a t interval over the nine seeds' means.

## Design and rule (fixed before the first cell)

- **No training, no GPU.** `scripts/experiments/indicator_utility.py`; tables, folds and learners are
  `sliced_utility.py`'s (E186's). Output under `/scratch/diogoneiss/trident-2026-11-24-indicator-utility`.
- **Tables per cell (seven):** `missing`, `naive` (mean/mode of the fold's training rows), `K`, `truth`, and three
  flagged ones, `naive+ind`, `K+ind`, `truth+ind`: the fill plus one 0/1 column per feature, 1 where the cell was
  a gap. The flags pass to the forest as numbers.
- **Learner and protocol:** the forest (500 trees), fitted on each table's own rows outside the test fold, macro-F1
  on the test fold. Logistic regression beside. The `complete` protocol is not run.
- **Cells (114):** the core, `credit-g` and `kr-vs-kp` at 40/60/80nan on the nine seeds (54 cells); and the 26 MCAR
  variants of E186 on E187's seeds 1717/8484/9595 (78 cells, 18 shared with the core).
- **Gate on the code:** the tally recomputes nothing but compares every cell's inherited fills (`missing`, `naive`,
  `K`, `truth`; both learners) with the F1 that E186, E187 or E188 stored, and refuses above 1e-12.
- **Primary, the core pooled per (seed, fold), 45 replicates, forest.** With the loss L = F1(missing) − F1(K):
  - **No read** if L's interval is not above 0.
  - **No read, the control fails** if the lower end of F1(naive+ind) − F1(missing) is at or below −L/2. Mean/mode
    with flags holds the same information as the table with gaps; if the forest scores it clearly lower, flag
    columns do not stand in for the forest's own treatment of gaps, and K with flags failing to recover says nothing.
  - Else the change F1(K+ind) − F1(K) is read:
    - below 0: **flags make K's fill worse, not the cause**;
    - above 0 and its upper end under L/2: **a minor part**;
    - above 0 and reaching L/2: **accounts for the loss** if the upper end of F1(K+ind) − F1(missing) is not below
      0, else **accounts for part of the loss**;
    - includes 0 with the upper end under L/2: **not shown, ruled out as the main cause**;
    - includes 0 otherwise: **inconclusive**.
  - L enters as its mean; its own uncertainty is not propagated.
- **Secondary, same rule, not findings unless they hold under the corrected interval:** each table's three
  variants pooled, each variant, and each seed set alone (15 replicates) as a consistency read.
- **Two contrasts beside the rule, pre-registered:**
  - F1(K+ind) − F1(naive+ind): do the decoder's values add anything once the forest can tell filled cells apart?
  - F1(truth+ind) − F1(truth): what the extra columns cost a forest on a complete table, where the flags are noise
    (they double the feature count and so change the forest's sampling of features).
- **The 26 variants, secondary, no cause rule:** per variant (15 fold values) the calls on F1(K+ind) − F1(K),
  F1(K+ind) − F1(missing) and F1(K+ind) − F1(naive+ind), and their counts. The question: is the fill with flags
  ever called worse than the fill alone, or than the gaps left missing?
- **Readings, written before the first cell:**
  - *Accounts for the loss:* given the flags the forest on K's fill is no longer below the gaps; the loss was the
    forest taking filled values for observed ones.
  - *Part, or a minor part:* that explains some of it; the rest is untested.
  - *Ruled out with the gate passed:* the forest loses on K's fill even when it can tell the filled cells; what
    costs is in the filled values themselves. The contrast with `naive+ind` then says whether a constant would
    serve the forest better than the decoder's values.
  - *No read by the gate:* the advantage of the table with gaps may come from how this forest splits on gaps, not
    from information; this design cannot separate them.
- **One read:** `--tally`, once, when the 114 cells are in. It refuses a partial set and a failing gate. No default
  and no ADR changes by this study.
- **Not in this study:** MAR and MNAR, where the gap pattern can predict the label; other learners (boosting with
  its own treatment of gaps), tuned forests; the checkpoint by `sw_score` (S); the `complete` protocol; other draws
  of the gaps; E188's second batch.
