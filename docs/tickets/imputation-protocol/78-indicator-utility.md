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

## Outcome (2026-11-24 02:38 GMT-3): no read on the core as pooled; on `kr-vs-kp` flags account for the loss; on `credit-g` the design cannot test it

The 114 cells were in at 2026-11-24 02:34 GMT-3, none failed. `--tally` was run once, at 2026-11-24 02:35 GMT-3. Its
gate on the code: the inherited fills differ from the F1 stored by E186, E187 and E188 by at most 0.00e+00.

- **Primary, the core pooled: no read, the control fails.** Loss F1(missing) − F1(K) +0.0263 (lower end +0.0225).
  Mean/mode with flags is −0.0190 [−0.0218, −0.0163] against the gaps left missing; the margin was −0.0131. The
  three seed sets alone give the same call. The pooled change F1(K+ind) − F1(K), +0.0016 [−0.0012, +0.0045], is the
  mean of two opposite tables (+0.028 and −0.025): the pre-registered pooling mixed them, and that number describes
  neither.
- **`kr-vs-kp`, its three variants pooled (secondary, holds under the corrected interval): flags account for the
  loss.** Loss +0.0269. The control passes: mean/mode with flags −0.0031 [−0.0049, −0.0012] against the gaps. K
  with flags against K alone **+0.0281 [+0.0252, +0.0309]**, corrected [+0.0174, +0.0387]: 104% of the loss
  [94%, 115%]. K with flags against the gaps +0.0012 [−0.0017, +0.0041], no difference.
  - The recovery is not specific to the decoder's values: mean/mode with flags is level with the gaps too. K with
    flags is above mean/mode with flags only at 40nan (+0.0162, holding corrected), not at 60nan or 80nan; pooled
    +0.0043 [+0.0013, +0.0072], fragile.
  - Per variant all three read "accounts for the loss" by the plain rule; only 60nan keeps it corrected, and there
    the remaining gap's upper end is on the boundary (+0.0000). The table-pooled line is the result.
- **`credit-g`: no read, and the design cannot test it there.** Flag columns cost this forest on every table: on
  the complete table −0.0424 [−0.0490, −0.0358] (holding corrected), on K's fill −0.0248, on mean/mode −0.034 (from
  the levels). The cost of the 20 extra columns is as large as the loss under test (+0.0257). The loss on `credit-g`
  stays unexplained.
- **What the flags cost on a complete table (the pre-registered contrast):** truth with flags is below truth on
  the core (−0.0311), on `kr-vs-kp` (−0.0197) and over the 26 variants (−0.0118), every one holding corrected. It is
  never above, as MCAR implies: the flags carry nothing about the label. On `kr-vs-kp` the recovery comes despite
  that cost.
- **The 26 variants on E187's seeds: flags are not a general recipe.** F1(K+ind) − F1(K): better on 5 (`kr-vs-kp`
  20/40/60/80nan, `pendigits_80nan`), no difference on 11, worse on 10 (`credit-g` 20/40/60/80nan,
  `spambase_20nan`, `pendigits` 20/40/60nan, `electricity_20nan`, `letter_20nan`). Holding corrected: three for the
  flags (`kr-vs-kp` 40 and 60nan, `pendigits_80nan`) and one against (`pendigits_40nan`, by 0.0015). Pooled +0.0006
  [−0.0021, +0.0033], no difference. Against the gaps left missing K with flags is better on 11, no difference on
  11, worse on 4 (`credit-g` 40/60/80nan, `spambase_20nan`); K alone and K with flags do not differ there in the
  pool. The `kr-vs-kp` and `credit-g` cells of this read at 40nan and above are the core's own, not new evidence.
- **Logistic regression, beside.** Its `missing` table is already the mean with a flag, so mean/mode with flags
  must equal it, and does: −0.0005 on the core, −0.0002 on the 26 variants. That is the only check the flagged
  code path has, since none ran before the pre-registration. On the core K with flags is +0.0080 above K alone
  (fragile) and −0.0121 below mean/mode with flags (fragile).
- **Levels only, no verdict:** on `credit-g` K sits below mean/mode at 40, 60 and 80nan, with and without flags.
  Only the flagged contrast has an interval: −0.0155 [−0.0213, −0.0097], fragile.
- **Against what was stated before the first cell:** the control failed on the core, where I expected it to pass
  (it passes on `kr-vs-kp`); the recovery on `kr-vs-kp` is full, where I expected part; K with flags is level with
  mean/mode with flags at 60 and 80nan only.

**Reading.** On `kr-vs-kp`, under MCAR and with this forest, a fill without flags loses to the gaps left missing and
the same fill with flags does not, whether the values are the decoder's or the mode: there the loss is the forest
taking filled values for observed ones. That is one table. On `credit-g` this design gives nothing, because the flag
columns themselves cost the forest as much as the loss, so the cause there is open. Flag columns are not a recipe to
adopt across tables: over 26 variants they are called worse than the fill alone more often than better. No default
and no ADR change.

**Disclosed, printout unedited:**
- `credit-g_40nan`'s gate threshold prints positive (+0.0035) because the loss there is negative; its line is "no
  read" by the first test.
- Seed-set lines exist only for the core pooled. None was computed per table afterwards.
- The first cell (`kc2_20nan` 1717) was run alone after the commit as a crash test; its output was "DONE" and its
  file's header, no score.

```
gate: 114 cells, the inherited fills (missing, naive, K, truth; forest and logistic regression) differ from the F1 stored by E186, E187 and E188 by at most 0.00e+00

## E189, the core: macro-F1 of a forest fitted on each table's own rows, 6 variants (credit-g and kr-vs-kp at 40/60/80nan), seeds 101/202/303/1717/8484/9595/2718/6464/7272; a difference is pooled over the variants per (seed, fold) replicate, 45 replicates, plain 95% t interval (NB: corrected)
F1(K+ind) − F1(K): +0.0016 [-0.0012, +0.0045] (NB [-0.0089, +0.0122]) -> no difference; over 9 seed means [-0.0027, +0.0060]
F1(K+ind) − F1(missing): -0.0247 [-0.0283, -0.0210] (NB [-0.0383, -0.0110]) -> missing; over 9 seed means [-0.0299, -0.0194]
F1(K) − F1(missing): -0.0263 [-0.0301, -0.0225] (NB [-0.0406, -0.0120]) -> missing; over 9 seed means [-0.0325, -0.0201]
F1(naive+ind) − F1(missing): -0.0190 [-0.0218, -0.0163] (NB [-0.0293, -0.0088]) -> missing; over 9 seed means [-0.0217, -0.0163]
F1(K+ind) − F1(naive+ind): -0.0056 [-0.0089, -0.0024] (NB [-0.0176, +0.0064]) -> naive+ind; over 9 seed means [-0.0091, -0.0022]
F1(truth+ind) − F1(truth): -0.0311 [-0.0346, -0.0276] (NB [-0.0441, -0.0180]) -> truth; over 9 seed means [-0.0348, -0.0273]
F1(naive+ind) − F1(naive): -0.0040 [-0.0068, -0.0013] (NB [-0.0143, +0.0062]) -> naive; over 9 seed means [-0.0056, -0.0025]
F1(naive) − F1(missing): -0.0150 [-0.0181, -0.0119] (NB [-0.0266, -0.0034]) -> missing; over 9 seed means [-0.0184, -0.0116]
F1(truth) − F1(missing): +0.2034 [+0.1983, +0.2085] (NB [+0.1844, +0.2223]) -> truth; over 9 seed means [+0.1999, +0.2068]
Levels: missing 0.6240, naive 0.6091, K 0.5978, truth 0.8274, naive+ind 0.6050, K+ind 0.5994, truth+ind 0.7964

### The rule (forest, own): does the forest lose on K's fill because it cannot tell a filled cell from an observed one?
- core, 6 variants pooled per (seed, fold): loss +0.0263 (lower end +0.0225); gate F1(naive+ind) − F1(missing) lower end -0.0218 against -0.0131; F1(K+ind) − F1(K) +0.0016 [-0.0012, +0.0045], as a share of the loss +0.06 [-0.05, +0.17]; F1(K+ind) − F1(missing) upper end -0.0210 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)
- kr-vs-kp, its three variants pooled: loss +0.0269 (lower end +0.0232); gate F1(naive+ind) − F1(missing) lower end -0.0049 against -0.0134; F1(K+ind) − F1(K) +0.0281 [+0.0252, +0.0309], as a share of the loss +1.04 [+0.94, +1.15]; F1(K+ind) − F1(missing) upper end +0.0041 -> accounts for the loss
- credit-g, its three variants pooled: loss +0.0257 (lower end +0.0184); gate F1(naive+ind) − F1(missing) lower end -0.0398 against -0.0128; F1(K+ind) − F1(K) -0.0248 [-0.0299, -0.0197], as a share of the loss -0.97 [-1.16, -0.77]; F1(K+ind) − F1(missing) upper end -0.0436 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing) (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- credit-g_40nan: loss -0.0070 (lower end -0.0183); gate F1(naive+ind) − F1(missing) lower end -0.0226 against +0.0035; F1(K+ind) − F1(K) -0.0345 [-0.0437, -0.0254]; F1(K+ind) − F1(missing) upper end -0.0168 -> no read (the loss F1(missing) − F1(K) is not above 0)
- credit-g_60nan: loss +0.0321 (lower end +0.0195); gate F1(naive+ind) − F1(missing) lower end -0.0535 against -0.0160; F1(K+ind) − F1(K) -0.0225 [-0.0307, -0.0143], as a share of the loss -0.70 [-0.96, -0.45]; F1(K+ind) − F1(missing) upper end -0.0433 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing) (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- credit-g_80nan: loss +0.0520 (lower end +0.0399); gate F1(naive+ind) − F1(missing) lower end -0.0550 against -0.0260; F1(K+ind) − F1(K) -0.0173 [-0.0245, -0.0101], as a share of the loss -0.33 [-0.47, -0.19]; F1(K+ind) − F1(missing) upper end -0.0574 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)
- kr-vs-kp_40nan: loss +0.0183 (lower end +0.0131); gate F1(naive+ind) − F1(missing) lower end -0.0089 against -0.0092; F1(K+ind) − F1(K) +0.0285 [+0.0244, +0.0326], as a share of the loss +1.55 [+1.33, +1.78]; F1(K+ind) − F1(missing) upper end +0.0140 -> accounts for the loss (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- kr-vs-kp_60nan: loss +0.0390 (lower end +0.0325); gate F1(naive+ind) − F1(missing) lower end -0.0054 against -0.0195; F1(K+ind) − F1(K) +0.0340 [+0.0289, +0.0392], as a share of the loss +0.87 [+0.74, +1.00]; F1(K+ind) − F1(missing) upper end +0.0000 -> accounts for the loss
- kr-vs-kp_80nan: loss +0.0233 (lower end +0.0178); gate F1(naive+ind) − F1(missing) lower end -0.0048 against -0.0117; F1(K+ind) − F1(K) +0.0217 [+0.0159, +0.0275], as a share of the loss +0.93 [+0.68, +1.18]; F1(K+ind) − F1(missing) upper end +0.0030 -> accounts for the loss (fragile; corrected: no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing))
Beside, each seed set alone (15 replicates):
- E186's seeds 101/202/303: loss +0.0294 (lower end +0.0229); gate F1(naive+ind) − F1(missing) lower end -0.0258 against -0.0147; F1(K+ind) − F1(K) +0.0002 [-0.0049, +0.0052], as a share of the loss +0.01 [-0.17, +0.18]; F1(K+ind) − F1(missing) upper end -0.0236 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)
- E187's seeds 1717/8484/9595: loss +0.0266 (lower end +0.0198); gate F1(naive+ind) − F1(missing) lower end -0.0237 against -0.0133; F1(K+ind) − F1(K) +0.0039 [-0.0015, +0.0092], as a share of the loss +0.14 [-0.06, +0.35]; F1(K+ind) − F1(missing) upper end -0.0174 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)
- E188's seeds 2718/6464/7272: loss +0.0229 (lower end +0.0150); gate F1(naive+ind) − F1(missing) lower end -0.0226 against -0.0114; F1(K+ind) − F1(K) +0.0009 [-0.0044, +0.0062], as a share of the loss +0.04 [-0.19, +0.27]; F1(K+ind) − F1(missing) upper end -0.0135 -> no read (flags do not stand in for the gaps: mean/mode with flags is not within half the loss of the gaps left missing)

### Per table and per variant (forest, own)
- kr-vs-kp: missing 0.7082, naive 0.6793, K 0.6813, truth 0.9905, naive+ind 0.7051, K+ind 0.7094, truth+ind 0.9708
  - F1(K+ind) − F1(K): +0.0281 [+0.0252, +0.0309] (NB [+0.0174, +0.0387]) -> K+ind
  - F1(K+ind) − F1(missing): +0.0012 [-0.0017, +0.0041] (NB [-0.0096, +0.0120]) -> no difference
  - F1(K) − F1(missing): -0.0269 [-0.0306, -0.0232] (NB [-0.0407, -0.0131]) -> missing
  - F1(naive+ind) − F1(missing): -0.0031 [-0.0049, -0.0012] (NB [-0.0100, +0.0038]) -> missing
  - F1(K+ind) − F1(naive+ind): +0.0043 [+0.0013, +0.0072] (NB [-0.0067, +0.0152]) -> K+ind
  - F1(truth+ind) − F1(truth): -0.0197 [-0.0214, -0.0180] (NB [-0.0261, -0.0134]) -> truth
- credit-g: missing 0.5399, naive 0.5388, K 0.5142, truth 0.6643, naive+ind 0.5049, K+ind 0.4894, truth+ind 0.6219
  - F1(K+ind) − F1(K): -0.0248 [-0.0299, -0.0197] (NB [-0.0439, -0.0057]) -> K
  - F1(K+ind) − F1(missing): -0.0505 [-0.0574, -0.0436] (NB [-0.0761, -0.0249]) -> missing
  - F1(K) − F1(missing): -0.0257 [-0.0330, -0.0184] (NB [-0.0530, +0.0016]) -> missing
  - F1(naive+ind) − F1(missing): -0.0350 [-0.0398, -0.0301] (NB [-0.0530, -0.0169]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0155 [-0.0213, -0.0097] (NB [-0.0370, +0.0060]) -> naive+ind
  - F1(truth+ind) − F1(truth): -0.0424 [-0.0490, -0.0358] (NB [-0.0668, -0.0180]) -> truth
- credit-g_40nan: missing 0.5551, naive 0.5822, K 0.5621, truth 0.6651, naive+ind 0.5406, K+ind 0.5276, truth+ind 0.6144
  - F1(K+ind) − F1(K): -0.0345 [-0.0437, -0.0254] (NB [-0.0685, -0.0005]) -> K
  - F1(K+ind) − F1(missing): -0.0276 [-0.0384, -0.0168] (NB [-0.0677, +0.0126]) -> missing
  - F1(K) − F1(missing): +0.0070 [-0.0044, +0.0183] (NB [-0.0353, +0.0493]) -> no difference
  - F1(naive+ind) − F1(missing): -0.0145 [-0.0226, -0.0064] (NB [-0.0448, +0.0157]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0130 [-0.0233, -0.0028] (NB [-0.0512, +0.0251]) -> naive+ind
  - F1(truth+ind) − F1(truth): -0.0507 [-0.0593, -0.0420] (NB [-0.0829, -0.0184]) -> truth
- credit-g_60nan: missing 0.5323, naive 0.5255, K 0.5003, truth 0.6636, naive+ind 0.4882, K+ind 0.4777, truth+ind 0.6217
  - F1(K+ind) − F1(K): -0.0225 [-0.0307, -0.0143] (NB [-0.0531, +0.0080]) -> K
  - F1(K+ind) − F1(missing): -0.0546 [-0.0659, -0.0433] (NB [-0.0967, -0.0125]) -> missing
  - F1(K) − F1(missing): -0.0321 [-0.0447, -0.0195] (NB [-0.0790, +0.0149]) -> missing
  - F1(naive+ind) − F1(missing): -0.0441 [-0.0535, -0.0348] (NB [-0.0789, -0.0094]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0105 [-0.0216, +0.0006] (NB [-0.0518, +0.0309]) -> no difference
  - F1(truth+ind) − F1(truth): -0.0419 [-0.0503, -0.0335] (NB [-0.0731, -0.0107]) -> truth
- credit-g_80nan: missing 0.5321, naive 0.5086, K 0.4801, truth 0.6643, naive+ind 0.4859, K+ind 0.4628, truth+ind 0.6296
  - F1(K+ind) − F1(K): -0.0173 [-0.0245, -0.0101] (NB [-0.0441, +0.0095]) -> K
  - F1(K+ind) − F1(missing): -0.0693 [-0.0812, -0.0574] (NB [-0.1136, -0.0251]) -> missing
  - F1(K) − F1(missing): -0.0520 [-0.0641, -0.0399] (NB [-0.0971, -0.0068]) -> missing
  - F1(naive+ind) − F1(missing): -0.0463 [-0.0550, -0.0376] (NB [-0.0787, -0.0139]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0230 [-0.0338, -0.0122] (NB [-0.0633, +0.0172]) -> naive+ind
  - F1(truth+ind) − F1(truth): -0.0347 [-0.0421, -0.0272] (NB [-0.0625, -0.0068]) -> truth
- kr-vs-kp_40nan: missing 0.8111, naive 0.7731, K 0.7927, truth 0.9905, naive+ind 0.8050, K+ind 0.8212, truth+ind 0.9695
  - F1(K+ind) − F1(K): +0.0285 [+0.0244, +0.0326] (NB [+0.0133, +0.0437]) -> K+ind
  - F1(K+ind) − F1(missing): +0.0101 [+0.0063, +0.0140] (NB [-0.0042, +0.0245]) -> K+ind
  - F1(K) − F1(missing): -0.0183 [-0.0236, -0.0131] (NB [-0.0378, +0.0011]) -> missing
  - F1(naive+ind) − F1(missing): -0.0061 [-0.0089, -0.0032] (NB [-0.0167, +0.0045]) -> missing
  - F1(K+ind) − F1(naive+ind): +0.0162 [+0.0129, +0.0195] (NB [+0.0040, +0.0284]) -> K+ind
  - F1(truth+ind) − F1(truth): -0.0210 [-0.0228, -0.0191] (NB [-0.0279, -0.0140]) -> truth
- kr-vs-kp_60nan: missing 0.7084, naive 0.6766, K 0.6694, truth 0.9905, naive+ind 0.7060, K+ind 0.7034, truth+ind 0.9691
  - F1(K+ind) − F1(K): +0.0340 [+0.0289, +0.0392] (NB [+0.0149, +0.0532]) -> K+ind
  - F1(K+ind) − F1(missing): -0.0050 [-0.0099, +0.0000] (NB [-0.0234, +0.0135]) -> no difference
  - F1(K) − F1(missing): -0.0390 [-0.0454, -0.0325] (NB [-0.0630, -0.0150]) -> missing
  - F1(naive+ind) − F1(missing): -0.0024 [-0.0054, +0.0006] (NB [-0.0136, +0.0087]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0025 [-0.0076, +0.0026] (NB [-0.0215, +0.0164]) -> no difference
  - F1(truth+ind) − F1(truth): -0.0214 [-0.0235, -0.0194] (NB [-0.0290, -0.0138]) -> truth
- kr-vs-kp_80nan: missing 0.6052, naive 0.5883, K 0.5819, truth 0.9905, naive+ind 0.6044, K+ind 0.6036, truth+ind 0.9737
  - F1(K+ind) − F1(K): +0.0217 [+0.0159, +0.0275] (NB [+0.0003, +0.0431]) -> K+ind
  - F1(K+ind) − F1(missing): -0.0016 [-0.0062, +0.0030] (NB [-0.0187, +0.0155]) -> no difference
  - F1(K) − F1(missing): -0.0233 [-0.0288, -0.0178] (NB [-0.0437, -0.0029]) -> missing
  - F1(naive+ind) − F1(missing): -0.0008 [-0.0048, +0.0033] (NB [-0.0158, +0.0142]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0008 [-0.0057, +0.0040] (NB [-0.0190, +0.0173]) -> no difference
  - F1(truth+ind) − F1(truth): -0.0167 [-0.0183, -0.0151] (NB [-0.0227, -0.0108]) -> truth

### Beside: logistic regression fitted on each table's own rows (a gap is the mean with an indicator, or its own category), the core
- F1(K+ind) − F1(K): +0.0080 [+0.0055, +0.0105] (NB [-0.0013, +0.0174]) -> K+ind
- F1(K+ind) − F1(missing): -0.0126 [-0.0159, -0.0093] (NB [-0.0248, -0.0004]) -> missing
- F1(K) − F1(missing): -0.0206 [-0.0244, -0.0169] (NB [-0.0345, -0.0068]) -> missing
- F1(naive+ind) − F1(missing): -0.0005 [-0.0025, +0.0014] (NB [-0.0077, +0.0066]) -> no difference
- F1(K+ind) − F1(naive+ind): -0.0121 [-0.0156, -0.0085] (NB [-0.0253, +0.0011]) -> naive+ind
- F1(truth+ind) − F1(truth): -0.0053 [-0.0077, -0.0028] (NB [-0.0144, +0.0038]) -> truth
- F1(naive+ind) − F1(naive): +0.0201 [+0.0169, +0.0234] (NB [+0.0080, +0.0323]) -> naive+ind
- F1(naive) − F1(missing): -0.0207 [-0.0240, -0.0173] (NB [-0.0332, -0.0082]) -> missing
- F1(truth) − F1(missing): +0.1726 [+0.1677, +0.1775] (NB [+0.1544, +0.1908]) -> truth
- Levels: missing 0.6509, naive 0.6302, K 0.6302, truth 0.8235, naive+ind 0.6503, K+ind 0.6383, truth+ind 0.8182

## E189, the 26 MCAR variants on E187's seeds 1717/8484/9595 (forest, own; 15 replicates): is the fill with flags ever worse than the fill alone, or than the gaps left missing?
F1(K+ind) − F1(K): +0.0006 [-0.0021, +0.0033] (NB [-0.0056, +0.0068]) -> no difference
F1(K+ind) − F1(missing): +0.0042 [+0.0003, +0.0081] (NB [-0.0048, +0.0132]) -> K+ind
F1(K) − F1(missing): +0.0036 [-0.0005, +0.0077] (NB [-0.0057, +0.0129]) -> no difference
F1(naive+ind) − F1(missing): -0.0107 [-0.0128, -0.0087] (NB [-0.0155, -0.0060]) -> missing
F1(K+ind) − F1(naive+ind): +0.0150 [+0.0102, +0.0197] (NB [+0.0040, +0.0260]) -> K+ind
F1(truth+ind) − F1(truth): -0.0118 [-0.0149, -0.0087] (NB [-0.0189, -0.0046]) -> truth
F1(naive+ind) − F1(naive): -0.0020 [-0.0043, +0.0003] (NB [-0.0073, +0.0033]) -> no difference
F1(naive) − F1(missing): -0.0088 [-0.0113, -0.0062] (NB [-0.0145, -0.0030]) -> missing
F1(truth) − F1(missing): +0.1379 [+0.1323, +0.1435] (NB [+0.1251, +0.1508]) -> truth
Levels: missing 0.7142, naive 0.7054, K 0.7178, truth 0.8521, naive+ind 0.7034, K+ind 0.7184, truth+ind 0.8403

### Per variant, 15 fold values a variant
- kc2_20nan: missing 0.7158, naive 0.7083, K 0.6949, truth 0.7133, naive+ind 0.7217, K+ind 0.7061, truth+ind 0.7159
  - F1(K+ind) − F1(K): +0.0112 [-0.0067, +0.0292] (NB [-0.0300, +0.0525]) -> no difference
  - F1(K+ind) − F1(missing): -0.0097 [-0.0237, +0.0043] (NB [-0.0418, +0.0224]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0155 [-0.0284, -0.0026] (NB [-0.0452, +0.0141]) -> naive+ind
- kc2_40nan: missing 0.7262, naive 0.7267, K 0.7170, truth 0.7083, naive+ind 0.7183, K+ind 0.7240, truth+ind 0.7193
  - F1(K+ind) − F1(K): +0.0070 [-0.0051, +0.0192] (NB [-0.0209, +0.0350]) -> no difference
  - F1(K+ind) − F1(missing): -0.0021 [-0.0233, +0.0191] (NB [-0.0509, +0.0466]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0057 [-0.0170, +0.0284] (NB [-0.0465, +0.0579]) -> no difference
- kc2_80nan: missing 0.7074, naive 0.7194, K 0.7037, truth 0.7119, naive+ind 0.7146, K+ind 0.7053, truth+ind 0.7190
  - F1(K+ind) − F1(K): +0.0015 [-0.0095, +0.0125] (NB [-0.0237, +0.0268]) -> no difference
  - F1(K+ind) − F1(missing): -0.0021 [-0.0277, +0.0234] (NB [-0.0609, +0.0566]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0093 [-0.0361, +0.0176] (NB [-0.0710, +0.0524]) -> no difference
- vehicle_20nan: missing 0.7036, naive 0.6886, K 0.7178, truth 0.7435, naive+ind 0.6873, K+ind 0.7144, truth+ind 0.7524
  - F1(K+ind) − F1(K): -0.0034 [-0.0107, +0.0040] (NB [-0.0203, +0.0136]) -> no difference
  - F1(K+ind) − F1(missing): +0.0108 [+0.0018, +0.0199] (NB [-0.0099, +0.0316]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0271 [+0.0181, +0.0361] (NB [+0.0064, +0.0478]) -> K+ind
- vehicle_40nan: missing 0.6559, naive 0.6475, K 0.6861, truth 0.7464, naive+ind 0.6448, K+ind 0.6840, truth+ind 0.7488
  - F1(K+ind) − F1(K): -0.0021 [-0.0130, +0.0088] (NB [-0.0272, +0.0230]) -> no difference
  - F1(K+ind) − F1(missing): +0.0281 [+0.0109, +0.0453] (NB [-0.0115, +0.0677]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0392 [+0.0154, +0.0631] (NB [-0.0156, +0.0941]) -> K+ind
- vehicle_80nan: missing 0.4391, naive 0.4482, K 0.4427, truth 0.7467, naive+ind 0.4381, K+ind 0.4480, truth+ind 0.7451
  - F1(K+ind) − F1(K): +0.0053 [-0.0122, +0.0229] (NB [-0.0351, +0.0458]) -> no difference
  - F1(K+ind) − F1(missing): +0.0090 [-0.0150, +0.0329] (NB [-0.0461, +0.0641]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0099 [-0.0181, +0.0379] (NB [-0.0545, +0.0743]) -> no difference
- biodeg_20nan: missing 0.8290, naive 0.8359, K 0.8446, truth 0.8531, naive+ind 0.8340, K+ind 0.8419, truth+ind 0.8514
  - F1(K+ind) − F1(K): -0.0026 [-0.0092, +0.0039] (NB [-0.0177, +0.0125]) -> no difference
  - F1(K+ind) − F1(missing): +0.0129 [+0.0078, +0.0181] (NB [+0.0011, +0.0248]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0080 [-0.0009, +0.0169] (NB [-0.0124, +0.0284]) -> no difference
- biodeg_40nan: missing 0.7986, naive 0.7886, K 0.8206, truth 0.8523, naive+ind 0.7836, K+ind 0.8235, truth+ind 0.8454
  - F1(K+ind) − F1(K): +0.0030 [-0.0020, +0.0080] (NB [-0.0085, +0.0144]) -> no difference
  - F1(K+ind) − F1(missing): +0.0249 [+0.0104, +0.0395] (NB [-0.0085, +0.0583]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0399 [+0.0198, +0.0600] (NB [-0.0064, +0.0861]) -> K+ind
- biodeg_80nan: missing 0.6672, naive 0.6709, K 0.6705, truth 0.8537, naive+ind 0.6516, K+ind 0.6752, truth+ind 0.8482
  - F1(K+ind) − F1(K): +0.0048 [-0.0084, +0.0180] (NB [-0.0256, +0.0351]) -> no difference
  - F1(K+ind) − F1(missing): +0.0080 [-0.0391, +0.0551] (NB [-0.1002, +0.1163]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0237 [-0.0165, +0.0638] (NB [-0.0686, +0.1159]) -> no difference
- credit-g_20nan: missing 0.5679, naive 0.5796, K 0.6068, truth 0.6657, naive+ind 0.5579, K+ind 0.5679, truth+ind 0.6243
  - F1(K+ind) − F1(K): -0.0390 [-0.0564, -0.0215] (NB [-0.0791, +0.0012]) -> K
  - F1(K+ind) − F1(missing): -0.0000 [-0.0178, +0.0177] (NB [-0.0408, +0.0408]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0099 [-0.0061, +0.0260] (NB [-0.0270, +0.0469]) -> no difference
- credit-g_40nan: missing 0.5535, naive 0.5836, K 0.5540, truth 0.6642, naive+ind 0.5433, K+ind 0.5281, truth+ind 0.6120
  - F1(K+ind) − F1(K): -0.0259 [-0.0461, -0.0057] (NB [-0.0724, +0.0206]) -> K
  - F1(K+ind) − F1(missing): -0.0254 [-0.0419, -0.0090] (NB [-0.0632, +0.0124]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0152 [-0.0329, +0.0025] (NB [-0.0559, +0.0255]) -> no difference
- credit-g_60nan: missing 0.5329, naive 0.5230, K 0.5071, truth 0.6647, naive+ind 0.4891, K+ind 0.4841, truth+ind 0.6210
  - F1(K+ind) − F1(K): -0.0230 [-0.0377, -0.0083] (NB [-0.0569, +0.0108]) -> K
  - F1(K+ind) − F1(missing): -0.0488 [-0.0722, -0.0254] (NB [-0.1026, +0.0050]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0051 [-0.0289, +0.0188] (NB [-0.0598, +0.0497]) -> no difference
- credit-g_80nan: missing 0.5297, naive 0.5194, K 0.4754, truth 0.6669, naive+ind 0.4799, K+ind 0.4640, truth+ind 0.6290
  - F1(K+ind) − F1(K): -0.0114 [-0.0206, -0.0022] (NB [-0.0326, +0.0098]) -> K
  - F1(K+ind) − F1(missing): -0.0657 [-0.0855, -0.0459] (NB [-0.1112, -0.0202]) -> missing
  - F1(K+ind) − F1(naive+ind): -0.0158 [-0.0375, +0.0058] (NB [-0.0657, +0.0340]) -> no difference
- kr-vs-kp_20nan: missing 0.8941, naive 0.8747, K 0.8958, truth 0.9906, naive+ind 0.8889, K+ind 0.9064, truth+ind 0.9757
  - F1(K+ind) − F1(K): +0.0106 [+0.0044, +0.0168] (NB [-0.0037, +0.0249]) -> K+ind
  - F1(K+ind) − F1(missing): +0.0123 [+0.0046, +0.0200] (NB [-0.0053, +0.0300]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0175 [+0.0099, +0.0251] (NB [+0.0000, +0.0350]) -> K+ind
- kr-vs-kp_40nan: missing 0.8114, naive 0.7729, K 0.7922, truth 0.9906, naive+ind 0.8064, K+ind 0.8215, truth+ind 0.9697
  - F1(K+ind) − F1(K): +0.0293 [+0.0224, +0.0362] (NB [+0.0134, +0.0452]) -> K+ind
  - F1(K+ind) − F1(missing): +0.0101 [+0.0026, +0.0176] (NB [-0.0072, +0.0274]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0151 [+0.0079, +0.0222] (NB [-0.0013, +0.0314]) -> K+ind
- kr-vs-kp_60nan: missing 0.7072, naive 0.6766, K 0.6692, truth 0.9906, naive+ind 0.7056, K+ind 0.7039, truth+ind 0.9699
  - F1(K+ind) − F1(K): +0.0347 [+0.0225, +0.0469] (NB [+0.0066, +0.0628]) -> K+ind
  - F1(K+ind) − F1(missing): -0.0033 [-0.0129, +0.0063] (NB [-0.0253, +0.0188]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0017 [-0.0110, +0.0075] (NB [-0.0230, +0.0195]) -> no difference
- kr-vs-kp_80nan: missing 0.6109, naive 0.5872, K 0.5878, truth 0.9906, naive+ind 0.6078, K+ind 0.6073, truth+ind 0.9728
  - F1(K+ind) − F1(K): +0.0195 [+0.0077, +0.0313] (NB [-0.0076, +0.0466]) -> K+ind
  - F1(K+ind) − F1(missing): -0.0036 [-0.0112, +0.0041] (NB [-0.0211, +0.0140]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0005 [-0.0095, +0.0084] (NB [-0.0211, +0.0200]) -> no difference
- spambase_20nan: missing 0.9240, naive 0.9206, K 0.9210, truth 0.9498, naive+ind 0.9191, K+ind 0.9191, truth+ind 0.9416
  - F1(K+ind) − F1(K): -0.0019 [-0.0036, -0.0001] (NB [-0.0059, +0.0021]) -> K
  - F1(K+ind) − F1(missing): -0.0049 [-0.0093, -0.0005] (NB [-0.0150, +0.0053]) -> missing
  - F1(K+ind) − F1(naive+ind): +0.0000 [-0.0044, +0.0045] (NB [-0.0103, +0.0103]) -> no difference
- pendigits_20nan: missing 0.9682, naive 0.9610, K 0.9837, truth 0.9916, naive+ind 0.9609, K+ind 0.9826, truth+ind 0.9907
  - F1(K+ind) − F1(K): -0.0011 [-0.0021, -0.0001] (NB [-0.0033, +0.0011]) -> K
  - F1(K+ind) − F1(missing): +0.0144 [+0.0128, +0.0159] (NB [+0.0108, +0.0180]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0217 [+0.0198, +0.0236] (NB [+0.0174, +0.0260]) -> K+ind
- pendigits_40nan: missing 0.9103, naive 0.8856, K 0.9519, truth 0.9917, naive+ind 0.8879, K+ind 0.9503, truth+ind 0.9900
  - F1(K+ind) − F1(K): -0.0015 [-0.0022, -0.0009] (NB [-0.0030, -0.0001]) -> K
  - F1(K+ind) − F1(missing): +0.0400 [+0.0369, +0.0432] (NB [+0.0328, +0.0473]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0624 [+0.0572, +0.0676] (NB [+0.0505, +0.0743]) -> K+ind
- pendigits_60nan: missing 0.7829, naive 0.7458, K 0.8416, truth 0.9915, naive+ind 0.7560, K+ind 0.8396, truth+ind 0.9896
  - F1(K+ind) − F1(K): -0.0019 [-0.0033, -0.0005] (NB [-0.0051, +0.0013]) -> K
  - F1(K+ind) − F1(missing): +0.0568 [+0.0515, +0.0620] (NB [+0.0447, +0.0688]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0836 [+0.0738, +0.0935] (NB [+0.0609, +0.1063]) -> K+ind
- pendigits_80nan: missing 0.5024, naive 0.4710, K 0.4991, truth 0.9915, naive+ind 0.4866, K+ind 0.5049, truth+ind 0.9905
  - F1(K+ind) − F1(K): +0.0058 [+0.0036, +0.0080] (NB [+0.0008, +0.0109]) -> K+ind
  - F1(K+ind) − F1(missing): +0.0024 [-0.0048, +0.0097] (NB [-0.0143, +0.0192]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0182 [-0.0057, +0.0422] (NB [-0.0368, +0.0732]) -> no difference
- electricity_20nan: missing 0.8227, naive 0.8197, K 0.8298, truth 0.9063, naive+ind 0.8156, K+ind 0.8274, truth+ind 0.8904
  - F1(K+ind) − F1(K): -0.0023 [-0.0036, -0.0011] (NB [-0.0052, +0.0006]) -> K
  - F1(K+ind) − F1(missing): +0.0048 [+0.0038, +0.0058] (NB [+0.0024, +0.0072]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0118 [+0.0107, +0.0130] (NB [+0.0092, +0.0144]) -> K+ind
- electricity_40nan: missing 0.7538, naive 0.7451, K 0.7555, truth 0.9063, naive+ind 0.7456, K+ind 0.7559, truth+ind 0.8838
  - F1(K+ind) − F1(K): +0.0004 [-0.0007, +0.0016] (NB [-0.0022, +0.0030]) -> no difference
  - F1(K+ind) − F1(missing): +0.0022 [-0.0001, +0.0044] (NB [-0.0031, +0.0074]) -> no difference
  - F1(K+ind) − F1(naive+ind): +0.0104 [+0.0086, +0.0122] (NB [+0.0062, +0.0145]) -> K+ind
- electricity_80nan: missing 0.5953, naive 0.5911, K 0.5935, truth 0.9064, naive+ind 0.5962, K+ind 0.5956, truth+ind 0.8908
  - F1(K+ind) − F1(K): +0.0021 [-0.0003, +0.0045] (NB [-0.0034, +0.0076]) -> no difference
  - F1(K+ind) − F1(missing): +0.0003 [-0.0041, +0.0047] (NB [-0.0098, +0.0103]) -> no difference
  - F1(K+ind) − F1(naive+ind): -0.0006 [-0.0053, +0.0040] (NB [-0.0113, +0.0100]) -> no difference
- letter_20nan: missing 0.8581, naive 0.8493, K 0.8994, truth 0.9654, naive+ind 0.8478, K+ind 0.8964, truth+ind 0.9607
  - F1(K+ind) − F1(K): -0.0030 [-0.0045, -0.0015] (NB [-0.0064, +0.0004]) -> K
  - F1(K+ind) − F1(missing): +0.0383 [+0.0360, +0.0406] (NB [+0.0330, +0.0436]) -> K+ind
  - F1(K+ind) − F1(naive+ind): +0.0486 [+0.0458, +0.0513] (NB [+0.0423, +0.0549]) -> K+ind
Variants, F1(K+ind) − F1(K): K+ind 5, no difference 11, K 10
Variants, F1(K+ind) − F1(missing): K+ind 11, no difference 11, missing 4
Variants, F1(K+ind) − F1(naive+ind): K+ind 11, no difference 14, naive+ind 1

### Beside: logistic regression, the 26 variants
- F1(K+ind) − F1(K): -0.0035 [-0.0056, -0.0013] (NB [-0.0085, +0.0016]) -> K
- F1(K+ind) − F1(missing): +0.0344 [+0.0315, +0.0374] (NB [+0.0277, +0.0412]) -> K+ind
- F1(K) − F1(missing): +0.0379 [+0.0348, +0.0410] (NB [+0.0307, +0.0451]) -> K
- F1(naive+ind) − F1(missing): -0.0002 [-0.0019, +0.0015] (NB [-0.0041, +0.0038]) -> no difference
- F1(K+ind) − F1(naive+ind): +0.0346 [+0.0320, +0.0371] (NB [+0.0287, +0.0405]) -> K+ind
- F1(truth+ind) − F1(truth): -0.0070 [-0.0098, -0.0042] (NB [-0.0134, -0.0006]) -> truth
- F1(naive+ind) − F1(naive): +0.0135 [+0.0112, +0.0159] (NB [+0.0081, +0.0190]) -> naive+ind
- F1(naive) − F1(missing): -0.0137 [-0.0168, -0.0106] (NB [-0.0208, -0.0066]) -> missing
- F1(truth) − F1(missing): +0.1500 [+0.1456, +0.1544] (NB [+0.1398, +0.1602]) -> truth
```
