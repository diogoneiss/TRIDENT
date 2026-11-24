# Is calibration why filling loses to the gaps left missing on kr-vs-kp and credit-g? (E188, T65; pre-registered 2026-11-23 22:05 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E186 and E187 (tickets 75, 76) found, on two seed sets, that a forest does worse on the decoder's fill (K,
  the default output) than on the table with its gaps left missing on `kr-vs-kp` 40–80nan and `credit-g` 60–80nan;
  the cause was not tested. The default output is calibrated from 40nan up (ADR 0027: a number is shrunk toward the
  training mean, a low-confidence category is answered with the mode), and the loss appears from 40nan up. This
  study tests that one candidate cause.
- **The user's go** (2026-11-23): "vamos na sua sugestão", answering my offer after E187: retrain K with
  `--decode_calibration none` on the two tables at 40–80nan, about 18 runs, and repeat the classification.
- **What changed from that offer, and why (mine, disclosed):**
  - **No `none` runs: one default run a cell, with `--score_remask`.** Under that diagnostic a fold ledger holds the
    calibrated output (`induced`) and the same model's output before the calibration (`induced_uncalibrated`,
    `decoding.py`). Checkpoint selection is uncalibrated (ADR 0027, decision 6), so those rows are what a `none` run
    of that model would output. A retrained `none` run would differ from K by retraining too: the same arm, variant
    and seed trained twice on gorgona8 (E183's S and E184's S) differ by 0.01 to 0.03 in a fold's `impute_score`,
    while the calibration moves it by 0.001 to 0.009 (below). The diagnostic draws from its own stream, so the run is
    the default run.
  - **A second batch behind the 18 runs** (the other 11 variants at 40nan and above, 33 runs, about 6.5 h), so the
    GPU is not idle overnight (the user's standing instruction). It asks whether the calibration changes downstream
    utility elsewhere; it does not answer the "why". `touch $S/STOP` ends the queues between cells.
- **Seen before the first cell (all from runs already read, none from this study):**
  - E187's levels (fresh seeds 1717/8484/9595, forest, `own`): the mean/mode fill, which the calibration never
    touches, is also below the gaps left missing on `kr-vs-kp` 40–80nan and `credit-g` 60–80nan (for example
    `kr-vs-kp_60nan`: missing 0.7072, mean/mode 0.6766, K 0.6692; `credit-g_80nan`: 0.5297, 0.5194, 0.4754). So "the
    loss starts where the calibration starts" is weak evidence: every single-value fill loses there.
  - On `credit-g_80nan` K is below mean/mode, and the calibration pulls K's numbers toward the mean (`mean_alpha`
    about 0.5 at 80nan, about 0.85 at 40nan): there the calibration may help. The rule below is two-sided.
  - K's logged scores with and without the calibration (seeds 101 and 1717): `acc_cat` differs by 0.0025 or less on
    all four variants looked at (`kr-vs-kp_40nan` 0.8634 → 0.8635); 7 to 14 of `kr-vs-kp`'s 36 columns carry a
    threshold. A net accuracy can hide offsetting changes, so the share of changed cells is read below.
  - My expectation, stated now: no downstream effect on `kr-vs-kp`; a small one, of either sign, on `credit-g`.
- **Disclosed:**
  - **Seeds 2718, 6464, 7272:** not among the store's `seed` tags and in no script, ticket or log entry (checked
    2026-11-23). Three seeds, fixed now, no extension. A fresh seed changes the folds, the initialisation and the
    training masks; a variant's gap pattern is its file on disk.
  - **E76's runs** (ticket 29: the default run with `--score_remask` and `--score_checkpoints`, seeds 101/202/303, state
    `/scratch2/diogoneiss/trident-2026-10-25-remask-dev`) hold both outputs for the six core variants. Their
    configuration equals today's default (the only differing keys are flags added since, at their defaults). They
    are read beside, labelled not fresh: same seeds as E186, so K − missing on them is a retraining of a known
    result. Before this commit only the gate below ran on them (flags, ledgers); no downstream cell.
  - Nothing in `src/` changes. `sliced_utility.py` (E186's) gains `_fill`, the two lines that fetch an arm's ledger
    moved into a function this study's read replaces; E186's cell `credit-g_40nan` seed 101 recomputed through it is
    byte-identical to the stored file.
- **Scripts (this commit):** runner
  [scripts/experiments/uncalibrated_utility_run.py](../../../scripts/experiments/uncalibrated_utility_run.py) (arm
  `KR`: the default run with the diagnostic); reads
  [scripts/experiments/uncalibrated_utility_read.py](../../../scripts/experiments/uncalibrated_utility_read.py), which
  the queues never import. State `/scratch/diogoneiss/trident-2026-11-23-uncalibrated-utility`, tag
  `experiment = uncalibrated-utility-2026-11-23`.

## Design and rule (fixed before the first cell)

- **Scenarios:** arm KR on fresh seeds 2718/6464/7272 × five folds, gorgona8's RTX 3090 Ti, four queues
  (`c15_queue.sh`), the core first, shortest variants first.
  - **Core, 18 runs, about 1.5 h:** `credit-g` and `kr-vs-kp` at 40, 60 and 80nan.
  - **Second batch, 33 runs, about 6.5 h:** `kc2`, `vehicle`, `biodeg` at 40 and 80nan, `pendigits` at 40, 60 and
    80nan, `electricity` at 40 and 80nan (the rest of E184's and E185's variants at 40nan and above; `pendigits`
    with its promoted files).
- **Fills, from one run:** **K** = the ledger's `induced` rows (calibrated, the default output); **N** = its
  `induced_uncalibrated` rows. Beside them `missing`, `naive` (the fold's mean/mode) and `truth`, as E186.
- **Check before a read (`--gate`, and the tally refuses on a failure):** each run's `DECODE_CALIBRATION gaps`,
  `DECODE_METRIC impute_score`, no loss term; in every fold the two outputs are on the same cells with the same
  truths; K's rows reproduce the run's `impute/induced/acc_cat` and `rmse_num_z`, N's rows its
  `impute/induced/uncalibrated/` ones. On E76's 18 runs the gate passes (run before this commit).
- **Downstream read, as E186 and E187:** each row's gaps filled by the fold in which it is a test row; macro-F1 on
  the test folds; a random forest of 500 trees (primary) and logistic regression (beside); protocol `own`
  (primary) and `complete` (beside). A pooled difference is over the variants per (seed, fold) replicate, 15
  replicates, plain 95% t interval; the Nadeau–Bengio interval beside, and a verdict it does not share is reported
  as fragile.
- **Primary, the core on the fresh seeds, forest, `own`.** Three pooled differences over the six variants:
  the loss F1(missing) − F1(K), the change F1(N) − F1(K), and the rest F1(N) − F1(missing).
  - **Precondition:** the loss's interval is above 0. Otherwise no read (nothing to explain on these seeds).
  - **Calibration helps here, not a cause:** the change's interval is below 0.
  - **A cause:** the change's interval is above 0 and its upper end reaches half the loss. It **explains the loss**
    if the rest's interval is not below 0, **part of the loss** if it still is.
  - **A minor cause:** the change's interval is above 0 and its upper end is under half the loss.
  - **Not shown, ruled out as the main cause:** the change's interval includes 0 and its upper end is under half
    the loss. This is the study's margin for a null: without it "no difference" would say nothing.
  - **Inconclusive:** the change's interval includes 0 and its upper end reaches half the loss.
  - The loss enters as its mean; its own uncertainty is not propagated. The change is also printed as a share of
    the loss, with its interval's ends.
- **Secondary, same rule:** each table's three variants pooled (`kr-vs-kp`: categories only, so only the mode
  fallback acts; `credit-g`: both parts act), and each variant alone (15 fold values).
- **Beside, labelled not fresh:** the same lines on E76's runs. Agreement with the fresh read is stated; the
  verdict is the fresh read's.
- **What the calibration changes (a manipulation read, no rule):** per variant, the share of the test folds' gap
  cells whose category differs between K and N, the share of numbers moved and their mean shift, with the runs'
  `n_thresholded_columns` and `mean_alpha`. An effect of zero with few cells changed means the calibration did
  little, not that a calibration could not matter.
- **Second batch, secondary:** F1(N) − F1(K) pooled over the 11 variants and per variant: N if the interval is
  above 0, K if below, no difference otherwise, with the interval as the bound. No cause rule (K does not lose to
  the gaps there). Beside: K − missing per variant on a third seed set, the changed cells, logistic regression,
  `complete`.
- **Two reads, each once:** `--tally core` when the 18 core runs and E76's 18 downstream cells are in;
  `--tally rest` when the 33 are in. Each refuses a partial set. No other read. No default and no ADR changes by
  this study, whatever it finds; ADR 0027's rule was set on `impute_score` and stays the user's decision.
- **Not in this study:** whether the forest loses the "this cell was missing" signal once a value stands in the gap
  (the leading alternative after the observation above: a CPU-only read on E186's and E187's ledgers, a separate
  study); the re-masked fill itself (in the same ledgers, not read); MAR and MNAR; tuned classifiers; other draws
  of the gaps.

## Core read (2026-11-23 23:47 GMT-3; first of two reads): calibration is ruled out as the main cause on the core; `credit-g` alone gives no robust read

The 18 core runs were in at 2026-11-23 23:35 GMT-3, none failed. The 36 downstream cells (18 fresh, 18 of E76's
runs) were computed without error and `--tally core` was run once, at 2026-11-23 23:42 GMT-3. Its gate: 36 finished
runs checked, 0 failed. The study stays in progress: the second batch (33 runs) is training and `--tally rest` is
the second read.

- **Primary, the core pooled (fresh seeds): not shown, ruled out as the main cause.** The loss F1(missing) − F1(K)
  is +0.0229 (lower end +0.0150). The change F1(N) − F1(K) is **−0.0003 [−0.0047, +0.0041]**, corrected
  [−0.0104, +0.0099]: its upper end is 18% of the loss, under the half that was the margin. The verdict is the
  same under the corrected interval.
- **Per table.** `kr-vs-kp`, its three variants pooled: the same verdict, also under the corrected interval (loss
  +0.0242; change +0.0006 [−0.0031, +0.0044]). `credit-g`, its three variants pooled: ruled out by the plain rule
  (loss +0.0216; change −0.0011 [−0.0092, +0.0069]), but the loss itself is not above 0 under the corrected interval,
  so `credit-g` alone gives **no robust read**.
- **Per variant: not findings.** On none of the six variants does the change differ from 0 (N 0, no difference 6,
  K 0). Of the nine secondary lines (two tables, six variants, and the core), only the core and `kr-vs-kp` pooled
  keep their verdict under the corrected interval; `credit-g_40nan` has no read because K does not lose there (seen
  before the first cell).
- **Beside, not fresh (E76's runs, seeds 101/202/303): the same verdict on the core** (loss +0.0310; change −0.0046
  [−0.0100, +0.0008]). Its `kr-vs-kp` line reads "calibration helps" (−0.0069 [−0.0135, −0.0002], fragile); the
  fresh seeds do not repeat it (+0.0006), so no claim that calibration helps is made.
- **What the calibration changes (fresh seeds):** 1.4 to 4.8% of the categorical gap cells take another category;
  73 to 87% of `credit-g`'s numbers move, by a mean of 0.05 to 0.10 z. `acc_cat` moves by 0.004 or less and
  `rmse_num_z` by 0.02 or less. The calibration as ADR 0027 applies it changes little of the fill, which is why its
  effect on the forest is near zero. The verdict is about this calibration, not about what any calibration could do.
- **The loss itself, a third seed set:** F1(K) − F1(missing) −0.0229 [−0.0307, −0.0150] on the core, missing on 5 of
  the 6 variants (`credit-g_40nan` no difference), as in E187. The mean/mode fill, which the calibration never
  touches, loses too (−0.0142 [−0.0217, −0.0067]).
- **Beside, an observation with no rule:** with the classifier fitted on the complete rows, the uncalibrated fill
  scores higher on both seed sets: forest +0.0064 [+0.0033, +0.0094] fresh and +0.0041 [+0.0004, +0.0078] on E76's
  runs; logistic regression +0.0076 [+0.0030, +0.0122] and +0.0033 [−0.0006, +0.0072]. Every corrected interval
  includes 0. It is recorded, and nothing about ADR 0027 follows from it here; that rule is the user's decision.
- **Beside, logistic regression on each table's own rows:** change +0.0005 [−0.0036, +0.0046], no difference.
- **Against what was stated before the first cell:** no effect on `kr-vs-kp`, as expected; on `credit-g` no
  robust read rather than a small effect.

**Reading.** On `kr-vs-kp` and on the core as a whole, the calibration of the default output is not why K's fill
serves the forest worse than the gaps left missing: undoing it on the same model recovers at most 18% of the loss
by the plain interval. What does cause the loss is not tested here. The leading alternative stays untested, and under
MCAR the gap pattern carries no information about the label, so a follow-up should compare the fill with gap
indicators against the fill alone, not assume a lost signal. No default and no ADR change.

**Disclosed, printout unedited:**
- Its title says "sliced selector": the read reuses E186's printer.
- The share's interval on `credit-g_40nan` is printed with its ends reversed, the loss there being negative.
- A standalone `--gate` run just before the tally counted 45 finished runs: the 36 above and nine of the second
  batch already in. It read those nine runs' logged scores and ledgers' cell errors, no downstream F1.
- Before the tally I tailed the downstream job's log: a cell's row and gap counts and the FAILED count (0), no F1.

```
gate: 36 finished runs checked, 0 failed (flags; both outputs on the same cells; each reproduces the run's own acc_cat and rmse_num_z)
## Downstream utility of the sliced selector (E188, core, fresh seeds): macro-F1 on the test folds of 6 MCAR variants, seeds 2718/6464/7272; a difference is pooled over the variants per (seed, fold) replicate, 15 replicates, plain 95% t interval (NB: corrected)
Primary, forest fitted on each table's own rows, F1(N) − F1(K): -0.0003 [-0.0047, +0.0041] (NB [-0.0104, +0.0099]) -> no difference
Three-way, same, F1(K) − F1(missing): -0.0229 [-0.0307, -0.0150] (NB [-0.0409, -0.0048]) -> missing
Three-way, same, F1(N) − F1(missing): -0.0231 [-0.0311, -0.0151] (NB [-0.0415, -0.0048]) -> missing
  beside, F1(naive) − F1(missing): -0.0142 [-0.0217, -0.0067] (NB [-0.0315, +0.0030]) -> missing
  beside, F1(truth) − F1(missing): +0.2016 [+0.1915, +0.2117] (NB [+0.1783, +0.2248]) -> truth
Levels (forest, own): missing 0.6231, naive 0.6088, K 0.6002, N 0.5999, truth 0.8246

### By missingness level (forest, own)
- 40nan (2 variants): missing 0.6856, naive 0.6756, K 0.6793, N 0.6797, truth 0.8255
  - F1(N) − F1(K): +0.0004 [-0.0073, +0.0081] (NB [-0.0172, +0.0181]) -> no difference
  - F1(K) − F1(missing): -0.0063 [-0.0180, +0.0054] (NB [-0.0332, +0.0206]) -> no difference
  - F1(N) − F1(missing): -0.0059 [-0.0158, +0.0040] (NB [-0.0286, +0.0169]) -> no difference
- 60nan (2 variants): missing 0.6193, naive 0.6023, K 0.5857, N 0.5882, truth 0.8246
  - F1(N) − F1(K): +0.0024 [-0.0056, +0.0105] (NB [-0.0160, +0.0209]) -> no difference
  - F1(K) − F1(missing): -0.0336 [-0.0477, -0.0195] (NB [-0.0660, -0.0012]) -> missing
  - F1(N) − F1(missing): -0.0311 [-0.0435, -0.0188] (NB [-0.0595, -0.0028]) -> missing
- 80nan (2 variants): missing 0.5642, naive 0.5486, K 0.5355, N 0.5319, truth 0.8238
  - F1(N) − F1(K): -0.0036 [-0.0148, +0.0076] (NB [-0.0294, +0.0221]) -> no difference
  - F1(K) − F1(missing): -0.0287 [-0.0436, -0.0139] (NB [-0.0629, +0.0054]) -> missing
  - F1(N) − F1(missing): -0.0324 [-0.0475, -0.0173] (NB [-0.0671, +0.0024]) -> missing

### Beside: logistic regression, own (a gap is the mean with an indicator, or its own category)
- F1(N) − F1(K): +0.0005 [-0.0036, +0.0046] (NB [-0.0089, +0.0099]) -> no difference
- F1(K) − F1(missing): -0.0160 [-0.0232, -0.0087] (NB [-0.0326, +0.0007]) -> missing
- F1(N) − F1(missing): -0.0155 [-0.0224, -0.0086] (NB [-0.0314, +0.0004]) -> missing
- Levels: missing 0.6487, naive 0.6289, K 0.6328, N 0.6332, truth 0.8206

### Beside: forest fitted on the complete rows, scored on each table's test fold
- F1(N) − F1(K): +0.0064 [+0.0033, +0.0094] (NB [-0.0007, +0.0134]) -> N
- F1(K) − F1(missing): -0.0066 [-0.0136, +0.0003] (NB [-0.0226, +0.0093]) -> no difference
- F1(N) − F1(missing): -0.0003 [-0.0082, +0.0076] (NB [-0.0185, +0.0180]) -> no difference
- Levels: missing 0.6139, naive 0.5783, K 0.6073, N 0.6136, truth 0.8246

### Beside: logistic regression fitted on the complete rows
- F1(N) − F1(K): +0.0076 [+0.0030, +0.0122] (NB [-0.0030, +0.0181]) -> N
- F1(K) − F1(missing): +0.0285 [+0.0206, +0.0363] (NB [+0.0104, +0.0466]) -> K
- F1(N) − F1(missing): +0.0360 [+0.0262, +0.0459] (NB [+0.0134, +0.0587]) -> N
- Levels: missing 0.5939, naive 0.5877, K 0.6223, N 0.6299, truth 0.8206

### Per variant (forest, own), 15 fold values a variant
- credit-g_40nan: missing 0.5624, naive 0.5778, K 0.5674, N 0.5678, truth 0.6611
  - F1(N) − F1(K): +0.0003 [-0.0145, +0.0152] (NB [-0.0338, +0.0345]) -> no difference
  - F1(K) − F1(missing): +0.0051 [-0.0167, +0.0268] (NB [-0.0450, +0.0551]) -> no difference
  - F1(N) − F1(missing): +0.0054 [-0.0121, +0.0229] (NB [-0.0349, +0.0457]) -> no difference
- credit-g_60nan: missing 0.5320, naive 0.5285, K 0.5008, N 0.5067, truth 0.6593
  - F1(N) − F1(K): +0.0058 [-0.0101, +0.0218] (NB [-0.0308, +0.0425]) -> no difference
  - F1(K) − F1(missing): -0.0312 [-0.0564, -0.0059] (NB [-0.0892, +0.0268]) -> missing
  - F1(N) − F1(missing): -0.0253 [-0.0470, -0.0037] (NB [-0.0752, +0.0245]) -> missing
- credit-g_80nan: missing 0.5278, naive 0.5083, K 0.4892, N 0.4796, truth 0.6578
  - F1(N) − F1(K): -0.0096 [-0.0285, +0.0093] (NB [-0.0530, +0.0339]) -> no difference
  - F1(K) − F1(missing): -0.0386 [-0.0635, -0.0137] (NB [-0.0958, +0.0186]) -> missing
  - F1(N) − F1(missing): -0.0482 [-0.0736, -0.0227] (NB [-0.1067, +0.0104]) -> missing
- kr-vs-kp_40nan: missing 0.8089, naive 0.7733, K 0.7912, N 0.7917, truth 0.9899
  - F1(N) − F1(K): +0.0005 [-0.0036, +0.0047] (NB [-0.0090, +0.0100]) -> no difference
  - F1(K) − F1(missing): -0.0177 [-0.0274, -0.0079] (NB [-0.0401, +0.0047]) -> missing
  - F1(N) − F1(missing): -0.0172 [-0.0264, -0.0079] (NB [-0.0384, +0.0041]) -> missing
- kr-vs-kp_60nan: missing 0.7066, naive 0.6762, K 0.6706, N 0.6697, truth 0.9899
  - F1(N) − F1(K): -0.0010 [-0.0092, +0.0073] (NB [-0.0199, +0.0180]) -> no difference
  - F1(K) − F1(missing): -0.0360 [-0.0462, -0.0257] (NB [-0.0595, -0.0125]) -> missing
  - F1(N) − F1(missing): -0.0369 [-0.0477, -0.0261] (NB [-0.0617, -0.0122]) -> missing
- kr-vs-kp_80nan: missing 0.6007, naive 0.5889, K 0.5818, N 0.5842, truth 0.9899
  - F1(N) − F1(K): +0.0023 [-0.0067, +0.0114] (NB [-0.0185, +0.0231]) -> no difference
  - F1(K) − F1(missing): -0.0189 [-0.0297, -0.0080] (NB [-0.0437, +0.0060]) -> missing
  - F1(N) − F1(missing): -0.0165 [-0.0304, -0.0027] (NB [-0.0484, +0.0153]) -> missing
Variants, F1(N) − F1(K): N 0, no difference 6, K 0
Variants, F1(K) − F1(missing): K 0, no difference 1, missing 5
Variants, F1(N) − F1(missing): N 0, no difference 1, missing 5

### The rule (E188, core, fresh seeds; forest, own): is calibration why K's fill loses to the gaps left missing?
- core, 6 variants pooled per (seed, fold): loss +0.0229 (lower end +0.0150); F1(N) − F1(K) -0.0003 [-0.0047, +0.0041], as a share of the loss -0.01 [-0.20, +0.18]; F1(N) − F1(missing) upper end -0.0151 -> not shown, ruled out as the main cause
- kr-vs-kp, its three variants pooled: loss +0.0242 (lower end +0.0177); F1(N) − F1(K) +0.0006 [-0.0031, +0.0044], as a share of the loss +0.03 [-0.13, +0.18]; F1(N) − F1(missing) upper end -0.0177 -> not shown, ruled out as the main cause
- credit-g, its three variants pooled: loss +0.0216 (lower end +0.0075); F1(N) − F1(K) -0.0011 [-0.0092, +0.0069], as a share of the loss -0.05 [-0.43, +0.32]; F1(N) − F1(missing) upper end -0.0073 -> not shown, ruled out as the main cause (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- credit-g_40nan: loss -0.0051 (lower end -0.0268); F1(N) − F1(K) +0.0003 [-0.0145, +0.0152], as a share of the loss -0.07 [+2.87, -3.00]; F1(N) − F1(missing) upper end +0.0229 -> no read (the loss F1(missing) − F1(K) is not above 0)
- credit-g_60nan: loss +0.0312 (lower end +0.0059); F1(N) − F1(K) +0.0058 [-0.0101, +0.0218], as a share of the loss +0.19 [-0.32, +0.70]; F1(N) − F1(missing) upper end -0.0037 -> inconclusive (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- credit-g_80nan: loss +0.0386 (lower end +0.0137); F1(N) − F1(K) -0.0096 [-0.0285, +0.0093], as a share of the loss -0.25 [-0.74, +0.24]; F1(N) − F1(missing) upper end -0.0227 -> not shown, ruled out as the main cause (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- kr-vs-kp_40nan: loss +0.0177 (lower end +0.0079); F1(N) − F1(K) +0.0005 [-0.0036, +0.0047], as a share of the loss +0.03 [-0.21, +0.26]; F1(N) − F1(missing) upper end -0.0079 -> not shown, ruled out as the main cause (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- kr-vs-kp_60nan: loss +0.0360 (lower end +0.0257); F1(N) − F1(K) -0.0010 [-0.0092, +0.0073], as a share of the loss -0.03 [-0.26, +0.20]; F1(N) − F1(missing) upper end -0.0261 -> not shown, ruled out as the main cause (fragile; corrected: inconclusive)
- kr-vs-kp_80nan: loss +0.0189 (lower end +0.0080); F1(N) − F1(K) +0.0023 [-0.0067, +0.0114], as a share of the loss +0.12 [-0.36, +0.60]; F1(N) − F1(missing) upper end -0.0027 -> inconclusive (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))

### What the calibration changes (E188, core, fresh seeds): gap cells of the test folds where the two outputs differ
- credit-g_40nan: categories changed 438 of 15600 (2.8%); numbers moved 6102 of 8400 (72.6%, mean shift 0.0611 z); n_thresholded_columns 4.60, mean_alpha 0.85, acc_cat K 0.6151, N 0.6150, rmse_num_z K 0.9602, N 0.9652, impute_score K 0.9185, N 0.9202
- credit-g_60nan: categories changed 1132 of 23400 (4.8%); numbers moved 10940 of 12600 (86.8%, mean shift 0.1025 z); n_thresholded_columns 5.07, mean_alpha 0.68, acc_cat K 0.5999, N 0.5961, rmse_num_z K 0.9542, N 0.9740, impute_score K 0.9762, N 0.9893
- credit-g_80nan: categories changed 478 of 31200 (1.5%); numbers moved 12301 of 16800 (73.2%, mean shift 0.0526 z); n_thresholded_columns 1.80, mean_alpha 0.57, acc_cat K 0.5802, N 0.5790, rmse_num_z K 0.9615, N 0.9696, impute_score K 1.0075, N 1.0123
- kr-vs-kp_40nan: categories changed 2947 of 138024 (2.1%); numbers moved 0 of 0; n_thresholded_columns 13.60, acc_cat K 0.8659, N 0.8652, impute_score K 0.7178, N 0.7215
- kr-vs-kp_60nan: categories changed 4927 of 207144 (2.4%); numbers moved 0 of 0; n_thresholded_columns 14.13, acc_cat K 0.8387, N 0.8375, impute_score K 0.8570, N 0.8633
- kr-vs-kp_80nan: categories changed 3876 of 276156 (1.4%); numbers moved 0 of 0; n_thresholded_columns 7.40, acc_cat K 0.8178, N 0.8165, impute_score K 0.9738, N 0.9808

## Downstream utility of the sliced selector (E188, core, not fresh: E76's runs): macro-F1 on the test folds of 6 MCAR variants, seeds 101/202/303; a difference is pooled over the variants per (seed, fold) replicate, 15 replicates, plain 95% t interval (NB: corrected)
Primary, forest fitted on each table's own rows, F1(N) − F1(K): -0.0046 [-0.0100, +0.0008] (NB [-0.0169, +0.0077]) -> no difference
Three-way, same, F1(K) − F1(missing): -0.0310 [-0.0363, -0.0256] (NB [-0.0432, -0.0187]) -> missing
Three-way, same, F1(N) − F1(missing): -0.0356 [-0.0424, -0.0287] (NB [-0.0512, -0.0199]) -> missing
  beside, F1(naive) − F1(missing): -0.0170 [-0.0204, -0.0135] (NB [-0.0248, -0.0091]) -> missing
  beside, F1(truth) − F1(missing): +0.2048 [+0.1976, +0.2120] (NB [+0.1883, +0.2214]) -> truth
Levels (forest, own): missing 0.6248, naive 0.6079, K 0.5939, N 0.5893, truth 0.8297

### By missingness level (forest, own)
- 40nan (2 variants): missing 0.6812, naive 0.6791, K 0.6742, N 0.6668, truth 0.8304
  - F1(N) − F1(K): -0.0073 [-0.0140, -0.0007] (NB [-0.0226, +0.0080]) -> K
  - F1(K) − F1(missing): -0.0071 [-0.0172, +0.0031] (NB [-0.0304, +0.0162]) -> no difference
  - F1(N) − F1(missing): -0.0144 [-0.0237, -0.0051] (NB [-0.0357, +0.0070]) -> missing
- 60nan (2 variants): missing 0.6218, naive 0.6011, K 0.5842, N 0.5820, truth 0.8289
  - F1(N) − F1(K): -0.0022 [-0.0116, +0.0072] (NB [-0.0238, +0.0194]) -> no difference
  - F1(K) − F1(missing): -0.0375 [-0.0454, -0.0297] (NB [-0.0555, -0.0195]) -> missing
  - F1(N) − F1(missing): -0.0398 [-0.0525, -0.0270] (NB [-0.0691, -0.0104]) -> missing
- 80nan (2 variants): missing 0.5715, naive 0.5434, K 0.5232, N 0.5189, truth 0.8296
  - F1(N) − F1(K): -0.0043 [-0.0174, +0.0088] (NB [-0.0343, +0.0258]) -> no difference
  - F1(K) − F1(missing): -0.0483 [-0.0610, -0.0356] (NB [-0.0774, -0.0191]) -> missing
  - F1(N) − F1(missing): -0.0525 [-0.0644, -0.0407] (NB [-0.0798, -0.0253]) -> missing

### Beside: logistic regression, own (a gap is the mean with an indicator, or its own category)
- F1(N) − F1(K): -0.0004 [-0.0041, +0.0032] (NB [-0.0088, +0.0079]) -> no difference
- F1(K) − F1(missing): -0.0251 [-0.0310, -0.0191] (NB [-0.0387, -0.0114]) -> missing
- F1(N) − F1(missing): -0.0255 [-0.0325, -0.0185] (NB [-0.0416, -0.0095]) -> missing
- Levels: missing 0.6518, naive 0.6319, K 0.6267, N 0.6263, truth 0.8256

### Beside: forest fitted on the complete rows, scored on each table's test fold
- F1(N) − F1(K): +0.0041 [+0.0004, +0.0078] (NB [-0.0045, +0.0126]) -> N
- F1(K) − F1(missing): -0.0202 [-0.0283, -0.0122] (NB [-0.0388, -0.0017]) -> missing
- F1(N) − F1(missing): -0.0162 [-0.0243, -0.0080] (NB [-0.0349, +0.0026]) -> missing
- Levels: missing 0.6203, naive 0.5786, K 0.6001, N 0.6042, truth 0.8297

### Beside: logistic regression fitted on the complete rows
- F1(N) − F1(K): +0.0033 [-0.0006, +0.0072] (NB [-0.0057, +0.0123]) -> no difference
- F1(K) − F1(missing): +0.0267 [+0.0201, +0.0333] (NB [+0.0116, +0.0419]) -> K
- F1(N) − F1(missing): +0.0300 [+0.0229, +0.0371] (NB [+0.0138, +0.0462]) -> N
- Levels: missing 0.5914, naive 0.5951, K 0.6181, N 0.6214, truth 0.8256

### Per variant (forest, own), 15 fold values a variant
- credit-g_40nan: missing 0.5495, naive 0.5853, K 0.5535, N 0.5436, truth 0.6699
  - F1(N) − F1(K): -0.0100 [-0.0240, +0.0041] (NB [-0.0422, +0.0223]) -> no difference
  - F1(K) − F1(missing): +0.0040 [-0.0185, +0.0266] (NB [-0.0478, +0.0559]) -> no difference
  - F1(N) − F1(missing): -0.0059 [-0.0257, +0.0139] (NB [-0.0514, +0.0396]) -> no difference
- credit-g_60nan: missing 0.5322, naive 0.5252, K 0.5001, N 0.5058, truth 0.6668
  - F1(N) − F1(K): +0.0057 [-0.0104, +0.0218] (NB [-0.0314, +0.0427]) -> no difference
  - F1(K) − F1(missing): -0.0321 [-0.0440, -0.0202] (NB [-0.0594, -0.0047]) -> missing
  - F1(N) − F1(missing): -0.0264 [-0.0481, -0.0047] (NB [-0.0763, +0.0235]) -> missing
- credit-g_80nan: missing 0.5389, naive 0.4980, K 0.4704, N 0.4676, truth 0.6682
  - F1(N) − F1(K): -0.0027 [-0.0207, +0.0152] (NB [-0.0440, +0.0385]) -> no difference
  - F1(K) − F1(missing): -0.0685 [-0.0893, -0.0477] (NB [-0.1163, -0.0207]) -> missing
  - F1(N) − F1(missing): -0.0713 [-0.0907, -0.0518] (NB [-0.1160, -0.0266]) -> missing
- kr-vs-kp_40nan: missing 0.8130, naive 0.7730, K 0.7948, N 0.7901, truth 0.9910
  - F1(N) − F1(K): -0.0047 [-0.0092, -0.0001] (NB [-0.0151, +0.0057]) -> K
  - F1(K) − F1(missing): -0.0182 [-0.0282, -0.0082] (NB [-0.0411, +0.0048]) -> missing
  - F1(N) − F1(missing): -0.0229 [-0.0311, -0.0146] (NB [-0.0418, -0.0039]) -> missing
- kr-vs-kp_60nan: missing 0.7114, naive 0.6770, K 0.6684, N 0.6583, truth 0.9910
  - F1(N) − F1(K): -0.0101 [-0.0207, +0.0005] (NB [-0.0345, +0.0143]) -> no difference
  - F1(K) − F1(missing): -0.0430 [-0.0546, -0.0314] (NB [-0.0697, -0.0163]) -> missing
  - F1(N) − F1(missing): -0.0531 [-0.0639, -0.0423] (NB [-0.0779, -0.0283]) -> missing
- kr-vs-kp_80nan: missing 0.6040, naive 0.5887, K 0.5760, N 0.5702, truth 0.9910
  - F1(N) − F1(K): -0.0058 [-0.0196, +0.0080] (NB [-0.0376, +0.0259]) -> no difference
  - F1(K) − F1(missing): -0.0280 [-0.0366, -0.0194] (NB [-0.0477, -0.0083]) -> missing
  - F1(N) − F1(missing): -0.0338 [-0.0457, -0.0219] (NB [-0.0611, -0.0065]) -> missing
Variants, F1(N) − F1(K): N 0, no difference 5, K 1
Variants, F1(K) − F1(missing): K 0, no difference 1, missing 5
Variants, F1(N) − F1(missing): N 0, no difference 1, missing 5

### The rule (E188, core, not fresh: E76's runs; forest, own): is calibration why K's fill loses to the gaps left missing?
- core, 6 variants pooled per (seed, fold): loss +0.0310 (lower end +0.0256); F1(N) − F1(K) -0.0046 [-0.0100, +0.0008], as a share of the loss -0.15 [-0.32, +0.02]; F1(N) − F1(missing) upper end -0.0287 -> not shown, ruled out as the main cause
- kr-vs-kp, its three variants pooled: loss +0.0297 (lower end +0.0241); F1(N) − F1(K) -0.0069 [-0.0135, -0.0002], as a share of the loss -0.23 [-0.45, -0.01]; F1(N) − F1(missing) upper end -0.0302 -> calibration helps here, not a cause (fragile; corrected: not shown, ruled out as the main cause)
- credit-g, its three variants pooled: loss +0.0322 (lower end +0.0198); F1(N) − F1(K) -0.0023 [-0.0103, +0.0056], as a share of the loss -0.07 [-0.32, +0.17]; F1(N) − F1(missing) upper end -0.0221 -> not shown, ruled out as the main cause
- credit-g_40nan: loss -0.0040 (lower end -0.0266); F1(N) − F1(K) -0.0100 [-0.0240, +0.0041], as a share of the loss +2.46 [+5.93, -1.00]; F1(N) − F1(missing) upper end +0.0139 -> no read (the loss F1(missing) − F1(K) is not above 0)
- credit-g_60nan: loss +0.0321 (lower end +0.0202); F1(N) − F1(K) +0.0057 [-0.0104, +0.0218], as a share of the loss +0.18 [-0.33, +0.68]; F1(N) − F1(missing) upper end -0.0047 -> inconclusive
- credit-g_80nan: loss +0.0685 (lower end +0.0477); F1(N) − F1(K) -0.0027 [-0.0207, +0.0152], as a share of the loss -0.04 [-0.30, +0.22]; F1(N) − F1(missing) upper end -0.0518 -> not shown, ruled out as the main cause (fragile; corrected: inconclusive)
- kr-vs-kp_40nan: loss +0.0182 (lower end +0.0082); F1(N) − F1(K) -0.0047 [-0.0092, -0.0001], as a share of the loss -0.26 [-0.51, -0.01]; F1(N) − F1(missing) upper end -0.0146 -> calibration helps here, not a cause (fragile; corrected: no read (the loss F1(missing) − F1(K) is not above 0))
- kr-vs-kp_60nan: loss +0.0430 (lower end +0.0314); F1(N) − F1(K) -0.0101 [-0.0207, +0.0005], as a share of the loss -0.23 [-0.48, +0.01]; F1(N) − F1(missing) upper end -0.0423 -> not shown, ruled out as the main cause
- kr-vs-kp_80nan: loss +0.0280 (lower end +0.0194); F1(N) − F1(K) -0.0058 [-0.0196, +0.0080], as a share of the loss -0.21 [-0.70, +0.29]; F1(N) − F1(missing) upper end -0.0219 -> not shown, ruled out as the main cause (fragile; corrected: inconclusive)

### What the calibration changes (E188, core, not fresh: E76's runs): gap cells of the test folds where the two outputs differ
- credit-g_40nan: categories changed 499 of 15600 (3.2%); numbers moved 6387 of 8400 (76.0%, mean shift 0.0637 z); n_thresholded_columns 4.53, mean_alpha 0.84, acc_cat K 0.6150, N 0.6147, rmse_num_z K 0.9589, N 0.9648, impute_score K 0.9183, N 0.9208
- credit-g_60nan: categories changed 801 of 23400 (3.4%); numbers moved 10067 of 12600 (79.9%, mean shift 0.0824 z); n_thresholded_columns 3.73, mean_alpha 0.70, acc_cat K 0.5957, N 0.5940, rmse_num_z K 0.9619, N 0.9777, impute_score K 0.9849, N 0.9932
- credit-g_80nan: categories changed 520 of 31200 (1.7%); numbers moved 11996 of 16800 (71.4%, mean shift 0.0530 z); n_thresholded_columns 1.60, mean_alpha 0.61, acc_cat K 0.5837, N 0.5821, rmse_num_z K 0.9603, N 0.9687, impute_score K 1.0048, N 1.0105
- kr-vs-kp_40nan: categories changed 3107 of 138024 (2.3%); numbers moved 0 of 0; n_thresholded_columns 13.93, acc_cat K 0.8644, N 0.8644, impute_score K 0.7254, N 0.7258
- kr-vs-kp_60nan: categories changed 4711 of 207144 (2.3%); numbers moved 0 of 0; n_thresholded_columns 11.93, acc_cat K 0.8380, N 0.8361, impute_score K 0.8609, N 0.8711
- kr-vs-kp_80nan: categories changed 3700 of 276156 (1.3%); numbers moved 0 of 0; n_thresholded_columns 7.40, acc_cat K 0.8179, N 0.8163, impute_score K 0.9733, N 0.9815
```
