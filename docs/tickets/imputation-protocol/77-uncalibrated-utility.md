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
