# K and S on fresh seeds: does E186's downstream reading repeat? (E187, T64; pre-registered 2026-11-22 17:27 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E186 (ticket 75) read a downstream classifier on fills whose seeds (101/202/303) had already been used to
  read `sw_score` (E183 to E185). Its primary gain was small and fragile, and its split by table (filling helps a
  forest on the numerical tables, hurts on `kr-vs-kp` and `credit-g`) was found after the read. Ticket 75 states that
  a claim needs fresh seeds. This study trains them.
- **The user's go** (2026-11-22): "sim, repita k e s em semenstes novas", answering my offer after E186. The reply
  names K and S, so the two arms (ticket 75's reading of the user's "H") are settled for this study. The design
  below is mine.
- **Disclosed:**
  - E186's printout was read before this design; every prediction below comes from it.
  - **Seeds 1717, 8484, 9595:** not among the store's `seed` tags and in no script, ticket or log entry (checked
    2026-11-22). A fresh seed changes the folds, the initialisation and the training masks; a variant's gap pattern
    is its file on disk and stays the same.
  - **Three seeds, fixed now, no extension.** The corrected (Nadeau–Bengio) interval's width is dominated by the
    split-ratio term (0.2/0.7 against 1/15), so more seeds would barely narrow it: a real gain of E186's size is
    expected to stay "fragile" under it. Stated before the first cell.
  - No smoke: the trainer is `sliced_loss_run.py`'s, unchanged. The first cells (`kc2`, minutes each) check the
    mechanics; a failing check is reported, not dropped.
  - Nothing in `src/` changes. `sliced_loss_run.py` gains one type annotation (`PAIRS`), no behaviour.
  - **The reads' script is not in this commit.** It will run `sliced_utility.py`'s functions (E186's) and E182's
    scorer with this study's state, seeds and arms, in a file the queues never import, and is committed with a dated
    note here before any of this study's ledgers is scored.
- **Runner (this commit):** [scripts/experiments/sliced_fresh_run.py](../../../scripts/experiments/sliced_fresh_run.py);
  state `/scratch/diogoneiss/trident-2026-11-22-sliced-fresh`, tag `experiment = sliced-fresh-2026-11-22`.

## Design and rule (fixed before the first cell)

- **Scenarios:** the 26 MCAR variants of E184 and E185 × seeds 1717/8484/9595 × five folds, arms **K** (the
  imputation task's default run: checkpoint by the validation gaps' `impute_score`, ADR 0015) and **S** (the same
  run with `--decode_metric sw_score`, E183). 156 runs, all on gorgona8's RTX 3090 Ti (a pair on one machine, ADR
  0035), four queues (`c15_queue.sh`, two threads each), a pair's two arms side by side, shortest variants first.
  `pendigits` and `letter` use their promoted files (ADR 0018 / 0036 / 0037), as in E183 to E185. Both arms carry the
  default output (calibrated from 40nan, ADR 0027).
  - Cost: about 28 h from E184's and E185's clocks (half of 19.6 h and of 37.4 h), so about 22:00 GMT-3 on
    2026-11-23.
- **Checks before the reads:** `--gate` (each run's `DECODE_METRIC`, no loss term) and `--pairing` (a pair's scored
  cells and hgb's score agree).
- **Downstream read, as E186:** per (variant, seed) the tables `missing`, `K`, `S`, beside `naive` (the fold's
  mean/mode fill) and `truth`; each row's gaps filled by the fold in which it is a test row; macro-F1 on the test
  folds; a random forest of 500 trees (primary, takes the gaps as they are) and logistic regression (beside; a gap
  is the mean with an indicator, or a level of its own); protocol `own` (fitted on the table's own rows outside the
  fold; primary) and `complete` (fitted on the truth-filled rows; beside). A pooled difference is over the 26
  variants per (seed, fold) replicate, 15 replicates, plain 95% t interval; the Nadeau–Bengio interval beside, and
  a verdict it does not share is reported as fragile. On the fresh seeds alone:
  - **Primary: F1(S) − F1(K)**, forest, `own`. E186: +0.0035 [+0.0009, +0.0061]. **Replicated** if the interval is
    above 0; **not replicated** if it includes 0; **reversed** if below 0.
  - **Three-way: F1(K) − F1(missing) and F1(S) − F1(missing)**, same rule. E186: +0.0033 and +0.0068.
  - **The by-table split, K − missing per variant** (15 fold values, plain interval), its first test. E186's calls:
    - K better on 9: `pendigits_20nan`, `pendigits_40nan`, `pendigits_60nan`, `letter_20nan`, `vehicle_20nan`,
      `vehicle_40nan`, `electricity_20nan`, `electricity_40nan`, `biodeg_40nan`;
    - missing better on 5: `kr-vs-kp_40nan`, `kr-vs-kp_60nan`, `kr-vs-kp_80nan`, `credit-g_60nan`, `credit-g_80nan`.
    - **Replicated** if at least 10 of these 14 repeat their call and none takes the opposite call; **not
      replicated** if fewer than 7 repeat or two or more take the opposite call; **partly** otherwise.
  - Beside, no rule: whether S − K's two variants (`credit-g_60nan`, `pendigits_80nan`) repeat; S − missing per
    variant (E186: S on 12, missing on 4); `naive` and `truth` against `missing`; the levels; by missingness level;
    logistic regression; `complete` (E186: S − K +0.0085 forest, +0.0060 logistic, both holding corrected).
  - Beside, labelled not fresh: the primary line pooled over the six seeds (E186's three and these), 30 replicates.
- **Manipulation check, secondary (E183 on fresh seeds):** each run's test-fold ledgers scored by E182's code
  (`sw_score`, `dist_score`, `impute_score`; the `reference` group with the run's ledger in TRIDENT's place, as
  E184). S − K per variant by the fold-pair rule with seed agreement, margin 0.01 of K's mean. E183's rule over the
  variants whose shuffled control reaches 0.10: supported if S is better on `sw_score` on half or more and worse on
  at most 2; refuted if better on a sixth or fewer; mixed otherwise. E183: better on 15 of 26, worse on none;
  `impute_score` worse on 6. Without it a downstream null could not be told from a selector that did nothing.
  The strong-tier comparisons are not repeated (no baseline files for these seeds).
- **Read once:** when all 156 runs are in, after the checks; the reads refuse a partial set. No interim read. No
  default and no ADR changes by this study; a replicated gain is input for the user's decision.
- **Not in this study:** the loss term (L, LS); ADR 0024's H; MAR and MNAR; why filling hurts on `kr-vs-kp` and
  `credit-g`; tuned classifiers.

## Amendment (2026-11-22 17:47 GMT-3): the reads' script, before any of this study's ledgers is scored

- **Script:** [scripts/experiments/sliced_fresh_read.py](../../../scripts/experiments/sliced_fresh_read.py), which the
  queues never import: `utility` (E186's downstream cell), `score` (E182's `reference` scorer on a run's ledgers),
  `--jobs`, and `--tally`, which refuses a partial set and prints E186's tally, the count against E186's 14 calls
  with the rule above, the two S − K carriers, the six-seed pooled line and the manipulation check.
- **`sliced_utility.py` (E186's) is parameterised, behaviour unchanged:** the header takes the study, variants and
  seeds from its globals and the loss-term pairs have a name. E186's `--tally` was rerun after the edit and its 160
  lines are identical to ticket 75's printout (`diff`, no output).
- **Self-check on seeds already read (101/202/303), not a read of this study:** with `FRESH_SELFCHECK` the same code
  recomputed the 78 downstream files from E184's and E185's K and S runs and reproduced E186: primary +0.0035
  [+0.0009, +0.0061], K − missing calls K 9 / missing 5, 14 of 14 repeat (trivially), the two carriers S. Its
  manipulation check, on those seeds with E184/E185's own K runs as control: S better on `sw_score` on 14 of 26,
  worse on none (E183, with E181's control runs: 15 and 0); `impute_score` worse on 5.
- **Mechanics so far:** the first 8 runs (`kc2_20nan`, `kc2_40nan` starting) passed `--gate` (8 checked, 0 failed)
  and `--pairing` (cells and hgb the same). No score of this study has been computed or read.

## Outcome

None yet.
