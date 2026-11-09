# The deployable hybrid on the MIMIC-IV transplant variants (E163, T55; pre-registered 2026-11-08 21:08 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** on E149's ([04](04-pattern-transplant.md)) `mimic-sofa-donor_tmatch_20nan` TabPFN's score was lower
  than the decoder's (+0.0265 [+0.0155, +0.0375]), the only clinical table where the strong tier led. T39's deployable
  hybrid (ADR 0031 decision 6: the `table` rule, weights chosen on fold k's hidden observed validation cells, no
  complete table) has never run on clinical data. On `mimic-sofa-donor_mcar_20nan` missForest led the decoder too
  (0.9151 against 0.9237), so the missForest side (E162) is read beside it.
- **The user's go:** "faça 1 2 e 3" (2026-11-08), on "1. Híbrido implantável na MIMIC".
- **Why not the natural-gap tables:** the hybrid is scored on induced cells, whose truth needs a complete sibling.
  Only the transplant variants, cut from the complete donor rows, have one. E148's natural tables do not.
- **Disclosure:** E149's decoder and TabPFN scores on these variants are on the record (above). E149's decoder is K,
  the induced checkpoint (ADR 0015). The hybrid's model side is H, the deployable readout (ADR 0024,
  `--decode_checkpoint loss --decode_calibration none`), as in T39. H on the SOFA `tmatch` may trail TabPFN by more
  than K did, and nothing here re-reads E149.

## Design

- **Cells:** E149's six variants, `mimic-{cardio,sofa}-donor_{tmatch,trand,mcar}_20nan`, seeds 42/7/13, five
  folds: 18 runs. Runner `scripts/experiments/mimic_hybrid.py` (`deployable_hybrid.train` under MLflow tag
  `experiment = mimic-hybrid-2026-11-08`), imputation defaults with H's flags.
- **Fills, per fold, offline after training (`mimic_hybrid.py cells`):**
  - H's induced test fills, from the run's ledger, asserted equal to the run's H score fold by fold;
  - H's fills of the validation split's observed cells, hidden in 5 random groups (E82's procedure);
  - TabPFN v2 (ADR 0028) and missForest (ADR 0016), refit on the training fold, on the same cells. missForest is
    asserted equal to the run's logged missForest score fold by fold.
- **Methods on the test induced cells:**
  - H, TabPFN and missForest alone;
  - `table_tabpfn`, ADR 0031 decision 6's rule with TabPFN as the other side;
  - `table_missforest`, the same rule with missForest (E162);
  - `pick`: per fold, whichever of the two hybrids scores lower on fold k's validation cells. These are the same
    cells that choose each hybrid's weights, one more in-sample choice; a tie keeps TabPFN. E165 tests this rule on
    the OpenML tables.
- **Machine (ADR 0035):** all 18 runs and every offline step on medusa1, beside the other user's job on its two GPUs.
  gorgona8's GPU is held by the user's other project. Four queues, two per GPU, through `c15_queue.sh`; state
  `/scratch/diogoneiss/trident-2026-11-08-mimic-hybrid`.
- **Smoke, off the record (scratchpad, no MLflow):** `deployable_hybrid.train` on
  `mimic-sofa-donor_tmatch_20nan` s42 with 3 decode epochs wrote all five selection files and five cell ledgers.
  Its numbers describe a 3-epoch model and are not read. The refactored missForest pass reproduced E162's
  `vehicle_80nan` s4545 cell files byte for byte.
- **Cost:** E149's cells took 19–33 minutes each beside other trainers. With four queues, training should take about
  2–3 h. The offline pass, TabPFN on the GPU and missForest on the CPU, is about 1–2 h more.

## Measures and rule (fixed now)

- **Verdict:** `common.verdict` on the 15 (seed, fold) differences of `impute_score`, plain 95% t interval, SESOI
  1% of the opponent's mean (E90's rule).
- **Primary,** on the two `tmatch` variants, the accepted rule's hybrid `table_tabpfn` against H and against TabPFN:
  - **supported** if it is better than H on at least one, worse than H on neither, and worse than TabPFN on neither;
  - **refuted** if it is better than H on neither, or worse than H on either;
  - **mixed** otherwise.
- **Descriptive:** the same pairs on `trand` and `mcar`; `table_missforest` − H and − missForest;
  `pick` − H, − `table_tabpfn` and − `table_missforest`, with how often the pick took missForest.
- **Read:** once, with all 18 cell tables in: `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python
  3.11 python -B scripts/experiments/mimic_hybrid.py --tally` on medusa1.
- **What it leads to:** a sentence for the user beside E149's. Any extension of ADR 0031 to clinical tables is the
  user's call.
- **Threats, as E149:** the donors are sicker than the cohort, and a transplant carries where the gaps fall, not
  why. `tmatch` is at most missing at random given the matching columns. One seed set.

## Outcome (read once 2026-11-09 00:09 GMT-3): supported

All 18 runs trained on medusa1 from 21:09 to 21:44 GMT-3; the offline pass finished 22:09 GMT-3 with no failure, and
both asserts (H's ledger, missForest's logged score) held on every fold. Read on medusa1 at commit 2601430. The
printout, unedited:

```
- `mimic-cardio-donor_tmatch_20nan` (primary): H 0.8566, tabpfn 0.9206, missforest 0.8964, table_tabpfn 0.8532, table_missforest 0.8529, pick 0.8534, picked_missforest 0.2000
  - table_missforest − H -0.0037 [-0.0077, +0.0004] equivalent
  - table_missforest − missforest -0.0435 [-0.0526, -0.0343] better
  - table_tabpfn − H -0.0034 [-0.0081, +0.0012] equivalent
  - table_tabpfn − tabpfn -0.0674 [-0.0769, -0.0580] better
  - pick − H -0.0033 [-0.0079, +0.0014] equivalent
  - pick − table_tabpfn +0.0002 [-0.0013, +0.0016] equivalent
  - pick − table_missforest +0.0004 [-0.0070, +0.0078] equivalent
  - the pick took missForest on 3 of 15 folds
- `mimic-cardio-donor_trand_20nan`: H 0.8624, tabpfn 0.8999, missforest 0.9089, table_tabpfn 0.8468, table_missforest 0.8561, pick 0.8516, picked_missforest 0.2667
  - table_missforest − H -0.0062 [-0.0115, -0.0010] better
  - table_missforest − missforest -0.0527 [-0.0620, -0.0434] better
  - table_tabpfn − H -0.0156 [-0.0211, -0.0101] better
  - table_tabpfn − tabpfn -0.0532 [-0.0652, -0.0412] better
  - pick − H -0.0108 [-0.0170, -0.0046] better
  - pick − table_tabpfn +0.0048 [-0.0006, +0.0102] none
  - pick − table_missforest -0.0046 [-0.0091, +0.0000] none
  - the pick took missForest on 4 of 15 folds
- `mimic-cardio-donor_mcar_20nan`: H 0.8003, tabpfn 0.7731, missforest 0.8004, table_tabpfn 0.7750, table_missforest 0.7844, pick 0.7740, picked_missforest 0.2667
  - table_missforest − H -0.0159 [-0.0236, -0.0081] better
  - table_missforest − missforest -0.0160 [-0.0232, -0.0087] better
  - table_tabpfn − H -0.0253 [-0.0354, -0.0153] better
  - table_tabpfn − tabpfn +0.0019 [-0.0048, +0.0085] none
  - pick − H -0.0263 [-0.0358, -0.0168] better
  - pick − table_tabpfn -0.0009 [-0.0027, +0.0008] equivalent
  - pick − table_missforest -0.0104 [-0.0159, -0.0049] better
  - the pick took missForest on 4 of 15 folds
- `mimic-sofa-donor_tmatch_20nan` (primary): H 0.9156, tabpfn 0.8951, missforest 0.9822, table_tabpfn 0.9005, table_missforest 0.9179, pick 0.9075, picked_missforest 0.4000
  - table_missforest − H +0.0023 [-0.0048, +0.0094] none
  - table_missforest − missforest -0.0643 [-0.0766, -0.0520] better
  - table_tabpfn − H -0.0151 [-0.0212, -0.0090] better
  - table_tabpfn − tabpfn +0.0054 [-0.0010, +0.0118] none
  - pick − H -0.0080 [-0.0168, +0.0007] none
  - pick − table_tabpfn +0.0071 [+0.0009, +0.0132] worse
  - pick − table_missforest -0.0104 [-0.0205, -0.0003] better
  - the pick took missForest on 6 of 15 folds
- `mimic-sofa-donor_trand_20nan`: H 0.9315, tabpfn 0.9001, missforest 0.9711, table_tabpfn 0.9113, table_missforest 0.9294, pick 0.9146, picked_missforest 0.2000
  - table_missforest − H -0.0020 [-0.0040, -0.0000] better
  - table_missforest − missforest -0.0417 [-0.0547, -0.0286] better
  - table_tabpfn − H -0.0202 [-0.0338, -0.0065] better
  - table_tabpfn − tabpfn +0.0111 [+0.0057, +0.0166] worse
  - pick − H -0.0169 [-0.0309, -0.0029] better
  - pick − table_tabpfn +0.0033 [-0.0008, +0.0073] equivalent
  - pick − table_missforest -0.0149 [-0.0288, -0.0010] better
  - the pick took missForest on 3 of 15 folds
- `mimic-sofa-donor_mcar_20nan`: H 0.9264, tabpfn 0.8925, missforest 0.9151, table_tabpfn 0.9025, table_missforest 0.9140, pick 0.9046, picked_missforest 0.2667
  - table_missforest − H -0.0123 [-0.0187, -0.0059] better
  - table_missforest − missforest -0.0011 [-0.0114, +0.0092] none
  - table_tabpfn − H -0.0239 [-0.0337, -0.0141] better
  - table_tabpfn − tabpfn +0.0100 [-0.0018, +0.0219] none
  - pick − H -0.0218 [-0.0322, -0.0114] better
  - pick − table_tabpfn +0.0021 [-0.0004, +0.0045] equivalent
  - pick − table_missforest -0.0095 [-0.0147, -0.0042] better
  - the pick took missForest on 4 of 15 folds

PRIMARY `table` (TabPFN side) on mimic-cardio-donor_tmatch_20nan, mimic-sofa-donor_tmatch_20nan: better than H on ['mimic-sofa-donor_tmatch_20nan'], worse than H on none, worse than TabPFN on none -> supported
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Primary, supported:** on the SOFA `tmatch` variant, where H trails TabPFN (0.9156 against 0.8951), the accepted
  rule's hybrid is better than H (−0.0151 [−0.0212, −0.0090]) and not worse than TabPFN (+0.0054 [−0.0010, +0.0118],
  no call). On the cardio `tmatch` it is equivalent to H (−0.0034) and far better than TabPFN (−0.0674).
- **Descriptive:**
  - `table_tabpfn` is better than H on the other four variants too, by 0.016 to 0.025. On `mimic-sofa-donor_trand_20nan`
    it is worse than TabPFN (+0.0111 [+0.0057, +0.0166]), outside the primary.
  - `table_missforest` is never worse than H or missForest; it is better than H on 4 of 6, but by less than
    `table_tabpfn` on every variant except the cardio `tmatch`, where both tie H.
  - `pick` is never worse than H, and equivalent to `table_tabpfn` or without a call on five; on the SOFA `tmatch`
    it is worse than `table_tabpfn` (+0.0071 [+0.0009, +0.0132]). It took missForest on 3 to 6 of 15 folds. On these
    tables the pick adds nothing over the TabPFN side.
- **What it leads to:** a sentence for the user beside E149's, and whether ADR 0031 decision 6 extends to clinical tables
  is the user's call. Threats as stated: sicker donors, one seed set, `tmatch` at most missing at random given the
  matching columns.

## Decision

- **2026-11-09 19:51 GMT-3:** ADR 0031 decision 6's scope extended to the MIMIC-IV transplant variants with E163 and E166, on
  the user's decision ("vamos no 2b e 3").
