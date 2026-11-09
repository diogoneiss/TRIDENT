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
