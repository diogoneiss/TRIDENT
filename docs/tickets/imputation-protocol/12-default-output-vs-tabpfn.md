# The default output against TabPFN: KC and H (E57; pre-registered 2026-10-21)

This ticket was written and committed before the reading was run.

- **Why:** E55 compared TabPFN against K, the uncalibrated checkpoint. Since ADR 0027, the default
  output at a missing rate of 0.3 or more is KC: K after the calibration fitted on the validation
  gaps. Every confirmation variant is at 40nan or above.
  - Calibration helped most at 60 and 80nan (E40), exactly where TabPFN beat K (E55).
  - H, the loss checkpoint, is the deployable readout for claims that must transfer (ADR 0024).
- **The user's go:** "Faça 1 e 2 em sequência agora" (2026-10-21, about 09:10 GMT-3).
- **No run.** The reading uses E44's runs, which logged KC (`--score_calibrated`) and H
  (`--score_checkpoints`), and E55's TabPFN CSVs. Code: a new section in
  `scripts/experiments/tabpfn_report.py`; E55's own printout is unchanged.

## Reading, fixed now

- **The measures:** KC − TabPFN and H − TabPFN, beside K − TabPFN.
  - On the induced `impute_score`, 15 fold pairs, the verdict rule and the 1% SESOI.
  - On E44's 15 confirmation variants, with the tally and every call that changes from K to KC.
  - The development variants at 40nan and above are read where their runs carry KC (descriptive).
- **What it states:** KC − TabPFN is the thesis's statement against TabPFN at the default output
  (ADR 0027). H − TabPFN is the deployable one (ADR 0024).
- **No decision follows.** ADR 0028 stands; this sets which readout the thesis's sentence uses.
