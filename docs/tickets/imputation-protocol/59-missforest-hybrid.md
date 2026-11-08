# The deployable hybrid with missForest where TRIDENT trails it (E162, T54; pre-registered 2026-11-08 11:01 GMT-3)

This ticket was written and committed before the study's first hybrid score.

- **Why:** on the confirmation set, `vehicle_80nan` and `biodeg_80nan` are the only variants where TRIDENT (K) is
  worse than the best of every imputer: missForest, by +0.043 and +0.049 (E159's 12-variant printout). T39's
  deployable hybrid (ADR 0031 decision 6, the `table` rule) closed TRIDENT's gap to TabPFN without the complete table.
  Here the same rule is applied with missForest (ADR 0016, external) as the other side. The result either becomes a
  practical rule or stays a stated limit.
- **The user's go:** "Execute d e c" (2026-11-08), on "C: o híbrido com o missForest nos 2 casos a 80%".
- **Disclosure:** the cells are E89's (seeds 1212/2323/3434) and E90's (4545/5656/6767). They hold H's fills of the test
  split's induced cells and of the hidden observed validation cells; H's and missForest's per-fold scores are on the
  record, the hybrid with missForest has never been scored. The cost first proposed (a GPU confirmation of 2–3 h) was
  wrong: both seed sets' cell tables already exist, so the study is CPU only.

## Design

- **Fills:** `missforest_hybrid.py cells` refits ADR 0016's missForest on each training fold of the 12
  (variant, seed) pairs and fills the same test and validation cells, the validation cells group by group as TabPFN's
  were. A fold whose missForest score on the test cells differs from the run's logged `missforest` score stops the
  study (the assert). State `/scratch2/diogoneiss/trident-2026-11-08-missforest-hybrid`, nothing in the store.
- **Arms, on the test induced cells:** H (the model side, ADR 0024's deployable readout, as T39); missForest; the
  `table` rule's hybrid, with its weights chosen on fold k's validation cells only.
- **Read:** once, with all 12 pairs in: `missforest_hybrid.py --tally`.

## Measures and rule (fixed now)

- Per variant and seed set, the fold-pair verdict with the 1% SESOI (`common.verdict`): hybrid − H, hybrid −
  missForest, H − missForest. The `column` rule is printed beside the primary and is not part of it.
- **Primary, on E90's seeds (the confirmation cells):** **supported** if, on both variants, the hybrid is better than H
  and not worse than missForest. **Refuted** if it is better than H on neither. **Mixed** otherwise. E89's seeds are read
  beside with the same calls; a call that differs between the two seed sets is stated.
- **What it leads to:** supported → a proposal, the user's to decide, to extend ADR 0031's deployable hybrid to
  missForest where TRIDENT trails it at 80nan. Otherwise the 80nan limit on these two tables is stated in the thesis.
  No default change from this study.
- **Limits, stated now:** two variants, both small tables at 80nan; H, not K, is the model side, as in T39.
