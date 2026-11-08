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

## Outcome (2026-11-08 11:46 GMT-3)

**Supported.** All 12 pairs' fills were written by 11:45 GMT-3. On every fold of all 12 (60 folds), the reproduction
assert held: the refit missForest's score equals the run's logged score. Read once.

- **Primary, E90's seeds:** the `table` hybrid is better than H on both variants (vehicle −0.064, biodeg −0.072) and
  better than missForest on both (−0.021, −0.012), not merely not worse. H trails missForest on both (+0.044, +0.060),
  as on the confirmation set.
- **E89's seeds agree on every call:** hybrid − H −0.065 and −0.069, hybrid − missForest −0.015 and −0.012.
- **Beside it, the `column` rule:** better than H on all four; against missForest, better on vehicle and equivalent on
  biodeg. The `table` rule beats it on every reading, as in T39.
- **What it leads to (fixed in the design):** a proposal, the user's to decide, to extend ADR 0031's deployable hybrid
  (decision 6) to missForest as the other side where TRIDENT trails it at 80nan. Not written. **Open for that
  decision:** these two variants were picked because TRIDENT trails missForest there, measured on the complete table.
  A deployable version needs a rule that picks the other side (TabPFN or missForest) without the complete table, for
  example from the same validation cells. This study did not test such a rule.

### Printout (unedited)

```
## e89 (seeds 1212/2323/3434)
- `vehicle_80nan`: H 0.8178, missforest 0.7673, table 0.7527, column 0.7604
  - table − H -0.0651 [-0.0733, -0.0569] better
  - table − missforest -0.0146 [-0.0203, -0.0090] better
  - H − missforest +0.0505 [+0.0427, +0.0582] worse
  - column − H -0.0574 [-0.0652, -0.0497] better
  - column − missforest -0.0070 [-0.0125, -0.0015] better
- `biodeg_80nan`: H 0.9777, missforest 0.9207, table 0.9087, column 0.9184
  - table − H -0.0689 [-0.0768, -0.0610] better
  - table − missforest -0.0119 [-0.0193, -0.0045] better
  - H − missforest +0.0570 [+0.0484, +0.0657] worse
  - column − H -0.0593 [-0.0669, -0.0517] better
  - column − missforest -0.0023 [-0.0067, +0.0022] equivalent

## e90 (seeds 4545/5656/6767)
- `vehicle_80nan`: H 0.8119, missforest 0.7683, table 0.7476, column 0.7605
  - table − H -0.0643 [-0.0720, -0.0566] better
  - table − missforest -0.0207 [-0.0262, -0.0152] better
  - H − missforest +0.0436 [+0.0346, +0.0526] worse
  - column − H -0.0515 [-0.0581, -0.0449] better
  - column − missforest -0.0079 [-0.0114, -0.0043] better
- `biodeg_80nan`: H 0.9855, missforest 0.9255, table 0.9137, column 0.9237
  - table − H -0.0718 [-0.0779, -0.0656] better
  - table − missforest -0.0118 [-0.0162, -0.0073] better
  - H − missforest +0.0600 [+0.0529, +0.0670] worse
  - column − H -0.0618 [-0.0688, -0.0548] better
  - column − missforest -0.0018 [-0.0069, +0.0033] equivalent
```
