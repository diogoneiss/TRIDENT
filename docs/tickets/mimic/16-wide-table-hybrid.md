# The deployable hybrid on `mimic-wide`: safe and still a gain where TabPFN alone is weak? (E178, T57; pre-registered 2026-11-13 08:28 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 decision 6 (the opt-in offline hybrid, `table_tabpfn`: H's fills and TabPFN's blended with one
  numerical weight and one categorical pick per table, chosen on the fold's hidden validation cells) covers the two
  11,082-stay MIMIC-IV tables, under transplanted patterns (E163, E166) and natural gaps (E173). On `mimic-wide`, six times larger, E176 ([mimic/14](14-wide-table.md)) found
  TabPFN v2 alone behind the decoder and, on the real patterns, behind the bar. This study asks whether the hybrid
  is still never worse than H there, and whether it still gains anything.
- **The user's go:** "2 e 1, vá" (2026-11-13); option 1 was this hybrid.
- **Disclosure:**
  - E176's scores on these variants and seeds were read (decoder − TabPFN −0.0991 natural, −0.0577 `tmatch`,
    −0.0044 `mcar`), with readouts M and K. H, the deployable readout this study blends, was not run on this table.
  - With TabPFN that far behind, the hybrid may choose H's fill nearly everywhere and read equivalent to H. E163's
    rule calls that refuted (no gain). Said now: a reading refuted by "better than H on neither, worse on neither"
    means the hybrid is harmless and unneeded on this table, and decision 6's scope does not extend to it.
  - No cell of this study has run.

## Design

- **Cells:** H (`deployable_hybrid.train`: `--decode_checkpoint loss --decode_calibration none`, the hidden observed
  validation cells read in place) on `mimic-wide_natural`, `mimic-wide-donor_tmatch_20nan` and
  `mimic-wide-donor_mcar_20nan`; seeds 42, 7, 13, E176's; 9 runs, five folds each. Study `e178` of
  `scripts/experiments/mimic_hybrid.py`, tag `mimic-wide-hybrid-2026-11-13`, state
  `/scratch/diogoneiss/trident-2026-11-13-mimic-wide-hybrid`.
- **Offline pass:** `mimic_hybrid.py --study e178 cells <variant> <seed>` per run: TabPFN v2 refit on the fold (H's
  ledger asserted equal to the run's H), then missForest refit (asserted equal to the run's logged missForest). The
  natural table is scored on its masked population, the donor variants on their induced cells.
- **Machine (ADR 0035, the split of studies between servers):** every run and the offline pass on medusa1; two
  queues, one a GPU.
- **Cost:** E176's runs took about 80 minutes on the donor table and 105 on the natural one; H's are the same
  shape, less where the baseline cache already holds E176's folds: at most 14 GPU-hours, about 7 h in two queues.
  The offline pass is about 8 minutes of TabPFN a run plus missForest's refit, not timed.

## Measures and rule (fixed now)

`impute_score` per fold on each variant's scored cells, lower is better; fold-pair rule (95% t interval, every seed
agreeing), equivalence margin 1% of the opponent's mean (`missforest_hybrid.SESOI`). E163's rule, unchanged, on the
primaries `mimic-wide_natural` and `mimic-wide-donor_tmatch_20nan`:

- **Refuted** if `table_tabpfn` is better than H on neither primary, or worse than H on either.
- **Supported** if, otherwise, it is worse than TabPFN on neither.
- **Mixed** otherwise.
- **Secondary:** the same lines on `mcar`; the missForest side (`table_missforest`), the pick and the cross-fit pick
  as `--tally` prints them; how often each side is chosen.
- Three seeds, fixed; no extension.
- **Read** once with the 9 runs and their cells in, on medusa1:
  `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py --study e178 --tally`
- **What it leads to:** whether ADR 0031 decision 6's scope reaches a clinical table of this size, the user's call.
