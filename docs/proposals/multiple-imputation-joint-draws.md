# Proposal: multiple imputation with draws joint across a row's gaps (would be E179, T56; drafted 2026-11-13 20:24 GMT-3)

A design for the user's decision. Nothing here is pre-registered and no cell has run; the ticket, the log entry and
its Index row are written only after the user's go.

## Why

E175 ([ticket 68](../tickets/imputation-protocol/68-multiple-imputation.md)) found that residual draws around H's fill
(`MI-D`) cover better than the single fill on 13 of 15 variants but are not proper: at 80% missing the slopes between
columns are covered 0.34 to 0.60 of the time. It handed the next study two requirements:

- **draws joint across a row's gaps.** `MI-D` adds independent noise per cell, which adds variance to each column
  and no covariance between two filled columns, so a slope is pulled further from the truth, not covered;
- **an interval whose control covers on every table.** With the truth in every gap the interval covered 0.574 on
  `kc2` and 0.88 on `biodeg`: the complete-data t interval fails there before any gap is filled.

## Design proposed

- **Cells (45), retrained:** H on `biodeg`, `kc2`, `vehicle`, `credit-g`, `pendigits` at 20, 40 and 80% MCAR,
  seeds 4545/5656/6767 (fresh: E175 read `MI-D` at 101/202/303). The models are not saved, so every arm that needs
  the model reads it in place after the decode stage, as `deployable_hybrid.train` does for the validation cells.
  No `src/` change: the script reads the categorical heads' logits as `predict` does and switches the `Dropout`
  modules on for the stochastic passes (the encoder has no batch norm).
- **Arms**, M = 20, each completing the fold's test rows, all from one run's model:
  - `MI-D`: E175's per-cell residual draws, recomputed on these runs (so this study also replicates E175's arm on
    fresh seeds);
  - `MI-S` (descriptive): per-cell draws from the model: a category sampled from the softmax, a number as the
    fill of a dropout pass plus a validation residual. Model draws without jointness;
  - `MI-R`: `MI-D` with the numerical residuals of a row drawn together, with the correlation the residuals have
    between columns among validation cells hidden in the same pass (available in every deployment);
  - `MI-J`: `MI-S` chained: a row's gaps are filled one at a time in random order, each draw written into the row
    before the next pass, so later draws condition on earlier ones;
  - `MICE`: E175's comparator, unchanged.
- **Target and interval (the change from E175):** the target is the fold's own complete-data estimate (the same test
  rows with the truth in every gap), and the interval is the missing-data part of Rubin's variance alone,
  Q̄ ± t(M−1) · sqrt((1 + 1/M) B). This is Rubin's condition for proper imputation read directly, and it needs no
  control: the complete-data arm is exact by construction, on every table. E175's reading (the whole table's value
  against the total variance) stays as a secondary on `vehicle`, `credit-g` and `pendigits`, where its control held.
- **Estimands:** E175's three families (means, shares, slopes), unchanged.
- **Rule, read for `MI-R` and for `MI-J` separately, each against `MI-D`** on the shortfall u = max(0, 0.95 − coverage),
  fold-pair verdict, SESOI 0.01:
  - supported if better on at least 4 of the five 80% variants and worse on none of the 15;
  - refuted if better on at most 1 of the five, or worse on 3 or more of the 15;
  - mixed otherwise;
  - proper enough: mean coverage at least 0.90 on at least 12 of the 15 (yes or no).
- **Machine and cost:** all 45 cells on gorgona8 after E178 leaves its GPU (medusa1 cannot start CUDA until it is
  rebooted); about 15 to 20 GPU-hours, by ticket 68's estimate for these cells plus the chained passes (at most
  33 gaps a row × 20 draws on `biodeg`, a few minutes a fold).

## Risks said now

- **The chain leaves the training regime.** At 80% missing the decoder was trained on rows with about a fifth of
  their cells present; by the end of a chain it reads a row that is nearly complete, most of it its own draws.
  `MI-J` may fail for that reason alone, which is why `MI-R` (no such drift) is read by the same rule.
- **The residual correlation is measured with more of the row visible** than a test row has (a validation pass
  hides a fifth of the observed cells), so `MI-R`'s covariance may be too small at 80%.
- `SI` has no between-draw variance, so it has no place under the new interval; the single fill is compared only in
  the secondary that keeps E175's reading.

## The user's calls

1. The target: the fold's complete-data estimate with the missing-data variance (proposed), or E175's whole-table
   value with the total variance (then `kc2` and `biodeg` are out, their control fails).
2. Seeds 4545/5656/6767 (proposed) or E175's 101/202/303.
3. The thresholds above (4 of 5, none worse, 12 of 15).
4. Script-side reading of the model, no `src/` change (proposed), or a `predict` that returns draws (needs the
   user's authorization for `src/`, a flag and a tag).
