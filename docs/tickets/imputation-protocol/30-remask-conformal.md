# Does the re-masking spread make intervals narrower at the same coverage? (E77; pre-registered 2026-10-25)

This ticket was written and committed before the study's reading was run on E75's data.

- **Why:** ADR 0030 (proposed) reads the spread as a rank, because its raw coverage at a nominal 90%
  was 17–66% (E75, [28](28-remask-ensemble.md)). A ranking that carries information should give,
  once scaled to a coverage target, intervals that are narrower where the model is sure and wider
  where it is not, so narrower on average than a constant width with the same coverage. This tests
  that directly, offline, on E75's ledgers. No training, nothing in `src/` or the store.
- **The user's go:** item 4 of the plan, "De acordo com sua sugestão" (2026-10-25, about 09:45 GMT-3).
- **Disclosure:** the reader's self-check ran on the flag-on smoke ledger of imputation-protocol/28's
  launch checks, which is the same deterministic cell as E75's vehicle_40nan at 707. On its folds 1
  and 2 the spread interval was wider than the constant one (mean width 1.88 and 1.74 against 1.24
  and 1.26, both near 90% coverage). The rule below was in the code before that check and is not
  changed by it. An oracle spread (the error itself) gave 0.60 against 1.22, so the reader can show
  a gain when there is one.

## Design (fixed now)

- **Cells:** E75's 15 variants × 707/808/909 × 5 folds, numerical gaps only.
- **Two 90% intervals per numerical column, cross-fitted:** for fold k, fitted on the same seed's
  other four folds' cells of that column:
  - **spread:** fill ± q × spread, with q the 90% quantile of |error| / spread there;
  - **constant:** fill ± w, with w the 90% quantile of |error| there.
- **Centres:**
  - **K, primary:** the run's headline fill, the ledger's `induced` rows: KC here, since every E75
    variant is calibrated. The spread comes from the passes at K, so the interval is around the fill
    ADR 0030 keeps as the output.
  - **H, descriptive:** the passes' own fill at H. The ledger holds no single-pass H fill.
- **Measure:** per fold, (mean spread width − mean constant width) / mean constant width. Over 15
  folds, the verdict rule with a 1% SESOI, so 0.01 of the constant width. Lower is better. Both
  coverages are printed.

## Reading, fixed now (`remask_ensemble.py --conformal`)

- **supported** (the spread can become an interval output): narrower on 10 or more of the 15 under
  K and wider on at most 1;
- **refuted:** narrower on 5 or fewer, or wider on 4 or more;
- **mixed:** otherwise.

**What each reading leads to:**
- **supported:** ADR 0030 is amended to offer a cross-fitted interval beside the rank, for the user's
  decision.
- **otherwise:** ADR 0030 stays a rank and says the spread does not beat a constant width.

## Amendment (2026-10-25, about 09:58 GMT-3, before any result was read)

- **What happened:** the first run of the reader warned that quantiles came out NaN.
  - A numerical cell whose 20 passes agree has spread 0, so |error| / spread is infinite there, and
    the 90% quantile can fall between two infinities.
  - The design did not say what to do with such cells.
- **What was looked at:** only those warnings and the share of zero-spread cells per variant. That
  share is under 2% at 40nan, 0.4–7% at 60nan, and 10–32% at 80nan (biodeg_80nan 0.5%). No width or
  coverage was read, and that run's output was discarded.
- **The fix:**
  - The spread is floored, per column, at the smallest positive spread on the fit folds, for both
    the fit and the scored cells.
  - A column with no positive spread on its fit folds gets a floor of 1, which makes its spread
    interval a constant one.
  - The rule and every other part of the design are unchanged.

## Outcome (2026-10-25, 09:59 GMT-3)

**The printout** (`scripts/experiments/remask_ensemble.py --conformal`, on the amended reader, unedited):

```text
- `pendigits_40nan`: K: width (spread − constant) / constant +0.316 [+0.308, +0.325] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +0.233 [+0.223, +0.243] worse; coverage spread 0.900, constant 0.900
- `pendigits_60nan`: K: width (spread − constant) / constant +0.468 [+0.455, +0.481] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +0.431 [+0.410, +0.451] worse; coverage spread 0.900, constant 0.900
- `pendigits_80nan`: K: width (spread − constant) / constant +2415.799 [+1225.109, +3606.488] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +1914.865 [+1566.428, +2263.301] worse; coverage spread 0.900, constant 0.900
- `letter_40nan`: K: width (spread − constant) / constant +0.443 [+0.435, +0.451] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +0.394 [+0.386, +0.401] worse; coverage spread 0.900, constant 0.900
- `letter_60nan`: K: width (spread − constant) / constant +0.729 [+0.718, +0.741] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +0.766 [+0.725, +0.808] worse; coverage spread 0.900, constant 0.900
- `letter_80nan`: K: width (spread − constant) / constant +1911.780 [+1675.251, +2148.310] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +3331.681 [+2312.246, +4351.115] worse; coverage spread 0.900, constant 0.900
- `electricity_40nan`: K: width (spread − constant) / constant +1.436 [+1.366, +1.507] worse; coverage spread 0.899, constant 0.900; H: width (spread − constant) / constant +1.361 [+1.282, +1.440] worse; coverage spread 0.900, constant 0.900
- `electricity_60nan`: K: width (spread − constant) / constant +11.038 [+6.955, +15.122] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +17.987 [+7.748, +28.226] worse; coverage spread 0.900, constant 0.900
- `electricity_80nan`: K: width (spread − constant) / constant +10794.694 [+9104.367, +12485.021] worse; coverage spread 0.900, constant 0.900; H: width (spread − constant) / constant +33750.872 [+20746.426, +46755.319] worse; coverage spread 0.900, constant 0.899
- `vehicle_40nan`: K: width (spread − constant) / constant +0.395 [+0.339, +0.451] worse; coverage spread 0.897, constant 0.895; H: width (spread − constant) / constant +0.408 [+0.335, +0.482] worse; coverage spread 0.895, constant 0.896
- `vehicle_80nan`: K: width (spread − constant) / constant +65.673 [+23.563, +107.783] worse; coverage spread 0.899, constant 0.898; H: width (spread − constant) / constant +33.882 [+20.901, +46.863] worse; coverage spread 0.898, constant 0.898
- `biodeg_40nan`: K: width (spread − constant) / constant +0.542 [+0.171, +0.913] worse; coverage spread 0.892, constant 0.891; H: width (spread − constant) / constant +0.078 [+0.025, +0.130] worse; coverage spread 0.898, constant 0.896
- `biodeg_80nan`: K: width (spread − constant) / constant +2.224 [+0.709, +3.739] worse; coverage spread 0.872, constant 0.893; H: width (spread − constant) / constant +16.354 [+4.562, +28.147] worse; coverage spread 0.844, constant 0.892
- `kc2_40nan`: K: width (spread − constant) / constant +0.092 [+0.036, +0.147] worse; coverage spread 0.893, constant 0.897; H: width (spread − constant) / constant +0.037 [-0.043, +0.116] none; coverage spread 0.892, constant 0.894
- `kc2_80nan`: K: width (spread − constant) / constant +3.748 [+3.050, +4.446] worse; coverage spread 0.897, constant 0.898; H: width (spread − constant) / constant +3.506 [+2.943, +4.069] worse; coverage spread 0.895, constant 0.896

- width, K: better 0 / no difference 0 / worse 15 (fold-pair t, seed agreement)
- width, H: better 0 / no difference 1 / worse 14 (fold-pair t, seed agreement)

Primary: the spread interval is narrower than the constant one at the same target on 0 of 15 (K), wider on ['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_40nan', 'electricity_60nan', 'electricity_80nan', 'vehicle_40nan', 'vehicle_80nan', 'biodeg_40nan', 'biodeg_80nan', 'kc2_40nan', 'kc2_80nan'] -> refuted (fold-pair t, seed agreement)
```

**Reading: refuted.** The spread interval is narrower on 0 of 15 and wider on all 15 under K (H:
wider on 14, no difference on kc2_40nan), at the same coverage (both 0.84–0.90, nearly all 0.89–0.90).

- **At 40nan, where zero spreads are 0–2%**, the floor plays no part, and the spread interval is still
  9–144% wider: kc2 +0.09, vehicle +0.40, pendigits +0.32, letter +0.44, biodeg +0.54,
  electricity +1.44.
- **At 60 and 80nan the ratios are dominated by the amendment's floor.** The zero-spread cells,
  scaled by the smallest positive spread, drive q up, so those magnitudes (up to ×10⁴) measure the
  floor, not the spread. The 40nan reading alone already decides the rule.
- **What it means:** the within-column ranking E75 found (ρ about 0.1–0.5) is too weak to adapt a
  width. A per-column constant interval, fitted the same way, is narrower at the same coverage.
- **What it leads to, as fixed above:** ADR 0030 stays a rank and says the spread does not beat a
  constant width.
