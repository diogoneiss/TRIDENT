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
