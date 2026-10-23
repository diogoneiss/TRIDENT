# Does E60's complementarity hold on other tables and seeds? The blend on E40's set (E66; pre-registered 2026-10-23)

This ticket was written and committed before the first cell.

- **Why:** E60 found TRIDENT's and TabPFN's errors complementary ([15](15-tabpfn-blend.md)). An
  equal-weight blend of their fills beat both on 10 of 15 confirmation variants, but on one seed
  set (707/808/909) and mostly numerical tables.
  - This asks whether the effect holds on other tables and seeds before the thesis leans on it.
- **The user's go:** "vamos 1" (2026-10-23, about 07:00 GMT-3), to "replicate E60 on another seed
  set".
- **Offline:** `scripts/experiments/tabpfn_blend.py --set e40`. The script is E60's, with a
  seed-set option. E60's own reading reproduces unchanged (10 of 15). Nothing is written to the
  store or to `src/`.

## Design (fixed before the first cell)

- **Cells:** E40's runs (calibration-2026-10-16) at seeds 404/505/606, 12 variants, 36 pairs.
  - kr-vs-kp, spambase and credit-g at 40, 60 and 80nan;
  - vehicle, biodeg and kc2 at 60nan.
  - Their fold ledgers are on disk, and E61 found them equal to K on kr-vs-kp and credit-g.
- **The fills, as in E60:**
  - **the model's:** K, from the runs' fold ledgers;
  - **TabPFN's:** v2, refitted on the same training folds and asked for the ledger's induced
    cells;
  - **the blend:** the average on every numerical cell, and the model's category on a categorical
    one.
- **Integrity:** the model's fold `impute_score` must equal the run's K, or the pair stops.
- **Running:** two GPU lanes, about 1 h; state in
  `/scratch2/diogoneiss/trident-2026-10-23-tabpfn-blend-e40/`.

## Reading, fixed now (`tabpfn_blend.py --tally --set e40`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Per variant:** blend − K and blend − TabPFN.
- **Counted variants:** those with numerical cells, so 9 of the 12. kr-vs-kp is all-categorical,
  so its blend is the model's fill by construction. Its three variants are printed but not
  counted.
- **Over the counted 9:**
  - **complementary:** the blend beats both on 3 or more, E60's third;
  - **not complementary:** on 1 or none;
  - **mixed:** otherwise.
- **What the zero "worse" calls mean:** as in E60, on an all-numerical variant the blend cannot be
  worse than the worse of the two (Minkowski). Only "better than the better one" counts.
- **What each reading leads to:** descriptive; no decision or default follows.
  - **complementary:** the thesis's hybrid argument rests on two seed sets and two table sets.
  - **not complementary:** E60's result is stated as specific to the large numerical tables of
    the confirmation set.
