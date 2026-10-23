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

## Outcome (2026-10-23, 07:58 GMT-3)

**The run:** 36 of 36 pairs, 07:03–07:57 GMT-3, with none failed. On every fold, the model's
`impute_score` from the ledger equals the run's K.

**The printout** (`scripts/experiments/tabpfn_blend.py --tally --set e40`, unedited):

```text
- `kr-vs-kp_40nan` (numeric share 0.00) (no numerical cell: the blend is the model's fill, not counted): blend − model +0.0000 [+0.0000, +0.0000] equivalent; blend − tabpfn +0.0110 [+0.0054, +0.0167] worse
- `kr-vs-kp_60nan` (numeric share 0.00) (no numerical cell: the blend is the model's fill, not counted): blend − model +0.0000 [+0.0000, +0.0000] equivalent; blend − tabpfn +0.0434 [+0.0366, +0.0501] worse
- `kr-vs-kp_80nan` (numeric share 0.00) (no numerical cell: the blend is the model's fill, not counted): blend − model +0.0000 [+0.0000, +0.0000] equivalent; blend − tabpfn +0.0365 [+0.0271, +0.0458] worse
- `spambase_40nan` (numeric share 1.00): blend − model +0.0033 [-0.0003, +0.0069] equivalent; blend − tabpfn -0.0506 [-0.0574, -0.0437] better
- `spambase_60nan` (numeric share 1.00): blend − model +0.0022 [-0.0003, +0.0047] equivalent; blend − tabpfn -0.0325 [-0.0393, -0.0256] better
- `spambase_80nan` (numeric share 1.00): blend − model -0.0073 [-0.0106, -0.0040] better; blend − tabpfn -0.0128 [-0.0150, -0.0105] better
- `credit-g_40nan` (numeric share 0.35): blend − model -0.0080 [-0.0101, -0.0059] better; blend − tabpfn +0.0139 [+0.0063, +0.0215] worse
- `credit-g_60nan` (numeric share 0.35): blend − model -0.0089 [-0.0105, -0.0072] better; blend − tabpfn +0.0302 [+0.0200, +0.0405] worse
- `credit-g_80nan` (numeric share 0.35): blend − model -0.0043 [-0.0056, -0.0031] better; blend − tabpfn +0.0095 [+0.0041, +0.0150] worse
- `vehicle_60nan` (numeric share 1.00): blend − model -0.0180 [-0.0287, -0.0072] better; blend − tabpfn -0.0161 [-0.0296, -0.0027] better
- `biodeg_60nan` (numeric share 1.00): blend − model -0.0221 [-0.0294, -0.0148] better; blend − tabpfn -0.0356 [-0.0415, -0.0298] better
- `kc2_60nan` (numeric share 1.00): blend − model -0.0012 [-0.0322, +0.0298] none; blend − tabpfn -0.0752 [-0.1071, -0.0432] better
  - blend − model: better 6 / no difference 1 / equivalent 5 / worse 0 (fold-pair t, seed agreement)
  - blend − tabpfn: better 6 / no difference 0 / worse 6 (fold-pair t, seed agreement)

Primary: the blend beats both on 3 of 9 counted (['spambase_80nan', 'vehicle_60nan', 'biodeg_60nan']) -> complementary (fold-pair t, seed agreement)
```

**Verdict: complementary, at the floor.** The blend beats both on 3 of the 9 counted variants
(spambase_80nan, vehicle_60nan, biodeg_60nan); the rule needed 3. E60 had 10 of 15.

**Reading:**

- **The blend beats both where the two are close on the numerical cells.**
  - vehicle_60nan and biodeg_60nan, the same tables where E60 found it at 40nan.
  - spambase_80nan, where TabPFN closes in on the model.
- **Where one method dominates, the blend lands at or near the better one:**
  - **spambase at 40 and 60nan:** K leads TabPFN by 3–5 points, and the blend is equivalent to K.
    It neither helps nor hurts.
  - **kc2_60nan:** the blend is better than TabPFN and not distinguishable from K (a wide interval).
- **On mixed tables the blend cannot reach TabPFN.**
  - On credit-g (35% numerical cells), the blend is better than K at every rate but worse than
    TabPFN. It touches only the numerical cells, and TabPFN's lead there sits in the categorical
    ones (E61).
  - On a mixed table Minkowski's bound covers only the numerical part, so these "worse" calls are
    real.
- **kr-vs-kp** (all-categorical, not counted): the blend is the model's fill, worse than TabPFN, as
  E58 and E61 found.
- **With E60:** the complementarity is real but narrower than E60's 10 of 15 suggested. It shows
  on the numerical tables where TRIDENT and TabPFN are of similar accuracy. Where one is clearly
  ahead, an equal-weight average lands near the better one at best. On categorical cells, this
  blend has nothing to average.

**What it leads to (fixed in the ticket):**

- The thesis's hybrid argument rests on two seed sets and two table sets, stated with its scope:
  numerical cells, and tables where the two are close.
- **Not pre-registered, a natural next step for the user:** a blend weighted on the validation
  gaps would let a dominant method keep its lead. Blending categorical cells needs both methods'
  class probabilities, which the ledgers do not hold.
- Descriptive; no default changes.
