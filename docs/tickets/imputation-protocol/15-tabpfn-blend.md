# Are TRIDENT's and TabPFN's errors complementary? A blend of their fills (E60; pre-registered 2026-10-21)

This ticket was written and committed before the first cell.

- **Why:** on the confirmation set, TRIDENT leads TabPFN at 40–60nan and trails it at 80nan.
  - TabPFN's 80nan fills are more shrunk (E56).
  - If the two err on different cells, an average of their fills beats both. That would be a
    "hybrid" argument for the thesis's discussion.
  - If they err on the same cells, it does not.
- **The user's go:** "sim, vamos" (2026-10-21, about 19:20 GMT-3).
- **Offline:** `scripts/experiments/tabpfn_blend.py`. Nothing is written to the store or to `src/`.

## Design (fixed before the first cell)

- **Cells:** E44's 15 confirmation variants × 707/808/909, 45 pairs, one GPU lane beside E59.
- **The fills:**
  - **the model's:** K, from the runs' fold ledgers;
  - **TabPFN's:** v2, refitted on the same training folds and asked for the ledger's induced
    cells;
  - **the blend:** the average of the two on every numerical cell, and the model's category on a
    categorical one, since the ledgers hold no class probabilities.
- **Integrity:** the model's `impute_score` must equal the run's K fold by fold.

## Reading, fixed now

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI
(`tabpfn_blend.py --tally`):

- **Per variant:** blend − K and blend − TabPFN.
- **Over the 15:**
  - **complementary:** the blend beats both on 5 or more;
  - **not complementary:** on 1 or none;
  - **mixed:** otherwise.
- **Descriptive:** it feeds the thesis's discussion, and no decision or default follows.
- **Limit:** the blend touches numerical cells only. Electricity's `day` keeps the model's
  category.

## Outcome (2026-10-22, 00:05 GMT-3)

**The run:** 45 of 45 pairs, 19:28–22:57 GMT-3 on 2026-10-21, one lane beside E59; none failed.
On every fold of every pair, the model's `impute_score` from the ledger equals the run's K.

**The printout** (`scripts/experiments/tabpfn_blend.py --tally`, unedited):

```text
- `pendigits_40nan` (numeric share 1.00): blend − model -0.0138 [-0.0151, -0.0124] better; blend − tabpfn -0.0245 [-0.0261, -0.0228] better
- `pendigits_60nan` (numeric share 1.00): blend − model -0.0122 [-0.0132, -0.0112] better; blend − tabpfn -0.0302 [-0.0316, -0.0289] better
- `pendigits_80nan` (numeric share 1.00): blend − model -0.0207 [-0.0218, -0.0196] better; blend − tabpfn -0.0045 [-0.0055, -0.0034] better
- `letter_40nan` (numeric share 1.00): blend − model -0.0022 [-0.0035, -0.0010] better; blend − tabpfn -0.0366 [-0.0380, -0.0353] better
- `letter_60nan` (numeric share 1.00): blend − model -0.0055 [-0.0063, -0.0046] better; blend − tabpfn -0.0355 [-0.0366, -0.0345] better
- `letter_80nan` (numeric share 1.00): blend − model -0.0103 [-0.0111, -0.0095] better; blend − tabpfn -0.0083 [-0.0092, -0.0074] better
- `electricity_40nan` (numeric share 0.88): blend − model -0.0337 [-0.0669, -0.0005] better; blend − tabpfn -0.0033 [-0.0172, +0.0106] none
- `electricity_60nan` (numeric share 0.87): blend − model -0.0099 [-0.0151, -0.0047] better; blend − tabpfn -0.0062 [-0.0084, -0.0039] better
- `electricity_80nan` (numeric share 0.87): blend − model -0.0102 [-0.0154, -0.0050] better; blend − tabpfn +0.0001 [-0.0019, +0.0021] equivalent
- `vehicle_40nan` (numeric share 1.00): blend − model -0.0165 [-0.0226, -0.0104] better; blend − tabpfn -0.0168 [-0.0247, -0.0090] better
- `vehicle_80nan` (numeric share 1.00): blend − model -0.0499 [-0.0556, -0.0443] better; blend − tabpfn +0.0031 [-0.0011, +0.0073] equivalent
- `biodeg_40nan` (numeric share 1.00): blend − model -0.0154 [-0.0253, -0.0056] better; blend − tabpfn -0.0568 [-0.0766, -0.0371] better
- `biodeg_80nan` (numeric share 1.00): blend − model -0.0530 [-0.0602, -0.0459] better; blend − tabpfn -0.0005 [-0.0067, +0.0056] equivalent
- `kc2_40nan` (numeric share 1.00): blend − model -0.0479 [-0.0821, -0.0137] better; blend − tabpfn -0.0496 [-0.0827, -0.0164] better
- `kc2_80nan` (numeric share 1.00): blend − model -0.0112 [-0.0405, +0.0182] none; blend − tabpfn -0.0689 [-0.0925, -0.0453] better
  - blend − model: better 14 / no difference 1 / worse 0 (fold-pair t, seed agreement)
  - blend − tabpfn: better 11 / no difference 1 / equivalent 3 / worse 0 (fold-pair t, seed agreement)

Primary: the blend beats both on 10 of 15 (['pendigits_40nan', 'pendigits_60nan', 'pendigits_80nan', 'letter_40nan', 'letter_60nan', 'letter_80nan', 'electricity_60nan', 'vehicle_40nan', 'biodeg_40nan', 'kc2_40nan']) -> complementary (fold-pair t, seed agreement)
```

**Verdict: complementary.** The blend beats both K and TabPFN on 10 of the 15 confirmation
variants; the rule's threshold was 5.

**Reading:**

- **What "beats both" measures.** On a numerical cell, the RMSE of the averaged fill is never
  above the average of the two RMSEs (Minkowski's inequality).
  - On an all-numerical variant, then, the blend can never be worse than the worse of the two.
    The zero "worse" calls are arithmetic, not a finding.
  - Only "better than the better of the two" is evidence. That is what the 10 of 15 counts: the
    two fills' errors are far enough from parallel that averaging them gains on both.
- **By rate:**
  - **At 40–60nan,** where K mostly leads TabPFN (E55), the blend beats both on 8 of 9.
    - The exception is electricity_40nan, where blend − TabPFN is not distinguishable (a wide
      interval).
  - **At 80nan:**
    - On pendigits and letter, the blend still beats both.
    - On electricity, vehicle and biodeg, it is equivalent to TabPFN: the blend lands on TabPFN's
      level, and averaging in K adds nothing.
    - On kc2, it beats TabPFN and is not distinguishable from K.
- **The weights** were fixed at one half before the first cell; nothing was tuned on scored cells.
- **Limits:**
  - **The model side is K.** E44 ran before ADR 0027, and all 15 variants are at 40nan or more,
    where the default output is KC.
    - E57 found KC changes one of K's 15 calls against TabPFN (letter_80nan, worse →
      equivalent).
  - **Electricity's `day`** keeps the model's category, so its blend is a blend of numerical
    cells only.

**What it means for the thesis:** TRIDENT's and TabPFN's errors are not the same errors. An
equal-weight average of the two fills beats both on most large tables and at 40–60nan. This is a
"hybrid" argument for the discussion. It is descriptive, and no decision or default follows.
