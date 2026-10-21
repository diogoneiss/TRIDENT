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
