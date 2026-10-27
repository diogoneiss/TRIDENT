# Does the spread grow when a cell's top sources are missing? (E80; pre-registered 2026-10-27)

This ticket was written and committed before the reading ran.

- **Why:** ADR 0030's spread says how sure a numerical fill is, and E78/E79's map says what it is
  filled from. If the two describe one mechanism, a cell should be less sure when the columns its
  fill relies on are missing in its row. That should hold beyond the row simply missing more.
- **The user's go:** "otimo, quero fazer E e F agora, vá" (2026-10-27, before 08:29 GMT-3). This is
  study E of T38's proposal.
- **Code:** no change to `src/` and no training. `scripts/experiments/occlusion_spread.py` reads E78's
  and E79's fold ledgers and validation maps.
- **What was already seen (disclosure):** both studies' printouts were read: the maps, D, the AUROC
  and the top sources. No spread-related measure was computed on these cells before this ticket.
  `--check` ran first, and printed only the alignment counts.

## Design (fixed before the reading)

- **Data:** E78's and E79's cells.
  - Scope: the 16 real variants with numerical gaps, the 8 at 40nan and the 8 at 60nan. kr-vs-kp
    has no numerical gaps and is out.
  - synthblock at 40 and 60nan is descriptive.
- **The row's gaps:** read from the fold ledger's `induced` rows, which hold every gap of the test
  split; categories the variant never shows are included.
  - `--check` on pendigits fold 1 at 40 and 60nan: no duplicate cells, and every `induced_remask`
    row is inside the gap rows. The ledger's `row` is the test split's position, the same index the
    occlusion views use.
- **The spread:** each numerical gap's `induced_remask` row, at K.
  - H is not read. These runs did not use `--score_checkpoints`, so there is no
    `induced_remask_loss_checkpoint`. A claim from this study says "K" (ADR 0024).

## Measure and rule, fixed now (`occlusion_spread.py`)

- **One fold's value:** for each numerical target j, take its top 3 sources by the fold's validation
  map, as E78's B ranks them.
  - Compute Spearman, over j's test gap cells, between the spread and how many of those 3 are
    missing in the cell's row.
  - Subtract the mean of the same correlation over 20 draws of 3 random other columns, from a private
    stream seeded by `[seed, fold, 80]`.
  - Average over the targets; a correlation is NaN when either side is constant or there are fewer
    than 10 cells.
- **The verdict on each variant (15 folds):**
  - **specific** if the 95% t-interval lies above 0 and every seed's mean is above 0;
  - **against** if it lies below 0 with every seed below 0;
  - no signal otherwise.
- **The rule over the 16:**
  - **supported:** specific on 12 or more, and against on none;
  - **refuted:** specific on 6 or fewer;
  - **mixed:** otherwise.
- **Descriptive:**
  - Spearman of the spread with the top-3 missing count, and with the row's total gaps;
  - Spearman of the absolute error with the top-3 missing count;
  - on synthblock, the same measure with the true block as the top 3.

**What each reading leads to:**
- **Supported:** the thesis can tie the two outputs together. The spread is higher where the map's
  sources are missing, more than the row's missingness explains.
- **Otherwise:** they are reported as separate signals. No default changes either way.
