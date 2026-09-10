# 14. A regression fixture for the imputation task

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 08, 09

## Question

`tests/fixtures/vehicle_00nan_regression.json` pins a short two-fold classification run
and is a reviewed baseline that `AGENTS.md` protects. The imputation task needs its own
answer to "did this change move the numbers?".

- **Is a fixture warranted at all**, or do unit tests on the decoder head and the metric
  functions suffice? The classification fixture exists because the whole pipeline is
  seeded and bit-reproducible; the decode stage inherits that property.
- **What does it pin**: which dataset and configuration (the mixed `credit-g_20nan`
  exercises both column types and the induced-missing population, which `vehicle_00nan`
  cannot), which metrics, and what tolerance.
- **Both populations or one?** Induced-missing metrics only exist on a variant with a
  sibling, so pinning them requires a non-`_00nan` dataset.
- **Marker**: another `@pytest.mark.integration` test, kept out of the default run.

Recommended starting answer: one fixture on a short `credit-g_20nan` run covering both
populations, marked integration, with the same tolerance style as the existing fixture.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

**The existing fixture's pattern**, read while resolving, is more careful than the ticket
assumed and is copied wholesale. `tests/fixtures/vehicle_00nan_regression.json` records the
environment it was generated on (python, platform, torch build, device, seed, cv_folds) plus
`"runs": 3`, and then:

| Field | Value |
|---|---|
| `observed_max_deviation` | 0.0 |
| `tolerance` | 0.01 |

Deviation is *measured* across repeats and the tolerance set far above it. Bit-identical in
practice, with headroom so the test survives a different device or torch build.

**Decisions:**

1. **A fixture is warranted, on `credit-g_20nan`.** The pipeline is seeded and
   bit-reproducible, which is what makes such a baseline meaningful, and `AGENTS.md`
   already treats the existing one as reviewed. The dataset is forced: `credit-g_20nan` is
   the only variant exercising **both column types** (13 categorical, 7 numerical) **and**
   the induced-missing population, since it has a row-aligned `_00nan` sibling.
   `vehicle_00nan` can do neither -- it is all numerical and has no missing cells.

2. **Six numbers per fold, plus their means**: `impute_score`, `rmse_num_z` and `acc_cat`,
   each for the self-masked and induced-missing populations. Not the whole metric set of
   ticket 05: the existing fixture pins three of the eleven or so numbers available, so
   selectivity is the precedent, and `mae_num_z` and `macro_f1_cat` move with their
   primaries.

3. **The degenerate compositions are unit-tested, not integration-tested.** The
   all-numerical datasets (`w_cat = 0`), the all-categorical `kr-vs-kp` (`w_num = 0`) and
   the zero-baseline guard are arithmetic properties of the metric functions, testable in
   milliseconds against synthetic inputs. An integration run per composition would triple
   the slow suite to re-verify the same formula.

4. **Structure and tolerance copy the existing fixture exactly**, environment block and
   measured-deviation field included. Run the configuration several times, record the
   deviation that actually appears, and set the tolerance well above it. Inventing a
   tighter tolerance than 0.01 would make the new test fail on hardware where the old one
   passes.

   Proposed configuration, mirroring the existing fixture's smallness so the test stays
   short: `DIM` 16, `HIDDEN_DIM` 8, `HEADS` 4, `LAYERS` 1, `DIM_FEED` 16, `DROPOUT` 0.1,
   `EPOCHS_PRE` 2, `EPOCHS_DECODE` 2, `BATCH` 64, `LAMBDA_NUM` 1.0, `EVAL_MASK_RATE` 0.2,
   `cv_folds` 2, seed 42, tracking disabled. A two-epoch decode stage is barely trained and
   its scores will sit near the mean/mode baseline; that is fine, because the fixture pins
   determinism rather than quality.

5. **`AGENTS.md` is extended to name both fixtures**, so the imputation baseline inherits
   the same rule: change it only for an intentional, documented behaviour change. The test
   carries its regeneration command so the next person need not reconstruct it.

Marked `@pytest.mark.integration`, keeping `pytest -m "not integration"` fast, as the
existing regression test is.

Consequence: unblocks ticket 10, the destination handoff.

## Comments
