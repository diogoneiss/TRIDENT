# Ticket 0004: Exact original units in the imputation preview

Backlog item: [C5](../BACKLOG.md#c5-numeric-precision-and-scaling--needs-research).
Raised by the author on 2026-09-10 after reading a real `spambase_20nan` preview.
Decided 2026-09-10; this ticket is the record and the execution brief.

## The reframing

C5's first concern reads as a dtype question -- "should `EncodedTable.num_values` be
float64, and what does that cost on the 45k-row datasets?" -- but the code says it is a
**provenance** question, and the dtype question dissolves.

`split_numeric_and_special` builds a float64 array and downcasts it at
`src/utils.py:123`, which is correct: the model's parameters are float32, so the tensor
should be too. The defect is that `_score_population` then reads the *displayed* truth
back out of that float32 tensor (`src/training/decoding.py:287`,
`truth.num_values[index][selected].tolist()`) when the exact float64 value is still
available upstream. We display a known truth at float32 precision for no reason, and the
inverse scaler amplifies the loss by the column's `scale_` -- 1.15e-01 on `kc2`'s `e`.

Nothing wants float64 in the tensor. The `imputed` side is genuinely float32, because
that is the model's real output precision; there is nothing to recover there.

## Decisions

1. **Retain the raw numerical columns.** `prepare_dataset` copies
   `frame[numerical_columns]` before `scaler.fit_transform` overwrites it, and
   `PreparedDataset` carries it as a **defaulted** field beside `scaler`. That field set
   the precedent: retaining it does not move when scaling happens, so no training
   behaviour changes. Cost is roughly 3 MB on the 45k-row datasets.

2. **Look the exact truth up in `decoding.py`, not `artifacts.py`,** because the two
   populations have different truth sources and only decoding has both in scope:

   | population | exact original-unit truth |
   |---|---|
   | `masked` | the variant's own raw value (`PreparedDataset.raw_numerical`) |
   | `induced` | the `_00nan` sibling's raw value, before `decoding.py:223` scales it |

   The variant's raw frame holds `NaN` at an induced-missing cell, so it cannot serve
   that population. `scored_cells["row"]` is positional within
   `features.iloc[fold.test_indices].reset_index(drop=True)` (`decoding.py:128`), so
   every raw lookup slices identically.

3. **Render each column at its own observed precision**, derived from the raw values: an
   integer-valued column prints 0 decimals, a 2-decimal column prints 2. This is what
   turns `spambase`'s exact zeros from `1.144e-09` into `0`, on both the actual and the
   imputed rows.

4. **Mark a cell whose rendered text is not the exact value.** Computed per cell by
   checking whether the rendering round-trips, not assumed per row. An `actual` at its
   column's own precision renders exactly and stays unmarked; an `imputed` value loses
   information (`-38.55` -> `-39`) and is marked. The preview header carries a legend
   pointing at the cell ledger for full precision, so the marker means "shortened for
   display, exact value in the CSV" rather than decorating every number.

## Constraints

- `scored_cells["actual"]` and `["imputed"]` must stay byte-identical. The imputation
  metrics read them directly (`src/training/imputation_metrics.py:50,55,79,86,95`), so
  changing them would move `rmse_num_z` and `impute_score`. Only the derived
  `*_original_units` columns and the preview rendering change.
- Therefore `tests/fixtures/credit-g_20nan_imputation_regression.json` stays green
  unedited, and `vehicle_00nan` is untouched because no training code moves.
- No flag, MLflow tag or backfill: this is not a training-behaviour change, so the
  standing gate-it-behind-a-flag preference does not apply.
- Every new field and argument defaults to today's behaviour.

## Explicitly not in this ticket

- **C5 concern 3, unbounded numerical imputations.** Clamping was considered and
  rejected by the author; the model should not be handed constraints it did not learn.
  Observed-precision rounding turns `-38.55` into `-39`, which is still negative, so it
  does not quietly reinstate clamping.
- **Backlog C1**, the scaler fitted before splitting. Orthogonal: the preview displays
  through whatever scaler produced the scaled space, while C1 is about scoring fairness.
- **Changing `EncodedTable.num_values` to float64.** Ruled out above; it would move
  training numerics and break the protected classification baseline.

## Acceptance

- A numerical `actual` equals the dataset's raw value exactly before rendering, for both
  populations. It also *displays* exactly wherever the dataset carries three decimals or
  fewer; where it carries more (`electricity`, `biodeg`) the marker says so.
- A column whose raw values are integers prints no decimal point; exact zeros print `0`.
- A rounded cell carries the marker; an exact one does not; the legend explains it.
- The cell ledger keeps full precision in both scalings.
- `pytest -m "not integration"` green; both integration regressions green, unedited.
- A real `spambase_20nan` and `kc2_20nan` preview shows the before/after that closes C5.

## Outcome

Done 2026-09-10, test-first across two seams. `prepare_dataset`'s one-line retention got
an assertion on an existing test rather than a seam of its own.

**Measured on real runs**, both isolated from `results/` and `metrics/`:

| dataset | what it shows |
|---|---|
| `kc2_20nan` | `e` (`scale_` 123,433) displayed **870848.556** where the dataset holds **870848.580**, a 2.4e-02 gap. Two of 21 columns moved at the display precision |
| `spambase_20nan` | **33,496 of 43,101** scored cells are exact zeros; each used to render as `1.414e-09`. All now print `0.000`, unmarked |

The rendering contract, checked directly:

| value | shown | why |
|---|---|---|
| `0.0` | `0.000` | exact zero, unmarked |
| `1.144e-09` | `0.000*` | snapped, and the marker says so |
| `-1e-09` | `0.000*` | never `-0.000` |
| `-38.55` | `-38.550` | concern 3: still negative, and exact, so unmarked |
| `870848.58` | `870848.580` | the kc2 truth, exact |
| `0.083073` | `0.083*` | `electricity`/`biodeg` carry more than three decimals |
| `1.2345e16` | `123450000000000..*` | clipping is lossy too |
| `"red"` | `red` | categories are already readable |

**Verification.** 132 unit tests green; both integration regressions green and unedited,
confirming the metrics never moved. `graphify update .` run.

**Left alone deliberately.** Concern 3 stands as recorded behaviour. `EncodedTable`
keeps float32, which the reframing showed was never the problem.
