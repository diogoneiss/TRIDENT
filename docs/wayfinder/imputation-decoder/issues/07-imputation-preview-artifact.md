# 07. Prototype: the human-friendly imputed-vs-actual preview

Type: prototype
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 03

## Question

What does a person want to see to judge imputations at a glance? Build a throwaway
rendering (fake numbers are fine) to react to, then fix the format.

Dimensions to settle:

- **Rows and cells**: a fixed-seed sample of N test rows; show every scored cell or only
  a few per row; group by column or by row.
- **Columns of the table**: row id, column name, type, actual value, imputed value,
  error / correct flag, and for categoricals the top-k probabilities.
- **Units**: original units (inverse `StandardScaler`, `LabelEncoder.inverse_transform`
  via `embedder.label_encoders`) vs scaled space.
- **Side-by-side view**: original row | masked row (with `[MASK]`) | imputed row.
- **Format and location**: markdown, CSV, or HTML; logged as an MLflow artifact under
  e.g. `imputation/preview.md` on diagnostic children and the single-split parent;
  also written under `results/`.
- Whether the preview covers both ground truths from ticket 03.

## Answer

Resolved 2026-09-10. A throwaway prototype rendered three structurally different layouts
over real `credit-g` rows with real `_00nan` ground truth and simulated decoder outputs;
the format was chosen by looking at them. Prototype captured on branch
`prototype/imputation-preview` at `scripts/prototype_imputation_preview.py`; run it with
`uv run --python 3.10 python scripts/prototype_imputation_preview.py [--variant A|B|C]`.

**Variants built:** A a cell ledger (one line per scored cell, cell-major), B a row
triptych (actual / model saw / imputed, row-major), C a column report card
(column-major).

**What the rendering settled that argument could not:**

- **B alone shows what the model saw.** Its middle line is where ticket 03's decision
  becomes visible: an induced-missing cell reads `[NULL]->[MASK]`, a self-masked cell
  reads `[MASK]`. Neither other layout can express the token path.
- **C is not a preview.** At four sampled rows its per-column counts were 1-3, making
  "100% correct" noise. To mean anything it needs the whole test fold, at which point it
  *is* the per-column metrics table ticket 05 already routes to an artifact.
- **Original units are unreadable as a quality signal.** The same simulated noise showed
  as `rmse 792` on `credit_amount` and `0.081` on `num_dependents`. That is a units
  artifact, which is exactly why ticket 05 ranks on scaled values. The preview's job is
  comprehension, not judgement.

**Decisions:**

1. **Layout.** Variant B is the human preview; Variant A is a companion machine-readable
   file holding every scored cell; Variant C is dropped as duplicated by ticket 05.
2. **Sampling.** A fixed-seed random sample of test-fold rows, seeded from the run seed
   and the fold ordinal so it is reproducible and comparable across runs. No selection by
   error, so the preview cannot flatter or damn the model. The companion file holds every
   scored cell, so failures remain findable without biasing the preview.
3. **Populations share one preview.** The model-saw line already distinguishes them, and
   splitting would destroy the row context that justifies the layout.
4. **Location** mirrors loss plots exactly, plus a copy under the results directory.
   Written for every imputation run without a flag, since previewing imputations is the
   point of the task rather than an extra.
5. **Units.** Original units in the preview; both scalings plus the model's confidence in
   the companion file.

**Artifact inventory** (the answer to "which layout appears where"):

| File | Layout | Contents | Written | Reaches MLflow |
|---|---|---|---|---|
| `imputation_preview_fold_N.md` | B | the sampled rows only | every fold | best and worst fold children |
| `imputation_cells_fold_N.csv` | A | every scored cell of the test fold | every fold | best and worst fold children |
| `metrics/per_column_imputation.csv` | C's content | column x population x metric, long format with a fold column | once per run | the parent run |

Each layout has exactly one home. The fold suffix follows the existing plot convention
and is dropped for a single-split run, whose artifacts attach to the parent because it
has no children.

**Sampled rows are not removed from the cell file**; it is a strict superset, so every
previewed cell is also in the complete record. The cell file carries an `in_preview`
boolean so joining the two is a filter rather than a hunt.

**Upload asymmetry worth knowing**: every fold writes its files to the results directory,
but only the diagnostic children upload them, because `BufferedFoldTracker` collects each
fold's artifacts and `_log_diagnostic_children` replays only the selected folds. A
five-fold run leaves five previews on disk and two in the tracking store, or one when
best and worst tie.

**Pipeline consequence.** Original units require the fitted `StandardScaler`, which
`prepare_dataset` currently fits and then discards, and the embedder's `LabelEncoder`s
for the inverse categorical mapping. Retaining a reference to the scaler on
`PreparedDataset` is an addition, not a change to the split-and-scale order that
`AGENTS.md` protects, so it stays inside the rules. Ticket 09 carries it.

## Comments
