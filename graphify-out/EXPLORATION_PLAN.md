# Graph-Driven Documentation Vault — Exploration Plan

**Goal:** build an Obsidian vault, one note per node, so every documentable entity in
TRIDENT carries its graph context (callers, callees, cross-community bridges, rationale,
open questions) as navigable `[[wikilinks]]` — without touching source docstrings.

**Mechanism:** `graphify.export.to_obsidian(G, communities, output_dir, community_labels)`.
It already does the hard part per node — YAML frontmatter, a `## Connections` section of
`[[neighbor]] - `relation` [CONFIDENCE]` lines, community/confidence tags — and tracks
ownership in `.graphify_obsidian_manifest.json` so re-running it later only adds/updates
notes, never clobbers a hand-edited one. That manifest is what makes a **phased** build
safe: each phase below is a separate call into the same `graphify-out/obsidian/` vault,
scoped to a growing node set, and nothing already written gets orphaned or duplicated.

Source of prioritization: `graphify-out/GRAPH_REPORT.md` (god nodes, communities) plus a
direct comparison of `code`-type nodes vs. `rationale`-type (docstring-derived) nodes per
file in `graph.json` — see the gap table below.

## Why this order

Two independent signals point at the same handful of files, which is the strongest
argument for starting there:

1. **God nodes** (highest graph degree = most other code depends on understanding this
   correctly): `FoldTrackingRecord` (29), `PreparedDataset` (27), `summarize_cross_validation()` (25),
   `FoldResult` (24), `run_training()` (20), `ArtifactWriter` (19), `MlflowTracker` (18),
   `CrossValidationSummary` (18), `train_and_evaluate_classifier()` (15), `train_pretrainer()` (15).
2. **Documentation gaps** (code entities the AST pass found with no paired docstring
   node at the same `source_location` — an existing docstring shows up in the graph as a
   `file_type: rationale` node at that entity's line):

   | File | Code entities | Docstring (rationale) nodes | Gap |
   |---|---|---|---|
   | `src/training/tracking.py` | 27 | 5 | 22 |
   | `src/training/types.py` | 22 | 2 | 20 |
   | `opt.py` | 16 | 5 | 11 |
   | `src/training/summary.py` | 12 | 4 | 8 |
   | `src/training/artifacts.py` | 11 | 3 | 8 |

   `types.py` holds 4 of the 10 god nodes; `tracking.py` holds 1 more (`MlflowTracker`) and
   is itself the single biggest gap. That overlap is why Phase 1 targets both files
   directly instead of picking from the gap table alone.

Six of the ten god nodes are plain dataclasses/typed values (`FoldTrackingRecord`,
`PreparedDataset`, `FoldResult`, `CrossValidationSummary`) — they have no behavior to
misdocument, but they're exactly the shared seams a wiki note is best at making legible:
who builds them, who consumes them, and — via the `INFERRED uses` edges the semantic
pass found — which relationships are structural inference rather than a literal
`import`/`call` and might warrant a second look.

## Phases

### Phase 1 — God nodes + the two gap files ✅ done
**Scope:** 10 god nodes ∪ every code entity in `src/training/types.py` and
`src/training/tracking.py`, plus each of those nodes' direct graph neighbors (needed so
the exported `[[wikilinks]]` resolve to real notes instead of dead links).
**Size:** 54 core nodes → 145 nodes once neighbors are included, spanning 8 of the 16
communities (MLflow Fold Tracking, Training Config & Data Pipeline, Cross-Validation
Summary Stats, Model & Training Core, MLflow Tracking Tests, CLI Entry & Hyperparameter
Search, Artifact Writing & Persistence, Fold Metrics Computation).
**Output:** `graphify-out/obsidian/` (the vault's default location, so later phases can
extend it in place). **153 notes written** (145 node + 8 community). Spot-checked
`PreparedDataset.md` against the manual trace done earlier in the session — exact match.

### Phase 2 — Remaining gap files ✅ done
**Scope:** every code entity in `opt.py`, `src/training/summary.py`,
`src/training/artifacts.py` (the next three rows of the gap table), plus their direct
neighbors. Run as a **cumulative** re-export (Phase 1 files + Phase 2 files together,
not just the delta) — `to_obsidian`'s manifest prunes any previously-owned note not
rewritten in the current call, so passing only the new files would have deleted Phase 1's
145 notes.
**Output:** vault grew to **196 notes** (187 node + 9 community); disk count matches the
reported count exactly, confirming no Phase-1 notes were pruned. Spot-checked
`ArtifactWriter.md` — its 7 methods, 6 `INFERRED uses` edges, and 2 structural edges all
present and correctly typed.

### Phase 3 — Everything else ✅ done
**Scope:** the remaining code files (`src/embedder.py`, `src/models.py`,
`src/transformer.py`, `src/utils.py`, `src/mlflow_utils.py`, `src/training/config.py`,
`src/training/data.py`, `src/training/discovery.py`, `src/training/finetuning.py`,
`src/training/pretraining.py`, `src/training/runner.py`, `src/training/cli.py`,
`datasets/generate_splits.py`, `main.py`, `train.py`) plus all `tests/*` files, plus the
16 doc/ADR/ticket/concept nodes not yet pulled in as neighbors. Run as the plain,
unscoped `graphify export obsidian` — the full graph, 367 nodes.
**Output:** vault grew to **383 notes** (367 node + 16 community). Disk count matches
exactly — Phases 1 and 2's notes all survived the unscoped re-export untouched.

## Status: complete

All three phases are in `graphify-out/obsidian/`. Open it as an Obsidian vault to browse.

## Verification per phase

- Confirm `to_obsidian`'s return count matches the planned node count before reporting done.
- Spot-check 2–3 notes for a god node against the source file at the cited `source_location`
  — the note is a map, not a source of truth; a wrong graph edge should not get amplified
  into a wrong-looking wiki note.
- Flag (don't silently resolve) any `AMBIGUOUS`-confidence connection surfaced in a note —
  those come from the honesty-rules audit trail and are exactly the ones worth a human
  second look before treating the note as settled documentation.
