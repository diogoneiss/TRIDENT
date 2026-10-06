# Tasks

The research and engineering tracker for the imputation effort: what to do next, what waits on a
decision, and what was done. Code-level bugs and pendencies stay in [BACKLOG.md](BACKLOG.md);
each experiment's contract is its ticket under `docs/tickets/`, and its result goes in the
[experimentation log](EXPERIMENTATION_LOG.md). Status is one of **decision**, **next**,
**backlog**, **in progress** or **done**; update a task when its status changes and link the
ticket or commit that moved it.

Created 2026-09-30 from the proposals after studies 04 and 05 (E27, E28).

## Waiting on a decision

Nothing waits on a decision. T00 was decided on 2026-10-01: the imputation default is now M, as recorded in [ADR 0013](adr/0013-imputation-defaults-from-the-confirmatory-study.md).

| ID | Task | Evidence | Status |
|---|---|---|---|
| T00 | Make `embedding_normalized` the imputation default, and choose its decode setting (150, 450, 450 with patience 50, or per variant) | E25: never worse than the current objective, better than no pre-training on 3 of 4 variants. E26: with decode 450, never worse than either half. E27: patience 50 keeps the scores on 3 of 4 tables at a fraction of the time, −0.006 on pendigits. E28: batch 1024 pays only on pendigits. **E30 (all 21 variants): M better than A on 9, worse on none; P worse than A on 2, so by the pre-stated rule the recommendation is M** (normalised pre-training + decode 450), at 1.65x A's time | done |

## Next up, in the recommended order

### T01 · Confirmatory study on all 21 variants — done (ticket [07](tickets/imputation-pretraining/07-confirmatory-21-variants.md), E30)

**Done 2026-10-01**, 00:22–13:19 GMT-3, all 189 cells (A, M, P on 21 variants, three seeds).
M − A: M better on 9 variants, A better on none; P − A: P better on 7, A better on 2
(kr-vs-kp_40nan, spambase_40nan); M better than P on 3. Recommendation by the pre-stated rule:
M, now T00's decision. Wall time 12 h 57 min against about 8.5 h planned (1.52x): small tables
took 3–5x their per-cell estimate, kr-vs-kp 3.5x.

The plan as written before launch:

The candidate default against the current one on every imputation variant, since E24–E28 used
only four. Three seeds (the verdict rule needs three), five folds, the same pairing and rule as
tickets 01–05. Reuses the server cells of credit-g_80nan, kr-vs-kp_40nan and pendigits
(credit-g_20nan changed configuration and runs again).

| candidate against the current default (A) | new cells | GPU time, summed | wall time, 3 queues |
|---|---|---|---|
| M: normalised pre-training + decode 450 | 108 | ~24 h | ~10 h |
| P: M + decode patience 50 | 108 | ~16.5 h | ~7 h |
| A, M and P together | 162 | ~32 h | ~14 h |

Estimated from the 2026-09-24/25 arm-A runs of all 21 variants (431 min per seed on Windows, 0.42x
on the server), M ≈ 2x A and P ≈ 0.26–0.86x M as measured on 2026-09-30. electricity, letter and
the four spambase variants are more than half of it; P on electricity and letter is the largest
uncertainty.

### T02 · Train/score token-shape mismatch — done (ADR 0014, replicated in E33)

**Follow-up 2026-10-05:** the pre-training stage's gap token was studied in ticket [06](tickets/imputation-token-shape/06-pretraining-gaps-as-mask.md), E36: showing the gaps as `[MASK]` there too gains nothing (better on no variant, worse on electricity_20nan), so it stays `null`. (tickets [01](tickets/imputation-token-shape/01-column-wise-scoring.md), [02](tickets/imputation-token-shape/02-gaps-as-mask-in-training.md); E31, E32)

**Step 2 done 2026-10-02 16:22 GMT-3:** training the decode stage with the gaps shown as `[MASK]`
(`--decode_gap_token mask`) is better than today's default on 11 of 21 variants and worse on none,
with gains at every missing level (mean over 21: 0.7745 → 0.7542; below the best baseline on 16
variants, from 11). By the pre-stated rule G qualifies as the imputation default. Wall time
5 h 38 min against 5.8 h planned (0.97x). **Decided 2026-10-03: adopted as the imputation
default ([ADR 0014](adr/0014-gaps-shown-as-mask-in-the-decode-stage.md)); a replication on fresh
seeds runs overnight (ticket [03](tickets/imputation-token-shape/03-replication-on-fresh-seeds.md),
E33).** **Replicated 2026-10-03 12:05 GMT-3 (E33):** on seeds 101, 202, 303, G better on 13 of 21
variants and worse on none (6 of the 9 heavy); mean 0.7786 → 0.7519; wall time 11 h 21 min, on
plan. T02 is done.

**Step 1 done 2026-10-02:** scoring the induced cells one gap column at a time, the other gaps as
`[NULL]`, is better than the headline on 5 of the 9 variants at 60nan or more and worse on none;
the effect grows with the missing level (+0.004 at 20nan, −0.022 at 60nan). Supported by the
pre-stated rule. Wall time 5 h 59 min against 5.2 h planned (1.15x). **The user decides** between
step 2 below and an ADR making column-wise scoring the headline (mean 0.7745 → 0.7673, at a cost
of up to 0.013 on six light variants). **The user chose step 2 (2026-10-02):** `--decode_gap_token
mask` shows every real gap as `[MASK]` in the decode stage; 63 cells against E30's M, pre-registered
in ticket 02.

The lead hypothesis for why the model loses to simple imputers (handoff items 1 and 2). In
training, real gaps are `[NULL]` and about 2% of cells are `[MASK]`; in induced scoring every gap
is `[MASK]` at once (about 80% on credit-g_80nan, where every arm loses to mean/mode).

1. **Scoring only, cheap:** score the induced cells one gap column at a time, the other gaps left
   as `[NULL]`. A diagnostic flag like `--score_null_path`; tests the hypothesis without training
   anything new.
2. **If step 1 confirms it:** train with decode rows shaped like induced scoring (part of the real
   gaps shown as `[MASK]`), behind a flag, as a pre-registered study.

### T03 · Architecture defaults — screening done (ticket [imputation-architecture/01](tickets/imputation-architecture/01-architecture-screening.md), E37); confirmation waits on a decision

**Screened 2026-10-06:** 4 layers + feed-forward 256 + a final LayerNorm (ALL) is better on 3 of 5
variants and worse on none (credit-g_20nan a near miss), at 1.82x the time; it advances to a
21-variant confirmation, which needs the user's go-ahead. F256 alone also advances (better on
pendigits only); depth alone hurts credit-g_20nan; the final LayerNorm alone hurts pendigits. **The user chose instead (2026-10-06)** to test whether the bigger encoder helps the large
tables, with a size-dependent default if so: ticket [02](tickets/imputation-architecture/02-bigger-encoder-by-table-size.md), E38.

Every full-profile winner used 4–6 layers (default 2); the feed-forward width (32) is narrower
than the model width (128), against the usual ~4x; there is no final LayerNorm before the heads
(handoff item 4). A study of `LAYERS` 4, `DIM_FEED` 256 and a final LayerNorm (behind a flag) on
the candidate configuration.

### T18 · Decide whether `--decoder_heads batched` becomes the default — done

Implemented 2026-09-30 (`c9be71f`) behind a flag: 1.4–1.6x faster a cell, equal to the
per-column path up to rounding, so it is another training draw of the same model, like a run on
another GPU. Flipping the default needs the user's decision. **Checked 2026-10-01** (ticket
[06](tickets/imputation-pretraining/06-batched-decoder-heads.md), E29): no detectable difference
on any of the four variants, so the rule licenses the switch; pendigits leans +0.0008 on all
three seeds (lower bound at zero), noise-sized. **Decided 2026-10-01: batched is the default** (ADR 0012); the
check cell's new reference is induced 0.8863449097353776, masked 0.9256312571312231.

## Backlog

### Speed

- **T06 · `torch.compile` or CUDA graphs** on the decode loop. Likely 1.5–3x on a launch-bound
  loop, but the per-batch count of hidden cells varies, which breaks CUDA graphs; numerics change
  slightly. Related: [BACKLOG P2](BACKLOG.md) (fused attention).

### Model selection

- **T07 · Checkpoint by validation `impute_score`** instead of the λ-weighted loss (critique
  F-02-1, handoff item 6); also a better criterion for early stopping. **Done:** measured (ticket [imputation-token-shape/04](tickets/imputation-token-shape/04-checkpoint-and-calibration.md), E34, 2026-10-03): the checkpoint by the validation rows' own gaps is better on 3 variants (all 80nan) and worse on none, so it qualifies; **Decided 2026-10-03: adopted as the imputation default ([ADR 0015](adr/0015-decode-checkpoint-chosen-on-the-validation-gaps.md)).** **Replicated 2026-10-04 (E35, ticket [05](tickets/imputation-token-shape/05-checkpoint-replication.md)):** on seeds 42, 7, 13, K better on 6 variants and worse on none; T07 is done. The calibration tested beside it failed on one variant (spambase_20nan).
- **T08 · A second validation mask:** one to choose the checkpoint, one to score the search
  objective (critique F-07-2).
- **T09 · Optuna with k-fold validation inside the search.** Selection on one split is "a
  lottery" (E14); only worth another search with a steadier objective, now cheaper after the
  speed-ups.
- **T10 · Per-column categorical loss weighting and an unbiased evaluation-mask draw** (handoff
  item 6).

### Specific follow-ups

- **T11 · pendigits: batch 1024 at LR x4 with a longer patience** (100, or a 600-epoch ceiling):
  its best arm (E28, 0.323) and a validation loss still falling near epoch 450 (E27).
- **T12 · A lower `LR_PRE` with the normalised objective:** every full-profile winner chose 2–34x
  below the default (E21).
- **T13 · An EMA teacher for the normalised objective** (data2vec), the anti-collapse ingredient
  it still lacks (ADR 0011, considered options).
- **T14 · A steps-aware batch rule** (at least ~8 steps an epoch) instead of one batch for every
  table (E28).
- **T15 · MAE-style architecture** (ADR 0004 Design C, handoff item 5): the most expensive; only if
  T02–T03 fall short.

### Process

- **T16 · One configuration-driven study report** instead of a copied report script per study.
- **T17 · An end-to-end test with a scratch MLflow store for every new flag**; it would have caught
  study 04's finalisation crash before 37 minutes of GPU time.

## Done (2026-09-30)

| Task | Commits |
|---|---|
| Pre-training ablation (E24) | outcome `9a77e4d` |
| `--pretrain_objective` (ADR 0011) and studies E25, E26 | `d42f07c`, outcomes `f58d795`, `2a026ad` |
| Decoder without per-column host synchronisation (exact) | `71c6a9b`, `6dd07a5` |
| CPU thread cap for concurrent queues | `dc96c6d` |
| Baseline cache (ADR 0007 decision 10, exact) | `8818f6d` |
| `--decode_patience` and study E27 | `521df3d`, fix `a1d54a9`, outcome `c5b7647` |
| Larger-batch study E28 | outcome `fc275cc` |
| `FAILED` runs no longer mirrored (ADR 0006 amended) | `a2a6a26` |
| Promoted `credit-g_20nan` imputation file removed | `8954217` |
| `[NULL]`-path defect D-1 fixed, gate re-measured | `918bcd6`, `1154f59` |
| Tag backfills `pretrain_objective`, `decode_patience` | applied 2026-09-30 |
| T05 · Batched per-column heads, behind `--decoder_heads` | `c9be71f` |
| Cache prune script (`scripts/prune_cache.py`) | `c9be71f` |
| Epoch masks drawn on the encoded tensors instead of the DataFrame (exact; pre-training's pandas re-encoding was 25–30% of an epoch on categorical tables) | `2e0e85a` |
| Tag backfill `decoder_heads` (1473 runs) | applied 2026-10-01 |
| ADR 0014 · gaps shown as `[MASK]` in the decode stage (E32, replicated in E33) | `e2f4e47`, 2026-10-03 |
| ADR 0015 · decode checkpoint chosen on the validation gaps (E34); tag `decode_checkpoint` backfilled on 1990 runs | `815f6d8`, 2026-10-03 |
| Speed check, kr-vs-kp_40nan arm A, 5 folds, one trainer: 20.1 min on 2026-09-29 (hard-disk store) → 4.0 min per-column (same scores to the last digit) → 2.95 min batched heads | 2026-10-01 |
| T04 · Pre-training cache: **dropped** by the user's decision (an exact cache hits only fold 1 in most studies; a per-fold reseed would change the numbers) | 2026-09-30 |
| Experimentation log (E01–E28) | `0849c4c` and later |
| T00 · Imputation defaults from E30: `embedding_normalized`, decode 450, `cosine` ([ADR 0013](adr/0013-imputation-defaults-from-the-confirmatory-study.md)); default runs reproduce E30's M cells to the last digit | 2026-10-01 |
