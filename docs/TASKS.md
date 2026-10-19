# Tasks

The research and engineering tracker for the imputation effort: what to do next, what waits on a
decision, and what was done. Code-level bugs and pendencies stay in [BACKLOG.md](BACKLOG.md);
each experiment's contract is its ticket under `docs/tickets/`, and its result goes in the
[experimentation log](EXPERIMENTATION_LOG.md). Status is one of **decision**, **next**,
**backlog**, **in progress** or **done**; update a task when its status changes and link the
ticket or commit that moved it.

Created 2026-09-30 from the proposals after studies 04 and 05 (E27, E28).

## Waiting on a decision

T21 to T27 come from [plans/2026-10-17-experiment-improvements.md](plans/2026-10-17-experiment-improvements.md),
which turns the 2026-10-16 review into workstreams. **Decided 2026-10-16, about 22:45 GMT-3:** the
user agreed with every recommendation in its section 7 (D1–D21) and chose path B (E43 first, then
the full E44). T00 was decided on 2026-10-01: the
imputation default is now M, as recorded in
[ADR 0013](adr/0013-imputation-defaults-from-the-confirmatory-study.md).

| ID | Task | Evidence | Status |
|---|---|---|---|
| T00 | Make `embedding_normalized` the imputation default, and choose its decode setting (150, 450, 450 with patience 50, or per variant) | E25: never worse than the current objective, better than no pre-training on 3 of 4 variants. E26: with decode 450, never worse than either half. E27: patience 50 keeps the scores on 3 of 4 tables at a fraction of the time, −0.006 on pendigits. E28: batch 1024 pays only on pendigits. **E30 (all 21 variants): M better than A on 9, worse on none; P worse than A on 2, so by the pre-stated rule the recommendation is M** (normalised pre-training + decode 450), at 1.65x A's time | done |
| T21 | **Reporting** (plan §2a). One shared report helper: the bar from `summary.OUTSIDE_THE_BAR`, K or H chosen by the `decode_checkpoint` tag, verdict tallies against the bar of 5, missForest, the best of six and the per-type bar, a cross-table Wilcoxon with Holm, and the test named beside every count. Then the thesis table on 101/202/303 (primary), 42/7/13 (labelled pre-ADR-0019) and 404/505/606 (E40). Proposed ADR 0020. Subsumes part of T16. | Review finding 1 and threat 5: counts change with the rule and the opponent (19 of 21 by means, 16/3/2 by verdict at today's default); three bar definitions coexist | done 2026-10-18: helper and thesis table (`4c95980`, `8c210de`; tuned and tuned-strong columns and the D4 count `97f148c`), ADR 0020 |
| T22 | **New baselines** (plan §2b, E41). condexp, inductive SoftImpute and ICE with the sequential transform, in a new module `src/training/linear_baselines.py`, scored offline on the decoder's cells. The same pass recomputes per-fold lgbm and missForest where the fold CSVs lack them. Then two baseline tiers, all logged (the user's design, 2026-10-16): the bar of 5 stays the weak tier; missForest and the new arms form the strong tier with `baseline/best_strong` and `gap_to_best_strong_baseline`, through a live family and a backfill (proposed ADR 0021). | Review findings 2–3: the probe found them setting the bar on about 7 induced variants by means; a candidate flip on kc2_20nan; the inherited missForest replay is invalid for linear learners | done: E41 (`8c210de`); ADR 0021 landed 2026-10-17 (`f21ee33`) and backfilled 2026-10-18 (1,176 runs) |
| T23 | **Fairness** (plan §2c, E42 and E43). An equal, validation-only tuning budget for the baselines, offline (proposed ADR 0022, amending ADR 0007 d5). `DECODE_CHECKPOINT=masked` with H and M readouts under the K default: protected, needs code authorization (proposed ADR 0024, amending ADR 0015). | Threat 1: baselines have no budget, and one hgb recipe moves the bar. Threat 3: K uses the validation gaps' truth; at 80nan the model beats the mean/mode fill on 0 of 3 variants under H and on 2 of 3 under K | done 2026-10-18: E43 (H is the deployable headline; ADR 0024) and E42 (tuned bar better on 17/21 and 9/12; 4 verdicts change, all kr-vs-kp but one; a tuned missForest moves 2 strong-tier verdicts of 48; ADR 0022) |
| T24 | **Metrics** (plan §2d). The per-kind panel and the per-type bar (items 7 and 7b): balanced accuracy, signed bias and per-column ratios, computed offline from the fold CSVs, the server's fold ledgers (all five folds are kept) and a recompute for the baselines' fills. No `score_cells` change for now (proposed ADRs 0023 and 0025). | Threat 4: against missForest the verdicts go from 14/6/1 to 9/6/6 when MAE replaces RMSE; heavy tails decide kc2 and biodeg | in progress: the per-type bar is in the helper (ADR 0020); the offline panel read on E40 (`4915515`) and E44 (`b7334c6`, with the twin split and context-free rows); ADR 0025 (metric suite, B) still to write |
| T25 | **Confirmation** (plan §2e, E44). The frozen post-E40 configuration on the 15 variants no decision used, on seeds 707/808/909, with the H, M and KC readouts. 45 cells, about 8–9 h on 4 queues. Proposed ADR 0026 (the confirmation protocol and its splits). | Threat 2: every default was chosen on the same 21 variants and one MCAR draw each; 14 variants on disk have no imputation run, and electricity_40nan has 2 failed ones | done 2026-10-17 (E44 mixed: 9 wins, 3 losses of 15; electricity loses; panel, twins and context-free rows read) |
| T26 | **MNAR** (plan §2f, E45). A smoke family with probit value-dependent missingness on biodeg, kc2 and vehicle at 60nan, arms G and Mx with MCAR controls (36 cells), and the t2 readout (the MCAR half from E31–E36, the MNAR half from E45). | Review items 8–9; todos t2, t5 and t7 | done 2026-10-17: E45 smoke (go) and E46 full family (supported: lead survives MNAR on 4 of 4 new pairs; lost only at biodeg's slope 2.0) |

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

### T03 · Architecture defaults — done (E37, E38; [ADR 0018](adr/0018-bigger-encoder-on-the-large-tables.md): bigger encoder on pendigits_20nan and letter_20nan)

**Screened 2026-10-06:** 4 layers + feed-forward 256 + a final LayerNorm (ALL) is better on 3 of 5
variants and worse on none (credit-g_20nan a near miss), at 1.82x the time; it advances to a
21-variant confirmation, which needs the user's go-ahead. F256 alone also advances (better on
pendigits only); depth alone hurts credit-g_20nan; the final LayerNorm alone hurts pendigits. **The user chose instead (2026-10-06)** to test whether the bigger encoder helps the large
tables, with a size-dependent default if so: ticket [02](tickets/imputation-architecture/02-bigger-encoder-by-table-size.md), E38. **Primary read 2026-10-06:** supported, ALL better on pendigits (−3.8%) and letter (−5.6%), no difference on electricity (+0.3%) at 1.77x its cost; **the size-dependent default waits on the user's decision.** **Completed 2026-10-07 (63 of 63):** medium tables −0.2% (neutral), small tables +1.9% (worse on 4 of 10), so the size dependence holds. **Decided 2026-10-07:** adopted on pendigits_20nan and letter_20nan by promoted files (ADR 0018); electricity, with no gain, keeps today's encoder.

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

### T19 · Re-tune the promoted files under today's defaults — done (ADR 0019; ticket [imputation-retune/01](tickets/imputation-retune/01-retune-promoted-files.md), E39)

The five promoted files keep sampled values from ADR 0005's studies of 2026-09-11, chosen under the
old configuration and the masked objective. The user chose to re-tune them on 2026-10-15. The
study runs five reduced studies ranked by the validation gaps, then compares each winner with the
current file and with no file at all, on seeds 101/202/303. It runs overnight on 2026-10-16.

**Done 2026-10-16 (35 of 35 jobs).** The current files are worse than having no file on three
variants: kr-vs-kp_40nan −4.8%, credit-g_40nan −4.7% and spambase_40nan −0.5%. The re-tuned
winners beat the defaults on none. The rule fixed in advance says:

- promote the winner on credit-g_40nan and kr-vs-kp_40nan, where it equals no file;
- remove spambase_40nan's file;
- keep the files of spambase_20nan and kr-vs-kp_20nan.

**Decided 2026-10-16:** the three files that lose to the defaults are removed
([ADR 0019](adr/0019-remove-the-stale-promoted-imputation-files.md)). The other two stay, and no
winner is promoted.

### T20 · Calibration at 40% missing and above — adopted 2026-10-17, ADR 0027 ([proposal](proposals/calibration-as-imputation-output.md); ticket [imputation-calibration/01](tickets/imputation-calibration/01-calibration-at-higher-missingness.md), E40)

E34 and E35 found the calibrated readout never worse than K at 40nan and above, and better in both
seed sets on 6 variants. It failed at 20nan. The user chose on 2026-10-16 to confirm the rule
"calibrate at 40% missing or more" on fresh seeds, 404, 505 and 606. Adopting it would need a code
change and an ADR.

**Done 2026-10-16 (36 of 36): supported.** KC is better on all 6 named variants (7 of 12) and K on
none. The gain grows with missingness, from −0.0009 at 40nan to −0.0059 at 80nan, and no 40nan
variant earns a verdict.

**Parked 2026-10-16** at the user's request, to try other ideas first. The idea, the evidence from
three seed sets and what adopting it takes are in
[proposals/calibration-as-imputation-output.md](proposals/calibration-as-imputation-output.md).

**Revived 2026-10-17 by the user,** after E44's confirmation. There KC was better than K on 8 of 15
variants and worse on none, and that includes pendigits_40nan and letter_40nan. An ADR (0027,
amending ADR 0015) makes the calibration the imputation default at 40% missing and above. A flag
and a tag keep today's output reachable.

**Implemented and landed 2026-10-17** (`--decode_calibration`, default `auto` for imputation;
`17121a5`). A default run equals E40's KC to the last digit on credit-g_60nan, and a 20nan run is
unchanged. Its tag was backfilled on 2026-10-18 (2,680 runs stamped `none`).

- **Follow-up, E47 done 2026-10-19:** the deployable analogue is a calibration fitted on the masked
  validation cells and read with H
  ([imputation-calibration/02](tickets/imputation-calibration/02-deployable-calibration.md)).
  - It is supported by the rule but small: at most 0.4% of H, keeping 3–29% of the oracle's gain.
  - **Closed by the user (2026-10-19):** stop. KC is an oracle readout, and HC_m's small gain is
    stated beside it.

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
- **T13 · An EMA target encoder and a predictor for pre-training** (data2vec, BYOL; SMART, wiki
  `references/smart-paper`). These are the anti-collapse guards pre-training still lacks (ADR 0011,
  considered options). TRIDENT's target is the clean cell's input embedding from the same embedder,
  gradient stopped, with no transformer layer (corrected 2026-10-19 by T28's report); SMART adds an EMA target
  encoder (decay 0.996) and an embedding decoder. E04, E05 and E24 suggest the pre-training
  collapses. **A future task, by the user's decision (2026-10-17):** plan §2h; after T28.
- **T28 · Why pre-training adds little to direct imputation, and what the ingested literature
  offers** (the user's request, 2026-10-17). This is an investigation, not a run.
  - **The fork's evidence:**
    - E04: the pre-training loss drop is norm-matching, and the encoder is worse than a
      per-column constant;
    - E05: 300 pre-training epochs equal 2;
    - E24: pre-training never helps against none, and hurts pendigits;
    - E25/E26: only the normalised objective helps, on 2 of 4 variants;
    - E36: showing gaps as `[MASK]` in pre-training gains nothing.
  - **Questions:**
    - Is it collapse, a target mismatch (embedding targets against value-space scoring), or a
      decode stage strong enough to wash the encoder out?
    - What do the wiki's references say? Among them: SMART, the JEPA and masked-embedding concept
      pages, MaCoDE's bin targets, ReMasker, and the distillation review of 2026-10-09.
  - **Output:** a short report with diagnosable hypotheses and the cheapest test for each. T13
    (EMA) is one candidate.
  - **Report drafted 2026-10-19:**
    [reports/2026-10-19-t28-why-pretraining-adds-little.md](reports/2026-10-19-t28-why-pretraining-adds-little.md).
    - Six hypotheses. The leading one is that the decode stage repeats the pretext and washes the
      encoder out.
    - The order is D1 (a free store read of the decode curves, pre-trained against not), then D2
      (collapse and linear probes under the normalised objective), then D3 (frozen encoder against
      random).
    - T13 fits only if D2 shows collapse and a frozen pre-trained encoder beats a random one.

- **T29 · Why the boosters win on electricity** (the user's decision on E44, 2026-10-17). electricity
  is a stated limit of the thesis. hgb/lgbm set the bar there, and the model loses at 20, 40 and
  60nan and ties at 80nan (E44, E35). This is a bounded investigation, later.
  - **Leads:** electricity's 8 columns with 45,312 rows; the bar's per-column wins; context-free
    rows (20.9% of cells at 80nan, no verdict changes without them).
  - **Output:** a short report, then at most one pre-registered test.

- **T14 · A steps-aware batch rule** (at least ~8 steps an epoch) instead of one batch for every
  table (E28).
- **T15 · MAE-style architecture** (ADR 0004 Design C, handoff item 5): the most expensive; only if
  T02–T03 fall short.

### Later baselines and classification (plan 2026-10-17)

- **T27 · MaCoDE, TabPFN v2 and classification step zero** (plan §2g; review items 10–12).
  **MaCoDE done 2026-10-19 (E48,** [imputation-protocol/07](tickets/imputation-protocol/07-macode.md)**):** TRIDENT beats it on 20/21 and 14/15, never worse; an external deep baseline only. TabPFN v2 and step zero not started.
  - MaCoDE vendored behind an inductive wrapper, with its bins bug fixed first: 10–14 h of code,
    3–5 GPU-h off-store.
  - A column-wise TabPFN v2 in a separate venv pinned to `ModelVersion.V2`: 5–8 h of code and
    6–10 GPU-h.
  - Classification step zero: a native-NaN GBDT bar, impute-plus-indicator arms and an
    `EPOCHS_PRE=0` arm, with `--cv_folds 10`.

  In GPU gaps only, after T21–T25; D21 is decided after E44.

### Process

- **T16 · One configuration-driven study report** instead of a copied report script per study.
  T21's shared helper is the first step.
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
