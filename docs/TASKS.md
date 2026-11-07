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
| T21 | **Reporting** (plan §2a). One shared report helper: the bar from `summary.OUTSIDE_THE_BAR`, K or H chosen by the `decode_checkpoint` tag, verdict tallies against the bar of 5, missForest, the best of six and the per-type bar, a cross-table Wilcoxon with Holm, and the test named beside every count. Then the thesis table on 101/202/303 (primary), 42/7/13 (labelled pre-ADR-0019) and 404/505/606 (E40). Proposed ADR 0020. Subsumes part of T16. | Review finding 1 and threat 5: counts change with the rule and the opponent (19 of 21 by means, 16/3/2 by verdict at today's default); three bar definitions coexist | done 2026-10-18: helper and thesis table (`4c95980`, `8c210de`; tuned and tuned-strong columns and the D4 count `97f148c`), ADR 0020. E51 done 2026-10-20 ([imputation-protocol/08](tickets/imputation-protocol/08-default-refresh.md)): a third seed set, 1001/2002/3003, for the nine 20nan rows; holds (K − bar 7/1/1, no reversal; K − strong 6/3/0) |
| T22 | **New baselines** (plan §2b, E41). condexp, inductive SoftImpute and ICE with the sequential transform, in a new module `src/training/linear_baselines.py`, scored offline on the decoder's cells. The same pass recomputes per-fold lgbm and missForest where the fold CSVs lack them. Then two baseline tiers, all logged (the user's design, 2026-10-16): the bar of 5 stays the weak tier; missForest and the new arms form the strong tier with `baseline/best_strong` and `gap_to_best_strong_baseline`, through a live family and a backfill (proposed ADR 0021). | Review findings 2–3: the probe found them setting the bar on about 7 induced variants by means; a candidate flip on kc2_20nan; the inherited missForest replay is invalid for linear learners | done: E41 (`8c210de`); ADR 0021 landed 2026-10-17 (`f21ee33`) and backfilled 2026-10-18 (1,176 runs) |
| T23 | **Fairness** (plan §2c, E42 and E43). An equal, validation-only tuning budget for the baselines, offline (proposed ADR 0022, amending ADR 0007 d5). `DECODE_CHECKPOINT=masked` with H and M readouts under the K default: protected, needs code authorization (proposed ADR 0024, amending ADR 0015). | Threat 1: baselines have no budget, and one hgb recipe moves the bar. Threat 3: K uses the validation gaps' truth; at 80nan the model beats the mean/mode fill on 0 of 3 variants under H and on 2 of 3 under K | done 2026-10-18: E43 (H is the deployable headline; ADR 0024) and E42 (tuned bar better on 17/21 and 9/12; 4 verdicts change, all kr-vs-kp but one; a tuned missForest moves 2 strong-tier verdicts of 48; ADR 0022) |
| T24 | **Metrics** (plan §2d). The per-kind panel and the per-type bar (items 7 and 7b): balanced accuracy, signed bias and per-column ratios, computed offline from the fold CSVs, the server's fold ledgers (all five folds are kept) and a recompute for the baselines' fills. No `score_cells` change for now (proposed ADRs 0023 and 0025). | Threat 4: against missForest the verdicts go from 14/6/1 to 9/6/6 when MAE replaces RMSE; heavy tails decide kc2 and biodeg | done 2026-10-19: the per-type bar in the helper (ADR 0020); the offline panel read on E40 (`4915515`), E44 (`b7334c6`) and E45/E46's signed bias (`5f1da3d`); ADR 0025 |
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

**E59 (2026-10-22, [imputation-protocol/14](tickets/imputation-protocol/14-large-encoder.md)):** ALL on pendigits and letter at 40–80nan is mixed on K (better 3/6, worse on letter_80nan by +0.4%); on KC 4/0. The gain fades with the rate and is gone at 80nan. **Decided by the user (2026-10-22):** ADR 0018 stays at 20nan; the bigger encoder at 40–60nan is a future possibility (T32).

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

- **T06 · `torch.compile` or CUDA graphs** on the decode loop. **Closed by the user (2026-10-23):** the
  batched lookup stays off by default, so imputation results stay comparable; see the measurements below. Likely 1.5–3x on a launch-bound
  loop, but the per-batch count of hidden cells varies, which breaks CUDA graphs; numerics change
  slightly. Related: [BACKLOG P2](BACKLOG.md) (fused attention).
  - **Measured 2026-10-23** (`scripts/experiments/compile_bench.py`, kr-vs-kp_40nan, seed 404, five
    folds, EPOCHS_PRE 30 and EPOCHS_DECODE 60, alone on the GPU). These are measurements, not a
    pre-registered study.
    - **Where the time goes** (cProfile): the backward pass (about 4.5 ms a batch), the per-column
      categorical embeddings (36 lookups a forward on kr-vs-kp), and the decoder's `nonzero` host
      syncs. The run is launch-bound.
    - **Compiling the transformer only:** about +10% a steady epoch (decode 10.5 → 12 it/s), and
      about 40 s of compilation per cell. Over a whole cell, about break-even.
    - **Compiling the embedder and the transformer:** about +23% a steady epoch (decode 10.5 → 13
      it/s, pre-training 13.5 → 16.6 it/s), and 40–60 s of compilation per cell.
      - Break-even on a medium cell (about 5.5 min alone); a likely net gain only on the large
        tables.
    - **Not rounding-only:** compiled dropout draws from its own generator. K moved from 0.770 to
      0.761 and 0.766 on this short run, so adoption would need a flag and an equivalence study.
    - **triton 3.1 needs `setuptools`,** which the lock does not have; the benchmark used
      `uv run --with setuptools`.
    - **The user chose the exact eager levers (2026-10-23).**
  - **`--embedding_lookup batched`** (`EMBEDDING_LOOKUP`, off by default; tag `embedding_lookup`;
    backfill script `scripts/backfill_embedding_lookup_tag.py`, not applied).
    - The categorical tables are read in one gather. The forward values are the same, and the
      gradients differ in float rounding only (at most 2e-7 relative, microbenchmark).
    - **Measured** (`compile_bench.py lookup`, same short runs):
      - kr-vs-kp_40nan: pre-training +21%, decode +12%, run −11%;
      - credit-g_40nan: −2%;
      - biodeg_60nan (no categorical column): unchanged, with identical scores.
    - On kr-vs-kp the rounding difference grows over training into a different trajectory (K
      0.7701 → 0.7649 on the short run), so adoption needs an equivalence reading like ADR 0012's.
  - **Host syncs:** the decoder's `nonzero` is about 4% of a run. It cannot be removed exactly.
  - **Where that leaves T06:**
    - The exact levers save little outside kr-vs-kp.
    - The numerical tables are bound by the transformer's and decoder's many small kernels. Only
      compilation fuses those, and it changes the dropout draws.
    - **Decided (2026-10-23, the user's "1"):** keep the flag off and close T06. Compilation, with
      the flag, stays an option for new work such as T27.

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
  - **Started 2026-10-22, by the user's go ("vamos no 3 agora, dispare").** E61
    ([imputation-protocol/16](tickets/imputation-protocol/16-categorical-deficit.md)) reads per
    column where K loses to TabPFN on kr-vs-kp and credit-g. A weighting follows only if the read
    finds a target. Under MCAR, a plain column mean equals today's pooled term.
  - **E61 done 2026-10-22: no target.** No column is stuck at the marginal on any of the 6
    variants. K learns every column TabPFN learns, a few points less (gain correlation +0.76 to
    +0.94). **The weighting half closes;** the evaluation-mask half stays open.

### Specific follow-ups

- **T11 · pendigits: batch 1024 at LR x4 with a longer patience** (100, or a 600-epoch ceiling):
  its best arm (E28, 0.323) and a validation loss still falling near epoch 450 (E27).
  **E101 done 2026-10-30 ([imputation-protocol/38](tickets/imputation-protocol/38-large-batch.md)): supported.** BATCH 1024, LR x4 and EPOCHS_DECODE 600 better on pendigits 20/40/60nan (−5.4%, −4.3%, −1.2%), equivalent at 80nan; a fresh-seed confirmation before any ADR. **E102 (fresh seeds, [39](tickets/imputation-protocol/39-large-batch-confirmation.md)): not replicated by its rule** — the 20–60nan gains repeat (−5.8%, −4.2%, −1.2%), but 80nan is worse (+0.42%); **E103 ([40](tickets/imputation-protocol/40-large-batch-large-tables.md)): letter mixed (−2.9%/−1.5% at 20/40nan, worse at 80nan), electricity refuted.** No default change; the user decides whether a table- and rate-limited rule is worth a new question. **E104 replicated it on pendigits and letter at 20–40nan (4 of 4); ADR 0036 accepted 2026-10-31: those four variants train with P600. T11 closed.**
- **T12 · A lower `LR_PRE` with the normalised objective:** every full-profile winner chose 2–34x
  below the default (E21).
  - **Skipped by the user's decision (2026-10-21),** on my recommendation. T28's decision table pointed here (E53's `BATCH 64` moved credit-g's probe), but E52 and E53 show the decode stage rebuilding the encoder from scratch, so a better pre-trained one would be overwritten too.
- **T13 · An EMA target encoder and a predictor for pre-training** (data2vec, BYOL; SMART, wiki
  `references/smart-paper`). These are the anti-collapse guards pre-training still lacks (ADR 0011,
  considered options). TRIDENT's target is the clean cell's input embedding from the same embedder,
  gradient stopped, with no transformer layer (corrected 2026-10-19 by T28's report); SMART adds an EMA target
  encoder (decay 0.996) and an embedding decoder. E04, E05 and E24 suggest the pre-training
  collapses. **A future task, by the user's decision (2026-10-17):** plan §2h; after T28.
  **Done 2026-11-03 for classification (E112 mixed, E114 partial; [classification-step-zero/12](tickets/classification-step-zero/12-ema-target.md), [13](tickets/classification-step-zero/13-ema-target-confirmation.md)):** `PRETRAIN_OBJECTIVE embedding_ema` beats no pre-training by under a point at 40–60nan, below the SESOI; closed, the flag stays off. Imputation was left out on T28's diagnosis (washout, not collapse).
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
    - **D1 done 2026-10-19 (E49,** [imputation-pretraining/08](tickets/imputation-pretraining/08-decode-curves.md)**):**
      - washout on pendigits: the 450-epoch decode erases a +58% head start by about epoch 185;
      - real transfer on kr-vs-kp (+2.5% at the minima);
      - credit-g mixed.
      - Next: a frozen-encoder or lower-encoder-LR decode arm on pendigits, or D3.
    - **E50 done 2026-10-20** ([imputation-pretraining/09](tickets/imputation-pretraining/09-encoder-levers.md)): refuted.
      - Freezing the encoder for 100 decode epochs is worse on pendigits (+3.4%); a tenth of the
        encoder learning rate is worse on both variants (+11.6%, +4.4%).
      - Without pre-training, pendigits is 1.2% worse; kr-vs-kp_40nan is equivalent.
      - The levers are closed. Next: D2 and D3.
    - **D2 and D3 done 2026-10-20 (E53, E52;** [imputation-pretraining/11](tickets/imputation-pretraining/11-collapse-probes.md), [10](tickets/imputation-pretraining/10-frozen-encoder.md)**):**
      - no collapse under N; the representation is real on pendigits and kr-vs-kp (frozen,
        −53% and −16% against random; probes +36% and +7.5%), and nothing on credit-g;
      - the decode stage keeps 40–60% of it (CKA) and builds an encoder as good from random
        (C0 − G within 1.7%): washout (H3).
      - By the decision table: T12 first, with a `BATCH 64` arm, for the user to decide.

- **T29 · Why the boosters win on electricity** (the user's decision on E44, 2026-10-17). electricity
  is a stated limit of the thesis. hgb/lgbm set the bar there, and the model loses at 20, 40 and
  60nan and ties at 80nan (E44, E35). This is a bounded investigation, later.
  - **Leads:** electricity's 8 columns with 45,312 rows; the bar's per-column wins; context-free
    rows (20.9% of cells at 80nan, no verdict changes without them).
  - **Output:** a short report, then at most one pre-registered test.
  - **Report written 2026-10-20** ([2026-10-20-t29-electricity](reports/2026-10-20-t29-electricity.md)):
    - the limit is against hgb and lgbm only;
    - the model trails them most on `day` and `nswdemand`;
    - `date` is a 933-level key.
  - **E54 done 2026-10-20** ([imputation-protocol/09](tickets/imputation-protocol/09-electricity.md)): supported.
    - With the row's `date`, K − bar is worse at every rate; without it, never worse (better at
      20 and 80nan).
    - The boosters look values up by `date`, a 933-level key.
    - **T29 closed.** A model-side encoding of `date` was set aside by the user (2026-10-21): the limit stays stated with its mechanism.

- **T30 · A MAR smoke** (the user's go to record it, 2026-10-21). **Running since 2026-10-22 as E63**
  ([imputation-protocol/18](tickets/imputation-protocol/18-mar-smoke.md)), against matched MCAR
  controls: the same drivers and per-column counts (about 42% of feature cells).
  - **Done 2026-10-22: loses ground.** (H − strong), MAR minus control, is worse on biodeg and
    kr-vs-kp and better on kc2; (H − bar) is never worse. **Proposed, for the user:** a fuller MAR
    family (the 20nan rung, a slope dial, spambase and credit-g), with an arm whose decode masks
    follow the drivers, to test the mask-mismatch hypothesis.
  - **The user chose both (2026-10-22).**
    - **E64** ([imputation-protocol/19](tickets/imputation-protocol/19-mar-family.md)), done
      2026-10-23: **refuted.** (H − strong), MAR minus control, is worse only on biodeg_60nan (1 of
      12), and the smoke's "loses ground" does not survive. (H − bar) is worse on the three small
      numerical tables at 60nan.
    - **E65** ([imputation-protocol/20](tickets/imputation-protocol/20-mar-propensity.md)), done
      2026-10-23: **refuted.** The propensity-mask arm behind `--decode_mask_weighting` is never
      better and is worse on 6 of 12 under MAR and 5 of 12 under the control. The flag stays, off
      by default; its tag was backfilled (`uniform`, 2,967 runs) on 2026-10-23 after a store backup.
  - **T30 closed (2026-10-23).** Under MAR, TRIDENT's standing against the strong tier holds on
    six tables at 20 and 60nan, with biodeg_60nan the exception. The bar gains at 60nan on the
    small numerical tables.
  - **The gap:** every variant is MCAR (`generate_splits.py`) or MNAR (E45/E46,
    `generate_mnar.py`). Nothing tests missingness that depends on *other* observed columns.
  - **Generator:** `datasets/generate_mar.py`. It draws from the `_00nan` table at the MCAR
    siblings' counts, as `generate_mnar.py` does, never on top of a `_XXnan` variant.
    - About 30% of the feature columns stay fully observed as drivers.
    - In every other column the cells are drawn without replacement, with weights from a logistic on
      the standardised drivers.
    - Categorical tables can enter, since the drivers can be numerical.
  - **Smoke, in E45's shape:** biodeg, kc2, vehicle and kr-vs-kp at 60nan, seeds 42/7/13, arm G.
    Each MAR variant is read against its MCAR control.
  - **Opponents:** the bar, the strong tier with TabPFN (ADR 0028, offline pass) and G's H.
  - **Measure:** (H − opponent) MAR minus MCAR.
  - **Caveat:** keeping drivers fully observed changes the per-column missing share against the
    MCAR sibling; the ticket must say which counts it matches.
  - **Cost:** the generator and a runner; a few GPU hours.
- **T31 · Fresh MCAR draws: is a result tied to one missingness pattern?** (the user's go to record
  it, 2026-10-21). **Running since 2026-10-22 as E62**
  ([imputation-protocol/17](tickets/imputation-protocol/17-mcar-draws.md)): 36 cells, with draw 1
  rerun beside draws 2 and 3; biodeg_80nan at 42/7/13 for the first time.
  - **Done 2026-10-22: holds.** No reversal on K − bar or K − strong across the three draws of the
    four variants; two drifts are named in the ticket. The draw moves a contrast about as much as
    the fold-and-seed interval (biodeg_80nan about twice), so the thesis's intervals are
    conditional on the draw. **T31 closed.**
  - **The gap:** each MCAR rate is a single draw from `_00nan` (`inject_nans`, seed 42 + rate). The
    seeds vary the folds and the model, never which cells are missing. The review of 2026-10-16
    named it ("one MCAR draw each").
    - The rates are drawn independently, not nested: 39.2% of biodeg_20nan's gaps are gaps at 40nan.
  - **Design:** new MCAR draws from `_00nan` with other seeds, written beside the originals
    (`<base>_<pp>nan_d2.csv`, …), on a subset spanning sizes and kinds.
    - pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan, biodeg_80nan;
    - two extra draws each, at seeds 42/7/13.
    - Today's default runs on them, with the bar, the strong tier and TabPFN.
  - **Reading:** each draw's calls against the original's, with reversals listed (E51's rule).
  - **Cost:** about 24 cells and a TabPFN pass; one GPU night.
  - **Related:** T30, which would reuse the same generator scaffolding.

- **T32 · The bigger encoder at 40–60nan on the large tables** (a future possibility, by the
  user's decision on E59, 2026-10-22: "por enquanto nao vale a pena").
  - **E59:** on pendigits and letter, about −4% on K at 40nan (−5% to −6% on KC), less at 60nan,
    and nothing at 80nan.
  - **Cost:** about 2× the training time (1.98× pendigits, 2.05× letter).
  - **To adopt it:** a fresh-seed confirmation, pre-registered, then an ADR amending 0018.

  **E113 done 2026-11-03 ([imputation-protocol/46](tickets/imputation-protocol/46-bigger-encoder-large-tables.md)): supported** on today's defaults, better on 4 of 4 (pendigits −6.0%/−3.1%, letter −5.5%/−1.1% at 40/60nan); **E116 ([47](tickets/imputation-protocol/47-bigger-encoder-confirmation.md)) replicated it on fresh seeds (4 of 4); ADR 0037 accepted 2026-11-03: pendigits and letter at 40–60nan train with ADR 0018's encoder. T32 closed.** **E119 ([49](tickets/imputation-protocol/49-bigger-encoder-mar-mnar.md)) done 2026-11-03: holds under MAR and MNAR (better 4/4 each, worse 0); ADR 0037 extended to all three mechanisms (decision 3).** **E121 ([50](tickets/imputation-protocol/50-bigger-encoder-20nan-mar-mnar.md)) done 2026-11-03: holds at 20nan under MAR and MNAR (better 2/2 each, worse 0); ADR 0018's scope awaits the user.**
- **T33 · A clean `[NULL]`/`[MASK]` ablation in the decode stage** (proposed 2026-10-22, about
  19:25 GMT-3, from the wiki's sync of the fork; accepted by the user; see below).
  - **Accepted by the user (2026-10-22, about 19:40 GMT-3):** "eu concordo com todas as suas ideias".
    It runs after E64 and E65 end, pre-registered in its own ticket first; nothing starts before then.
  - **Unblocked (2026-10-23):** E64 and E65 ended, both refuted ([19](tickets/imputation-protocol/19-mar-family.md), [20](tickets/imputation-protocol/20-mar-propensity.md)).
  - **The gap:** E45/E46's t2 readout, Mx, changes three things at once (the gaps as `[NULL]` in the
    decode stage, the loss checkpoint and column-wise scoring), so it does not isolate the split
    ([imputation-protocol/06](tickets/imputation-protocol/06-mnar-family.md), "For the user").
  - **The signal worth checking:** on biodeg's dial, Mx's deficit to G's H shrinks against MCAR at
    every slope (−0.014 to −0.026), and at slope 2.0 Mx beats G (−0.006), the only variant of E45 and
    E46 where the untied arm wins.
  - **What it would be:** the ablation the ticket names, `[NULL]` gaps alone, without column-wise
    scoring and the loss checkpoint, on the dial and its MCAR control and on E46's four new pairs.
    The design and the reading are the ticket's job.
  - **Why it matters:** the user takes the `[NULL]`/`[MASK]` split as TRIDENT's main claim (wiki
    `projects/trident/trident`, Positioning, 2026-10-15), and this is its only imputation-side test.
    The classification-side ablation is the wiki's todo t2.
  - **Cost, for scale:** E46's 66 cells took 3 h 33 min on four queues.
  - **Running since 2026-10-23 as E67** ([imputation-protocol/22](tickets/imputation-protocol/22-null-gaps.md)): arm N, `DECODE_GAP_TOKEN null` alone, on 8 MNAR pairs against E45/E46's G.
  - **Closed 2026-10-23 by E67: refuted.** The interaction on H is better on 1 of 8 pairs and worse
    on 2. On biodeg's dial, `[NULL]` gaps alone cost more as the tilt steepens (+0.030 at slope
    0.25 to +0.099 at 2.0, against +0.058 under MCAR), so Mx's signal came from its other two
    changes. ADR 0014 stands; the split's test moves to the classification side (wiki t2).

- **T34 · The tuned bar on the MAR folds** (proposed 2026-10-22, about 19:25 GMT-3, from the
  wiki's sync; accepted by the user; see below).
  - **Accepted by the user (2026-10-22, about 19:40 GMT-3):** "eu concordo com todas as suas ideias".
    It runs after E64 and E65 end, pre-registered in its own ticket first; nothing starts before then.
  - **Unblocked (2026-10-23):** E64 and E65 ended, both refuted ([19](tickets/imputation-protocol/19-mar-family.md), [20](tickets/imputation-protocol/20-mar-propensity.md)).
  - **The gap:** ADR 0022 decision 5 says every new study that makes a claim against the bar runs
    E42's driver on its own folds, and D4 counts a win only against both bars. E63 and E64 read
    against the untuned bar only, so "the standing against the bar holds under MAR" has no D4
    reading yet.
  - **What it would be:** `scripts/experiments/baseline_budget_run.py` on E63's and E64's folds,
    offline and on the CPU, with nothing written to the store, after E64 and E65 end so that it does
    not compete with them. The MNAR family (E45/E46) predates ADR 0022 and could be added.
  - **Cost, for scale:** E44's pass took about 4 hours on 4 CPU lanes.
  - **Running since 2026-10-23 as E68** ([imputation-protocol/23](tickets/imputation-protocol/23-mar-tuned-bar.md)): E42's driver on E63's and E64's folds, 96 pairs, offline CPU.
  - **Done 2026-10-23 by E68: mixed** (worse on 3 of 12, better on 2). The model is never behind
    the tuned bar on E64's family under MAR; at 60nan on biodeg, kc2 and vehicle its lead over both
    bars shrinks. **The user accepted that wording (2026-10-24, "1 ok"); T34 closed.**

- **T35 · An MNAR-aware imputer on the MNAR family** (proposed 2026-10-22, about 19:25 GMT-3,
  from the wiki's sync; accepted by the user; see below).
  - **Accepted by the user (2026-10-22, about 19:40 GMT-3):** "eu concordo com todas as suas ideias".
    It runs after E64 and E65 end, pre-registered in its own ticket first; nothing starts before then.
  - **Unblocked (2026-10-23):** E64 and E65 ended, both refuted ([19](tickets/imputation-protocol/19-mar-family.md), [20](tickets/imputation-protocol/20-mar-propensity.md)).
  - **The gap:** E45/E46's signed-bias panel shows every imputer under-predicting the hidden values
    and the model's bias moving least of 10 methods on 9 of 10 pairs. None of the 10 models the
    mechanism, so the claim "an imputer that models the mechanism shrinks the bias" (the wiki's H1d,
    `projects/trident/thesis-goals`) is untested.
  - **Candidates, from the wiki** (`projects/trident/findings-2026-10-16-baselines-methodology`, "Not
    quick"; `references/not-miwae-paper`, `references/nimiwae-paper`): not-MIWAE (TensorFlow code
    with no license, so a clean-room rewrite), then NIMIWAE (a GPL-3 R package driving PyTorch 1.4 through
    `reticulate`, CUDA required). Either would sit behind an inductive wrapper, as MaCoDE does (E48).
  - **What it would read:** the signed bias and `impute_score` on E45/E46's ten MNAR files and their
    controls, offline, beside the existing panel (`metric_panel.py --baselines`).
  - **Running from 2026-10-24 as E72** ([imputation-protocol/25](tickets/imputation-protocol/25-not-miwae.md)): clean-room MIWAE and not-MIWAE, on the user's go ("sim aprovo", 2026-10-23, 23:44 GMT-3).
  - **Done 2026-10-24 by E72: mixed.** The interaction is better on 3 of 10 and worse on none. On
    vehicle not-MIWAE matches the model's error and its MNAR bias shift is the smallest of 12
    methods; on biodeg and spambase its fills fail (worse than the mean fill), so those pairs do not
    test H1d. A better-conditioned fill would be a new pre-registered study; the user decides.
  - **Running from 2026-10-24 as E73** ([imputation-protocol/26](tickets/imputation-protocol/26-notmiwae-fill.md)), on the user's go: the decoder's weighted means as the fill, on E72's six failed pairs.
  - **Done 2026-10-24 by E73:** the count says supported (3 of 6, none worse), but two of the three
    are flagged pairs where the fill is not below the mean fill. The mean fill fixes E72's failure;
    both VAEs sit near the mean fill on biodeg and spambase. H1d holds where the VAE imputes
    competently (vehicle, biodeg_probit_60nan). T35 closed unless the user wants more.

- **T36 · A test-time re-masking ensemble: an uncertainty signal from the decoder** (proposed
  2026-10-22, about 23:59 GMT-3, from the wiki's claims ledger, `projects/trident/claims`, P1;
  accepted by the user; see below).
  - **Accepted by the user (2026-10-23, about 00:03 GMT-3):** "Aprovo T36 e T37". It runs after
    T33–T35, pre-registered in its own ticket first; nothing starts before then.
  - **Code authorization:** the in-run flag changes `src/`, so the ticket asks for it separately.
  - **The gap:** the imputer gives one fill per cell and no uncertainty. The thesis's clinical
    framing and multiple imputation both need one, and no task covers it.
  - **Why no new training objective is needed:** the decode stage already trains on masks drawn
    uniformly over observed cells (the hypothesis in
    [imputation-protocol/18](tickets/imputation-protocol/18-mar-smoke.md)). A test row with some
    of its observed cells also shown as `[MASK]` is therefore a row the decoder has seen in
    training.
  - **What it would be:** for each test row, R passes. Each pass shows a random share of the row's
    observed cells as `[MASK]`, beside its real gaps (already `[MASK]`, ADR 0014). The pass then:
    - averages the gap fills (numerical cells);
    - takes the majority category or the mean probability (categorical cells);
    - records the spread of the R fills per cell, the uncertainty proxy.
    R, the re-mask share and the categorical rule are the ticket's job.
  - **What it would read:**
    - **(a) Accuracy:** the averaged fill against the single pass, on `impute_score` with the
      verdict rule and the 1% SESOI. Both readouts: K, and H, since an uncertainty claim must
      transfer (ADR 0024).
    - **(b) Uncertainty:** whether the per-cell spread ranks the per-cell errors, for example a rank
      correlation between spread and absolute error, and the coverage of spread-based intervals.
    - **(c) The panel's dispersion (ADR 0025):** averaging shrinks fills, and keeping the spread
      better than TabPFN (14 of 15, E56) is one of the model's few distributional wins. The
      ticket says how much of it the average may spend.
  - **What it needs:** the decoder's weights, which no run has kept.
    - A run saves its model only under `--save_model`, which is off by default
      (`src/training/runner.py`).
    - T28's report found no imputation `.pt` file under `results/` or `mlruns/`.
    - `save_model` pickles the whole object, with no loader in the repository (BACKLOG H3).
    - So either the pass runs inside the run behind a new flag (a change to `src/`, needing the
      user's code authorization), or the cells rerun with a weights-only save and a loader
      (BACKLOG H3 first).
    - Either way the cells are retrained.
  - **Cells:**
    - **Descriptive (as in E56, E57, E60):** E44's confirmation variants.
    - **Feeding a default or an output:** a development seed set first, pre-registered.
  - **Cost, for scale:**
    - E44's 45 cells took about 8–9 h on 4 queues (T25), and the pass needs them retrained.
    - On top of that, R decoder forward passes per test fold.
  - **Attaches to:** the wiki's RQ1, todo t4 and ADR 0025's panel.
  - **Running from 2026-10-24 as E75** ([imputation-protocol/28](tickets/imputation-protocol/28-remask-ensemble.md)), on the user's code authorization ("autorizo o 1, vá", about 18:15 GMT-3): the in-run flag `--score_remask` (off by default), 20 passes at K and H, on E44's 15 variants at 707/808/909.
  - **Done 2026-10-25 by E75: mixed on accuracy.** Averaging is worse on pendigits and letter at
    40/60nan and better at 80nan, where it is shrinkage that calibration already takes. The numerical
    spread ranks the errors on 12 of 15 under K and H (the pre-stated bar) but covers 17–66% at a
    nominal 90%. An opt-in uncertainty output (a ranking signal) is the user's call.
  - **ADR 0030 proposed 2026-10-25** on the user's request ("Podemos propor essa ADR"): the numerical spread as an opt-in per-cell ranking signal up to 60nan (a post-hoc within-column check holds on 9 of 9 there), categorical uncertainty from the single pass's confidence, the fill unchanged. Awaits the user's decision.
  - **E77 done 2026-10-25, refuted** ([imputation-protocol/30](tickets/imputation-protocol/30-remask-conformal.md)): a cross-fitted spread interval is wider than a per-column constant one on 15 of 15 at the same 90% coverage; ADR 0030 stays a rank (amended).
  - **Running from 2026-10-25 as E76** ([imputation-protocol/29](tickets/imputation-protocol/29-remask-development.md)): the within-column reading, pre-registered, on the 21 development variants at 101/202/303.
  - **Done 2026-10-26 by E76: supported.** Within-column signal on 14 of 15 development variants up to 60nan (K and H), none against; ADR 0030 updated.
  - **ADR 0030 accepted by the user (2026-10-26, 22:30 GMT-3, "ok, aceito").** The numerical spread under `--score_remask` (off by default) is the opt-in uncertainty output, a rank up to 60nan; no default changes. T36 closed.

- **T37 · Per-column weights for the K + TabPFN blend** (proposed 2026-10-22, about 23:59
  GMT-3, from the wiki's claims ledger, `projects/trident/claims`, P2; accepted by the user; see
  below).
  - **Accepted by the user (2026-10-23, about 00:03 GMT-3):** "Aprovo T36 e T37". It runs after
    T33–T35, pre-registered in its own ticket first; nothing starts before then.
  - **The gap:**
    - E60's equal-weight blend beats both K and TabPFN on 10 of 15 confirmation variants
      ([imputation-protocol/15](tickets/imputation-protocol/15-tabpfn-blend.md), complementary,
      descriptive).
    - At 80nan on electricity, vehicle and biodeg it lands on TabPFN's level (equivalent).
    - Categorical cells keep the model's category, since the ledgers hold no class probabilities.
    - E61 found the categorical term carries 71–100% of K's gap to TabPFN on kr-vs-kp and
      credit-g, spread over columns K already learns, so T10's per-column weighting has no target.
    - A blend weight is a different lever, and it is untested.
  - **What it would be:** a weight per column, from a small grid fixed in the ticket, never chosen
    on the scored fold's truth. Two routes:
    - **(a) Cross-fitted, offline, from what exists:** the weight for fold k is the one that
      minimises the error on the other four folds' induced cells. It uses the existing fold
      ledgers and TabPFN refits on the same folds, as `tabpfn_blend.py` does. Nothing is written to
      the store or to `src/`, as in E60.
    - **(b) On the validation rows' own gaps:** the signal K and the tuned bar use. The fold ledger
      appears to hold the scored test cells only (`artifacts.write_imputation_preview`; the ticket
      checks), so this would need K's and TabPFN's validation fills: a rerun or a new output.
    - **Categorical columns:** a weight of 0 or 1 per column (pick K's or TabPFN's category, as
      the per-variable hybrid selection does in the wiki's `concepts/hybrid-imputation-selection`),
      unless class probabilities are logged.
  - **What it would read:** weighted blend − equal blend and weighted blend − TabPFN, per variant,
    on `impute_score` with the verdict rule and the 1% SESOI. The model side is K, as in E60.
    - **The question that matters:** does it beat TabPFN where the equal blend only matched it (the
      three 80nan variants) or lost to it? Losses: kr-vs-kp and credit-g from 40nan, on every
      seed set (E55, E58).
  - **Cells:**
    - E44's 15 confirmation variants (as E60);
    - E40's kr-vs-kp and credit-g runs at 40/60/80nan on 404/505/606 (E61's set), where TabPFN
      leads and the categorical deficit sits.
  - **Descriptive or a decision:** descriptive, like E60, unless the user wants a hybrid output.
    That would need its own pre-registered rule on a fresh seed set.
  - **Cost, for scale:** E60's 45 pairs ran in about 3.5 h on one lane, TabPFN refits included.
  - **Attaches to:** the wiki's H1d, todo t4, and its C5/C7 claims (competitive with TabPFN;
    complementary errors).
  - **Since (2026-10-23):** E66 ([21](tickets/imputation-protocol/21-tabpfn-blend-replication.md)) replicated the equal-weight blend on E40's tables and seeds at the rule's floor (3 of 9).
    Where one method dominates, the equal-weight blend lands near the better one, and on credit-g it
    cannot reach TabPFN, which is the case per-column weights address.
  - **Running since 2026-10-23 as E69** ([imputation-protocol/24](tickets/imputation-protocol/24-blend-weights.md)), on the user's go ("vamos com t37", about 22:50 GMT-3), ahead of T35: route (a), cross-fitted, on E60's 15 variants and E61's 6.
  - **Done 2026-10-24 by E69: mixed.** Better than TabPFN on 3 of 10 hard variants (the 80nan
    ties), worse on 1 by 0.0016; better than both on 13 of 21. On kr-vs-kp and credit-g the
    per-column pick reaches TabPFN, not past it. A hybrid output would need a fresh-seed rule; the
    user decides.
  - **Running from 2026-10-24 as E74** ([imputation-protocol/27](tickets/imputation-protocol/27-hybrid-output.md)), on the user's go: the hybrid over today's default (KC) on fresh seeds 1212/2323/3434, the 10 hard variants.
  - **Done 2026-10-24 by E74: mixed.** Better than KC on 7 of 10 (rule needed 8), worse than
    TabPFN on 2 by ≤0.005 (kr-vs-kp). E69's three numerical 80nan wins replicate; TabPFN alone beats
    KC on 8. Whether to propose an opt-in hybrid output is the user's call.

- **T38 · Interpretability: which columns does the decoder fill each column from?** (proposed
  2026-10-26 on the user's ask, "gostaria de trabalhar com interpretabilidade"; accepted the same
  evening, "sim, vamos!", studies A + B + C).
  - **The gap:** ADR 0030 says how sure a fill is; nothing says from what it was filled.
  - **What it is:** an occlusion map. Each source column's observed cells are hidden as `[MASK]`,
    the token the decode stage trains on, and the change in each gap's fill is measured. It is
    tested for faithfulness by deletion on 9 real tables at 40nan (B, primary). It is tested for
    truth on a synthetic table with known blocks (C, gate). Attention is set beside it,
    descriptively.
  - **Running from 2026-10-26 as E78**
    ([imputation-interpretability/01](tickets/imputation-interpretability/01-occlusion-map.md)),
    30 cells, no `src/` change.
  - **Done 2026-10-27 by E78: supported.** Faithful on 9 of 9 real tables at 40nan (kc2 weakest,
    D +0.036). The synthetic blocks are recovered (AUROC 0.979), including pairs with Pearson ≈ 0.
  - **E79 pre-registered 2026-10-26**
    ([imputation-interpretability/02](tickets/imputation-interpretability/02-occlusion-map-60nan.md)):
    the same design at 60nan, run after E78 on the user's overnight GPU time (no cell after 07:00
    GMT-3).
  - **Done 2026-10-27 by E79: supported, at the bar.** Faithful on 7 of 9 at 60nan, with kc2 and
    credit-g inconclusive and none against. Synthetic AUROC 0.914; effects shrink with
    missingness. T38's A + B + C are done. E (spread vs missing top sources) and F (MNAR) are open
    and wait for the user.
  - **E and F started 2026-10-27 on the user's go** ("otimo, quero fazer E e F agora, vá"):
    - E80 ([03](tickets/imputation-interpretability/03-spread-and-missing-sources.md)), offline;
    - E81 ([04](tickets/imputation-interpretability/04-gap-reliance.md)), 66 cells with a
      `decode_gap_token null` arm, since the default cannot be asked the question.
  - **Done 2026-10-27 by E80: mixed.** Specific on 7 of 16 and against on 6. The spread falls as a
    row's gaps grow, while error rises with missing top sources. That is a limit on ADR 0030 for the
    user to decide whether to record.
  - **Done 2026-10-27 by E81: mixed.**
    - The default cannot read the gap pattern by construction.
    - A `decode_gap_token null` decoder reads it at tilts 0.25–1.0 and on vehicle/kc2 MNAR, and not
      at MCAR. It shows only +0.005 at slope 2.0, the rule's point.
    - It is less accurate than the default on 11 of 11, so ADR 0014 stands.
    - T38's studies A–F are all done.
  - **E85 pre-registered 2026-10-27**
    ([imputation-interpretability/05](tickets/imputation-interpretability/05-occlusion-map-grid.md)):
    E78's design at 20nan and 80nan, 60 cells, run after E84.
  - **E85, 2026-10-28: partial.**
    - 80nan is complete and refuted: faithful on 4 of 9, failing on the small tables and spambase.
      The map's claim is "up to 60nan".
    - 20nan has 10 of 30 cells and no verdict; its remaining 20 cells wait for the user.
  - **E85 done 2026-10-28:** 20nan supported (8 of 9) and 80nan refuted (4 of 9). The map's claim
    is "20 to 60nan".
  - **E88, running from 2026-10-28**
    ([imputation-interpretability/06](tickets/imputation-interpretability/06-occlusion-map-mechanisms.md)),
    on the user's go: the map on E84's 12 MNAR and MAR variants at 60nan, on gorgona8.
  - **E88 done 2026-10-28: supported** (faithful on 11 of 12, kc2_probit inconclusive). The map's
    claim is "20 to 60nan under MCAR, and 60nan under MNAR and MAR".
  - **Follow-ups E and F:** run as E80 and E81, below.

- **T39 · Propose an opt-in hybrid output: KC + TabPFN per column, as an ADR** (future; the user,
  2026-10-27: "Gosto da ideia de propor uma ADR, deixe isso como tarefa futura mesmo").
  - **From E69 and E74**
    ([imputation-protocol/24](tickets/imputation-protocol/24-blend-weights.md),
    [27](tickets/imputation-protocol/27-hybrid-output.md)).
    - Per column, a weight in {0, 0.25, 0.5, 0.75, 1} on the model's numerical fill, or a pick of
      either model's category, chosen on the other four folds.
    - E74 was mixed. It was better than KC on 7 of 10 hard variants, where the rule needed 8. It was
      worse than TabPFN on 2 (kr-vs-kp 40/60nan, ≤0.005), and beat both on 3 (electricity,
      vehicle and biodeg at 80nan).
  - **The proposal must state:**
    - E74's pre-registered rule was not met;
    - TabPFN alone beat KC on 8 of those 10, so most of the hybrid's gain there is TabPFN's own. Its
      own gain shows on the three numerical 80nan variants.
  - **Shape:** opt-in and off by default. TabPFN stays an offline optional extra (ADR 0028), so the
    flag needs the `tabpfn` extra and one TabPFN refit per column a fold.
  - **Open, to settle before or inside the ADR:**
    - E69/E74 chose the weights on the other folds' *induced* cells, which need the complete table.
      A deployed run has no complete table, so it needs weights from observed cells: the masked
      population, or the validation rows' own fills.
    - That route is untested and may need its own pre-registered study.
  - **[ADR 0031](adr/0031-an-opt-in-offline-hybrid-output-with-tabpfn.md) proposed 2026-10-27,
    12:44 GMT-3, and accepted by the user (2026-10-27, 12:51 GMT-3, "eu apoio a 031").**
    - It is offline, in E74's exact shape. Its weights read the same information class as KC, so
      it is a benchmark output.
    - Deferred, each a separate study: E82, in-run validation-gap weights (about 4 h of GPU plus
      refits); a deployable masked-weight hybrid with H.
  - **The deployable route, running from 2026-10-27 as E82**
    ([imputation-protocol/31](tickets/imputation-protocol/31-deployable-hybrid.md)), on the user's
    go: H + TabPFN with weights chosen on hidden observed validation cells; 30 cells at E74's
    seeds.
  - **Done 2026-10-27 by E82: refuted.** Better than H on 10 of 10 and than KC on 7 (never worse),
    with no complete table. It is worse than TabPFN on 5 and than the oracle weights on 6.
    - The selection is the loss: thin cells (vehicle, biodeg at 80nan) and categorical picks
      (kr-vs-kp, credit-g).
    - Candidates for the user: more selection cells, or fewer weights (one per table, or shrinkage
      toward the equal blend).
  - **Running from 2026-10-28 as E89**
    ([imputation-protocol/33](tickets/imputation-protocol/33-hybrid-selection-rules.md)), on the
    user's go ("continue o e82 então"). It scores the table, shrinkage (primary) and pooled rules
    offline on E82's cells. It is a development read, so fresh seeds confirm a positive result.
  - **E89 done 2026-10-28: supported (development).** `shrunk` is behind TabPFN on 2 of 10, and
    `table` on none, both better than H on 10 of 10. A fresh-seed confirmation with one rule waits
    on the user.
  - **Running from 2026-10-28 as E90 and E91**
    ([imputation-protocol/34](tickets/imputation-protocol/34-hybrid-confirmation.md)), on the
    user's go: `table` on fresh seeds 4545/5656/6767, on the 10 hard variants (E90) and on 7 where
    the hybrid is not needed (E91). ADR 0031 carries E89 as evidence, with no decision changed.
  - **E90 and E91 done 2026-10-29: both supported.** `table` on fresh seeds is never worse than H
    or TabPFN on 17 variants (E90: better than H on 9 of 10; E91: better than TabPFN on 6 of 7).
    `shrunk` does not replicate. A deployable output in ADR 0031 is the user's decision.
  - **ADR 0031 decision 6 proposed 2026-10-29** on the user's request ("sim, proponha"): the
    deployable H + TabPFN hybrid with the `table` rule, opt-in and offline. **Accepted 2026-10-29,
    10:42 GMT-3** ("aprovo"). T39 is done.
  - **E93, running from 2026-10-29**
    ([imputation-protocol/35](tickets/imputation-protocol/35-hybrid-mechanisms.md)): decision 6's
    hybrid under MNAR and MAR at 60nan, on gorgona8.
  - **E93 done 2026-10-29: mixed.** The hybrid is better than H on 6 of 12. Under MNAR and MAR, H
    already leads TabPFN, so decision 6 stays MCAR.
  - **E95, running from 2026-10-29**
    ([imputation-protocol/36](tickets/imputation-protocol/36-hybrid-20nan.md)): decision 6's hybrid
    at 20nan, whole on medusa1.
  - **E95 done 2026-10-29: supported.** At 20nan the hybrid is never worse than H or TabPFN (9 of
    9). Extending decision 6 to 20nan is the user's decision. **Extended 2026-10-31 on the user's decision (ADR 0031).**

- **T14 · A steps-aware batch rule** (at least ~8 steps an epoch) instead of one batch for every
  table (E28).
  **Done 2026-10-30 by E100 ([imputation-protocol/37](tickets/imputation-protocol/37-steps-rule.md)): refuted.** `--min_steps_per_epoch 8` is better on 0 of 16 small-table variants and worse on 4; closed, the flag stays off.
- **T15 · MAE-style architecture** (ADR 0004 Design C, handoff item 5): the most expensive; only if
  T02–T03 fall short.

### Later baselines and classification (plan 2026-10-17)

- **T27 · MaCoDE, TabPFN v2 and classification step zero** (plan §2g; review items 10–12).
  **MaCoDE done 2026-10-19 (E48,** [imputation-protocol/07](tickets/imputation-protocol/07-macode.md)**):** TRIDENT beats it on 20/21 and 14/15, never worse; an external deep baseline only. TabPFN v2: E55 done 2026-10-21 ([imputation-protocol/10](tickets/imputation-protocol/10-tabpfn.md)): better than the strong tier on 11/15 confirmation variants. **The user accepted it into the strong tier (ADR 0028, 2026-10-21)** as an offline member; K − strong now reads 5/9/7 on development and 6/4/5 on confirmation. E56's panel: TabPFN's 80nan wins come with more shrunk fills. E57 (KC − TabPFN 6/4/1/4) and E58 (every other seed set, MNAR included) done; the thesis table is rebuilt in [2026-10-21-thesis-table-with-tabpfn](reports/2026-10-21-thesis-table-with-tabpfn.md), and the baselines page (artifact 7GTHy9rjKfPK2dWASRnJgo) is at its third version. E60 done 2026-10-22 ([imputation-protocol/15](tickets/imputation-protocol/15-tabpfn-blend.md)): an equal-weight blend of K's and TabPFN's fills beats both on 10/15 confirmation variants (complementary; descriptive). E66 done 2026-10-23 ([imputation-protocol/21](tickets/imputation-protocol/21-tabpfn-blend-replication.md)): on E40's tables and seeds it beats both on 3/9, complementary at the floor; the effect shows on numerical tables where the two are close. **E99 done 2026-11-01 ([classification-step-zero/08](tickets/classification-step-zero/08-deep-baselines.md)): supported.** Tuned TRIDENT leads MLP, SAINT and TabNet at 40–80nan by the majority rule; over MLP by about a point at 40/60nan and a tie at 80nan; TabNet is a weak baseline here. **Stage 4 done 2026-10-30 as E98 ([classification-step-zero/07](tickets/classification-step-zero/07-step-zero.md)): H1c refuted.** The tuned GBDT bar beats tuned TRIDENT at every level, by 0.024–0.032 macro-F1 at 40–80nan, and tuning leaves TRIDENT where the fork budget had it; what H1c becomes in the thesis, and the deep-baseline amendment, wait on the user. **Stage 1 done 2026-10-29 as E97 ([classification-step-zero/06](tickets/classification-step-zero/06-cpu-bar.md), descriptive):** untuned native HGB leads TRIDENT's classification by +0.015 to +0.033 macro-F1 at 20–80nan. ADR 0029's code landed (629aef4). **Step zero: ADR 0029 accepted and the plan's eight decisions settled 2026-10-29, 21:20 GMT-3** (deep baselines later, as an amendment); **plan and ADR proposed 2026-10-23** ([plan](plans/2026-10-23-classification-step-zero.md), [ADR 0029](adr/0029-classification-search-ranked-on-validation.md), proposed); waits on the user's eight decisions. **Stage 0 running since 2026-10-23 as E70** ([classification-step-zero/01](tickets/classification-step-zero/01-determinism.md)), on the user's go ("sim aprovo", 23:44 GMT-3). **Done: does not reproduce** (fold differences up to 0.076 macro-F1 against a 0.009 duplicate spread; the regression test passes), so Stage 2 reruns arm A on gorgona8. **Stage 2 running from 2026-10-24 as E71** ([classification-step-zero/02](tickets/classification-step-zero/02-no-pretraining.md)): arms A/B/C on 45 variants, seeds 42/7/13 as the GPU window allows. **Done 2026-10-24: no pre-training gain detected** at any level (B − A and C − A no verdict, means −0.005 to +0.004; seeds 42 and 7).
  - MaCoDE vendored behind an inductive wrapper, with its bins bug fixed first: 10–14 h of code,
    3–5 GPU-h off-store.
  - A column-wise TabPFN v2 in a separate venv pinned to `ModelVersion.V2`: 5–8 h of code and
    6–10 GPU-h.
  - Classification step zero: a native-NaN GBDT bar, impute-plus-indicator arms and an
    `EPOCHS_PRE=0` arm, with `--cv_folds 10`.

  In GPU gaps only, after T21–T25; D21 is decided after E44.

- **T40 · TabPFN v2 as a strong baseline for classification** (future; the user, 2026-10-27: "usar
  o tabpfn como baseline forte para classificação também").
  - **The gap:**
    - Step zero (T27; [plan](plans/2026-10-23-classification-step-zero.md); ADR 0029 proposed) has
      a GBDT bar with native NaN (LightGBM and HistGradientBoosting) and impute-then-learn arms.
    - It has no foundation-model classifier.
    - On imputation, TabPFN v2 is the strongest rival (E55, E57; ADR 0028's strong tier).
  - **What it would be:** TabPFN v2's classifier, untuned, as TabPFN is used, on step zero's own
    folds (`--cv_folds 10`) and rates.
    - It reads NaN natively, so it joins the GBDT family with no imputation step.
    - An impute-then-learn arm, with TRIDENT's or TabPFN's own fills before the classifier, is
      optional and decided in the ticket.
  - **Reuse:** the `tabpfn` extra pinned to 9.2.0 and v2's checkpoints, as in
    `src/training/tabpfn_baseline.py`.
  - **To check in the ticket, against TabPFN v2's documentation:**
    - its class and row limits: letter has 26 classes, and electricity has more rows than the
      context the imputer already caps (`CONTEXT_ROWS`);
    - how a table past them is handled: a context subsample, or a many-class wrapper.
  - **Decision needed:** whether it joins step zero's bar or stands as a separate strong tier. That
    is the same question ADR 0028 settled for imputation, so it belongs with ADR 0029's decisions or
    in an ADR of its own.
  - **When:** after step zero's decisions (ADR 0029's 8), in GPU gaps.
  - **Started 2026-10-27 as E83** ([classification-step-zero/03](tickets/classification-step-zero/03-tabpfn-classifier.md)),
    on the user's go, ahead of step zero:
    - untuned TabPFN v2 on E71's 3-fold folds (seeds 42/7, 8 tables × 5 levels), paired with E71's
      TRIDENT arm A;
    - letter is out (26 classes; v2 allows 10).
  - **Done 2026-10-27 by E83:** TabPFN ahead at every level, by about 2 points of macro-F1.
    - TRIDENT leads only on credit-g and electricity, through its class-weighted loss.
    - **Decided 2026-10-27, 20:16 GMT-3: a separate strong tier, as in imputation**
      ([ADR 0032](adr/0032-tabpfn-as-a-separate-strong-tier-for-classification.md), accepted).
      H1c's bar stays the best GBDT. Whether to add a class-balanced TabPFN arm is decided with
      step zero's design.
  - **E92, chained after E87 on medusa1 from 2026-10-28** ([classification-step-zero/05](tickets/classification-step-zero/05-tabpfn-tenfold.md)),
    on the user's go: plain and class-balanced TabPFN on E87's ten folds and five seeds.

- **T41 · Replicate the paper's pre-training gain on classification, with the old parameters and
  more folds** (future; the user, 2026-10-27: "deixe uma tarefa futura para repetir a execução
  usando os parametros antigos e mais folds, tentando replicar o que existia antes").
  - **Why:** E71 found no detectable gain from pre-training on classification (B − A −0.005 to
    +0.004 macro-F1, no verdict at any level), while the paper's Table 4 reports about +1.7 points.
    - E71 ran 3 folds and two complete seeds (42, 7; seed 13 stopped at 89 of 135 cells).
    - It ran the fork's old sweep configuration (EPOCHS_PRE 300, EPOCH_FINE 150, `cosine_legacy`,
      untuned), not the paper's protocol.
    - So it does not yet say whether the paper's gain fails to replicate or only fails to appear
      under other settings.
  - **What it would be:** arms A (with pre-training) and B (`EPOCHS_PRE 0`), paired by fold and
    seed, under two parameter sets read separately:
    - **the paper's:** the upstream 40/40 epochs, the paper's per-dataset hyperparameters and its
      80/10/10 split, as close to Table 4 as the code allows;
    - **the old sweep's**, as in E71, with more folds (`--cv_folds 10`, as step zero plans) and at
      least five seeds.
  - **To settle in the ticket:**
    - **Hyperparameters:** where the paper's per-dataset values live. No tuned configuration file
      is tracked in this repository, so they must be asked of the paper's authors; the advisor is
      one of them.
    - **Split:** whether the paper's single split is rebuilt as is, or replaced by repeated
      80/10/10 splits, so that the result has an interval.
    - **Code version:** which commit stands for "the paper's code". E70 found that HEAD on
      gorgona8 does not reproduce the old sweep fold by fold (differences up to 0.076 macro-F1), so
      a replication may need the upstream commit.
    - **The decision rule:** pre-registered before the first cell. For example, the gain counts as
      replicated if the 95% interval of the paired B − A excludes zero in the paper's direction at
      the rates Table 4 reports.
  - **Related:** T27 (step zero, which owns the classification arms and folds), T40 (TabPFN for
    classification), and the wiki's t3 (the Table 3 rerun with five or more seeds).
  - **When:** after the advisor meeting of 2026-10-27, once the paper's hyperparameters are in
    hand; in GPU gaps.
  - **The old-sweep half, running from 2026-10-28 as E87**
    ([classification-step-zero/04](tickets/classification-step-zero/04-tenfold-pretraining.md)),
    on the user's go ("aprovo! rode os 2 em paralelo"): arms A and B, 9 tables × 5 levels, ten
    folds, five seeds, on medusa1.
  - **E87 done 2026-10-29: refuted.** B − A is equivalent (inside ±0.01) at every level, from
    −0.0035 to +0.0022. Under the old sweep, pre-training adds nothing, and the paper's gain is
    excluded. The paper's-parameters half stays open.
  - **The upstream-defaults half, running from 2026-11-03 as E118**
    ([classification-step-zero/15](tickets/classification-step-zero/15-upstream-defaults-pretraining.md)),
    on the user's go ("rode o t41 agora nas duas"): the upstream code's built-in defaults (40/40 epochs), E87's
    arms, cells and rule, split by seed between the machines. The paper's tuned values stay untested.
  - **E118 done 2026-11-03: refuted.** B − A equivalent at every level (−0.0009 to +0.0023). Neither half shows the
    paper's gain; only the paper's tuned per-pair values, not in the repository, remain untested. **T41 closed.**

- **T42 · Can one study's cells run on both servers?** (the user's go, 2026-10-28: "vamos testar o
  0!").
  - **Done 2026-10-28 by E86** ([cross-machine/01](tickets/cross-machine/01-gorgona8-medusa1.md)):
    no verdict.
    - Both machines are deterministic, but they differ from each other fold by fold, by up to
      0.164 (imputation, relative) and 0.074 (classification macro-F1).
    - So each study runs whole on one machine. Two studies can run in parallel, one per machine.
  - **E94, running from 2026-10-29** ([cross-machine/02](tickets/cross-machine/02-equivalence.md)):
    the same check with 350 and 150 fold pairs. If a task is equivalent, its studies may be split by
    seed.
  - **E94 done 2026-10-29:** classification is equivalent (+0.0009 [−0.0056, +0.0073]), so its
    studies may split by seed. Imputation has no verdict (+0.54% [−0.06%, +1.13%]) and stays on one
    machine a study.
  - **E96, running from 2026-10-29** ([cross-machine/03](tickets/cross-machine/03-equivalence-without-kc2.md)):
    E94's imputation variants without kc2, on fresh seeds.
  - **E96 done 2026-10-29: equivalent** (+0.04% [−0.30%, +0.38%]). Imputation studies on credit-g,
    vehicle and biodeg may split by seed; kc2 and the unchecked tables may not.
  - **Decided 2026-10-29 (ADR 0035, accepted):** any study, on any table, may split between the
    servers by seed, with both arms of a pair on one machine. The user's words: "podemos dividir em
    qualquer maquina ... nao precisamos testar em mais datasets". T42 is done.
  - Runs now record `host` ([ADR 0034](adr/0034-a-host-tag-on-every-run.md)).

- **T43 · Conformal intervals for the imputer's fills, calibrated by row missingness** (proposed
  2026-10-29 from the wiki's claims ledger, `projects/trident/claims`, C18 and P6, batch t49; for
  the user's decision).
  - **The gap:** ADR 0030 gives a per-cell rank, not an interval. E77 found that the spread, scaled
    to 90% by cross-fitting, makes wider intervals than a per-column constant width on 15 of 15
    variants. E80 found why a single scale fails: the spread *falls* as a row's gaps grow, while the
    error *rises* when a cell's top sources are missing.
  - **The literature's answer:** missing values make the error heteroskedastic by pattern, so one
    correction for every pattern under-covers the rows that hide the most informative columns.
    Conformal prediction with missing values (Zaffran et al., ICML 2023) restores per-pattern
    coverage by calibrating on rows that carry the test row's pattern; conformalized matrix
    completion (Gui et al., NeurIPS 2023) calibrates cell-wise intervals for missing entries.
  - **What it would be:** offline, on ledgers that already hold fills and spreads (E75, E76, E84),
    with no new training.
    - **Groups (Mondrian conformal):** calibrate per numerical column and per bin of the row's
      missingness, either the count of gaps in the row or the count of the cell's top-3 sources
      missing from E78's map.
    - **Two scores per group:** a constant width (|error|), and a spread-scaled width
      (|error| / (spread + floor), the floor chosen on calibration cells).
    - **Calibration cells:** other folds' induced cells, as E77 (an oracle reading under K), and,
      for a deployable reading, observed validation cells hidden on purpose, as E82.
  - **What it would read:** coverage at a nominal 90% overall and per bin (the Zaffran-style
    per-pattern check); mean width against E77's per-column constant; both under K and H
    (ADR 0024); MCAR first, then E84's MNAR and MAR variants. The decision rule is the ticket's.
  - **What it would settle:** whether the decoder's uncertainty can become an interval at all
    (ADR 0030's "a rank, not an interval"), and whether binning by row missingness fixes E80.
  - **Started 2026-11-06 as E151** ([imputation-protocol/54](tickets/imputation-protocol/54-conformal-by-row-gaps.md)), on
    the user's go ("vou na sua sugestão, vá"): bins by the row's gap count only; the top-3-sources bin and the
    deployable calibration on hidden validation cells are left out (the latter needs new runs).
  - **Done 2026-11-06 13:05 GMT-3 by E151: Q1 supported** (binning by the row's gaps restores the top tertile's
    coverage on 11 of 15, from 0.81–0.87 to 0.90, no width cost on the large tables), **Q2 refuted** (the spread
    still loses to a constant inside a bin, 15 of 15). **ADR 0030 amended 2026-11-06 16:55 GMT-3 (decision 8),
    the user's "faça a emenda da 030":** the binned interval is the fill's interval, oracle-calibrated; a deployable
    reading needs new runs. **E153 pre-registered 2026-11-06** ([mimic/07](tickets/mimic/07-conformal-real-patterns.md),
    the user's go): decision 8 read on E149's real transplanted patterns, offline. **Done 2026-11-06 17:13 GMT-3:
    refuted, one width per column already covers** (MC equivalent to C on both tables, top tertile 0.900 and 0.909
    under C, MC 0.6–3% wider); stating it in decision 8 is the user's call. **E154 pre-registered 2026-11-06**
    ([mimic/08](tickets/mimic/08-conformal-natural-gaps.md), the user's "ok va"): the same reading on E148's natural
    tables, hidden observed cells, offline. **Done 2026-11-06 18:08 GMT-3: refuted, one width per column already covers** (MC
    equivalent on both natural tables, u(C) under 0.01, MC 1.2–1.3% wider).

- **T44 · A TabPFN arm in the MNAR prediction protocol** (proposed 2026-10-29 from the wiki's
  claims ledger, C14 and C15, batch t49; for the user's decision; after the protocol's own fork
  changes, wiki todo t41).
  - **The gap:** untuned TabPFN v2 leads TRIDENT's classification under MCAR (E83; ADR 0032), but
    its prior has only MCAR gaps. The wiki's search (t46) found no published test of TabPFN's
    native missing-value handling (a mean fill plus an indicator channel) under a controlled
    MNAR mechanism. The closest, Liang et al. (arXiv 2601.18500, a preprint), uses natural
    missingness only: there, TabPFN with raw NaNs and TabPFN after 12 imputers are within about
    0.003 mean AUC.
  - **What it would be:** two offline, untuned arms beside the protocol's A1 (`[NULL]`) and A2
    (missForest + indicator), on the same MNAR family, folds and seeds:
    - **TabPFN, native:** raw NaNs, ADR 0032's settings;
    - **TabPFN, explicit mask:** the 0/1 missingness mask appended as ordinary features, which the
      in-context learner can use as predictors (its built-in indicator was trained only where
      gaps carry no signal).
    - Class balancing follows E92's outcome.
  - **What it would read:** each arm against A1 on macro-F1 and log loss, and the interaction
    (arm − A1) MNAR minus MCAR: does TabPFN's MCAR lead shrink, hold or grow when missingness
    carries the label? Explicit mask minus native, on the MNAR variants only, measures what the
    MCAR prior leaves unused.
  - **What it would settle:** whether TabPFN's lead is robust to informative missingness, which
    no published study shows, and whether TRIDENT's per-column `[NULL]` gains ground there (H2).
  - **Pre-registered 2026-11-04 as E137 ([c15-mnar-prediction/08](tickets/c15-mnar-prediction/08-tabpfn-arm.md)), the user's go ("vamos no 1 e 2 paralelos"): native and explicit-mask arms, balanced primary, on E124's 69 A1 pairs; primary the interaction (arm − A1) MNAR − MCAR over the seven probit/MCAR sibling pairs, Holm over the two arms. Proposal text ported from `origin/docs/wiki-proposed-tasks-t49` (fbcc010) with T43.** **Done 2026-11-04: no verdict on the interaction for both arms (native +0.008 [−0.001, +0.017], mask +0.004); TabPFN's lead holds or grows under probit MNAR (+0.028 MCAR → +0.036 MNAR → +0.048 probit scales), the explicit mask adds nothing (+0.0003), and the lead vanishes only under outcome-dependent MNAR (F3 −0.003, A1 ahead on kc2). ADR 0032 unchanged; T44 closed.**

- **T45 · C15: `[NULL]` tokens against missForest + indicator under informative missingness** (the user's
  choice, 2026-11-03: "prefiro começar pelo c15"; the wiki's protocol C15, H2 and H2.2 of the thesis).
  [Plan](plans/2026-11-03-c15-mnar-prediction.md): Stages 0–3 approved (Stage 4 optional), objective
  `embedding_normalized` by explicit flag, GBDT internal only, margins fixed (−0.01 macro-F1, +2% log loss).
  **Stage 0 (E122, [c15-mnar-prediction/01](tickets/c15-mnar-prediction/01-stage0.md)), from 2026-11-03:** the
  driving columns and gate G2 (CPU), which define the primary set P. **Done 2026-11-03: G2 invalid** (positive controls fail on biodeg and kc2, intervals below 0); by the rule C15 stops; the next step is the user's. **E123 ([02](tickets/c15-mnar-prediction/02-g2-out-of-fold.md)), from 2026-11-03: G2 read out of fold over the training and validation rows (the user's "opção 1"); if the controls fail again, C15 stops for good.** **Done 2026-11-04: G2 valid, P = 5** (biodeg_probit_60nan, vehicle_probit_60nan, the odds-3 outcome variants of biodeg, kc2, spambase). **Stage 1 (E124, [03](tickets/c15-mnar-prediction/03-stage1.md)), from 2026-11-04:** `C15_ARM` behind a default-off setting; G1 on F5 and A1 − A2 on the development set. **G1 failed 2026-11-04 (1 of 3): C15 stops before Stage 2; E125–E127 withdrawn unrun; Stage 1b finishes descriptively.** **E128 ([07](tickets/c15-mnar-prediction/07-g1-a3-decision-mode.md)), from 2026-11-04: G1 again, A2 against A3 in decision mode (the user's option 4, a deviation); if it fails, C15 stops for good.** **Done 2026-11-04: G1 fails again (1 of 3: A1's contrast holds on 3 of 3, A2 beats A3 only on spambase); by the binding clause C15 stops for good. T45 closed.** Note 2026-11-06: E152 ran the same G1 on the SOFA table's
  natural gaps (T50, [mimic/06](tickets/mimic/06-sofa-positive-control.md)) and it failed the same way. Superseded for H2
  only on 2026-11-06 by T51 (the user's go): the gate moves outside the pipeline.

- **T46 · C1: does the `[NULL]`/`[MASK]` split earn its place?** (the user's choice after C15 closed,
  2026-11-04: "vamos trabalhar no c1"; the wiki's protocol C1, the thesis's main claim). [Plan](plans/2026-11-04-c1-split.md):
  `embedding_normalized` by explicit override (no ADR), gates condition only the Null row, G2 by interval, cosine
  threshold 0.9, scope cuts 1–3 and the anchor. S reuses E124's A1 cells. **Cut 1 (E129, [c1-split/01](tickets/c1-split/01-g1-probe.md)),
  cut 2 (E130, [02](tickets/c1-split/02-integrity-smoke.md)) and cut 3 (E131, [03](tickets/c1-split/03-primary.md)), from 2026-11-04.** **Done 2026-11-04: G1 passes (4 of 4), the smoke passes, the primary is Inconclusive: S − T −0.003 [−0.011, +0.004], G2 fails (S − B +0.004, no pre-training channel, as E71); cosines at initialisation.** **E132 ([04](tickets/c1-split/04-channel-probe.md)), from 2026-11-04 (the user's "vamos no 2"): the channel probe, a frozen pre-trained encoder against a frozen random one, head only; it says whether a fine-tuning-side fix is worth a ticket.** **Done 2026-11-04: content present but small (FP − FR +0.009 [+0.001, +0.017] MNAR; S − FP +0.10): washout; a fix is bounded near +0.01.** **E133 ([05](tickets/c1-split/05-finetuning-fix.md)), from 2026-11-04 (the user's "1!", after the loss curves: fine-tuning overfits after epoch ~20, frozen heads unconverged): LRS (encoder LR x0.1) and FRZ (head alone for 50 epochs) against S, Holm over the two; FP10/FR10 beside.** **Done 2026-11-04: neither helps (LRS −0.001, FRZ −0.003, no verdict); FP10/FR10 diverged. Fine-tuning side closed.** **E134 ([06](tickets/c1-split/06-finetuning-budget.md)) and E135 ([07](tickets/c1-split/07-finetuning-scheduler.md)), from 2026-11-04 (the user's "rode o 134 entao, alem de estudar o melhor metodo"): the fine-tuning budget (30/50 against 150) and fine-tuning's schedule (cosine, warmup_cosine, plateau, constant against cosine_legacy's sawtooth); 240 cells; a winner goes to an ADR proposal with a 21-variant confirmation.** **Done 2026-11-04: no verdict everywhere. E134: S30 −0.008 (Holm 0.058, vehicle worse ×2), S50 −0.001 (no file worse: a cost argument only, the user's call); E135: six schedules within −0.003…+0.001 of `cosine_legacy`, none better, plateau ≡ constant on 5/8 files. `cosine_legacy` at 150 epochs stays; no ADR. C1's fine-tuning side is closed on every axis tried (fix, budget, schedule).** **Closed 2026-11-04 (the user: "podemos encerrar ela então no momento"): Inconclusive for want of a channel. Pre-training leaves about +0.01 to a frozen head (E132) and fine-tuning rebuilds +0.10 from any start, so S ≈ T ≈ B on fully labelled small files; the split's claim is untestable in this regime, not refuted. Reopen only where a pre-training channel exists (a label-scarce fine-tuning regime is the candidate).** **Reopened 2026-11-04 in the label-scarce regime (the user's "vamos no 1"): E136 ([08](tickets/c1-split/08-label-scarce.md)), S/T/B with fine-tuning on a tenth and a quarter of the training labels (`FINETUNE_LABEL_FRACTION`, new, default 1.0), 144 cells; gate S(f) − B(f) with Holm over the two fractions, S(f) − T(f) read only past the gate.** **Done 2026-11-04: the gate opens at a quarter of the labels (S25 − B25 +0.016 [+0.007, +0.024], Holm 0.003; 4 files better, none worse), the first positive pre-training channel under full fine-tuning on classification here (E132's frozen-head +0.009 was the hint); closed at a tenth (+0.011, Holm 0.15). Past the gate S25 − T25 is Inconclusive under MNAR (−0.002 [−0.012, +0.007]) and T is ahead under MCAR (−0.012 [−0.023, −0.001], descriptive). No evidence for the split where it could be tested; T never worse than S. Next steps the user's.** **Closed 2026-11-04 (the user: "concordo com voce"): not Supported where testable; T never worse than S; T recorded as the simpler equivalent, no default change proposed. Follow-ups: T47 (the scarce-label channel) and T48 (H2 on the outcome-dependent family).**

- **T47 · Pre-training's channel at scarce labels, confirmed and sized** (from E136, the user's agreement
  2026-11-04). **E138 ([c1-split/09](tickets/c1-split/09-scarce-confirmation.md)), from 2026-11-04:** S25V − B25V on
  seeds 1515/2626/3737 with the validation split scarce too, 48 cells; holds if the plain interval is above 0. Then,
  on the user's go, stage 2 on the 21 confirmatory variants (630 cells, the big tables the bulk). A classification
  result the thesis can state without GBDT or TabPFN in the sentence.
  **E138 done 2026-11-05 00:29 GMT-3: HOLDS, +0.022 [+0.009, +0.035], 3 files better, none worse.** **Night of 2026-11-05 (the user: "quero te manter ocupado no mínimo até as 7"): E140 ([10](tickets/c1-split/10-label-fraction-curve.md), the fraction curve 0.1/0.5, 96 cells), E141 ([11](tickets/c1-split/11-stage2-21-variants.md), stage 2 on the 21 variants, 5 seeds, 210 cells) and E143 ([12](tickets/c1-split/12-missingness-levels.md), 20/40/80nan, 72 cells) chained on both machines.**
  **E140 done 2026-11-05 01:25 GMT-3: the channel is open at a tenth (+0.029 [+0.017, +0.042]) and at a half (+0.016 [+0.008, +0.024]); with E138's quarter (+0.022) and E131's full labels (+0.004) the curve is monotone: the scarcer the labels, the more pre-training pays; no file worse at any fraction.**
  **E141 done 2026-11-05 02:43 GMT-3: HOLDS on the 21 confirmatory variants, five seeds: +0.010 [+0.005, +0.016] pooled (NB crosses 0), 8 variants better (spambase ×4, the three large, kr-vs-kp_40nan), none worse; largest and tightest on the large tables (+0.016). Thesis-claim proposal and optional ADR for the label-scarce path put to the user; no default change.** **E143 done 2026-11-05 02:43 GMT-3: the channel is flat (+0.023…+0.026) at 20/40/60% missingness and closed at 80% (+0.004); no table worse. Descriptive, no ADR.** T47's pre-registered programme is complete; what remains is the user's wording. **E144 ([13](tickets/c1-split/13-selector-squares.md)), from 2026-11-05 ~03:00 GMT-3: the selector × seeds squares at a tenth and a quarter (168 cells, cuts 13/14), reading E140's selector hypothesis as a paired contrast; E136/E140 verdicts stand.** **E144 done 2026-11-05 09:40 GMT-3: not supported at a tenth and a quarter (selector effect −0.006 at both, Holm 0.57); the gain at scarce labels is pre-training's, not B's epoch selection; E136/E140 differ by seeds. T47 closed on the evidence side; the thesis sentence and an optional ADR for the label-scarce path are the user's.** **2026-11-05 16:02 GMT-3: [ADR 0038](adr/0038-label-scarce-fine-tuning-as-a-documented-regime.md) written as a proposal (not accepted) and the proposed sentences, T48's too, put in the consolidated report's section 3d; both await the user's word.** **ADR 0038 accepted 2026-11-05 by the user ("aceito a adr"): the `finetune_label_fraction` / `finetune_scarce_validation` tags are stamped on every classification run and backfilled (`1.0` / `off`) on 2906 runs and their 2888 active mirrors through the leader, verified 17:39 GMT-3; the two thesis sentences still await the user's word.**

- **T48 · H2 on the outcome-dependent family: same-seed MCAR siblings** (from E137, the user's agreement
  2026-11-04). **E139 ([c15-mnar-prediction/09](tickets/c15-mnar-prediction/09-h2-outcome-siblings.md)), from
  2026-11-04:** `{biodeg,kc2,spambase}_mcar_s2060_60nan` generated; A1 on them (9 cells); TabPFN arms offline; the
  interaction (arm − A1) F3 − sibling, Holm over the two arms: shrinks / grows / holds / no verdict.
  **E142 ([c15-mnar-prediction/10](tickets/c15-mnar-prediction/10-h2-fresh-draws.md)) pre-registered 2026-11-05, conditional on E139 reading "shrinks": the s4848 fresh draws (odds 3) on the confirmation seeds, 18 A1 cells plus TabPFN.**
  **E139 done 2026-11-05 00:34 GMT-3: the lead SHRINKS under outcome-dependent MNAR for both TabPFN arms (native −0.045 [−0.056, −0.035], mask −0.036 [−0.047, −0.026], Holm < 0.0001; 12 of 12 file intervals below zero): the first positive H2 reading. E142 launched on medusa1 (stage h2c).**
  **E142 done 2026-11-05 01:24 GMT-3: confirmed on the s4848 draw and seeds 1515/2626/3737 (native −0.039 [−0.051, −0.028], mask −0.028 [−0.040, −0.016], Holm ≤ 0.0002). H2 holds on the outcome-dependent family: `[NULL]` gains 0.03–0.05 of ground on TabPFN relative to MCAR. T48 done; the thesis sentence is the user's call.**

### Process


- **T49 · A distributional score to select by** (the user, 2026-11-05: "usar outra métrica pra imputação que não seja
  o impute_score ... incluindo uma nova flag para optimizar por essas novas métricas"). ADR 0025's offline panel
  already reads the distribution of the fills (Wasserstein, total variation, dispersion) for the model, the baselines
  and TabPFN; the new part is selecting by it. `DECODE_METRIC` / `--decode_metric dist_score` (off by default; tag
  `decode_metric`, backfilled) switches the checkpoint's score and a search's validation key; `score_cells` is
  untouched. **E145 ([imputation-protocol/51](tickets/imputation-protocol/51-decode-metric.md)), from 2026-11-05:**
  checkpoint by `dist_score` against by `impute_score` on E40's 12 variants, seeds 42/7/13, 72 cells on both machines.
  Cut for now: energy distance, I-Scores, row-level sliced Wasserstein, downstream utility, CRPS.
  **E145 done 2026-11-05 20:22 GMT-3: improves the distribution at a cost (`dist_score` −0.19, better on 12 of 12;
  `impute_score` +0.040, worse on 7, all at 40–80nan on kr-vs-kp, spambase and credit-g; free on the other 5). At
  80nan the two selectors keep opposite ends of the trajectory (a near-naive early epoch against a spread-out late
  one that loses to the naive fill). The flag stays off; no ADR. The `decode_metric` tag is backfilled on 3690 runs.
  Next, the user's: a criterion between the two, KC, large tables, fresh seeds.**
  **Downstream use of the fills, two pilots kept on the user's decision of 2026-11-06 (exploratory; a claim needs a
  pre-registered rerun on fresh seeds).** E146 ([imputation-protocol/52](tickets/imputation-protocol/52-downstream-utility.md)),
  on E145's cells: a classifier fitted on complete rows reads the `dist_score` checkpoint's fills better than the
  `impute_score` one's (macro-F1 +0.014 [+0.011, +0.018]; better on 5 of 12 variants, worse on none), with no
  difference when it is fitted on the fills themselves. E147
  ([imputation-protocol/53](tickets/imputation-protocol/53-imputer-utility.md); run as E149), twelve imputers a
  fold under MCAR, MAR and MNAR: mixed. `dist_score` orders the imputers more like the classifier under MCAR
  (+0.154), `impute_score` under MAR (−0.120) and MNAR (−0.221); at 60nan MCAR shows no difference, and post hoc the
  MCAR verdict is carried by 80nan, where `impute_score`'s order turns against the classifier's. Taking each
  score's best imputer, `impute_score` loses less F1 in every mechanism. Nothing argues for replacing
  `impute_score` (ADR 0025 stands). Scripts: `scripts/experiments/downstream_utility.py` and `imputer_utility.py`.
- **T50 · Tables with natural gaps, from MIMIC-IV** (the user, 2026-11-05). Two tables on the kg-impute cohort
  (11,082 ICU stays, label death in hospital): `mimic-cardio` (the 19 cardiovascular variables over 30 h) and
  `mimic-sofa` (the SOFA components' raw values over 24 h). The user's rule: the full tables are read from outside
  the checkout; a 1,000-stay real cohort (`mimic-cardio-toy`) lives in the repository as the toy problem.
  **Stage 0, done 2026-11-06:** the datasets root ([mimic/01](tickets/mimic/01-external-datasets-root.md)) and the
  tables ([mimic/02](tickets/mimic/02-tables.md)). **Pre-registered 2026-11-05:** E148
  ([03](tickets/mimic/03-masked-cells-natural-gaps.md), observed cells hidden; done 2026-11-06, the decoder better
  than the bar on both tables), E149 ([04](tickets/mimic/04-pattern-transplant.md), real patterns transplanted onto
  complete rows; done 2026-11-06, the decoder better than the bar on both donor tables) and E150
  ([05](tickets/mimic/05-downstream-mortality.md), fills and patterns judged by a mortality classifier; done
  2026-11-06, the fills do not help, the indicators help on the SOFA table only).
  **E152 pre-registered 2026-11-06** ([06](tickets/mimic/06-sofa-positive-control.md), the user's go): C15's gate G1
  (T45's, E128's form) on the SOFA table's natural gaps, A1/A2/A3 × 42/7/13; if it passes, RQ2's H2 has its first
  table; it starts on gorgona8 when E149's cells end. **Done 2026-11-06 16:34 GMT-3: G1 fails** (A1's bilirubin
  contrast −0.651 logit, below 0; A2 − A3 −0.0153, corrected interval across 0, seeds disagree): RQ2 stops at the
  gate on natural gaps too, as for C15.
- **T51 · H2 behind an external gate** (the user's "vamos no 1", 2026-11-06). C15's G1 failed three times on the
  comparator's indicator (A2 against A3), never on A1's reading of the gap, so H2 could never run. The gate moves to the
  data (E123's G2 for P, E150's logistic regression for the SOFA table); a named deviation from E128's binding stop.
  **E155 pre-registered 2026-11-06** ([c15-mnar-prediction/11](tickets/c15-mnar-prediction/11-h2-external-gate.md)):
  A1 against A2 at fresh seeds 8282/8383/9292 on P's s4848 draws and siblings (54 cells) and on the SOFA and cardio
  numerical tables (12), all on gorgona8; two primaries with Holm, read once.
  **E155 done 2026-11-06 21:48 GMT-3: inconclusive on both primaries** (pooled over P F1 −0.027, *reversed* missed
  by 0.0024, A1 inferior on vehicle probit; SOFA +0.002); the secondary reads *gains* (+0.016, plain interval; the
  corrected one crosses 0): A1 reaches parity with A2 on the outcome-dependent files and trails under MCAR. No seeds
  added, no ADR; the thesis sentence is the user's call. T51 done.
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
