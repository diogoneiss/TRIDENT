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
- **T12 · A lower `LR_PRE` with the normalised objective:** every full-profile winner chose 2–34x
  below the default (E21).
  - **Skipped by the user's decision (2026-10-21),** on my recommendation. T28's decision table pointed here (E53's `BATCH 64` moved credit-g's probe), but E52 and E53 show the decode stage rebuilding the encoder from scratch, so a better pre-trained one would be overwritten too.
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
  - **Next, if it holds, each its own ticket:**
    - E: does the spread grow when a cell's top sources are missing in its row?
    - F: reliance on the gap pattern under MNAR.

- **T14 · A steps-aware batch rule** (at least ~8 steps an epoch) instead of one batch for every
  table (E28).
- **T15 · MAE-style architecture** (ADR 0004 Design C, handoff item 5): the most expensive; only if
  T02–T03 fall short.

### Later baselines and classification (plan 2026-10-17)

- **T27 · MaCoDE, TabPFN v2 and classification step zero** (plan §2g; review items 10–12).
  **MaCoDE done 2026-10-19 (E48,** [imputation-protocol/07](tickets/imputation-protocol/07-macode.md)**):** TRIDENT beats it on 20/21 and 14/15, never worse; an external deep baseline only. TabPFN v2: E55 done 2026-10-21 ([imputation-protocol/10](tickets/imputation-protocol/10-tabpfn.md)): better than the strong tier on 11/15 confirmation variants. **The user accepted it into the strong tier (ADR 0028, 2026-10-21)** as an offline member; K − strong now reads 5/9/7 on development and 6/4/5 on confirmation. E56's panel: TabPFN's 80nan wins come with more shrunk fills. E57 (KC − TabPFN 6/4/1/4) and E58 (every other seed set, MNAR included) done; the thesis table is rebuilt in [2026-10-21-thesis-table-with-tabpfn](reports/2026-10-21-thesis-table-with-tabpfn.md), and the baselines page (artifact 7GTHy9rjKfPK2dWASRnJgo) is at its third version. E60 done 2026-10-22 ([imputation-protocol/15](tickets/imputation-protocol/15-tabpfn-blend.md)): an equal-weight blend of K's and TabPFN's fills beats both on 10/15 confirmation variants (complementary; descriptive). E66 done 2026-10-23 ([imputation-protocol/21](tickets/imputation-protocol/21-tabpfn-blend-replication.md)): on E40's tables and seeds it beats both on 3/9, complementary at the floor; the effect shows on numerical tables where the two are close. **Step zero: plan and ADR proposed 2026-10-23** ([plan](plans/2026-10-23-classification-step-zero.md), [ADR 0029](adr/0029-classification-search-ranked-on-validation.md), proposed); waits on the user's eight decisions. **Stage 0 running since 2026-10-23 as E70** ([classification-step-zero/01](tickets/classification-step-zero/01-determinism.md)), on the user's go ("sim aprovo", 23:44 GMT-3). **Done: does not reproduce** (fold differences up to 0.076 macro-F1 against a 0.009 duplicate spread; the regression test passes), so Stage 2 reruns arm A on gorgona8. **Stage 2 running from 2026-10-24 as E71** ([classification-step-zero/02](tickets/classification-step-zero/02-no-pretraining.md)): arms A/B/C on 45 variants, seeds 42/7/13 as the GPU window allows. **Done 2026-10-24: no pre-training gain detected** at any level (B − A and C − A no verdict, means −0.005 to +0.004; seeds 42 and 7).
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
