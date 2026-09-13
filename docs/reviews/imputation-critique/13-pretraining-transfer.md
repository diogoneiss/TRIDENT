# 13 - What the shared pre-training stage contributes to imputation

_Critique of the imputation work on branch feat/imputation-task. 2026-09-11._

**Scope:** `src/models.py:21-76` (`TridentPretrainer`), `src/training/pretraining.py:1-126`
(`train_pretrainer`), `src/training/decoding.py:32-100` (the consumer), `src/embedder.py:30-57`
and `:161-318` (`EncodedTable`, `encode`, `forward`), `src/transformer.py:69-110`
(`EncoderLayer`, `TabularTransformerEncoder`), `src/utils.py:18-60` (`preprocess_table`),
`src/training/schedulers.py:24-108` (`StageScheduler`, `PLATEAU_*`), `opt.py:82-145`
(`define_search_space`), ADR 0004 decisions 2/4, ADR 0005 decisions 2/4/6, the imputation
runs in `mlflow.db`, and `tests/unit/` + `tests/integration/` for pre-training coverage.

**Method:** oriented with `graphify query "what does pre-training contribute to the
imputation decode stage"`, then read the source. Three empirical exercises, all under
`uv run --python 3.10`, all scripts in
`C:\Users\DIOGON~1\AppData\Local\Temp\claude\C--Users-Diogo-Neiss-Documents-Mestrado-TRIDENT\f48d5c06-b3ce-4977-b950-7a95c1d2053f\scratchpad\critic13`:

1. **`probe.py`** — instruments the *real* `train_pretrainer` (no repo edit: the module
   global `TridentPretrainer` is swapped for a capturing subclass) through a full
   300-epoch run at `Hyperparameters()` defaults on `credit-g_20nan`, CPU, and measures
   three scale-free properties of the representation each epoch against a mask that is
   drawn once and never re-rolled. Its reconstruction of the loss matches the loop's own
   `pretrain/val_loss` to 0.4% at both ends, so it measures exactly the logged quantity.
2. **`ablate.py`** — a paired `EPOCHS_PRE` ablation, 300 vs 2 at fixed
   `EPOCHS_DECODE=150`, two seeds, `credit-g_20nan`, `cv_folds=2`, `lr_scheduler=cosine`,
   `tracking_enabled=False` and both `output_dir`/`metrics_dir` inside the scratchpad. I
   ran this as a probe script rather than as the one permitted `main.py` invocation
   because a transfer question needs two arms; it carries the same isolation the permitted
   form does (MLflow off, nothing written under `metrics/`, `results/` or `mlflow.db`), and
   I did not spend the `main.py` allowance at all.
3. **sqlite reads of a scratchpad *copy* of `mlflow.db`** (never the file itself) for the
   `plateau` learning-rate histories.

Plus `git diff main...HEAD` over `src/models.py`, `src/training/pretraining.py`,
`src/embedder.py` and `src/transformer.py`, to separate what this branch decided from what
it inherited — which turns out to matter for how F-13-1 should be read.

Not checked: whether a *differently designed* pre-training objective would help (out of
scope), and GPU behaviour — every number below is CPU float32.

## Findings

### F-13-1 - 300 epochs of pre-training end 4.2x worse than a 20-vector constant lookup on pre-training's own objective; the 98.5% loss drop is the encoder's output norm shrinking, not learning

- **Kind:** methodology
- **Severity:** critical
- **Where:** `src/models.py:40-76`; `src/training/pretraining.py:69-120`
- **Evidence:** the loss is
  `mse_loss(encoded[:,1:,:][mask], emb_target[:,1:,:][mask])` with
  `emb_target = self.embedder(original)` under `no_grad` (`models.py:53-72`). Both sides are
  free to shrink: `TabularEmbedder.forward` ends at
  `final_embeddings = final_embeddings + pos_embeds` (`embedder.py:301`) with no
  `LayerNorm` and no `F.normalize` anywhere in it, and the `normalize` calls that would
  have made the objective scale-free are sitting commented out two lines above the loss
  (`models.py:66-67`, with a now-dead `self.eps` at `:35` labelled "to avoid division by
  zero in optional normalization"). There is no predictor head, no EMA target, no
  variance or covariance term — the only asymmetry is the stop-gradient.

  I ran the real stage for the full 300 default epochs on `credit-g_20nan` and measured,
  on a mask drawn once:

  - **NMSE** = (squared error on hidden cells) / (squared error a *per-column mean target
    embedding* would make on the same cells). 1.0 = as good as a constant per column;
    above 1.0 = worse than that constant.
  - **centred cosine** = `cos(enc − mean_col, tgt − mean_col)`, i.e. agreement about which
    value was hidden, with the free column-constant part removed.
  - the two norms.

  ```
  epoch | pretrain/val_loss |   NMSE | centred cos | ||tgt|| | ||enc||
      0 |           1.60803 | 187.61 |     -0.0109 |   2.384 |  14.530
     10 |           0.76391 |  90.91 |     -0.0152 |   2.402 |  10.227
     50 |           0.20019 |  26.63 |     -0.0103 |   2.468 |   5.692
    150 |           0.10430 |  14.85 |     -0.0057 |   2.501 |   4.506
    299 |           0.02412 |   4.21 |      0.0001 |   2.622 |   3.204
  ```

  The probe's own MSE/dim is 1.60655 vs the loop's logged 1.60803 at epoch 0 and 0.02402
  vs 0.02412 at epoch 299 — the same quantity, to 0.4%.

  Read the columns together. The logged loss falls 98.5%, matching the 95-99.7% drops in
  the store. But `||enc||` falls 14.53 → 3.20 while `||tgt||` *rises* 2.38 → 2.62: the
  reduction is the transformer's output being squeezed down onto the target cloud. The
  centred cosine goes from −0.011 to +0.0001 — after 300 epochs the output at a hidden
  cell is **orthogonal** to what was hidden there. And NMSE finishes at 4.21, i.e. the
  trained encoder makes 4.2x the squared error that a lookup table of twenty constant
  vectors (one per column) would make on the same cells. In absolute terms, that trivial
  predictor scores `val_loss` 0.0057; the trained model scores 0.0241.

  The mechanism is visible in the same numbers. At epoch 299 the target's second moment is
  `2.622²/128 = 0.0537` per dimension while its variance around the column mean is 0.0057:
  **89% of the target's energy is the column-constant part** — the positional embedding
  plus the column's mean value embedding — and the transformer is *given* the positional
  embedding in its own input. A pooled raw MSE cannot tell "reproduced the free part" from
  "predicted the hidden value", and the shipped stage does the first (badly) and not the
  second at all. The per-kind split is the giveaway: categorical cells carry 0.641 of the
  pooled squared error and are 858/1339 = 0.641 of the hidden cells, so per-cell error is
  identical for the two kinds — the residual is a kind-agnostic scale mismatch, not
  reconstruction of anything.

  *"It just needs more optimizer steps" does not rescue it.* `credit-g_20nan` at the
  default `BATCH=256` gives 3 steps per epoch, so 300 epochs is only 900 steps. I re-ran
  the identical probe at `BATCH=64` — same rows, same epochs, 3300 steps:

  ```
  epoch | pretrain/val_loss |   NMSE | centred cos | ||tgt|| | ||enc||
      0 |           1.22716 | 142.85 |     -0.0155 |   2.387 |  12.743
     50 |           0.06669 |   9.75 |     -0.0020 |   2.529 |   3.922
    150 |           0.01441 |   2.64 |      0.0104 |   2.698 |   3.005
    299 |           0.00635 |   1.51 |      0.0257 |   2.944 |   3.029
  ```

  3.7x the steps buys a 3.8x lower loss (0.0241 → 0.0064) and NMSE **1.51** — still worse
  than the constant, with a centred cosine of 0.026. The direction of travel is right and I
  will not claim the objective is unlearnable; what the two runs establish is that at and
  around the shipped configuration the stage does not reach parity with a twenty-vector
  lookup table, and that the loss number moves 3.8x between two configurations that are
  both on the wrong side of that line. A metric that ranks 0.0064 above 0.0241 while both
  mean "worse than constant" is not measuring what its readers think it measures.
- **Consequence:** every pre-training number the branch produces is uninterpretable as
  evidence of learning. That is `pretrain/train_loss`, `pretrain/val_loss`, the
  `cv/pretrain/*/mean` and `ci95` aggregates (`types.py:52-53`), the pre-training half of
  every loss-curve artifact, and the "1.765 → 0.051" style summaries a reader takes from
  the store. It is also not comparable *between* runs: the target is produced by trainable
  parameters, so two configurations with different `DIM`, `WEIGHT_DECAY_PRE` or vocabulary
  sizes have differently-scaled targets and their `pretrain/val_loss` values are on
  different axes. Concretely: ADR 0005 decision 2 holds `EPOCHS_PRE` at 300 on the ground
  that "pre-training is shared and protected", and ADR 0004 decision 4 forbids the decode
  stage from re-running the Xavier sweep because it "would erase the pretrained encoder" —
  both protect an artefact that, measured on the only objective that produced it, is worse
  than a constant.

  **This branch did not introduce the flaw.** `git diff main...HEAD -- src/models.py`
  shows the objective arriving unchanged from `main`: the `mse_loss(enc_sel, tgt_sel)`
  line, the two commented-out `normalize` calls above it and the unused `self.eps` are all
  byte-identical on `main`, and the branch's edits to `TridentPretrainer.forward` are the
  tensor-native rewrite only (`EncodedTable` inputs, `no_grad` in place of `.detach()`,
  `masked.masked_positions` in place of the per-column pandas comparison, a detached metric
  tensor) — mathematically the same loss. `pretraining.py`'s diff is likewise the
  encode-once loop and the `StageScheduler` swap. That makes the finding *about the
  branch's own decisions*, not about inherited code: the branch built a second task on this
  stage, made "pre-training is untouched / shared and protected" a load-bearing premise in
  two ADRs, forbade the decode stage from disturbing its output, and is about to spend
  nineteen GPU-hours of studies around it — all without ever measuring what it produces.
- **Direction:** two separable repairs. (a) Make the objective scale-free or grounded —
  cosine/normalised target, or a predictor head with a stop-gradient target, or (ADR
  0004's own deferred "value loss only") reuse the decoder heads during pre-training so
  the stage has a value-space anchor. (b) Independently of (a), log NMSE-against-column-mean
  and centred cosine next to the loss, so the failure is visible in the run record; see
  F-13-3. Until one of these lands, the honest reading of a pre-training loss curve is
  "the two sides met", and the ADRs' protection of the stage should be restated as a
  choice made without evidence.

### F-13-2 - A paired ablation cannot detect any benefit from 298 of the 300 pre-training epochs: the arms sit inside fold-to-fold noise with the sign flipping between seeds, and nothing in the branch has ever run this comparison

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/decoding.py:54-58` (the seam); ADR 0005 decision 2 (`EPOCHS_PRE`
  held at 300); ADR 0004 decision 4
- **Evidence:** nothing in the branch attributes any decoded number to pre-training. ADR
  0004 calls the stage "protected"; ADR 0005 holds `EPOCHS_PRE` on the ground that it is
  "shared and protected"; the only measurement anywhere is critic 02's open question — a
  one-fold, 40-epoch probe whose *no*-pre-training arm came out ahead (0.9435 vs 0.9604),
  which they correctly called too weak to conclude anything.

  I ran the paired ablation it asks for. `credit-g_20nan`, `cv_folds=2`,
  `lr_scheduler=cosine`, `EPOCHS_DECODE=150` fixed, every other hyper-parameter at its
  default, two seeds, only `EPOCHS_PRE` moved (lower is better throughout; 1.0 is
  mean/mode parity):

  ```
  seed | EPOCHS_PRE | masked impute_score | induced impute_score | acc_cat | rmse_num_z | s
    42 |        300 |              0.9358 |               0.9218 |  0.6288 |     0.9111 | 305.8
    42 |          2 |              0.9454 |               0.9095 |  0.6261 |     0.9226 |  71.4
    43 |        300 |              0.9058 |               0.9173 |  0.6300 |     0.9296 | 149.7
    43 |          2 |              0.9241 |               0.9311 |  0.6190 |     0.9306 |  80.8
  ```

  On `impute/induced/impute_score` — ADR 0005's own headline, and the only population with
  real ground truth — the arm means are 0.9196 (300 epochs) and 0.9203 (2 epochs): a
  difference of **0.0007**, and the sign flips between the two seeds (seed 42 favours the
  2-epoch arm by 0.012, seed 43 the 300-epoch arm by 0.014). On
  `impute/masked/impute_score` the 300-epoch arm is ahead in both seeds, by 0.0096 and
  0.0183, mean 0.014. Put that against the noise measured in the same runs: the two folds
  of a *single* run differ by 0.042 (seed 42, 2 epochs: 0.9244 / 0.9663) and 0.047 (seed
  43, 300 epochs: 0.9295 / 0.8822), and the same arm moves 0.030 between seeds. Both arms
  land in the band the store already shows for this variant (median 0.927, min 0.895).

  The one place the arms differ in the same direction on both seeds is the *masked*
  population — self-masked cells, presented to the model as `[MASK]`, which is exactly
  pre-training's pretext. On `induced`, the `[NULL]`-origin population that is the actual
  imputation benchmark, there is nothing. That pattern is what F-13-1 predicts: the stage
  familiarises the encoder with the `[MASK]` token and teaches it nothing about values.

  One confound to name, because it is real: cutting `EPOCHS_PRE` from 300 to 2 changes how
  much of the `np.random` and `torch` streams pre-training consumes, so at the same seed
  the decode stage's re-rolled training masks (`decoding.py:92-94`) and its `randperm`
  orders differ between the arms. The pairing is therefore "only `EPOCHS_PRE` was
  configured differently", not "the decode stage saw identical corruption". The two-seed
  design folds that into the noise it is being compared against, which is the right place
  for it, but a definitive run should draw the decode-stage masks from a stream seeded
  independently of the pre-training loop.
- **Consequence:** the default imputation recipe spends 300 of 450 training epochs on a
  stage that, measured, does not move its headline metric outside noise. ADR 0005's six
  studies are about nineteen GPU-hours; the same ratio applies inside every trial. More
  importantly, the branch ships and documents a two-stage story — ADR 0004 decision 2, "a
  decode stage replaces fine-tuning; pre-training is untouched" — in which the first stage's
  contribution has never been measured, and the one number a reader could reach for
  (`pretrain/val_loss` falling two orders of magnitude) does not mean what it appears to
  mean. Anyone reading a promoted configuration will believe the pretrained encoder is
  doing work.
- **Direction:** two seeds and one dataset is suggestive, not conclusive, and I say so
  plainly — but it is the same direction as critic 02's independent probe, and the
  mechanism in F-13-1 explains it. The decisive version is cheap next to the studies it
  would inform: the same ablation at `cv_folds=5` on three seeds across `credit-g_20nan`,
  `kr-vs-kp_20nan` and `spambase_20nan`, roughly the cost of one of the six studies. Run it
  *before* the nineteen hours, not after, and record the result in ADR 0005 next to the
  sentence that holds `EPOCHS_PRE` at 300.

### F-13-3 - Nothing in the branch logs, tests or asserts any scale-free property of the pretrained representation, so "pre-training worked" is unfalsifiable from a run's record

- **Kind:** methodology
- **Severity:** high
- **Where:** `src/training/pretraining.py:112-119`; `tests/unit/` (no pre-training test file)
- **Evidence:** the stage logs exactly three scalars per epoch — `pretrain/train_loss`,
  `pretrain/val_loss`, `pretrain/learning_rate` — and nothing else. `grep -rn
  "norm()\|cosine_similarity\|F.normalize" src/` returns **no hits at all**: no embedding
  norm, no cosine, no value-space reconstruction rate is computed anywhere in the package
  for the pre-training stage. The decode stage, by contrast, reports `acc_cat`,
  `rmse_num_z` and a baseline-relative `impute_score`, so the repository clearly knows how
  to state a scale-free result — it just never does it for the stage that runs first and
  costs the most.

  Coverage is the same story. There is no `tests/unit/test_pretraining.py` and no
  `test_models.py`; `train_pretrainer` is never executed by any unit test, only
  monkeypatched away (`test_training_runtime.py:134,177,449`). The single construction of
  `TridentPretrainer` in the suite is
  `PretrainingOutcome(model=TridentPretrainer(embedder, transformer), ...)` at
  `tests/unit/test_training_decoding.py:79` — an **untrained** pretrainer used as a
  container, which incidentally shows the decode stage runs fine on a random encoder. The
  only executions of the real stage are the two integration fixtures, and the imputation
  one pins `EPOCHS_PRE: 2` at `DIM: 16`
  (`tests/fixtures/credit-g_20nan_imputation_regression.json`) — it "pins determinism, not
  quality", in its own docstring's words.
- **Consequence:** F-13-1 is diagnosed on one dataset, but its *symptom* — a 95-99.7% loss
  reduction over 300 epochs — is what every one of the store's nine datasets shows, and no
  artifact, metric, assertion or review gate in this repository can tell that symptom apart
  from success on any of them. A reader of the MLflow store sees a monotone loss curve
  dropping two orders of magnitude and concludes the stage is working; there is no number
  anywhere that would say otherwise. The same blind
  spot covers any future change to the stage: swap in a different masking scheme, a
  different `LR_PRE`, a different embedder, and the only feedback is a loss whose units
  move with the change.
- **Direction:** compute the two diagnostics from F-13-1 on the fixed validation mask once
  per epoch (both are a few lines on tensors already in hand) and log them as
  `pretrain/nmse_vs_column_mean` and `pretrain/centred_cosine`. Then one unit test on a
  small synthetic table with a deterministic relationship between two columns, asserting
  NMSE drops below some threshold clearly under 1.0 — that is the assertion that would
  have failed on day one.

### F-13-4 - The reduced profile spends two thirds of every trial's epoch budget on a held stage whose contribution is unmeasured, and its one cross-stage knob makes the two stages' corruption inseparable

- **Kind:** design
- **Severity:** high
- **Where:** `opt.py:96-112`; ADR 0005 decision 2 and decision 6;
  `src/training/pretraining.py:71,74` and `src/training/decoding.py:93`
- **Evidence:** the reduced branch returns `PROB_MASCARA`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `DROPOUT` and conditionally `LAMBDA_NUM`. `EPOCHS_PRE` (300) and
  `EPOCHS_DECODE` (150) are held, so every trial pays `300/(300+150) = 2/3` of its epoch
  budget on a stage the search cannot touch. Wall-clock is lower than the epoch fraction,
  because a decode epoch is heavier than a pre-training one: the cleanest pair from
  F-13-2's ablation (seed 43, nothing else running) puts pre-training at 68.9 s of a
  149.7 s two-fold run, 46%. Either way it is the single largest fixed cost in a trial, and
  ADR 0005's context notes that the old full space, which sampled `EPOCHS_PRE` in 20..60,
  made trials "five to ten times cheaper than a default-configuration run" — so this cost
  is a deliberate, quantified increase, taken on a stage F-13-2 could not detect a benefit
  from.

  Decision 6 closes the loop the wrong way: `optuna.importance.get_param_importances`
  "ranks only the sampled knobs". A held knob has zero variance and is structurally
  invisible to fANOVA, so the mechanism the ADR installs to "check the reduction" can never
  report that two thirds of the budget bought nothing.

  Meanwhile the one sampled knob that is *not* stage-local is `PROB_MASCARA`. Both stages
  read the same field: `preprocess_table(train_frame.copy(),
  p_base=hyperparameters.mask_probability, fine_tunning=False)` at `pretraining.py:71` and
  `:74`, and the identical call at `decoding.py:93`. `LR_DECODE` and
  `WEIGHT_DECAY_DECODE` are stage-local by construction (they only reach the decode
  `AdamW`), and `LR_PRE`/`WEIGHT_DECAY_PRE` are held — so the search has *no* way to move
  pre-training alone, and the one dimension that moves both moves them by the same amount.
- **Consequence:** a study cannot attribute its win. If the reduced profile's best trial
  lands at `PROB_MASCARA=0.6`, nothing in the record says whether 60% corruption suited the
  decoder's training masks or pre-training's, and the fANOVA importance for `PROB_MASCARA`
  is the sum of two effects that may have opposite signs. Combined with F-13-1, the likely
  reading is worse: twelve and a half GPU-hours of the nineteen are spent on a stage
  measured to produce a representation worse than a constant, and the study's protocol
  cannot notice.
- **Direction:** cheapest first — run one full-profile pilot that *does* sample
  `EPOCHS_PRE` (the ADR defers this "to the fog"; it is the measurement that decides
  whether the held set is right), or simply put `EPOCHS_PRE ∈ {0, 300}` into the reduced
  profile as a two-level factor, which costs one extra dimension and answers the question
  the profile is silent about. Separately, split the corruption rate into
  `PROB_MASCARA_PRE` and `PROB_MASCARA_DECODE` so the sampled dimension is attributable;
  defaulting both to the current field keeps every existing run comparable.

### F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions: measured 11 halvings to 1.66e-07 on spambase, from a signal with a 9% noise floor

- **Kind:** bug
- **Severity:** medium
- **Where:** `src/training/pretraining.py:73-75` and `:120`;
  `src/training/schedulers.py:99-104`
- **Evidence:** the decode stage draws its validation mask once per fold and says why in
  its module docstring — "selects its checkpoint on a validation mask that never changes so
  that the loss moves only when the model does" (`decoding.py:3-5`, implemented at
  `:71-82`). Pre-training re-rolls both masks every epoch (`pretraining.py:70-75`) and then
  feeds the resulting number to `scheduler.after_epoch(average_validation_loss)`
  (`:120`), which under `--lr_scheduler plateau` is
  `ReduceLROnPlateau(..., factor=0.5, patience=10).step(validation_loss)`
  (`schedulers.py:90-93,103-104`).

  It is not hypothetical. Four of the six imputation parents in `mlflow.db` ran `plateau`.
  Reading `pretrain/learning_rate` from a scratchpad copy of the store:

  ```
  credit-g_20nan  fold 92278ca2  300 epochs  lr 3.40e-04 -> 3.40e-04  halvings 0
  credit-g_20nan  fold f20f7ad1  300 epochs  lr 3.40e-04 -> 3.40e-04  halvings 0
  spambase_20nan  fold e764df47  300 epochs  lr 3.40e-04 -> 1.66e-07  halvings 11, first at epoch 154
  spambase_20nan  fold f1077a34  300 epochs  lr 3.40e-04 -> 2.66e-06  halvings  7, first at epoch 204
  ```

  Fold `e764df47`'s halvings land at epochs 154, 178, 189, 209, 227, 238, 249, 260, 271,
  282, 293. From epoch 238 onward they arrive **every 11 epochs exactly** — `patience=10`
  plus one — which means the loop never once saw a new best over the last sixty-two epochs
  and halved at the earliest opportunity every time. The rate is under 1e-5 from epoch 238
  (62 epochs, a fifth of the stage) and under 1e-6 from epoch 271, finishing 2048x below
  the configured 3.4e-4.

  The noise floor is measurable on that same fold: across the 29 epochs where the rate is
  under 1e-6 and the model is therefore effectively frozen, the logged `pretrain/val_loss`
  still ranges 0.01265..0.01762 — a 9.0% coefficient of variation and a 1.39x spread,
  produced entirely by the mask re-roll. So the "best" the scheduler was trying to beat was
  a lucky mask draw, and beating it was a coin flip the model could not influence.
- **Consequence:** two things. First, a `plateau` pre-training run is not the configured
  run: on spambase the schedule froze the stage on mask noise, and any comparison of that
  run against a `cosine` one attributes to the schedule what is partly a truncation of
  training. Second, the same fold shows the stage's kept state is worse than one it passed
  through — best `val_loss` 0.0121 at epoch 215, last-epoch 0.0174 — and pre-training keeps
  the last epoch by design (ADR 0005 context), so the noisy signal is good enough to cut the
  learning rate but is never allowed to choose a checkpoint. The asymmetry with the decode
  stage, which fixes its mask precisely so its selection signal is clean, is inside one
  runner and undocumented.
- **Direction:** draw pre-training's validation mask once per fold, exactly as
  `decoding.py:71-82` does — the training mask re-roll is the pretext task and should stay.
  That is a behaviour change to a protected stage, so it wants the repository's usual
  flag-and-tag treatment; the cheap intermediate is to feed `plateau` a signal it can trust
  by averaging over several mask draws, or to document that `plateau` is not a supported
  schedule for pre-training.


## Checked and cleared

- **The embedder does not collapse.** The obvious BYOL-style failure — shrink the target
  to zero — did not happen: `||tgt||` *grew* 2.384 → 2.622 over 300 epochs, despite
  `AdamW(weight_decay=0.005)` acting on the embedder that produces the target
  (`pretraining.py:50-54`). The loss reduction is entirely on the encoder-output side. The
  reason is structural and worth recording: `TabularTransformerEncoder.forward` begins
  `x = self.pre_norm(x)` (`transformer.py:108`), so the embedder's output scale is
  normalised away at the transformer input and shrinking it buys nothing on the prediction
  side while making the target harder to hit in relative terms. The collapse the design
  invites is real but it takes the other route — F-13-1.
- **The per-kind loss balance is not skewed by the Xavier sweep.** I expected
  `model.apply(initialize_weights)` (`pretraining.py:45-49`), whose
  `xavier_uniform_` on `nn.Embedding(num_categories, dim)` scales with the column's
  cardinality, to give categorical and numerical targets very different magnitudes and so
  let one kind dominate the pooled `mse_loss` the way `TridentDecoder` deliberately
  prevents. Measured, it does not: categorical cells carry 0.641 of the pooled squared
  error and are 0.641 of the hidden cells, at every epoch from 0 to 299. Per-cell error is
  the same for both kinds.
- **The Xavier sweep's coverage gap has no reachable consequence through the decoder.**
  `initialize_weights` matches only `nn.Embedding` and `nn.Linear`, so `cls_token` and the
  numerical `[MASK]`/`[NULL]` parameters keep their `torch.randn` initialisation
  (`embedder.py:144-153`) at roughly 8x the norm of a Xavier-scaled category embedding.
  That gap reaches the transformer only through `pre_norm`, which is a per-token
  `LayerNorm` and removes it; and it reaches the decoder not at all, because those
  parameters are inputs, never targets — `TridentDecoder` reads `targets.cat_indices` and
  `targets.num_values` as label tensors, never an embedding. The decoder's heads taking
  framework defaults (`models.py:135,139-148`) is therefore an init inconsistency with no
  measured effect, not a defect.
- **The probe measures the shipped objective, not a re-implementation.** `probe.py` swaps
  the module global `src.training.pretraining.TridentPretrainer` for a capturing subclass
  and otherwise calls the real `train_pretrainer` with a real `PreparedDataset` from
  `prepare_dataset`; its independently computed MSE/dim reproduces the loop's own
  `pretrain/val_loss` to within 0.4% at epoch 0 and epoch 299. The NMSE denominator is the
  per-column sample mean over the same hidden cells, which is optimistic for the baseline
  by a factor `(1 − 1/k)` with `k ≈ 40..70` cells per column — under 3%, nowhere near the
  4.21 it would have to explain away.
- **Pre-training keeps its last epoch, and nothing else selects on the noisy signal.** I
  checked whether the re-rolled validation loss chooses a checkpoint, a stopping point, or
  a fold ranking: `train_pretrainer` returns `PretrainingOutcome(model=model, ...)` with no
  best-state bookkeeping (`pretraining.py:122-126`), there is no early stopping, and
  `validation_losses` only reaches the summary as a loss band. The one decision it drives
  is the `plateau` learning rate — F-13-5.

## Open questions

- **Is the small, consistent `masked` gap in F-13-2 real?** The 300-epoch arm is ahead on
  `impute/masked/impute_score` in both seeds (0.0096 and 0.0183) while `induced` is a coin
  flip. If that survives more seeds it would mean pre-training buys familiarity with the
  `[MASK]` token and nothing about values — a finding with a clear consequence, since the
  branch's benchmark population is `induced`. Two seeds cannot separate 0.014 from a
  seed-to-seed spread of 0.030. Five seeds at `cv_folds=5` on `credit-g_20nan` alone would
  settle this particular gap for a fraction of one study's cost.
- **Is the failure dataset-dependent?** Both probes are `credit-g_20nan`. The `BATCH=64`
  run rules out step count as the explanation at 3.7x the shipped budget, but a wider table
  with a different categorical/numerical mix could behave differently, and the NMSE trend
  (4.21 → 1.51 as steps rise) leaves open where it plateaus. A 300-epoch probe on
  `spambase_20nan` (57 numerical columns, 18 steps/epoch) and on `kr-vs-kp_20nan` (the only
  all-categorical table) would close it; neither fits on CPU inside this review. Both reach
  the same 98-99% loss reduction in the store, which is the point: that number cannot
  distinguish the cases.
- **Would the deferred "Design B" ablation have caught this?** ADR 0004 lists tied heads as
  the design that "tells whether the embedding geometry is already decodable". Given
  F-13-1, running B is no longer an ablation of a working stage — it is the test of
  whether the stage produces a geometry at all, and it is now the cheapest such test.
