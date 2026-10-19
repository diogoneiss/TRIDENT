# T28 · Why pre-training adds little to direct imputation, and what the literature offers

**Status: a draft for the user, 2026-10-19.** This is a read-only investigation for task T28
(`docs/TASKS.md`). It trained nothing, wrote nothing to the MLflow store and ran no `--apply`
script. It read the code, the experimentation log, the critique and the wiki. The store was read
twice, read-only, only to confirm which metrics it holds; no number below comes from those reads.
Every number is cited to its E-number or file, and anything never measured is marked as such. The
output asked for: diagnosable hypotheses, the cheapest test for each, and where T13 (an EMA target
encoder) fits.

## In five lines

1. Under the shipped `embedding` objective the stage learns the column-constant part of its
   target and almost nothing else (E04). Its epochs buy nothing measurable on induced cells (E05,
   E24).
2. The normalised objective (today's default, ADR 0013) is the only one that beat no pre-training
   on more than one variant (3 of 4; the value target on pendigits only, the shipped one on none).
   At equal total epochs it wins on 1 of 4 variants, loses on 1 and ties on 2 (E25, N − C).
   Whether it still collapses has never been measured.
3. The decode stage repeats the pretext (hidden cells, the same rows, the same mask rate) with a
   value loss, about 3x the learning rate and 450 epochs on every parameter. A redundant first
   stage whose effect is washed out fits all the evidence. It is the leading hypothesis, not a
   proven one.
4. Two cheap diagnostics come before any new objective: a free store read of the per-epoch decode
   curves (pre-trained against not), and a short probe of collapse and linear decodability under
   the normalised objective. Neither writes to the store.
5. T13 is justified for imputation only if that probe shows the normalised objective still
   collapsing **and** a frozen pre-trained encoder beats a random one. Otherwise the cheaper move
   is to shorten or drop pre-training for imputation, by a pre-registered study and an ADR.

## 1. The evidence

Lower `impute_score` is better. "X − Y" is paired by seed and fold over 15 pairs. A verdict needs
the interval to exclude zero and all three seeds to agree in sign (E24 onward).

### What the fork's studies show

- **E04 (F-13-1): the loss drop is norm-matching.** Shipped objective, credit-g_20nan, 300 epochs,
  on a fixed mask: the loss falls 98.5%, NMSE against a per-column mean target goes from 187.61 to
  4.21, and the centred cosine ends at 0.0001. The output norm falls from 14.530 to 3.204, against
  a target norm of 2.622.
  - The verifier's independent probe found NMSE 7.70, a centred cosine at 1.9 SE, and only 19% over
    predicting the zero vector.
  - The column-constant part was 89% of the target's energy (critic; 89.5% for the verifier). It
    includes the positional embedding, which the transformer also receives as input.
- **E04, verifier addendum: the embedder erodes its own discriminative part.**
  - From the same initialisation, the within-column variance of the target fell from 0.00807 to
    0.00593 (−26%). The column-constant share rose from 83.3% to 89.5%.
  - Weight decay explains about 0.15% of the fall; the rest comes from the gradient through the
    input path.
  - The verifier's view: normalising the loss would not stop this, and a value-space anchor would
    (`13-pretraining-transfer.verdict.md`, "What this critique missed" 1).
- **E04: not unlearnable.** At `BATCH=64` (3,300 steps instead of 900), NMSE fell to 1.51 and the
  centred cosine reached 0.0257: still worse than a per-column constant, but moving the right way.
- **E05 (F-13-2): 300 pre-training epochs equal 2.**
  - Induced arm means were 0.9196 and 0.9203, the sign flipping between seeds, against 0.042–0.047
    of fold-to-fold noise.
  - On masked cells 300 epochs led on both seeds (by 0.0096 and 0.0183). That fits pre-training
    teaching familiarity with `[MASK]`, not values.
- **E21: the search shrank pre-training.** Every full-profile winner chose `LR_PRE` 2 to 34 times
  below the default, under the shipped objective, with 20–60 pre-training epochs sampled. The
  search preferred a stage that barely moves the encoder (T12).
- **E24: pre-training never helps against none.**
  - B − A (no pre-training): no difference on three variants. On pendigits_20nan pre-training
    *hurts* (−0.0216).
  - C − A (all 450 epochs on the decode loss): C better on pendigits (−0.0664), clearing `knn10`.
    A better on kr-vs-kp_40nan (+0.0184), resting on the cross-machine pair.
- **E25: only the normalised target beats no pre-training on more than one variant.** N − B is
  better on 3 of 4; V − B only on pendigits (−0.0206); A − B on none (E24).

  | variant | N − A | N − B | N − C | V − C |
  |---|---|---|---|---|
  | credit-g_20nan | +0.0060, none | −0.0054, none | −0.0044, none | −0.0106, none |
  | credit-g_80nan | −0.0147, **N** | −0.0290, **N** | −0.0099, none | +0.0315, **C** |
  | kr-vs-kp_40nan | +0.0001, none | −0.0191, **N** | −0.0288, **N** | −0.0056, none |
  | pendigits_20nan | −0.0373, **N** | −0.0157, **N** | +0.0291, **C** | +0.0243, **C** |

  At equal total epochs (N: 300 + 150; C: 0 + 450), normalised pre-training wins on 1 of 4, loses
  on 1 and ties on 2. So N's three wins over B (0 + 150) are partly "more epochs". The value target
  V never beats C.
- **E26: with the long decode, one variant keeps a gain.** M − C (300 + 450 against 0 + 450):
  kr-vs-kp −0.0205 [−0.0373, −0.0037], **M better**; none on the other three. M trains 750 epochs
  against C's 450, so even this win confounds objective and budget.
- **E36: aligning pre-training's tokens changes nothing.** With the gaps shown as `[MASK]` in
  pre-training, the arm was better on none of 21 variants and worse on electricity_20nan
  (+0.0105). Means: 0.7450 against 0.7446.
- **E07 (F-13-5): the stage selects nothing.** Pre-training re-rolls its validation mask every
  epoch and keeps its last epoch. Under `plateau` that noise halved the rate 11 times on
  spambase. Under `cosine`, the imputation default, nothing reads the signal.
- **E37/E38: capacity depends on table size.** The bigger encoder helps the large tables (−3.0%),
  is neutral on the medium ones (−0.2%) and hurts the small ones (+1.9%). What pre-training adds
  under it was never measured.

### What the evidence does not show

- **Collapse under the default objective.** Every collapse measurement (E04) is on `embedding`.
  For `embedding_normalized`, the default since ADR 0013, these are unmeasured: NMSE, the centred
  cosine, the within-column variance and the rank.
- **Pre-training's worth under today's defaults.** The only comparison against no pre-training
  with the long decode is E26, on four variants. It predates three changes: gaps shown as `[MASK]`
  (ADR 0014), the gap-chosen checkpoint (ADR 0015) and the bigger encoder (ADR 0018).
- **Its share of training time today.** The critique's 51.4% median predates the 450-epoch
  decode; today's share is unmeasured here. The parents hold `cv/time/pretrain_seconds/mean` and
  `cv/time/total_seconds/mean` (checked on an E36 parent), so one read-only query answers it.
- **A checkpoint to probe.** `--save_model` is off by default, and no imputation `.pt` file exists
  under `results/` or `mlruns/`. Every weight-level diagnostic below needs a short re-training
  outside the store.
- **Transfer to classification.** TRIDENT's paper credits pre-training with +1.7 macro-F1 (wiki
  `concepts/masked-embedding-reconstruction`), unreplicated. The verifier's `vehicle_00nan`
  contrast (+0.033 for 300 against 40 epochs) is confounded by the fine-tuning epochs.

## 2. What pre-training optimises now

The code is precise where T13's entry and plan §2h are loose. **The stop-gradient is on the same
*embedder*, not the same encoder, and the target is an input embedding, not an encoder output.**

- **Corrupted view.** `EpochMasker.draw` (`src/embedder.py`) hides observed cells and never a
  real gap.
  - The rate is `PROB_MASCARA` (0.5), scaled down by each row's missing share, and re-rolled
    every epoch.
  - A hidden categorical cell becomes its column's `[MASK]` id; a hidden numerical cell becomes a
    learned per-column mask vector.
  - Real gaps stay `[NULL]` unless `--pretrain_gap_token mask` (E36).
- **Target.** `self.embedder(original)` under `torch.no_grad()` (`src/models.py:54`): the clean
  cell's *input token*.
  - **Categorical column:** a row of an `nn.Embedding` codebook.
  - **Numerical column:** a per-column `Linear(1→16)–ReLU–Linear(16→128)` of the scaled value.
  - **Both,** plus the column's positional embedding (`src/embedder.py:379`).
  - No transformer layer touches it. This is a denoising autoencoder in input-embedding space, not
    a JEPA, which predicts encoder outputs (the distillation review's P12 says the same).
  - The embedder that makes the target is the one trained through the input path. So the target
    is detached but moves every step, and there is no EMA copy.
- **Prediction.** The transformer reads the corrupted tokens: a `pre_norm` LayerNorm, then 2
  pre-LN layers, `DIM` 128, 16 heads and feed-forward 32. The loss is read at the hidden cells
  only; `[CLS]` and the visible cells carry none.
- **Loss, `embedding`** (classification's default): pooled MSE between raw vectors, with no
  predictor (`src/models.py:72`).
- **Loss, `embedding_normalized`** (imputation's default, ADR 0011/0013).
  - The target is layer-normalised per cell over its 128 dimensions, with no affine
    (`src/models.py:109`).
  - A `Linear–GELU–Linear` predictor sits in front, discarded after the stage.
  - Predicting zero scores 1 at any scale, so E04's norm route is closed.
  - **What stays open:** normalising fixes each vector's norm, not its content. The embedder can
    still shrink the within-column differences against the column-constant part. If it does, the
    normalised target tends to `LN(pos_j + the mean embedding of column j)`: a per-column constant
    that the position alone predicts.
- **What the target means per kind.**
  - For a categorical column it is one of k codebook vectors, so the pretext is close to k-way
    classification (by MSE on a moving codebook).
  - For a numerical column it is a point on a learned one-dimensional curve in 128 dimensions.
  - So "embedding target against value scoring" is a weaker mismatch for categories than for
    numbers.
- **Optimisation.**
  - AdamW at `LR_PRE` 3.4e-4, weight decay 0.005, batch 256, 300 epochs, `cosine`. These are the
    defaults in `src/training/types.py`; a promoted file can override them.
  - A Xavier sweep re-initialises every `Embedding` and `Linear` first.
  - Credit-g_20nan gets 3 steps an epoch, 900 in all (F-13-1).
  - The last epoch is kept, and the validation mask is re-rolled every epoch
    (`src/training/pretraining.py:107`, `:150`).
- **The decode stage then** (`src/training/decoding.py:65–76`):
  - wraps the same embedder and transformer with fresh per-column heads: cross-entropy over the
    real categories, an MLP regression for numbers;
  - trains **every parameter**, with no freeze and no warm-up of the heads, at `LR_DECODE` 1e-3
    (about 3x `LR_PRE`) for 450 epochs;
  - draws its masks at the **same** `PROB_MASCARA` from the **same** training rows (F-13-4);
  - shows the gaps as `[MASK]` (ADR 0014) and chooses its checkpoint on the validation gaps (ADR
    0015).

  It is the pre-training pretext with a value target.

## 3. Hypotheses

Six hypotheses, roughly ordered by how much evidence each explains. They are not exclusive: H3 can
hold whatever H1 says.

### H1 · The objective collapses onto the column-constant part of its target

The stage lowers its loss by matching what it already knows (the position, the column's mean),
while the embedder erodes the part that identifies the value.

- **For:** E04 (NMSE 4.21–7.70, a centred cosine near zero, within-column variance −26%, 89% of
  the energy column-constant); E05 and E24 (nothing transfers); E21 (the search preferred less
  pre-training). The positional embedding sits in both the input and the target (Section 2).
- **Against:**
  - N, which closes the norm route, beats A on 2 of 4 variants and B on 3 of 4 (E25), so it is
    not a total collapse.
  - E04's `BATCH=64` run moves the right way.
- **Under N:** unmeasured. Layer normalisation does not stop the within-column erosion.
- **Literature:**
  - The standard guards are a stop-gradient, a predictor and an EMA target. N has the first two,
    and the EMA is the missing third (ADR 0011, considered options).
  - SimSiam (Chen and He 2021) shows the first two can suffice. BYOL (Grill et al. 2020), data2vec
    (Baevski et al. 2022) and I-JEPA (Assran et al. 2023) add the EMA teacher. This is background,
    not in the wiki.
  - All four take the target from an encoder *output*, never from an input embedding that the
    online branch also trains.
  - The wiki flags this exact risk (`concepts/masked-embedding-reconstruction` § Risks).
- **Cheapest test (D2, a collapse probe):**
  - **What to run.** Re-run `train_pretrainer` alone, with MLflow off and nothing written to the
    repository, for both objectives. Use credit-g_20nan, kr-vs-kp_40nan and pendigits_20nan, one
    fold and one seed each.
  - **What to log each epoch,** on a fixed mask and split by kind:
    - E04's two measures: NMSE against a per-column mean target, and the centred cosine;
    - the within-column share of the (normalised) target's variance;
    - the effective rank of each column's targets and predictions.
  - **Reuse.** E04's definitions are written out in the verdict file. Its scripts lived outside the
    repo and must be rewritten.
  - **Cost:** 2–3 code-hours. On the CPU, minutes per table: the verifier's 300 epochs on credit-g
    took 120.6 s. On pendigits, about a minute of GPU per fold (E24: A 13.5 against B 7.6 min a
    five-fold cell), plus the probe's own overhead.
  - **Reading.** It is collapse under N if the within-column share falls while the centred cosine
    stays near zero and NMSE stays at or above 1.

### H2 · The target is in the wrong space, so what the encoder learns does not transfer

The stage regresses input embeddings, while the decode stage and the scoring want values. A
representation that predicts embeddings need not decode linearly into values.

- **For:** the value target V beats A only on the all-numerical pendigits (E25 V − A −0.0421).
  That is the table where the embedding target is a 1-D curve rather than a codebook.
- **Against:**
  - An aligned target in its own stage does not help. V never beats C, and it is worse than A on
    credit-g_80nan (+0.0267) and kr-vs-kp (+0.0233) (E25).
  - For categorical columns the two targets are close (Section 2).
- **Literature:**
  - SMART's latent target beats input-space pre-training by 0.4–1.1 AUPRC, within one standard
    deviation (`references/smart-paper`, Table 2).
  - No ingested paper compares latent and value targets on imputation or on static tables
    (`concepts/masked-embedding-reconstruction`).
- **Cheapest test (a linear probe, added to D2):**
  - **What to fit.** At the hidden cells of the fixed mask, take the pre-trained encoder's output
    and fit per-column probes for the hidden value: logistic regression for categories, ridge for
    numbers. Fit on half the cells and score on the other half.
  - **Compare** against the same probe on a freshly Xavier-initialised encoder (B and C's
    starting point), and against mean/mode.
  - **Cost:** about 1 code-hour on top of D2, CPU minutes.
  - **Reading:**
    - pre-trained ≈ random-init: nothing value-relevant was learned (H1 or H2);
    - pre-trained clearly better: the information is there, and the limit is downstream (H3).

### H3 · The decode stage repeats the pretext and washes the encoder out

Pre-training sees no rows and no signal that the decode stage lacks. The decode stage then
retrains every parameter at about 3x the rate, for 450 epochs, from fresh heads. So pre-training
can only matter as an initialisation, and one that this stage can move far from.

- **For:**
  - E05 (300 epochs ≈ 2); E24 (on pendigits the epochs are worth more as decode epochs); E25 (N's
    gains over B mostly vanish against C); E26 (M ≈ C on 3 of 4); E36 (a change to pre-training's
    input does not reach the scores).
  - E05's masked-only gap ("familiarity with `[MASK]`") is now taught by the decode stage itself,
    which has shown every gap as `[MASK]` since ADR 0014.
- **Against:** kr-vs-kp keeps a gain both at equal epochs (E25 N − C −0.0288) and with the long
  decode (E26 −0.0205).
- **Literature:**
  - Mazrouei 2026 (`references/ssl-pretraining-tabular-paper`): mask-and-recover pre-training
    gains +0.0074 AUC at 5% labels and +0.0055 at 10%, neither significant, and turns slightly
    negative at 20%. Here every observed cell is a "label" for the decode loss, the far end of that
    curve.
  - MaskTab (`references/masktab-paper`) trains one hybrid stage against two stages' "objective
    misalignment". 20x more unlabeled rows add 1.9 KS points; 40x lose some.
  - ReMasker and MaCoDE are single-stage imputers.
  - SMART freezes its encoder for the first 5 fine-tuning epochs.
- **Cheapest test (D1, a store read; no GPU, no new run):**
  - **Data.** The parents hold `cv/decode/val_loss/mean` for every epoch: checked on an E26 parent
    (decode 450); E24/E25 (decode 150) log through the same path. Per-fold curves exist only on
    the best/worst-fold children, so use the parents' cross-fold mean.
  - **Pairing.** Compare arms with the same decode schedule, by variant and seed: A, B, N and V
    (150 epochs); C and M (450).
  - **Why the pairs compare.** The validation mask is drawn from its own stream per seed and fold
    (`evaluation_mask`, `src/training/data.py`), so every arm is scored on the same cells. The
    training masks do differ between arms (F-13-2's confound).
  - **Measures:** the gap at decode epochs 0, 10 and 50; the first epoch at which B or C reaches
    the pre-trained arm's loss; the gap at each arm's minimum.
  - **Filters.** Tags `ablation=pretraining-2026-09-29`, `experiment=pretrain-objective-2026-09-30`
    and `experiment=pretrain-decode-budget-2026-09-30`, with `run_role=parent` and
    `is_mirror=false`.
  - **Caveat.** The decode loss is the λ-weighted value loss, not `impute_score`.
  - **Cost:** 1–2 code-hours for a script under `scripts/experiments/`, which needs your go-ahead.
  - **Reading:**
    - a head start that closes within tens of epochs: washout;
    - none even at epoch 0: pre-training left nothing usable (back to H1 or H2);
    - a gap that persists to the minimum (expected on kr-vs-kp only): real transfer.
- **Second test (D3, a frozen-encoder probe and drift):**
  - **What to run.** A decode stage that trains the heads only, with the embedder and transformer
    frozen, for a fixed budget. Start it from three encoders: N-pretrained, A-pretrained and
    random (Xavier).
  - **What to record,** for the N-pretrained and random starts:
    - the linear CKA between the encoder's outputs before and after a normal decode stage;
    - the relative weight change per module.
  - **What it borrows.** It is the distillation review's "fixed-budget probe decoder" (l.221), and
    a cleaner form of ADR 0004's deferred Design B.
  - **Cost:**
    - about 2 code-hours, in a probe script outside the protected path (a flag would need the
      flag-and-tag treatment);
    - four variants × three seeds is E26's size: 12 cells in 03:00–05:00 GMT-3 while E25 shared
      the GPU. With no backward pass through the encoder it should cost less;
    - the drift half needs one full decode run per cell, and credit-g fits on the CPU.
  - **Reading:**
    - frozen-pretrained ≈ frozen-random: no usable representation (H1);
    - frozen-pretrained clearly better while a full decode ties C: washout (H3).

### H4 · The pretext suits categorical columns, not numerical ones

- **For:** pre-training's sign follows the column kind.
  - On the all-categorical kr-vs-kp it helps: E25 N − C −0.0288 and E26 M − C −0.0205. In E24, A
    beat C there (+0.0184, with the cross-machine caveat).
  - On the all-numerical pendigits it hurts or loses: E24 B − A −0.0216, E25 N − C +0.0291.
  - A mechanism that would fit is Section 2's: a codebook target is close to a class label, a 1-D
    curve is not. This mechanism is unmeasured.
- **Against:**
  - Two tables carry it all, and column kind is confounded with size (E38 puts pendigits in the
    large group, kr-vs-kp in the medium one).
  - pendigits also wants more decode epochs (E27: best epoch about 390).
  - Under the shipped objective, per-cell error was the same for both kinds on credit-g (F-13-1,
    "checked and cleared").
  - kr-vs-kp is the only all-categorical table (critique 13, open questions).
- **Literature:** MaCoDE puts numbers into 50 quantile bins with cross-entropy, so every column
  becomes categorical (`references/macode-paper`). MaskTab uses a type-aware value loss.
- **Cheapest test:**
  - D2's split by kind on credit-g (mixed), under N.
  - Free from the store: the per-kind test metrics (`acc_cat`, `rmse_num_z`) of the E25 and E26
    arms on the two credit-g variants.
  - Cost: inside D1 and D2.

### H5 · The stage is under-optimised or mis-scheduled

- **For:** E04 (900 steps on credit-g; 3.7x the steps took NMSE from 4.21 to 1.51); E28 (steps
  per epoch decided the decode stage's batch result); E07 (no epoch selection); E21 (the search
  moved `LR_PRE` far from the default).
- **Against:** E21 moved it *down*, with short pre-training: less pre-training, not more. E05: more
  of the same does not help.
- **Literature:** SMART pre-trains 25 epochs at 1e-3 on 11,988–41,903 patients. The wiki has no
  tabular evidence on pre-training budgets.
- **Cheapest test:**
  - Add two settings to D2 on credit-g: `LR_PRE` ÷ 4, and `BATCH` 64. CPU minutes.
  - If neither moves the diagnostics, H5 is not the limit.
  - If one does, T12 (a lower `LR_PRE` with N) becomes a screening arm in the E25 shape (12
    cells, E26's size).

### H6 · The tables, or the encoder, are too small for pre-training to pay

- **For:** the encoder is small (2 layers, feed-forward 32), and the bigger one helps only the large
  tables (E38). SMART's and MaskTab's gains come with far more rows than credit-g has.
- **Against:**
  - Pre-training uses the same rows as the decode stage, so no table size brings new information
    (H3).
  - The worst case for pre-training is the large pendigits (E24).
- **Literature:** MaskTab's scaling with unlabeled rows is small and non-monotone. Mazrouei's
  small MLP shows small gains.
- **Cheapest test:**
  - **What to run.** Reuse E38's ALL cells on pendigits_20nan and letter_20nan (seeds 101, 202,
    303) as the pre-trained arm. Run only the twin with `EPOCHS_PRE 0`: six cells.
  - **Cost:** roughly 4–6 GPU-hours, an estimate. A pendigits M cell is 35.4 min of fold time
    (E27), and ALL costs 1.84x (E38).
  - **Priority:** last. It writes to the store, needs a pre-registration, and tests the
    hypothesis with the least support.

### Tested and ruled out

- **The `[MASK]`/`[NULL]` mismatch between the stages:** E36 aligned them, with no gain on 21
  variants.
- **Scale-matching as the only failure:** N closes that route (ADR 0011), and still gains little
  against C (E25, E26).

## 4. What the literature offers

| source (wiki page) | what it offers | hypothesis | cost here |
|---|---|---|---|
| SMART (`references/smart-paper`) | EMA target encoder (decay 0.996); an embedding decoder; the target from the target *encoder*; L1 on removed entries; the encoder frozen for 5 fine-tuning epochs | H1, H3 | T13 as plan §2h: about a day of code and one GPU night. The unfreeze epoch: about 2 code-hours and a screening night |
| JEPA family (`concepts/joint-embedding-predictive-architecture`; data2vec, BYOL and SimSiam as background) | the three collapse guards; contextual (encoder-output) targets | H1 | inside T13, once its target depth is chosen (below) |
| `jeon2026jepa` (not ingested; selection review l.439–442) | a latent loss beside a value loss collapsed until the value head read the encoder; value-only led at plateau; one seed | H1, H2 | ADR 0011's rejected "combined loss": N's loss as an auxiliary decode-stage term. About half a day of code and a screening night |
| MaCoDE (`references/macode-paper`) | cross-entropy over 50 quantile bins for numbers; one stage; a per-row mask rate drawn from U(0, 1) | H2, H4 | bin heads in the decoder: about a day of code. As a baseline it is T27: 10–14 h of code, 3–5 GPU-h |
| ReMasker (`references/remasker-paper`) | one value-space stage; the loss on hidden **and** visible cells (letter RMSE 0.0629 → 0.0616, california 0.0840 → 0.0663); mask ratio 0.3–0.5 best | H2, H3 | a visible-cell term in the decode loss: about half a day of code and a screening night. An RQ3 arm, not a pre-training fix |
| MaskTab (`references/masktab-paper`) | one hybrid stage against "objective misalignment"; a type-aware value loss; small gains from unlabeled rows | H3, H4 | the same auxiliary-loss arm as `jeon2026jepa` |
| Mazrouei 2026 (`references/ssl-pretraining-tabular-paper`) | pre-training gains small, non-significant, fading as supervision grows; weakest on natively incomplete tables | H3 (a prior) | none: it sets expectations |
| Distillation review, 2026-10-09 (`_raw/review-2026-10-09/03-distillation-review.md`) | the RQ3 protocol (l.221, l.274): a 2×2 {latent, value} × {hidden-only, hidden + visible}, a fixed-budget probe decoder for the latent arm, collapse diagnostics on every arm, an EMA + predictor arm. The normalised objective is the latent arm; the shipped one is only a reference. P12 corrects "JEPA-style" for the shipped objective | all | D2 and D3 are its probe and its diagnostics. Adopting them now keeps the later RQ3 arms comparable |

**What the literature does not settle.**

- **SMART's guards are unproven.** SMART never ablates its EMA or its decoder, never mentions
  collapse, and its latent-against-input gap is within one standard deviation.
- **Pre-training's gains are small and come with labels.** SMART adds 1.5–2.1 AUPRC and TRIDENT's
  paper 1.7 macro-F1, both on prediction tasks whose labels the pretext never sees.
- **Imputation is the opposite case.** The decode stage *is* the pretext with a value target, so
  no ingested paper predicts a large gain from an EMA teacher here.

**T13 changes two things unless it is split.** SMART takes its target from the target encoder's
output, data2vec from its top layers; TRIDENT takes the embedder's input token. "Add an EMA" can
therefore mean three arms:

- **An EMA embedder behind today's target:** the smallest change. It slows the erosion of the
  target (H1).
- **An EMA encoder whose output is the target:** SMART's and data2vec's design, which also changes
  the target's depth.
- **Both,** with the predictor kept.

A cheaper fourth arm addresses E04's 89% column-constant share directly: take the positional
embedding and the column mean out of the target before normalising. That is about 2 code-hours. It
is this report's idea, derived from F-13-1, not a published recipe.

## 5. Recommended order

1. **D1, the decode curves from the store** (H3). No GPU, 1–2 code-hours, read-only. It shows
   whether there is a head start and how fast it closes: four variants, five arms.
2. **D2, the collapse and linear-probe script** (H1, H2, H4, H5), both objectives, three tables.
   It costs 3–4 code-hours plus CPU and GPU minutes, and writes nothing to the store. It is the
   measurement that ADR 0011 and T13 both assume and that nobody has made under N.
3. **D3, the frozen-encoder probe and drift** (H1 against H3). About 2 code-hours, under one
   E26-sized batch of GPU.
4. **Then decide.** Each "propose" below is a pre-registered study and an ADR for you to decide.
   - **D2 finds no value-relevant learning under N, and frozen-pretrained ≈ frozen-random:**
     propose shortening or dropping pre-training for imputation. E05 and E24 already point that
     way.
     - The study would be a no-pre-training arm (`EPOCHS_PRE 0`) under today's defaults on all 21
       variants. It costs less than E36's 63 cells (5 h 52 min), having no pre-training stage.
     - T13 then serves classification and RQ3 only.
   - **D3 finds a usable representation, but D1 shows it washing out:** try the washout levers
     before T13. They are SMART's unfreeze epoch (heads alone for the first k decode epochs) and a
     lower decode rate on the encoder. Each costs about 2 code-hours and a screening night.
   - **D2 shows N still collapsing, and frozen-pretrained > frozen-random:** this is where **T13**
     belongs, in plan §2h's screening shape.
     - Arms: an EMA embedder, an EMA encoder with a contextual target, and the centred target.
     - D2's diagnostics logged on every arm, as plan §2h and the distillation review ask.
   - **A diagnostic moves with `LR_PRE` or the batch:** run T12 first; it is cheaper than T13.
5. **Value-space and single-stage alternatives** belong to RQ3 and T27, not to the pre-training
   fix. They are ReMasker's visible cells, MaCoDE's bins, and MaskTab's and `jeon2026jepa`'s
   auxiliary loss.
6. **H6 last,** and only if the large tables matter for the thesis claim.

**Where T13 fits: after D1–D3.** Two results together would justify it for imputation:

- the within-column share of the normalised target falling during pre-training, with the centred
  cosine near zero (collapse under today's default);
- an N-pretrained frozen encoder beating a random one, so there is something worth protecting.

Without the first, an EMA has nothing to fix. Without the second, a guarded stage still feeds a
decode stage that ignores it. Classification or RQ3 can still justify T13, but that is a different
question from T28's.

## Sources

**Repository** (branch `feat/imputation-task`, read 2026-10-19):

- **Plans and decisions:** `docs/TASKS.md` (T12, T13, T27, T28),
  `docs/plans/2026-10-17-experiment-improvements.md` (§2h), ADRs 0004, 0011 and 0013.
- **Results:** `docs/EXPERIMENTATION_LOG.md` (E04, E05, E07, E21, E24–E28, E30, E36–E38);
  `docs/tickets/imputation-pretraining/02-…` and `03-…`;
  `docs/reviews/imputation-critique/13-pretraining-transfer.md` and `.verdict.md`;
  `docs/reports/2026-09-30-overnight.md`.
- **Code:** `src/models.py`, `src/embedder.py`, `src/transformer.py`, `src/utils.py`;
  `src/training/pretraining.py`, `decoding.py`, `types.py`, `data.py`, `runner.py`.
- **The MLflow store, read-only.** Two reads checked the tag counts of the E24/E25/E26/E36 runs and
  the metric keys of one E26 parent with its children and one E36 parent. No values were read.

**Wiki** (`/scratch2/diogoneiss/Wiki`, read-only): `references/` `smart-paper`, `macode-paper`,
`remasker-paper`, `masktab-paper` and `ssl-pretraining-tabular-paper`; `concepts/`
`joint-embedding-predictive-architecture` and `masked-embedding-reconstruction`;
`projects/trident/fork-experiments`; `_raw/review-2026-10-09/03-distillation-review.md` (the
distillation review) and `02-selection-and-gaps.md` (`jeon2026jepa`, `ssl2026incomplete`).

**Not found or not ingested:**

- No wiki page exists for data2vec, BYOL, SimSiam, I-JEPA, T-JEPA or `jeon2026jepa`. What this
  report says about the first four is background knowledge. `jeon2026jepa` is known only from two
  lines of the selection review.
- There is no stored imputation checkpoint to probe.
- E04's probe scripts lived outside the repository and are gone.
