# 13 - What the shared pre-training stage contributes to imputation: verdicts

_Adversarial verification of `13-pretraining-transfer.md`. 2026-09-11._

**Method:** read `src/models.py` (whole file), `src/training/pretraining.py` (whole file),
`src/training/decoding.py:1-110`, `src/training/schedulers.py` (whole file),
`src/transformer.py:92-110`, `src/embedder.py:30-57,275-305`, `src/utils.py:18-60`,
`opt.py:82-145`, `src/training/types.py:88-160`, ADR 0004 decisions 2/4, ADR 0005
context/decision 6/deferred alternatives, and
`docs/wayfinder/imputation-optuna-reduced/issues/01-held-set-and-ranges.md`. Oriented with
`graphify query "what does pre-training contribute to the imputation decode stage"` first.

Four empirical exercises:

1. **An independent pre-training probe**, `scratchpad/verify13/probe.py`, run as
   `PYTHONPATH=<repo> uv run --python 3.10 python .../probe.py` from the repo root. It does
   not monkeypatch anything: it calls the real `train_pretrainer` on
   `credit-g_20nan`, fold 0 of `build_folds(..., cv_folds=2, seed=42)`, at
   `Hyperparameters()` defaults (300 epochs, DIM 128, BATCH 256, `cosine_legacy`, p=0.5),
   CPU, with a no-op tracker, then measures the representation on **one** validation mask
   drawn once under `np.random.seed(1234)`. The same measurement is taken on an untrained
   model built through the identical constructor + Xavier `apply` path, as the epoch-0
   reference. 120.6 s.
2. **SQLite reads of a scratchpad copy of `mlflow.db`** (`cp mlflow.db <scratchpad>`; the
   store itself was never opened) for the `plateau` learning-rate histories, the
   `EPOCHS_PRE` values of every run, and `time/pretrain_seconds` on 185 GPU trials.
3. **`git diff main...HEAD`** and `git show main:<path>` over `src/models.py`,
   `src/training/pretraining.py`, `src/training/schedulers.py`.
4. Targeted greps for the coverage and instrumentation claims.

I did **not** re-run F-13-2's paired `EPOCHS_PRE` ablation (four training runs is outside
this review's budget). I verified its premise instead, and say so in that verdict.

Probe output, verbatim:

```
defaults: 300 epochs, dim 128 batch 256 sched cosine_legacy p_mask 0.5
UNTRAINED {'mse_model': 1.95824, 'mse_base': 0.00807, 'nmse': 242.72649, 'cos': -0.00598,
           'raw_cos': 0.19973, 'n_enc': 15.98494, 'n_tgt': 2.38759,
           'second_moment': 0.04845, 'within_var': 0.00807, 'const_share': 0.83347,
           'n_cells': 623}
train_pretrainer: 120.6s, val_loss[0]=1.70959 val_loss[-1]=0.04488 drop=97.4%
TRAINED   {'mse_model': 0.04564, 'mse_base': 0.00593, 'nmse': 7.70142, 'cos': 0.00678,
           'raw_cos': 0.74983, 'n_enc': 3.58597, 'n_tgt': 2.56105,
           'second_moment': 0.05627, 'within_var': 0.00593, 'const_share': 0.89468,
           'n_cells': 623}
```

All quantities are over the hidden cells of that one mask, with `enc = transformer(embedder(
masked))[:, 1:, :][mask]` and `tgt = embedder(clean)[:, 1:, :][mask]`, and `base` the mean of
`tgt` over the hidden cells of the same column: `mse_model = mean((enc-tgt)^2)` — the pooled
`mse_loss` the stage optimises; `mse_base = mean((base-tgt)^2)`; `nmse = mse_model/mse_base`;
`cos = mean(cosine(enc-base, tgt-base))`; `second_moment = mean(tgt^2)` (what predicting the
zero vector scores); `within_var = mse_base`; `const_share = 1 - within_var/second_moment`.
`mse_model` 0.04564 against the loop's own last-epoch `pretrain/val_loss` of 0.04488 — 1.7%
apart, the difference being the mask draw, so this is the logged quantity.

## Verdicts

### F-13-1 - Pre-training's own loss falls 97-99% while the encoder ends worse than a constant-per-column lookup, and carries no per-cell information about the hidden value

- **Verdict:** SOUND
- **Severity after review:** high _(critic said critical)_
- **Basis:** every premise checks out in the source, and the measurement reproduces
  independently on a different fold with a different mask.

  The objective is `nn.functional.mse_loss(enc_sel, tgt_sel)` (`models.py:72`) on
  un-normalised vectors; the two `nn.functional.normalize` calls sit commented out directly
  above at `models.py:66-67`, and `grep -rn "self.eps" src/` returns exactly one line —
  the assignment at `models.py:35` — so it is dead. `grep -n "LayerNorm\|BatchNorm"
  src/embedder.py` returns **nothing**, and `TabularEmbedder.forward` does end at
  `final_embeddings = final_embeddings + pos_embeds` (`embedder.py:301`). The target is
  `self.embedder(original)` under `no_grad` (`models.py:53-54`) — trainable parameters, no
  predictor head, no EMA, stop-gradient as the only asymmetry.

  My run, on hidden cells of a fixed mask: the logged loss falls 1.70959 → 0.04488
  (−97.4%), `||enc||` falls 15.98 → 3.59 while `||tgt||` **rises** 2.388 → 2.561, the
  centred cosine moves −0.0060 → +0.0068, and NMSE against a per-column constant ends at
  **7.70** (the critic measured 4.21 on their fold; both are far above 1). The trivial
  constant predictor scores 0.00593 against the trained model's 0.04564.

  Two numbers of my own make the conclusion harder to argue with than the critic's NMSE:

  - **Against the zero vector.** Predicting nothing at all scores `second_moment` =
    0.05627. The trained encoder scores 0.04564. Three hundred epochs buy **19% over
    outputting zero**, and leave the model 7.7x worse than twenty constants. No oracle
    baseline is needed for that comparison, which makes it the one to quote.
  - **The centred cosine is statistically zero.** With d = 128 and 623 cells the standard
    error of a mean cosine under no alignment is 0.0035; the measured +0.0068 is 1.9 SE.
    Meanwhile the *raw* cosine reaches 0.75 — the output aligns with the column-constant
    part (89.5% of the target's energy, which includes the positional embedding the
    transformer is handed in its own input) and with nothing else.

  So "the drop is scale matching, not reconstruction" survives hostile checking. The
  consequence — that `pretrain/val_loss`, the `cv/pretrain/*` aggregates and every
  loss-curve artifact are uninterpretable as evidence of learning, and are not comparable
  across configurations whose targets are differently scaled — follows.
- **Correction (severity, and one framing):** cut critical → high for four reasons the
  finding should carry.
  1. **Nothing reported by this branch is invalidated.** `impute_score`, `acc_cat` and
     `rmse_num_z` come from the decode stage, which trains against real cell values and is
     scored against a mean/mode baseline. What the finding invalidates is the *narrative*
     that the encoder arrives pre-trained, not any published number.
  2. **It is inherited, and it is not an imputation issue.** `git diff main...HEAD --
     src/models.py` confirms the critic's own reading: the loss line, the commented
     `normalize` calls and the dead `eps` arrive unchanged; the branch's edit is the
     `EncodedTable`/`no_grad` rewrite. `train_pretrainer` is shared with classification, so
     this is a project-level measurement problem that the imputation branch inherits, not a
     defect it introduced.
  3. **"Worse than a constant" is a statement about the shipped configuration, not the
     objective.** 300 epochs at BATCH 256 on a 500-row fold is 900 optimizer steps. The
     critic's own BATCH=64 rerun (3.7x steps) reaches NMSE 1.51 with a centred cosine of
     0.026 — the same direction of travel, not yet across the line. "Undertrained, and the
     logged metric cannot show it" is what is established; "the objective cannot work" is
     not, and the critique is careful to say so in the body even though the heading is not.
  4. One dataset, and in my case one fold.

  The heading's "the 98.5% loss drop is ... not learning" is defensible but reads stronger
  than the body supports: the encoder does learn the free, column-constant part of the
  target (raw cosine 0.20 → 0.75). What it never learns is the part that identifies the
  hidden value. Say it that way.

### F-13-2 - The "shared and protected" `EPOCHS_PRE = 300` has no measurement behind it anywhere in the branch, and the critic's first ablation cannot detect a benefit

- **Verdict:** SOUND _(premise verified; the ablation's numbers are the critic's, not
  reproduced here)_
- **Severity after review:** high
- **Basis:** the premise that matters — *nothing in the branch has ever measured this* —
  is verifiable without running anything, and it holds:
  - Every one of the **186 imputation Optuna trials** in the store carries
    `EPOCHS_PRE = 300` (`Counter({'300': 186})`), and so does every one of the 6 imputation
    parents. The store contains zero imputation runs at any other value.
  - The justification on record is cost and sharing, never evidence.
    `docs/wayfinder/imputation-optuna-reduced/issues/01-held-set-and-ranges.md:40`:
    "*Pre-training is shared and protected; tuning it doubles the cost of a stage both
    tasks use. Pre-training keeps its last epoch, so `EPOCHS_PRE` is a real knob, held by
    decision.*" ADR 0005 line 86 repeats it. ADR 0004 decision 4 forbids the decode stage
    from re-running the Xavier sweep because it "would erase the pretrained encoder" —
    protecting an artefact whose value has not been measured.
  - `src/training/decoding.py:54-57` is indeed the whole seam: `TridentDecoder(embedder=
    pretraining.model.embedder, transformer=pretraining.model.transformer, ...)`.
  So the finding's durable content — a load-bearing premise in two ADRs, about to gate
  nineteen GPU-hours, with no supporting measurement — is confirmed, and high severity is
  right because the fix is cheap *only before* the studies run.
- **Correction:** the *negative result* is weaker than the heading implies, and the
  critique should be read with the caveat it half-states. Two seeds x two folds on one
  dataset, a 0.0007 arm difference against a 0.030 seed-to-seed spread and a 0.042-0.047
  fold-to-fold spread, plus the RNG-stream confound the critic names themselves, cannot
  distinguish "no benefit" from "a benefit smaller than this design can see". The heading
  "cannot detect any benefit from 298 of the 300 epochs" is literally true and is exactly
  how a null result should be phrased; the risk is a reader converting it into "298 epochs
  do nothing", which the data does not support. I did not re-run the ablation, so the four
  reported numbers are unverified here; the premise underneath them is not.

### F-13-3 - The stage logs three scalars and nothing scale-free, and no unit test ever executes it

- **Verdict:** SOUND
- **Severity after review:** medium _(critic said high)_
- **Basis:** all three premises are literally true.
  - `pretraining.py:112-119` logs exactly `pretrain/train_loss`, `pretrain/val_loss`,
    `pretrain/learning_rate`, and nothing else.
  - `grep -rn "norm()\|cosine_similarity\|F.normalize\|functional.normalize" src/` returns
    only `models.py:66` and `:67` — the two commented-out lines. No embedding norm, no
    cosine, no value-space reconstruction rate is computed anywhere in the package.
  - `grep -rn "train_pretrainer\|TridentPretrainer" tests/` returns five hits: an import
    and one untrained construction at `test_training_decoding.py:79`, and three
    monkeypatches at `test_training_runtime.py:134,177,449`. No unit test executes the
    loop. `tests/fixtures/credit-g_20nan_imputation_regression.json` pins `"EPOCHS_PRE": 2`
    at `"DIM": 16`.
- **Correction:** two adjustments, both severity-relevant.
  - "Unfalsifiable from a run's record" is too broad. The record *does* carry
    `impute/induced/impute_score` against a mean/mode baseline, so a reader can falsify
    "the pipeline works". What is unfalsifiable is specifically "the pre-training stage
    worked", which is the narrower claim the finding should make.
  - As a finding it is the instrumentation half of F-13-1's Direction (b) rather than an
    independent defect — the two overlap almost entirely, and counting both at high
    double-counts one problem. Medium, and worth keeping only for the test-coverage half,
    which is a real gap: a future change to the shared stage has no feedback other than a
    loss whose units move with the change.

### F-13-4 - The reduced profile holds two thirds of each trial's epoch budget on a stage the search cannot touch, and no sampled dimension moves pre-training alone

- **Verdict:** SOUND, with one premise detail false and two quantities overstated
- **Severity after review:** medium _(critic said high)_
- **Basis:** the structural core is exactly as described. `opt.py:96-112`'s reduced branch
  returns `PROB_MASCARA`, `LR_DECODE`, `WEIGHT_DECAY_DECODE`, `DROPOUT` and conditionally
  `LAMBDA_NUM`; `EPOCHS_PRE` (300) and `EPOCHS_DECODE` (150) are `Hyperparameters` defaults
  (`types.py:96,111`) and are held, so 300/450 of the epoch budget is unreachable.
  `PROB_MASCARA` is genuinely cross-stage: `pretraining.py:71,74` and `decoding.py:93` all
  pass `p_base=hyperparameters.mask_probability`. `LR_DECODE`/`WEIGHT_DECAY_DECODE` reach
  only the decode `AdamW` (`decoding.py:59-63`), and `LR_PRE`/`WEIGHT_DECAY_PRE` are held —
  so **no sampled dimension moves pre-training alone**, and fANOVA over sampled knobs
  cannot rank a held one. Both stand.
- **Correction:** three things.
  1. **"`PROB_MASCARA` is the one sampled knob that is not stage-local" is false.**
     `DROPOUT` is cross-stage too, and more tightly so: the transformer is constructed once,
     in `pretraining.py:36-42`, with `dropout=hyperparameters.dropout`, and the decode stage
     reuses *that object* (`decoding.py:54-57`). There is no decode-stage dropout at all.
     `opt.py`'s own comment says it — "The one regulariser acting on both stages". Two of
     the five sampled dimensions are cross-stage, not one. The finding gets stronger, but
     the sentence is wrong as written.
  2. **The GPU-hours figure is overstated by about 30%.** Every trial already logs
     `time/pretrain_seconds` and `time/training_seconds` (the latter is
     `pretrain + decode` stage seconds — `summary.py:40-46`, `tracking.py:177-178` — so
     this is the share of *stage* time, excluding preparation and scoring). Across the
     **185 imputation trials in the store that have both**, pre-training's median share is
     **0.514** (min 0.392, max 0.548; by dataset: credit-g_20nan 0.526,
     credit-g_40nan 0.529, kr-vs-kp_20nan 0.512, kr-vs-kp_40nan 0.512, spambase_20nan
     0.465). The consequence's "twelve and a half GPU-hours of the nineteen" applies the
     2/3 epoch ratio to wall clock; on the study's own recorded timings it is at most
     about 9.8. Two thirds is the epoch count; about half is the cost. Quote the store.
  3. **ADR 0005 names the fANOVA gap itself.** Decision 6 reads "*It ranks only the sampled
     knobs; a full-profile pilot to rank the held ones is deferred to the fog*", and the
     alternatives section records "*A full-profile pilot with fANOVA to rank the held knobs.
     Deferred: fifteen dimensions at 50 trials rank noisily.*" That is an acknowledged,
     reasoned trade-off, not a loop the ADR closed the wrong way without noticing. Holding
     architecture and pre-training knobs to cut dimensionality is also ordinary search
     design. What remains is the attribution problem (two cross-stage knobs, no pre-training
     dimension), which is real but medium.

### F-13-5 - Pre-training's re-rolled validation mask drives real `plateau` learning-rate decisions

- **Verdict:** CONFIRMED
- **Severity after review:** medium
- **Basis:** the code path is unambiguous and the store reproduces every number.
  `pretraining.py:70-75` re-rolls **both** masks every epoch; `:108` averages the resulting
  validation loss; `:120` passes it to `scheduler.after_epoch(average_validation_loss)`,
  which under `plateau` is `ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
  patience=10).step(validation_loss)` (`schedulers.py:90-93,103-104`). The decode stage does
  the opposite and says why: "*a re-rolled one would move the loss for reasons that have
  nothing to do with the model, making checkpoint selection meaningless*"
  (`decoding.py:71-73`, implemented at `:77-82`).

  Reading a scratchpad copy of `mlflow.db`, every figure matches to the digit:

  ```
  credit-g_20nan 92278ca2  300 epochs  lr 0.00034 -> 0.00034  halvings 0
  credit-g_20nan f20f7ad1  300 epochs  lr 0.00034 -> 0.00034  halvings 0
  spambase_20nan e764df47  300 epochs  lr 0.00034 -> 1.66e-07 halvings 11
        at epochs [154, 178, 189, 209, 227, 238, 249, 260, 271, 282, 293]
        frozen (lr<1e-6) n=29  val_loss 0.01265..0.01762  CV=9.2%  ratio=1.39
        best val_loss 0.01206 @ep215 ; last 0.01741 @ep299
  spambase_20nan f1077a34  300 epochs  lr 0.00034 -> 2.66e-06 halvings 7
  ```

  Four of the six imputation parents are tagged `lr_scheduler = plateau` — confirmed by
  listing them: two electricity, one credit-g, one spambase. The 9% noise floor on a frozen
  model is exactly what the finding claims.
- **Correction:** none to the substance — one addition that makes it stronger, in "missed"
  below. Medium is right: `plateau` is not the default (`DEFAULT_LR_SCHEDULER =
  "cosine_legacy"`, `types.py:17`), the two credit-g folds never triggered a single halving,
  and ADR 0005 decision 6 already refuses to use the existing plateau runs as the comparison
  baseline ("*fresh default runs, because the two imputation runs already in the store used
  `plateau` and 2 folds*"), which caps the damage to runs nobody will cite.

## What this critique missed

1. **The embedder *is* collapsing — in the only component that carries information.** The
   "Checked and cleared / the embedder does not collapse" item rests on `||tgt||` growing,
   which is true but is the wrong statistic. Decompose the target's energy on the same fixed
   mask, untrained vs after 300 epochs, from the same init (identical seed, identical
   constructor path):

   | | untrained | trained | change |
   |---|---|---|---|
   | target second moment per dim | 0.04845 | 0.05627 | +16% |
   | within-column variance (the discriminative part) | 0.00807 | 0.00593 | **−26%** |
   | column-constant share of energy | 83.3% | 89.5% | +6.2 pts |

   The part of the target that distinguishes one category or value from another in the same
   column *shrank by a quarter* while the free, column-constant part grew. The target is
   computed under `no_grad`, but the embedder is the same object that is trained through the
   input path, so the optimisation can and does make its own target easier. Weight decay
   cannot be the route: AdamW's decoupled decay shrinks a parameter by
   `(1 - lr*wd)^steps = (1 - 3.4e-4 * 0.005)^900 ~ 0.15%` over the whole stage, three orders
   of magnitude short of 26%, which leaves the gradient through the input path as the only
   remaining one. That is a collapse — just not the one BYOL warns about — and it lowers the loss for free, which is
   a second mechanism behind the 97-99% drop alongside the norm squeeze the critique names.
   It also matters for the repair: normalising the loss (Direction (a)) does not stop the
   informative variance from shrinking; a value-space anchor does.

2. **The provenance check was done for F-13-1 and not for F-13-5, where it changes the
   verdict's weight.** `git show main:src/training/schedulers.py` fails — *the file does not
   exist on `main`*, and `git diff --stat main...HEAD` shows `schedulers.py` (+113) and ADR
   0003 (+95) as pure additions. Not a stale local `main` either: the file's adding commit is
   `87d8c82` (2026-09-09, "feat(training): selectable learning-rate schedule with legacy
   default") and `git branch -r --contains 87d8c82` returns nothing, so `origin/main` does not
   have it. `main`'s `train_pretrainer` builds
   `CosineAnnealingLR(optimizer, T_max=pretraining_epochs)` directly and has no plateau
   option. So the re-rolled mask is inherited, but **the consumer that turns it into a
   training decision is this branch's own addition** — made in the same branch that gave its
   new decode stage a fixed validation mask and wrote down why. That is the strongest form
   of F-13-5 and it is absent from the finding.

3. **The store already answers F-13-4's cost question, and F-13-2's within limits.**
   `time/pretrain_seconds` is logged on every trial (185 usable): median 51.4% of training
   wall clock on the user's actual GPU, which is better evidence than one CPU-measured pair
   and corrects the nineteen-hour arithmetic. Separately, the `EPOCHS_PRE` column across
   **87 classification parents** is not constant — 300 (56 runs), 200 (20), 40 (11) — and
   the pre-training stage is shared. The one within-dataset, within-scheduler contrast
   (`vehicle_00nan`, `cosine_legacy`, 3 folds, seed 42, identical architecture) is
   `EPOCHS_PRE=40` → accuracy 0.7210 against `EPOCHS_PRE=300` → 0.7577 / 0.7530 / 0.7530, a
   +0.033 gap in favour of more pre-training. It is **confounded** — `EPOCH_FINE` moves with
   it, 40 vs 150 — so it settles nothing, but it points the other way from the imputation
   probe and it is free to look at. A critique arguing that pre-training contributes nothing
   owes a sentence to the task the stage was built for, and to the store rows that already
   bear on it.

4. **`DROPOUT` has no decode-stage counterpart at all.** Beyond F-13-4's attribution point:
   the decode stage cannot set its own dropout even in the `full` profile, because it reuses
   the transformer instance pre-training constructed. If the two stages want different
   regularisation — plausible, since one reconstructs embeddings and the other reconstructs
   values — the search space cannot express it, and no finding says so.

5. **Immaterial, for the record:** the critique's `grep -rn "norm()\|cosine_similarity\|
   F.normalize" src/` returns "no hits" partly because `F.normalize` is case-sensitive and
   the code writes `nn.functional.normalize`. The conclusion is unaffected — the only two
   hits are the commented-out lines the finding already quotes — but the grep as written
   would have missed a live call.
