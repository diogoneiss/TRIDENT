# 04. Decoder architecture and the stage that trains it

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 02

## Question

How does a decoder reconstruct actual cell values from the transformer output, and when
is it trained?

**Stage placement** (pick one):

- **Stage-2 head**, mirroring classification: keep the embedding-MSE pretraining stage
  bit-identical, then attach the decoder and train it (with masking re-rolled every
  epoch, like pretraining) as a *decode stage* that replaces classifier fine-tuning.
- **Joint pretraining loss**: add the value-reconstruction term to `TridentPretrainer`
  behind a flag; changes the pretraining path, so it needs its own tag.
- **From scratch**: train only the value loss, no embedding-MSE stage (an ablation).

**Head design** (informed by the survey in ticket 02):

- Categorical column c: `Linear(d, |V_c|)` per column, with `[MASK]`, `[NULL]` and the
  literal `"nan"` category removed or masked out of the output space (they live in the
  LabelEncoder vocabulary today, see backlog C2).
- Numerical column: `Linear(d, 1)` per column, or a 2-layer MLP mirroring the input MLP
  of the embedder; output in scaled space.
- Per-column vs shared head; whether heads are batched like the embedder's MLPs are.

**Training details**: encoder frozen vs fine-tuned during the decode stage; loss =
CE + MSE with what weighting; which parameters get Xavier init (the pretraining
`initialize_weights` sweep re-inits every Linear, note backlog C4); whether the decode
stage reuses `StageScheduler` and the best-validation-loss checkpoint rule from
`finetuning.py`.

Consequence to record: which new hyperparameters exist (feeds ticket 08) and which
random draws the decode stage consumes (feeds ticket 09).

Recommended starting answer: stage-2 head, per-column heads, encoder fine-tuned,
CE + MSE averaged per scored cell, reuse the scheduler and checkpoint rule of the
fine-tuning loop.

## Answer

Resolved 2026-09-10 over two grilling rounds; every recommendation was accepted.

**Stage placement.** A **decode stage replaces fine-tuning**. Pre-training keeps its
current embedding-reconstruction objective bit-identical; the decoder then attaches to
the pretrained encoder and trains on re-rolled masks. The runner's shape becomes
`train_pretrainer` followed by either `train_and_evaluate_classifier` or
`train_and_evaluate_decoder`. A joint pre-training objective and a value-loss-only
variant stay reachable later as extra flag values, not rewrites.

**Head design: per-column heads on the encoder output** (the research note's Design A,
the SAINT/XTab pattern). Design B (tied heads) is the cheap ablation of whether the
embedding geometry is already decodable; Design C (MAE-style decoder) only if A proves
capacity-bound. Note that the note's "capacity-bound at d = 4" warning misreads the
constructor default: training `DIM` defaults to 128.

**Categorical output space and target encoding.** The head over categorical column `c`
emits only the **real categories**: `[MASK]`, `[NULL]` and the dead literal `"nan"`
entry are all excluded, so a 7-entry vocabulary yields a 4-way head. Targets are encoded
from the fine-tuning-style processed frame, `preprocess_table(frame, p_base=0.0,
fine_tunning=True)`, **not** the raw frame that pre-training encodes.

Verified on `credit-g_20nan` while resolving this ticket:

| Fact | Value |
|---|---|
| Categorical columns whose vocabulary holds the dead `"nan"` entry | 13 of 13 |
| Numerical NaN cells when encoding the raw frame | 1400 |
| Numerical NaN cells when encoding the processed frame | 0 |
| Null categorical cell, raw encode | the dead `"nan"` id |
| Null categorical cell, processed encode | the `[NULL]` id |

Pre-training survives both hazards only because masking never covers an already-null
cell, so the poisoned positions are never selected. The decoder inherits that protection
for its loss but not for its output space, which is why all three tokens are excluded.

**Numerical head.** A two-layer network mirroring the embedder's input MLP in reverse,
`Linear(d, hidden) -> ReLU -> Linear(hidden, 1)` per column, batched across columns the
way `TabularEmbedder.forward` already batches its input MLPs, so cost stays a constant
number of kernel launches rather than two per column. Output is in scaled space.

**Loss.** Average each term over its own count of scored cells, then sum:

```text
L = L_cat + lambda_num * L_num
L_cat = mean over masked categorical cells of CE(head_c(H_c), local_id(y_c))
L_num = mean over masked numerical cells of (head_k(H_k) - x_k)^2
```

`lambda_num` defaults to 1. Per-type count averaging stops a table with 20 numerical and
2 categorical columns from drowning the categorical term. **No auxiliary
embedding-reconstruction term** in the decode stage: under the stage-placement decision
that objective belongs to pre-training, which is untouched.

**Encoder is fine-tuned, not frozen.** The optimizer takes every parameter, mirroring
what classification does with the same shared modules. Freezing is a later ablation.
This is safe in a way the pre-training objective is not: targets are label ids and scaled
floats, fixed points of reference, so the embedder can move without the target drifting
underneath it (today's target is a detached embedding the embedder itself produces).

**Mask rates.** Decode-stage *training* reuses the existing `mask_probability`
(`PROB_MASCARA`, default 0.5), re-rolled every epoch exactly as pre-training does, so
both stages corrupt rows identically. Only *evaluation* uses the separate evaluation mask
rate fixed by ticket 03. Recorded caveat: `preprocess_table`'s at-least-one-mask-per-row
fallback is a per-row Python loop (backlog P1), cheap at 0.5 but roughly 20x slower at
0.05, so a low training rate is a performance cliff. The evaluation draw happens once per
fold, so the cliff does not apply there.

**Learning-rate schedule.** The decode stage joins the existing `lr_scheduler`
hyperparameter; no new knob, no per-stage schedule. Consequence for the ADR: the default
is `cosine_legacy`, which anneals per batch, and the decode stage has no published runs
to preserve, so imputation experiments should be launched with `--lr_scheduler cosine`
even though the global default stays legacy for consistency.

**Checkpoint selection.** Track the best validation decode loss across all decode epochs
and restore it at the end, mirroring `finetuning.py`. This is meaningful only because
ticket 03 fixed the validation mask to a single draw, so every epoch is scored on the
same problem.

**Frame.** Feature-only, label column dropped, as pre-training does. The token layout is
then identical across all three stages, and the model never needs labels at imputation
time. Label-as-context is a separate experiment.

**Recorded hazard.** The decode stage must **not** copy `train_pretrainer`'s
`model.apply(initialize_weights)` sweep. It re-initialises every `nn.Linear` and
`nn.Embedding` in the whole model; run after pre-training it would erase the pretrained
encoder. New heads take framework defaults, exactly as the classifier head does today.

Consequences for other tickets:

- Ticket 08 (unblocked) gains these hyperparameters: decode epoch count, decode learning
  rate and weight decay, the evaluation mask rate from ticket 03, and `lambda_num`.
- Ticket 09 gains a `src/training/decoding.py` with `train_and_evaluate_decoder`, a
  `DecodingOutcome` type, and decode-stage names for the loss-band and stage-timing keys.
- Ticket 05 inherits the loss shape above and needs only to settle the reported metrics
  and the fold-ranking rule.
- Ticket 12 is unaffected: excluding `[NULL]` from the *output* space says nothing about
  feeding `[NULL]` as an *input* token, which is what that ticket asks.

## Comments
