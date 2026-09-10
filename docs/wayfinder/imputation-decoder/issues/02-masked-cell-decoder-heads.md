# 02. Research: how do tabular masked-modeling transformers decode a masked cell back to a value?

Type: research
Status: resolved
Blocked by: none
Assignee: research subagent (charting session, 2026-09-10); resolved 2026-09-10
Context pointer: branch `research/masked-cell-decoder-heads`, file `docs/research/masked-cell-decoder-heads.md`

## Question

In masked-cell-modeling transformers for tabular data, how is the reconstruction head
designed that maps a contextual token embedding at a masked cell back to a value?

1. Per-column heads vs one shared head (and how a shared head copes with differing
   categorical vocabularies).
2. Categorical: softmax over the column vocabulary with CE; how special tokens are
   excluded from the output space.
3. Numerical: linear/MLP regression (MSE, Huber, Gaussian NLL) vs binning + CE.
4. How the CE and regression terms are weighted into one loss.
5. Head used only for pretraining vs also as the imputer at inference; encoder frozen
   or fine-tuned; per-position heads vs an MAE-style lightweight transformer decoder.

Sources: ReMasker, SAINT, TabTransformer, VIME, TransTab, MET, XTab, UniTabE, BERT/MAE.

Feeds ticket 04 (decoder architecture and training stage).

## Answer

Resolved 2026-09-10 by a research subagent (19 fetches, arXiv pages and official repos).
Full note: branch `research/masked-cell-decoder-heads`, file
`docs/research/masked-cell-decoder-heads.md`, commit `529529b`. Read it with
`git show research/masked-cell-decoder-heads:docs/research/masked-cell-decoder-heads.md`.

Gist, by sub-question:

1. **Per-column output heads are the norm** for categorical cells (SAINT Eq. 5, XTab,
   the AutoGluon TabTransformer pretext: one `nn.Linear` per column). Shared heads cope
   with differing vocabularies by slicing one global softmax to the column's own ids
   (IBM TabFormer), by tying the output matrix to the input embedding table (BERT), or
   by generating the value as text (UniTabE).
2. **Special tokens** are excluded by construction (the head has only the real classes;
   TabFormer slicing) or merely never occur as labels (BERT keeps `[MASK]` in the
   softmax and ignores positions, not classes).
3. **Numerical cells** are regressed to a scalar with MSE on standardised values in
   every surveyed tabular paper (SAINT, XTab, MET, VIME, ReMasker's shared
   `Linear(d_dec, 1)`); no Huber or Gaussian NLL; bins + CE only in TabBERT.
4. **Loss combination** is a plain sum with at most one fixed scalar (SAINT
   `lambda_pt`, VIME `alpha`, MET `lambda = 1`); MAE and ReMasker use count-based means
   per term; ReMasker adds an equally weighted unmasked-cell term (its Table 3 says it
   helps imputation).
5. **Only imputation-oriented papers keep a decoder and use it at inference** (ReMasker:
   shallow MAE decoder, nothing frozen, fit on the incomplete table itself; UniTabE:
   prompt-driven generation). MAE discards its decoder, XTab keeps only the backbone
   and re-initialises heads, MET freezes the encoder, TransTab has no reconstruction
   head at all. SAINT's repo was unreachable, so its claims rest on the paper.

Candidate designs for TRIDENT (note section 3; the choice is ticket 04's):

- **Design A** (the note's first pick): per-column `Linear(d, V_c - 2)` categorical
  heads with the two special ids removed via a `valid_ids` buffer and a local-id remap;
  per-column `Linear(d, hidden) -> ReLU -> Linear(hidden, 1)` numerical heads, batched
  like the embedder's MLPs; `L = L_cat + lambda_num * L_num` with a count-based mean per
  type, optionally `+ lambda_emb * L_embedding_mse` to keep today's objective as a
  special case. Roughly 2k new parameters on a typical table. At inference, missing cells
  are fed as `[MASK]` and decoded with `argmax` / `inverse_transform`.
- **Design B**: BERT/TabFormer-style tied heads reusing the embedding tables plus one
  shared transform; about 100 parameters; the cheapest ablation of whether the embedding
  geometry is already decodable.
- **Design C**: ReMasker/MAE lightweight decoder (1-2 reused `EncoderLayer`s) feeding
  Design A's heads, optionally with the unmasked-cell term; most capacity and code; the
  only pattern validated as a tabular imputer in the literature.
- A loss-combination table lists: unweighted sum, one fixed scalar, count-based mean
  (recommended default), per-type chance normalisation (divide CE by `log(V_c - 2)`),
  uncertainty weighting (Kendall et al. 2018), and bins + CE.

Corrections for readers of the note:

- It states `d` defaults to 4. That is only the `TabularEmbedder` constructor default;
  training uses `Hyperparameters.dimension` (JSON `DIM`), whose default is 128, and the
  integration fixture uses 16. The "capacity-bound at d = 4" caveat does not apply to
  real runs.
- Its suggestion to expose the objective as a `pretrain_objective` flag (a joint
  pretraining loss with `lambda_emb`) is one of the stage-placement options in ticket
  04, not a settled decision.

Unblocks: ticket 04.

## Comments
