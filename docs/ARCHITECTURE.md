# TRIDENT Architecture

A layer-by-layer walkthrough of how TRIDENT turns a table row into a classification, and how the data gets there. Written for someone who knows PyTorch and transformers in general but has never opened this repo.

This is a deep dive, not the whole story. For the paper citation, CLI flags, and the config JSON format, see [README.md](../README.md) — this document links back to it rather than repeating it. For MLflow run/tag conventions, see [CONTEXT.md](../CONTEXT.md) and [ADR 0002](adr/0002-curated-cross-validation-mlflow-runs.md).

Every diagram in this doc comes in two flavors: a **theory** version (block names, no numbers — how you'd whiteboard it) and an **implementation** version (real tensor shapes and the actual PyTorch calls that produce them). Skip to whichever one you need.

## Contents

1. [The big picture](#the-big-picture)
2. [Data engineering pipeline](#data-engineering-pipeline)
3. [TabularEmbedder — a table row becomes a token sequence](#tabularembedder--a-table-row-becomes-a-token-sequence)
4. [TabularTransformerEncoder — the shared backbone](#tabulartransformerencoder--the-shared-backbone)
5. [TridentPretrainer — the self-supervised objective](#tridentpretrainer--the-self-supervised-objective)
6. [TridentModel — the fine-tuning / classification head](#tridentmodel--the-fine-tuning--classification-head)
7. [TridentDecoder — reconstructing cell values](#tridentdecoder--reconstructing-cell-values)
8. [End-to-end training orchestration](#end-to-end-training-orchestration)
9. [Hyperparameter → code wiring reference](#hyperparameter--code-wiring-reference)
10. [Running it](#running-it)

---

## The big picture

TRIDENT trains in two stages that **share the same embedder and transformer weights** — fine-tuning doesn't start from scratch, it continues training the exact same `TabularEmbedder`/`TabularTransformerEncoder` instances that pretraining produced, within a given fold.

```mermaid
flowchart LR
    A["Raw CSV"] --> B["Data engineering<br/>encode + scale"]
    B --> C["Stage 1: Pretraining<br/>TridentPretrainer<br/>(src/models.py:10)"]
    C -->|"embedder + transformer<br/>objects carried over as-is"| D["Stage 2: Fine-tuning<br/>TridentModel<br/>(src/models.py:60)"]
    D --> E["Test-set evaluation"]
```

Two things to hold onto before the details:

- **The "sequence" a transformer sees isn't rows over time — it's columns of one row.** Every categorical and numerical column becomes one token, a `[CLS]` token is prepended, and `TabularTransformerEncoder` attends across *columns*, not across time steps or other rows.
- **"Fresh per fold, shared within a fold."** `run_training` (`src/training/runner.py:33`) builds a brand-new `TabularEmbedder` + `TabularTransformerEncoder` for every cross-validation fold, but *within* a fold, `train_and_evaluate_classifier` re-wraps the exact same pretrained embedder/transformer objects — not copies (`src/training/finetuning.py:102-107`).

---

## Data engineering pipeline

Order matters here — `AGENTS.md` calls out preserving the split/scaling order specifically because it's easy to get backwards.

### Loading, encoding, scaling — `prepare_dataset` (`src/training/data.py:15`)

```mermaid
flowchart TD
    Load["prepare_dataset — data.py:15<br/>Load CSV<br/>datasets/processed_datasets/{base}/{name}.csv"] --> LabelEnc["LabelEncoder.fit_transform<br/>on the target column"]
    LabelEnc --> Cols["categorical_columns loaded from file<br/>numerical_columns = every other feature column"]
    Cols --> Scale["StandardScaler.fit_transform<br/>over numerical_columns,<br/>on the WHOLE frame — before any split"]
    Scale --> PDS["PreparedDataset<br/>(frame, columns, label_classes, ...)"]
```

The scaler is fit on the **entire dataset**, before train/val/test indices exist. That's the exact order `AGENTS.md:6` protects — changing it (e.g. fitting the scaler per-fold or per-split) is an algorithm change, not a refactor.

### Building folds — `build_folds` (`src/training/data.py:69`)

```mermaid
flowchart TD
    PDS["PreparedDataset.frame"] --> BF{"build_folds"}
    BF -->|"cv_folds is None"| Predefined["Load predefined split JSON<br/>datasets/processed_datasets/splits/{base}_split.json<br/>train_indices / val_indices / test_indices"]
    BF -->|"cv_folds is set"| CV["StratifiedKFold<br/>(falls back to plain KFold if a class<br/>has fewer samples than cv_folds)<br/>→ train+val vs test"]
    CV --> ValSplit["StratifiedShuffleSplit<br/>(falls back to ShuffleSplit)<br/>carves validation out of train+val,<br/>ratio ≈ 0.1 / (1 − 1/cv_folds), clamped [0.01, 0.5]"]
    Predefined --> FS["FoldSplit(train_indices, validation_indices, test_indices)"]
    ValSplit --> FS
```

With `cv_folds=None` you get exactly one `FoldSplit` (the legacy fixed split). With `cv_folds=k` you get `k` folds, each with its own train/val/test index sets — this is what the `tags.run_role=parent`/`best_fold`/`worst_fold` MLflow structure in [CONTEXT.md](../CONTEXT.md) is tracking one of.

### Per-epoch masking and batching

This part runs fresh every training step, inside `train_pretrainer` (`src/training/pretraining.py`) and `train_and_evaluate_classifier` (`src/training/finetuning.py`), using `preprocess_table` (`src/utils.py:19`).

```mermaid
flowchart TD
    FS["FoldSplit indices"] --> Frame["frame.iloc[indices]"]
    Frame --> PP{"preprocess_table()<br/>(src/utils.py:19)"}
    PP -->|"pretraining<br/>p_base = mask_probability"| Dyn["1. Replace nulls with '[NULL]'<br/>2. Dynamically mask cells to '[MASK]',<br/>probability scaled DOWN by the row's<br/>existing null density<br/>3. Guarantee ≥ 1 masked cell per row<br/>— re-rolled every epoch"]
    PP -->|"fine-tuning<br/>p_base = 0, fine_tunning=True"| NullOnly["Replace nulls with '[NULL]' only.<br/>No masking. Computed once,<br/>not per epoch (no randomness to refresh)"]
    Dyn --> Encode["TabularEmbedder.encode(frame)<br/>— whole frame to tensors, ONCE per epoch"]
    NullOnly --> EncodeOnce["TabularEmbedder.encode(frame)<br/>— ONCE per fold, never re-encoded"]
    Encode --> Shuffle["torch.randperm(len(frame))<br/>drawn on CPU, moved to device once"]
    EncodeOnce --> Shuffle
    Shuffle --> Batch["encoded[start : start+batch_size]<br/>— tensor slicing, no DataLoader"]
    Batch --> ToEmb["TabularEmbedder.forward(EncodedTable)"]
```

Notes worth knowing:

- Masking never overwrites an already-null cell — `[MASK]` and `[NULL]` are mutually exclusive per cell.
- Pretraining validation is batched the same way as training; fine-tuning validation and test sets run as a **single full-batch forward pass**, no loop.
- **Pandas work happens once per frame, not once per batch.** `TabularEmbedder.encode` (`src/embedder.py`) turns a whole DataFrame into an `EncodedTable`: label-encoded categorical indices, numerical values, `[MASK]`/`[NULL]` flags, and the `[MASK]` position matrix the pre-training loss needs. Pre-training re-encodes the masked frames each epoch because the masks are re-rolled; the clean targets and every fine-tuning frame are encoded a single time.
- `EncodedTable` stores its per-column tensors feature-major, so slicing a batch keeps each column contiguous. `forward` accepts either an `EncodedTable` or a raw DataFrame, so `model(df)` still works for saved models and ad-hoc inference.
- `split_numeric_and_special` (`src/utils.py`) still produces `numeric_values` (float, `0.0` at any special token), `mask_flags` and `null_flags`, but vectorized over whole columns instead of looping cell by cell.

---

## TabularEmbedder — a table row becomes a token sequence

`src/embedder.py:9`. Constructor: `TabularEmbedder(df, categorical_columns, numerical_columns, dimensao, hidden_dim)`. Every categorical column gets **its own** `nn.Embedding` table (not one shared vocabulary across columns); every numerical column gets **its own** 2-layer MLP. `dimensao` is the one embedding width shared by everything downstream — it's what README calls `DIM` and what the transformer calls `d_model`.

### Theory

```mermaid
flowchart TD
    Row["One row"] --> Cat["Categorical columns"]
    Row --> Num["Numerical columns"]
    Cat --> CatEmb["Per-column embedding table<br/>(vocabulary = its own values + [MASK] + [NULL])"]
    Num --> IsSpecial{"Is the cell [MASK] or [NULL]?"}
    IsSpecial -->|"no"| NumMLP["Per-column MLP:<br/>Linear → ReLU → Linear"]
    IsSpecial -->|"yes"| SpecialVec["Learned special vector<br/>(one per column, shared across rows)"]
    CatEmb --> Concat["One embedding per column, concatenated"]
    NumMLP --> Concat
    SpecialVec --> Concat
    Concat --> Reshape["Reshape into a sequence of column-tokens"]
    CLS["Learned [CLS] token"] --> Prepend["Prepend to the sequence"]
    Reshape --> Prepend
    Prepend --> PosAdd["+ learned positional embedding<br/>(position = column index, not time)"]
    PosAdd --> Out["Token sequence: one vector per column, plus CLS"]
```

### Implementation

```mermaid
flowchart TD
    Row["Batch of B rows"] --> Cat["n_cat categorical columns"]
    Row --> Num["n_num numerical columns"]
    Cat --> CatIdx["LabelEncoder.transform<br/>→ long indices, (B,) per column"]
    CatIdx --> CatEmb["nn.Embedding(num_categories, dimensao)<br/>→ (B, dimensao) per column"]
    Num --> Split["value / mask_flag / null_flag<br/>from the EncodedTable, each (n_num, B)"]
    Split --> MLP["Per-column MLP, evaluated for ALL columns at once:<br/>broadcast multiply-add → ReLU → baddbmm<br/>→ (n_num, B, dimensao)"]
    Split --> Where["torch.where: use the learned '_mask'/'_null'<br/>vector wherever the cell was special<br/>→ (n_num, B, dimensao)"]
    MLP --> Where
    CatEmb --> CatStack["torch.stack(dim=1) over columns<br/>→ (B, n_cat, dimensao)"]
    Where --> NumT["transpose → (B, n_num, dimensao)"]
    CatStack --> AllCat["torch.cat(dim=1)<br/>→ (B, n_tokens, dimensao)"]
    NumT --> AllCat
    CLS["cls_token: (dimensao,)"] --> Expand["expand → (B, 1, dimensao)"]
    AllCat --> PrependCLS["cat(dim=1) → (B, n_tokens+1, dimensao)"]
    Expand --> PrependCLS
    PrependCLS --> PosAdd2["+ pos_embedding_layer.weight[:n_tokens+1]<br/>broadcast over the batch"]
    PosAdd2 --> Out["Output: (B, n_tokens+1, dimensao)"]
```

Each numerical column still owns its own `Linear(1, hidden_dim) → ReLU → Linear(hidden_dim, dimensao)` parameters, and they still appear per column in `state_dict`. `forward` gathers those parameters and applies them as one batched operation, so the cost is a fixed number of kernel launches instead of two per column.

`n_tokens = len(categorical_columns) + len(numerical_columns)` is fixed at construction time, and column order is always *categorical columns, then numerical columns*, in list order — that ordering **is** the positional embedding's meaning, so it has to stay identical between training and inference.

> **Initialization gotcha**: `embedder.py` doesn't initialize its own layers. Immediately after construction, `train_pretrainer` (`src/training/pretraining.py:44-48`) runs `model.apply(...)` and re-initializes every `nn.Embedding`/`nn.Linear` in the whole model (embedder included) with Xavier-uniform. But `cls_token` and the `[MASK]`/`[NULL]` special vectors are plain `nn.Parameter(torch.randn(dimensao))` — not `nn.Embedding` or `nn.Linear` — so that sweep skips them. They stay standard-normal initialized.

---

## TabularTransformerEncoder — the shared backbone

`src/transformer.py:92`. A pre-norm transformer encoder, structurally close to a standard BERT-style encoder but bidirectional (no causal mask) and with one global `LayerNorm` before the stack in addition to each layer's own pre-norm.

### The stack

**Theory**

```mermaid
flowchart LR
    In["Input sequence"] --> PN["LayerNorm (applied once, pre_norm)"]
    PN --> E1["EncoderLayer 1"] --> E2["EncoderLayer 2"] --> Edots["... num_layers total"] --> Out["Output sequence"]
```

**Implementation**

```mermaid
flowchart LR
    In["x: (B, T, d_model)<br/>T = n_tokens + 1"] --> PN["pre_norm = LayerNorm(d_model)"]
    PN --> Stack["num_layers × EncoderLayer<br/>(B, T, d_model) → (B, T, d_model)"]
    Stack --> Out["Output: (B, T, d_model)<br/>shape never changes"]
```

### Inside one `EncoderLayer` (`src/transformer.py:69`)

A standard pre-LN residual block: normalize, sublayer, dropout, add — twice (attention, then feed-forward).

**Theory**

```mermaid
flowchart TD
    X["x"] --> N1["LayerNorm"] --> MHA["Multi-Head Self-Attention"] --> D1["Dropout"]
    X --> R1(("+"))
    D1 --> R1
    R1 --> N2["LayerNorm"] --> FFN["Feed-Forward Network"] --> D2["Dropout"]
    R1 --> R2(("+"))
    D2 --> R2
    R2 --> Y["output"]
```

**Implementation**

```mermaid
flowchart TD
    X["x: (B, T, d)"] --> N1["norm1 = LayerNorm(d)"] --> MHA["MultiHeadAttention(d, nhead)<br/>→ (B, T, d)"] --> D1["dropout1"]
    X --> R1(("+"))
    D1 --> R1
    R1 --> N2["norm2 = LayerNorm(d)"] --> FFN["FeedForwardNetwork(d, dim_feedforward)<br/>→ (B, T, d)"] --> D2["dropout2"]
    R1 --> R2(("+"))
    D2 --> R2
    R2 --> Y["output: (B, T, d)"]
```

### Inside `MultiHeadAttention` (`src/transformer.py:34`)

`head_dim = hidden_size // num_heads`; `wq`/`wk`/`wv`/`out` are all bias-free linear projections, Xavier-initialized.

**Theory**

```mermaid
flowchart TD
    X["x"] --> Q["Query projection"]
    X --> K["Key projection"]
    X --> V["Value projection"]
    Q --> Heads["Split into num_heads heads"]
    K --> Heads
    V --> Heads
    Heads --> Scores["Scaled dot-product: Q·Kᵀ"]
    Scores --> Soft["Softmax over keys"]
    Soft --> Weighted["Weighted sum of Values"]
    Weighted --> Merge["Merge heads back together"]
    Merge --> Out["Output projection"]
```

**Implementation**

```mermaid
flowchart TD
    X["x: (B, T, d)"] --> WQ["wq: Linear(d,d,bias=False)"] --> QV["q: (B, T, d)"]
    X --> WK["wk: Linear(d,d,bias=False)"] --> KV["k: (B, T, d)"]
    X --> WV["wv: Linear(d,d,bias=False)"] --> VV["v: (B, T, d)"]
    QV --> QH["view+transpose → (B, nhead, T, head_dim)"]
    KV --> KH["view+transpose → (B, nhead, T, head_dim)"]
    VV --> VH["view+transpose → (B, nhead, T, head_dim)"]
    QH --> Scores["q @ kᵀ × scale (scale = head_dim⁻⁰·⁵)<br/>→ (B, nhead, T, T)"]
    KH --> Scores
    Scores --> Soft["softmax(dim=-1) + dropout"]
    Soft --> Weighted["attn @ v → (B, nhead, T, head_dim)"]
    VH --> Weighted
    Weighted --> Merge["transpose+reshape → (B, T, d)"]
    Merge --> OutProj["out: Linear(d,d,bias=False) → (B, T, d)"]
```

There's no attention mask used anywhere in the current training loops (`mask=None` at every call site) — the `mask` plumbing exists in the code but padding/causal masking isn't part of this pipeline today, since every "sequence" is exactly `n_tokens+1` long for a given dataset.

---

## TridentPretrainer — the self-supervised objective

`src/models.py:10`. Wraps an embedder + transformer. The training signal: run the transformer on a *masked* row, and ask its output at each `[MASK]` position to match what the embedder alone would have produced for that same cell if it hadn't been masked.

### Theory

```mermaid
flowchart TD
    Masked["Masked row"] --> EmbIn["TabularEmbedder"]
    Original["Original (unmasked) row"] --> EmbTgt["TabularEmbedder<br/>(no gradient — this is the answer key)"]
    EmbIn --> Enc["TabularTransformerEncoder"]
    Enc --> AtMask["Keep only the [MASK] positions"]
    EmbTgt --> TgtMask["Keep only the same [MASK] positions"]
    AtMask --> MSE["MSE Loss"]
    TgtMask --> MSE
```

### Implementation

```mermaid
flowchart TD
    Masked["df_masked"] --> EmbIn["embedder(df_masked) → (B, L, d)<br/>L = n_tokens+1"]
    Original["df_original"] --> EmbTgt["embedder(df_original).detach() → (B, L, d)"]
    EmbIn --> Enc["transformer(emb_in) → encoded: (B, L, d)"]
    Enc --> Sel1["encoded[:, 1:, :][mask_tensor]<br/>→ (N_mask, d)"]
    EmbTgt --> Sel2["emb_target[:, 1:, :][mask_tensor]<br/>→ (N_mask, d)"]
    Sel1 --> MSE["mse_loss(enc_sel, tgt_sel)"]
    Sel2 --> MSE
    MSE --> Loss["scalar loss"]
```

`mask_tensor` (shape `(B, L-1)`) is built by checking, per non-CLS column, whether `df_masked` holds the literal string `"[MASK]"` there — it's derived from the dataframe, not from the embedder's internal `mask_flags`. There's no separate reconstruction/decoder head: the transformer's own contextual output at the masked position *is* the prediction, regressed straight against the clean embedding. If a batch happens to contain zero masks, `forward` returns a zero loss (with a stray debug `print("a")` — a minor rough edge, not a design point).

---

## TridentModel — the fine-tuning / classification head

`src/models.py:60`. This is the class README currently calls `TridentClassifier` — the code names it `TridentModel`. It wraps the *same* embedder and transformer objects `TridentPretrainer` was just using (passed in directly from `pretraining.model.embedder` / `.transformer`, `src/training/finetuning.py:102-107` — not a copy) and adds a small classification head on top of the `[CLS]` output.

### Theory

```mermaid
flowchart TD
    Row["Row (nulls replaced, no masking)"] --> Emb["TabularEmbedder<br/>(weights carried over from pretraining)"]
    Emb --> Enc["TabularTransformerEncoder<br/>(weights carried over from pretraining)"]
    Enc --> CLS["Take the [CLS] token's output"]
    CLS --> Head["Linear → ReLU → Dropout → Linear"]
    Head --> Logits["Class logits"]
    Logits --> Loss["Class-weighted Cross-Entropy<br/>(when labels are given)"]
```

### Implementation

```mermaid
flowchart TD
    Row["df"] --> Emb["embedder(df) → x: (B, L, d)"]
    Emb --> Enc["transformer(x) → encoded_output: (B, L, d)"]
    Enc --> CLS["encoded_output[:, 0, :] → (B, d)"]
    CLS --> H1["Linear(d, d//2)"] --> H2["ReLU"] --> H3["Dropout(0.3)"] --> H4["Linear(d//2, num_labels)"]
    H4 --> Logits["logits: (B, num_labels)"]
    Logits --> CE["CrossEntropyLoss(weight=class_weights)<br/>when labels given"]
```

`class_weights` come from `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` computed over the fold's *training* labels only (`finetuning.py:96-101`). The best-validation-loss checkpoint is tracked across all `finetuning_epochs` and restored at the end (a detached clone of `model.state_dict()`) — training always runs the full epoch budget, it just doesn't keep the last epoch's weights if an earlier one scored better on validation loss.

---

## TridentDecoder — reconstructing cell values

`src/models.py`. The imputation task's counterpart to `TridentModel`, added by
[ADR 0004](adr/0004-imputation-decoder-task.md). Where the classifier reads one token
(`[CLS]`) and answers one question, the decoder reads *every* column token and answers a
question per hidden cell: what value was taken away here?

It wraps the *same* embedder and transformer that pre-training produced, exactly as the
classifier does, and adds one head per column. It never re-initialises them.

### Theory

```mermaid
flowchart TD
    Row["Row with cells hidden<br/>([MASK] where a value was taken)"] --> Emb["TabularEmbedder<br/>(weights carried over from pretraining)"]
    Emb --> Enc["TabularTransformerEncoder"]
    Enc --> Drop["Drop the [CLS] position;<br/>keep one vector per column"]
    Drop --> CatHead["Categorical column:<br/>score its real categories only"]
    Drop --> NumHead["Numerical column:<br/>predict a scalar"]
    CatHead --> CE["Cross-entropy at hidden cells"]
    NumHead --> MSE["Squared error at hidden cells"]
    CE --> Sum["mean(CE) + lambda_num * mean(MSE)"]
    MSE --> Sum
```

The two terms are averaged over **their own** hidden cells before being summed, so a table
of twenty numerical and two categorical columns cannot drown the categorical term.

### Implementation

```mermaid
flowchart TD
    Hidden["hidden: EncodedTable<br/>masked_positions (B, n_tokens)"] --> Emb["embedder(hidden) → (B, L, d)"]
    Emb --> Enc["transformer(...) → (B, L, d)"]
    Enc --> Slice["[:, 1:, :] → (B, n_tokens, d)<br/>column j at index j, cat-then-num"]
    Slice --> Cat["cat_heads[key]: Linear(d, V_col − 3)<br/>→ (N_hidden, V_col − 3)"]
    Slice --> Num["num_heads[key]: Linear(d, hidden_dim)<br/>→ ReLU → Linear(hidden_dim, 1)"]
    Cat --> CEsel["cross_entropy vs local_of[target id]"]
    Num --> MSEsel["mse vs targets.num_values"]
```

**The output space excludes three vocabulary entries**, and this is the point of the
design rather than a detail. Every categorical vocabulary contains `[MASK]`, `[NULL]` and
the placeholder a missing cell stringifies to (`"nan"`; `as_category_strings` in
`embedder.py` collapses every pandas missing sentinel to that one, so naming it is enough).
None is ever a legitimate imputation, so the head has `V_col − 3` outputs and the target is
remapped through a `local_of` buffer. The excluded ids are looked up through the column's
own `LabelEncoder`, never assumed: classes are sorted, so they land at different positions
in every column.

Two `EncodedTable`s go in. The **hidden** view supplies the context *and*, through
`masked_positions`, says which cells to score; the **clean** view supplies their true
values. Targets are encoded from `preprocess_table(..., fine_tunning=True)` rather than the
raw frame, which is what pre-training encodes: the raw frame puts nulls on the dead
placeholder id and leaves `NaN` in the numerical targets (1400 of them on
`credit-g_20nan`), while the processed frame has neither.

`predict` returns the same three things a caller needs and nothing about how the heads
work: the column's own vocabulary id for each categorical cell, a scalar for each numerical
one, and how sure the head was.

---

## End-to-end training orchestration

`run_training` (`src/training/runner.py:33`) is the function everything above gets called from, once per fold. MLflow's parent-run/fold-run/best-fold/worst-fold tagging is a separate concern, documented in full in [CONTEXT.md](../CONTEXT.md) and [ADR 0002](adr/0002-curated-cross-validation-mlflow-runs.md) — this diagram only shows where the model code above plugs in.

```mermaid
flowchart TD
    Seed["set_global_seed(seed)"] --> Prep["prepare_dataset(spec)"]
    Prep --> Folds["build_folds(...)"]
    Folds --> Parent["Open MLflow parent run<br/>(logs hyperparameters)"]
    Parent --> FoldLoop{"for each fold"}
    FoldLoop --> FoldRun["Open MLflow fold run"]
    FoldRun --> Pretrain["train_pretrainer(...)<br/>fresh embedder + transformer,<br/>pretraining_epochs"]
    Pretrain --> Task{"request.task"}
    Task -->|"classification"| Finetune["train_and_evaluate_classifier(...)<br/>reuses the pretrained embedder + transformer,<br/>finetuning_epochs, restores best-val checkpoint,<br/>evaluates once on the test split"]
    Task -->|"imputation"| Decode["train_and_evaluate_decoder(...)<br/>same reused encoder, decode_epochs,<br/>masks re-rolled each epoch, fixed validation mask,<br/>scores self-masked and induced-missing cells"]
    Decode --> Artifacts
    Finetune --> Artifacts["Optional: loss plots, saved model weights;<br/>imputation also writes a preview and a cell ledger"]
    Artifacts --> Record["Record FoldResult"]
    Record --> FoldLoop
    FoldLoop -->|"all folds done"| Aggregate{"cv_folds set?"}
    Aggregate -->|"yes"| CVSummary["summarize_cross_validation(...)<br/>mean / 95% CI across folds"]
    Aggregate -->|"no"| SingleSplit["Single-split provenance logging"]
```

---

## Hyperparameter → code wiring reference

`Hyperparameters` (`src/training/types.py:9`) is the single source of truth; `from_mapping()` translates the legacy JSON keys (the format used in `datasets/hiperparams/{base}/{name}.json` and documented in README's Configuration System) into it.

| Dataclass field | JSON key | Where it actually lands |
|---|---|---|
| `dimension` | `DIM` | `TabularEmbedder.dimensao` = `TabularTransformerEncoder.d_model` — the one embedding width shared by the embedder, the transformer, and (halved) the classifier head |
| `hidden_dimension` | `HIDDEN_DIM` | Hidden width of each numerical column's 2-layer MLP inside `TabularEmbedder` |
| `heads` | `HEADS` | `TabularTransformerEncoder.nhead` → `head_dim = dimension // heads` (must divide evenly) |
| `layers` | `LAYERS` | Number of stacked `EncoderLayer`s |
| `feedforward_dimension` | `DIM_FEED` | `filter_size` inside each `EncoderLayer`'s `FeedForwardNetwork` |
| `dropout` | `DROPOUT` | Dropout rate inside every `EncoderLayer` (attention output + FFN output) |
| `pretraining_epochs` / `finetuning_epochs` | `EPOCHS_PRE` / `EPOCH_FINE` | Epoch counts, and the horizon of each stage's `StageScheduler` |
| `lr_scheduler` | `LR_SCHEDULER` | Which schedule `StageScheduler` (`src/training/schedulers.py`) builds for both stages: `cosine_legacy` (default, per-batch cosine with `T_max=epochs`), `cosine`, `warmup_cosine`, `constant`, `plateau`. Overridable with `--lr_scheduler`; tagged on MLflow runs. See [ADR 0003](adr/0003-selectable-learning-rate-schedule.md) |
| `batch_size` | `BATCH` | Manual batch-slice size in both training loops |
| `pretraining_learning_rate` / `_weight_decay` | `LR_PRE` / `WEIGHT_DECAY_PRE` | `AdamW` for `TridentPretrainer` |
| `finetuning_learning_rate` / `_weight_decay` | `LR_FINE` / `WEIGHT_DECAY_FINE` | `AdamW` for `TridentModel` |
| `mask_probability` | `PROB_MASCARA` | `p_base` passed to `preprocess_table` during pretraining |
| `labels` | `LABELS` | **Metadata only** — logged to MLflow (`runner.py:29`), but the model's real `num_labels` is derived at runtime from `len(dataset.label_classes)` (`finetuning.py:92`), not from this field |

Two gotchas worth remembering:

- The classification head's dropout is **hardcoded to `0.3`** (`src/models.py:88`), independent of the `dropout`/`DROPOUT` hyperparameter that controls the transformer.
- `labels`/`LABELS` doesn't configure anything about the model shape — it's a records-only field. Changing it doesn't change `num_labels`; the dataset does.

---

## Running it

This doc stops at the model and data-pipeline boundary. For actually invoking training — CLI flags, `main.py`/`train.py`/`opt.py`, `--metrics_dir`/`--disable_mlflow`, MLflow run comparisons — see [README.md](../README.md) and [AGENTS.md](../AGENTS.md); those stay the single source of truth for operational usage so this document doesn't drift out of sync with them.
