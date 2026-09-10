from dataclasses import dataclass

import torch
import torch.nn as nn

from .embedder import EncodedTable, TabularEmbedder
from .transformer import TabularTransformerEncoder

# Vocabulary entries that are never a legitimate imputation: the two masking tokens,
# plus the placeholder a missing cell stringifies to. ``as_category_strings`` in
# embedder.py collapses every pandas missing sentinel to "nan", so naming one is enough.
NON_CATEGORY_TOKENS = ("[MASK]", "[NULL]", "nan")


def _stack(columns: list, rows: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Stack per-column tensors feature-major, keeping the shape when there are none."""
    if columns:
        return torch.stack(columns)
    return torch.zeros((0, rows), dtype=dtype, device=device)

class TridentPretrainer(nn.Module):
    """
    Given a masked DataFrame, asks the Transformer to reconstruct,
    only at [MASK] positions, the same embedding vector that would exist
    if the column was not masked.

    The target is the embedding (detached) produced by the TabularEmbedder itself
    when it receives the original DF (without [MASK]).
    """
    def __init__(self, embedder: TabularEmbedder, transformer: TabularTransformerEncoder):
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.d_model = embedder.dimensao
        self.eps = 1e-8   # to avoid division by zero in optional normalization

    def _as_encoded(self, data, device) -> EncodedTable:
        return data if isinstance(data, EncodedTable) else self.embedder.encode(data, device)

    def forward(self, masked, original):
        """
        masked / original : pd.DataFrame or EncodedTable
            The corrupted view and the clean view of the same rows.
        """
        device = next(self.parameters()).device
        masked = self._as_encoded(masked, device)
        original = self._as_encoded(original, device)

        # 1) "Corrupted" embeddings (Transformer input)
        emb_in = self.embedder(masked)                 # (B, L, d)
        # 2) "Pure" embeddings (target) — never backpropagated through, so the
        #    graph is not built for them at all
        with torch.no_grad():
            emb_target = self.embedder(original)       # (B, L, d)

        # 3) Pass through Transformer
        encoded = self.transformer(emb_in)             # (B, L, d)

        # 4) Boolean mask of where [MASK] existed, resolved at encoding time
        #    → shape (B, L‑1)   (ignoring CLS at column 0)
        mask_tensor = masked.masked_positions

        # 5) Select only masked positions (flatten)
        enc_sel  = encoded[:, 1:, :][mask_tensor]      # (N_mask, d)
        tgt_sel  = emb_target[:, 1:, :][mask_tensor]   # (N_mask, d)
        #enc_sel = nn.functional.normalize(enc_sel, dim=-1)
        #tgt_sel = nn.functional.normalize(tgt_sel, dim=-1)
        if enc_sel.numel() == 0:          # no [MASK] in the batch
            return torch.tensor(0., device=device, requires_grad=True), {}

        # 6) Loss = MSE between vectors
        loss = nn.functional.mse_loss(enc_sel, tgt_sel)

        # The metric is returned as a detached tensor rather than a Python float so
        # that reading it does not force a device synchronisation on every batch.
        return loss, {"mse_embedding": loss.detach()}


@dataclass(frozen=True)
class DecodedCells:
    """What the decoder would fill every cell of a batch with.

    ``categorical_ids`` are the column's own LabelEncoder ids, so a caller compares them
    with target ids and decodes them with ``inverse_transform`` without knowing that the
    heads score a narrower space than the vocabulary.
    """

    categorical_ids: torch.Tensor  # (n_categorical, batch) long
    numerical_values: torch.Tensor  # (n_numerical, batch) float, scaled space
    # How sure the head was of the category it chose, for a reader judging a preview.
    categorical_confidence: torch.Tensor  # (n_categorical, batch) float


class TridentDecoder(nn.Module):
    """Reconstructs actual cell values from the encoder's output at each column.

    One head per column, so no two columns share parameters. A categorical head scores
    only the column's real categories: ``[MASK]``, ``[NULL]`` and the missing-value
    placeholder live in the vocabulary but are never a legitimate answer, so they are
    excluded from the output space rather than merely discouraged.

    The excluded ids are looked up through the column's encoder, never assumed: classes
    are sorted, so the special tokens land at different positions in every column.
    """

    def __init__(
        self,
        embedder: TabularEmbedder,
        transformer: TabularTransformerEncoder,
        lambda_num: float = 1.0,
    ):
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.lambda_num = lambda_num

        self.categorical_heads = nn.ModuleDict()
        for column, key in zip(embedder.categorical_columns, embedder.categorical_keys):
            encoder = embedder.label_encoders[column]
            excluded = {
                int(index)
                for token, index in zip(encoder.classes_, range(len(encoder.classes_)))
                if token in NON_CATEGORY_TOKENS
            }
            valid_ids = torch.tensor(
                [index for index in range(len(encoder.classes_)) if index not in excluded],
                dtype=torch.long,
            )
            self.register_buffer(f"valid_ids_{key}", valid_ids, persistent=False)
            # Vocabulary id -> position in this head's output, and -1 for the excluded
            # tokens, which can never be a target because masking never hides a null.
            local_of = torch.full((len(encoder.classes_),), -1, dtype=torch.long)
            local_of[valid_ids] = torch.arange(len(valid_ids))
            self.register_buffer(f"local_of_{key}", local_of, persistent=False)
            self.categorical_heads[key] = nn.Linear(embedder.dimensao, len(valid_ids))

        # The embedder's input MLP in reverse, one per column, so no two columns share
        # parameters. Output is a scalar in the same scaled space the embedder consumed.
        self.numerical_heads = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(embedder.dimensao, embedder.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(embedder.hidden_dim, 1),
                )
                for key in embedder.numerical_keys
            }
        )

    def _encode(self, data) -> torch.Tensor:
        """Contextual output for every column token, dropping the [CLS] position."""
        return self.transformer(self.embedder(data))[:, 1:, :]

    def forward(self, hidden, targets) -> tuple[torch.Tensor, dict]:
        """Reconstruction loss over the cells hidden from the model.

        ``hidden`` is the corrupted view whose ``masked_positions`` say which cells were
        taken away; ``targets`` is the clean view holding their true values. A cell the
        model can already see carries no loss, so a batch with nothing hidden is free.
        """
        device = next(self.parameters()).device
        hidden = self._as_encoded(hidden, device)
        targets = self._as_encoded(targets, device)
        context = self._encode(hidden)
        mask = hidden.masked_positions  # (batch, n_columns), categorical then numerical

        categorical_loss = torch.zeros((), device=device)
        categorical_cells = 0
        for index, key in enumerate(self.embedder.categorical_keys):
            selected = mask[:, index]
            if not bool(selected.any()):
                continue
            logits = self.categorical_heads[key](context[selected, index, :])
            local_of = getattr(self, f"local_of_{key}")
            expected = local_of[targets.cat_indices[index][selected]]
            categorical_loss = categorical_loss + nn.functional.cross_entropy(
                logits, expected, reduction="sum"
            )
            categorical_cells += int(selected.sum())

        numerical_loss = torch.zeros((), device=device)
        numerical_cells = 0
        offset = len(self.embedder.categorical_keys)
        for index, key in enumerate(self.embedder.numerical_keys):
            selected = mask[:, offset + index]
            if not bool(selected.any()):
                continue
            predicted = self.numerical_heads[key](context[selected, offset + index, :]).squeeze(-1)
            expected = targets.num_values[index][selected]
            numerical_loss = numerical_loss + nn.functional.mse_loss(
                predicted, expected, reduction="sum"
            )
            numerical_cells += int(selected.sum())

        if categorical_cells == 0 and numerical_cells == 0:
            return torch.zeros((), device=device, requires_grad=True), {}

        # Each kind is averaged over its own hidden cells, so a table dominated by one
        # kind cannot drown the other's term.
        loss = torch.zeros((), device=device)
        metrics: dict[str, torch.Tensor] = {}
        if categorical_cells:
            categorical_mean = categorical_loss / categorical_cells
            loss = loss + categorical_mean
            metrics["cross_entropy"] = categorical_mean.detach()
        if numerical_cells:
            numerical_mean = numerical_loss / numerical_cells
            loss = loss + self.lambda_num * numerical_mean
            metrics["mse"] = numerical_mean.detach()
        return loss, metrics

    def _as_encoded(self, data, device) -> EncodedTable:
        return data if isinstance(data, EncodedTable) else self.embedder.encode(data, device)

    def predict(self, data) -> DecodedCells:
        """Fill every cell of the batch, in the column's own vocabulary."""
        encoded = self._encode(data)
        categorical_ids = []
        categorical_confidence = []
        for index, key in enumerate(self.embedder.categorical_keys):
            logits = self.categorical_heads[key](encoded[:, index, :])
            valid_ids = getattr(self, f"valid_ids_{key}")
            chosen = logits.argmax(dim=-1)
            categorical_ids.append(valid_ids[chosen])
            categorical_confidence.append(
                logits.softmax(dim=-1).gather(1, chosen.unsqueeze(1)).squeeze(1)
            )
        offset = len(self.embedder.categorical_keys)
        numerical_values = [
            self.numerical_heads[key](encoded[:, offset + index, :]).squeeze(-1)
            for index, key in enumerate(self.embedder.numerical_keys)
        ]
        rows = encoded.shape[0]
        return DecodedCells(
            categorical_ids=_stack(categorical_ids, rows, torch.long, encoded.device),
            numerical_values=_stack(numerical_values, rows, torch.float32, encoded.device),
            categorical_confidence=_stack(
                categorical_confidence, rows, torch.float32, encoded.device
            ),
        )


class TridentModel(nn.Module):
    """
    Unified model for the classification task:
      1) Generates tabular embeddings (TabularEmbedder).
      2) Passes through Transformer encoders (TabularTransformerEncoder).
      3) Takes [CLS] and passes it through a linear layer for classification (nn.Linear).
    """
    def __init__(self, embedder, transformer, num_labels=2, class_weights=None):
        """
        Parameters
        ----------
        embedder : TabularEmbedder
            Responsible for generating embeddings (dimensao = d_model).
        transformer : TabularTransformerEncoder
            Bidirectional transformer encoder.
        num_labels : int
            Number of classes for classification. If 2 => binary.
        class_weights : torch.Tensor, optional
            Class weights for CrossEntropy, if you want to handle imbalance.
        """
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(embedder.dimensao, embedder.dimensao // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedder.dimensao // 2, num_labels),
        )
        
        # Store class weights if provided
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, data, labels=None):
        """
        data : pd.DataFrame or EncodedTable
            Input rows (possibly masked/null, but in fine-tuning usually without mask).
        labels : Tensor (optional), shape (batch_size,) with true classes, 
                 if we want to calculate CrossEntropy loss.

        Returns
        -------
        logits : torch.Tensor
            shape (batch_size, num_labels)
        loss (optional) : torch.Tensor
            If 'labels' is provided, also returns the CrossEntropyLoss.
        """
        x = self.embedder(data)              # (batch_size, seq_len, d_model)
        encoded_output = self.transformer(x) # (batch_size, seq_len, d_model)
        
        # Extract [CLS], which is at encoded_output[:, 0, :]
        cls_representation = encoded_output[:, 0, :]  # (batch_size, d_model)
        
        # Final classification
        logits = self.classifier(cls_representation)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            # If we have class weights, apply them in CrossEntropy
            if self.class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            return logits, loss
        
        return logits 