import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from .utils import split_numeric_and_special


def _sanitize(column: str) -> str:
    """Reduce a column name to a valid ModuleDict/ParameterDict key."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', column)


def as_category_strings(column) -> np.ndarray:
    """Stringify a categorical column with every kind of missing value collapsed to one.

    ``astype(str)`` renders pandas' missing sentinels differently: ``NaN`` becomes
    ``"nan"``, ``None`` becomes ``"None"``, ``NaT`` becomes ``"NaT"``. Left alone they
    would become distinct categories with distinct embeddings, so a frame built in code
    would disagree with the same frame read from a CSV. Reading a CSV only ever yields
    ``NaN``, so collapsing to ``"nan"`` keeps every existing run byte for byte while
    making the representation single-valued for both training tasks.
    """
    return column.where(column.notna(), np.nan).astype(str).to_numpy()


@dataclass(frozen=True)
class EncodedTable:
    """A whole DataFrame converted to tensors once, then sliced per batch.

    The per-column tensors are stored feature-major (n_columns, n_rows) so that a
    batch slice keeps each individual column contiguous, exactly like the per-batch
    tensors this replaces. ``masked_positions`` is row-major because the
    pre-training loss indexes it by row.
    """

    cat_indices: torch.Tensor       # (n_categorical, n_rows)  long
    num_values: torch.Tensor        # (n_numerical, n_rows)    float32
    mask_flags: torch.Tensor        # (n_numerical, n_rows)    bool
    null_flags: torch.Tensor        # (n_numerical, n_rows)    bool
    masked_positions: torch.Tensor  # (n_rows, n_categorical + n_numerical) bool

    def __len__(self) -> int:
        return self.masked_positions.shape[0]

    def __getitem__(self, index) -> "EncodedTable":
        """Select rows with a slice or an index tensor, keeping the layout."""
        return EncodedTable(
            cat_indices=self.cat_indices[:, index],
            num_values=self.num_values[:, index],
            mask_flags=self.mask_flags[:, index],
            null_flags=self.null_flags[:, index],
            masked_positions=self.masked_positions[index],
        )


class EpochMasker:
    """An epoch's fresh mask over a frame encoded once, without going back through pandas.

    The training loops re-roll their masks every epoch, and encoding the masked DataFrame
    each time was a quarter to a third of a pre-training epoch on tables with many categorical
    columns. The mask is drawn by the same ``draw_dynamic_mask`` ``preprocess_table`` calls,
    so the global random stream moves exactly as before, and applied to the tensors of the
    frame with its nulls as ``[NULL]``: a hidden categorical cell becomes its column's
    ``[MASK]`` id, a hidden numerical cell a zero with its mask flag. The result equals
    ``encode(preprocess_table(frame, p_base))`` tensor for tensor.
    """

    def __init__(self, embedder: "TabularEmbedder", frame, device) -> None:
        from .utils import preprocess_table

        self._null_values = frame.isnull().to_numpy()
        self._base = embedder.encode(preprocess_table(frame.copy(), fine_tunning=True), device)
        columns = list(embedder.categorical_columns) + list(embedder.numerical_columns)
        self._order = np.array([frame.columns.get_loc(column) for column in columns], dtype=np.int64)
        self._n_categorical = len(embedder.categorical_columns)
        self._mask_ids = torch.tensor(
            [int(embedder.label_encoders[column].transform(["[MASK]"])[0]) for column in embedder.categorical_columns],
            dtype=torch.long,
            device=device,
        )
        self._device = device

    def draw(self, p_base: float) -> "EncodedTable":
        from .utils import draw_dynamic_mask

        dynamic_mask = draw_dynamic_mask(self._null_values, p_base)
        positions = torch.tensor(dynamic_mask[:, self._order], dtype=torch.bool, device=self._device)
        categorical = positions[:, : self._n_categorical].t()
        numerical = positions[:, self._n_categorical :].t()
        base = self._base
        return EncodedTable(
            cat_indices=torch.where(categorical, self._mask_ids.unsqueeze(1), base.cat_indices),
            num_values=torch.where(numerical, torch.zeros((), dtype=base.num_values.dtype, device=self._device), base.num_values),
            mask_flags=numerical,
            null_flags=base.null_flags,
            masked_positions=positions,
        )


class TabularEmbedder(nn.Module):
    """
    Class that encapsulates the creation of embeddings for tabular data:
      - Categorical columns: Uses nn.Embedding + LabelEncoder (with [MASK]/[NULL])
      - Numerical columns: MLP + special embeddings for [MASK] and [NULL]
      - [CLS] token + Positional Embedding
    """
    def __init__(self, df, categorical_columns, numerical_columns,
                 dimensao=4, hidden_dim=32):
        super().__init__()

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dimensao = dimensao
        self.hidden_dim = hidden_dim

        # Store original column names and their sanitized versions for ModuleDict keys
        self.column_to_key = {}
        for col in self.numerical_columns:
            # Replace invalid characters with underscore for module names
            self.column_to_key[col] = _sanitize(col)

        # Sanitized keys resolved once, so forward() never runs a regex per batch.
        self.categorical_keys = [_sanitize(col) for col in self.categorical_columns]
        self.numerical_keys = [self.column_to_key[col] for col in self.numerical_columns]
        for kind, columns, keys in (
            ("categorical", self.categorical_columns, self.categorical_keys),
            ("numerical", self.numerical_columns, self.numerical_keys),
        ):
            if len(set(keys)) != len(keys):
                seen: dict[str, str] = {}
                for column, key in zip(columns, keys):
                    if key in seen:
                        raise ValueError(
                            f"{kind} columns {seen[key]!r} and {column!r} both reduce to the "
                            f"module key {key!r}; rename one of them so each column keeps its "
                            f"own parameters."
                        )
                    seen[key] = column

        # -----------------------
        # 1) LabelEncoders
        #    (including [MASK] and [NULL] in the vocabulary of each categorical column)
        # -----------------------
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()

            # Collect original categories + special tokens
            orig_vals = np.unique(as_category_strings(df[col]))
            special_tokens = ["[MASK]", "[NULL]"]
            categories = np.unique(np.concatenate([orig_vals, special_tokens]))

            # Fit with everything (original values + [MASK], [NULL])
            le.fit(categories)
            self.label_encoders[col] = le

        # -----------------------
        # 2) Categorical Embeddings
        # -----------------------
        self.num_categories = {
            col: len(self.label_encoders[col].classes_)
            for col in self.categorical_columns
        }

        self.embedding_layers = nn.ModuleDict({
            key: nn.Embedding(self.num_categories[col], self.dimensao)
            for col, key in zip(self.categorical_columns, self.categorical_keys)
        })

        # -----------------------
        # 3) MLP for Numerical columns
        # -----------------------
        self.mlp_layers = nn.ModuleDict({
            self.column_to_key[col]: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dimensao)
            ) for col in self.numerical_columns
        })

        # -----------------------
        # 4) Special Embeddings for Numerical columns
        # -----------------------
        self.special_embeddings = nn.ParameterDict()
        for col in self.numerical_columns:
            safe_key = self.column_to_key[col]
            self.special_embeddings[f"{safe_key}_mask"] = nn.Parameter(torch.randn(dimensao))
            self.special_embeddings[f"{safe_key}_null"] = nn.Parameter(torch.randn(dimensao))

        # -----------------------
        # 5) [CLS] Token
        # -----------------------
        self.cls_token = nn.Parameter(torch.randn(dimensao))
        # Per device, the [NULL] and [MASK] ids of every categorical column, for ``gaps_as_mask``.
        self._gap_token_ids: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # -----------------------
        # 6) Positional Embedding
        # -----------------------
        self.n_tokens = len(self.categorical_columns) + len(self.numerical_columns)
        self.pos_embedding_layer = nn.Embedding(self.n_tokens + 1, self.dimensao)

    def gaps_as_mask(self, encoded: "EncodedTable") -> "EncodedTable":
        """The same table with every real gap shown as its column's ``[MASK]``.

        ADR 0004 shows a gap as ``[NULL]``; the decode stage may show it as ``[MASK]`` instead,
        the token the induced scoring presents every gap as, so that it trains on the row shape
        it is scored in (``DECODE_GAP_TOKEN``). Only what the model sees changes: a categorical
        gap takes the column's ``[MASK]`` id, a numerical one its mask flag. ``masked_positions``
        is left alone, because a gap has no truth and must never enter the loss.
        """
        device = encoded.cat_indices.device
        if str(device) not in self._gap_token_ids:
            null_token, mask_token = (
                torch.tensor(
                    [int(self.label_encoders[column].transform([token])[0]) for column in self.categorical_columns],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(1)
                for token in ("[NULL]", "[MASK]")
            )
            self._gap_token_ids[str(device)] = (null_token, mask_token)
        null_ids, mask_ids = self._gap_token_ids[str(device)]
        cat_indices = encoded.cat_indices
        if len(self.categorical_columns):
            cat_indices = torch.where(cat_indices == null_ids, mask_ids, cat_indices)
        return EncodedTable(
            cat_indices=cat_indices,
            num_values=encoded.num_values,
            mask_flags=encoded.mask_flags | encoded.null_flags,
            null_flags=torch.zeros_like(encoded.null_flags),
            masked_positions=encoded.masked_positions,
        )

    def encode(self, df, device=None) -> EncodedTable:
        """
        Convert a whole DataFrame into the tensors ``forward`` consumes.

        Doing this once per frame instead of once per mini-batch is what makes the
        training loop tensor-native: the pandas work, the LabelEncoder lookups and
        the host-to-device copies all stop scaling with the number of batches.
        """
        if device is None:
            device = self.cls_token.device
        n_rows = len(df)

        # 1) Categorical columns -> one (n_categorical, n_rows) index matrix.
        if self.categorical_columns:
            cat_array = np.empty((len(self.categorical_columns), n_rows), dtype=np.int64)
            for index, col in enumerate(self.categorical_columns):
                cat_array[index] = self.label_encoders[col].transform(as_category_strings(df[col]))
            cat_indices = torch.tensor(cat_array, dtype=torch.long, device=device)
        else:
            cat_indices = torch.zeros((0, n_rows), dtype=torch.long, device=device)

        # 2) Numerical columns -> values plus [MASK]/[NULL] flags.
        numeric_values, mask_flags, null_flags = split_numeric_and_special(
            df, self.numerical_columns, device=device
        )

        # 3) [MASK] positions in categorical-then-numerical order, for the
        #    pre-training reconstruction loss.
        columns = list(self.categorical_columns) + list(self.numerical_columns)
        masked_array = np.zeros((n_rows, len(columns)), dtype=bool)
        for index, col in enumerate(columns):
            masked_array[:, index] = (df[col] == "[MASK]").to_numpy()

        return EncodedTable(
            cat_indices=cat_indices,
            num_values=numeric_values.t(),
            mask_flags=mask_flags.t(),
            null_flags=null_flags.t(),
            masked_positions=torch.tensor(masked_array, dtype=torch.bool, device=device),
        )

    def forward(self, data):
        """
        For each row, generates the resulting embedding:
          1) Transforms each categorical column into indices and passes through nn.Embedding.
          2) For each numerical column:
               - If it's a real value, pass through MLP.
               - If it's [MASK], use the special "mask" embedding.
               - If it's [NULL], use the special "null" embedding.
          3) Stack the embeddings of all columns (categorical+numerical) into
             (batch_size, n_tokens, dimension) and insert [CLS] at the top.
          4) Add the positional embedding, returning shape
             (batch_size, n_tokens+1, dimension).

        ``data`` is either a pandas DataFrame or an already encoded ``EncodedTable``.
        """
        dev = self.cls_token.device
        encoded = data if isinstance(data, EncodedTable) else self.encode(data, dev)

        token_groups = []

        # =============== #
        # 1) EMBEDDINGS OF CATEGORICAL COLUMNS
        # =============== #
        if self.categorical_keys:
            token_groups.append(
                torch.stack(
                    [
                        self.embedding_layers[key](encoded.cat_indices[index])
                        for index, key in enumerate(self.categorical_keys)
                    ],
                    dim=1,
                )
            )

        # =============== #
        # 2) EMBEDDINGS OF NUMERICAL COLUMNS
        # =============== #
        # Each numerical column owns a Linear(1 -> hidden) / ReLU / Linear(hidden -> d)
        # MLP. The parameters stay per column, but they are gathered and applied as one
        # batched operation, so a table with C numerical columns costs a constant number
        # of kernel launches instead of 2 * C of them.
        if self.numerical_keys:
            first = [self.mlp_layers[key][0] for key in self.numerical_keys]
            second = [self.mlp_layers[key][2] for key in self.numerical_keys]
            first_weight = torch.stack([layer.weight for layer in first]).squeeze(-1)
            first_bias = torch.stack([layer.bias for layer in first])
            second_weight = torch.stack([layer.weight for layer in second])
            second_bias = torch.stack([layer.bias for layer in second])

            # The first layer takes a single scalar per column, so it is a plain
            # broadcast multiply-add rather than a matrix product.
            values = encoded.num_values.unsqueeze(-1)                     # (C, B, 1)
            hidden = torch.relu(
                values * first_weight.unsqueeze(1) + first_bias.unsqueeze(1)
            )                                                             # (C, B, hidden)
            numerical = torch.baddbmm(
                second_bias.unsqueeze(1), hidden, second_weight.transpose(1, 2)
            )                                                             # (C, B, d)

            # Select the special embedding wherever the cell was [MASK] or [NULL],
            # otherwise the MLP output. The three cases are mutually exclusive, so
            # this is the same choice the old scatter-by-index version made, minus
            # the three device-to-host synchronisations it needed per column.
            mask_embeddings = torch.stack(
                [self.special_embeddings[f"{key}_mask"] for key in self.numerical_keys]
            )
            null_embeddings = torch.stack(
                [self.special_embeddings[f"{key}_null"] for key in self.numerical_keys]
            )
            numerical = torch.where(
                encoded.mask_flags.unsqueeze(-1),
                mask_embeddings.unsqueeze(1),
                torch.where(
                    encoded.null_flags.unsqueeze(-1),
                    null_embeddings.unsqueeze(1),
                    numerical,
                ),
            )
            token_groups.append(numerical.transpose(0, 1))                # (B, C, d)

        # =============== #
        # 3) INSERT [CLS] TOKEN
        # =============== #
        final_embeddings = (
            token_groups[0] if len(token_groups) == 1 else torch.cat(token_groups, dim=1)
        )
        batch_size = final_embeddings.shape[0]

        cls_token_expanded = self.cls_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.dimensao)
        final_embeddings = torch.cat([cls_token_expanded, final_embeddings], dim=1)

        # =============== #
        # 4) ADD POSITIONAL EMBEDDING
        # =============== #
        # Every row uses positions 0..seq_len-1, so the table is sliced and
        # broadcast instead of gathered once per row.
        seq_len = final_embeddings.shape[1]
        pos_embeds = self.pos_embedding_layer.weight[:seq_len].unsqueeze(0)

        final_embeddings = final_embeddings + pos_embeds

        return final_embeddings
