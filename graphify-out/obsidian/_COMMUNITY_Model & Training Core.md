---
type: community
members: 54
---

# Model & Training Core

**Members:** 54 nodes

## Members
- [[dot-__init__()]] - code - src/embedder.py
- [[dot-__init__()_1]] - code - src/models.py
- [[dot-__init__()_2]] - code - src/models.py
- [[dot-forward()]] - code - src/embedder.py
- [[dot-forward()_1]] - code - src/models.py
- [[dot-forward()_2]] - code - src/models.py
- [[dot-forward()_3]] - code - src/transformer.py
- [[dot-log_metrics()]] - code - src/training/types.py
- [[Class that encapsulates the creation of embeddings for tabular data -…]] - rationale - src/embedder.py
- [[Classifier fine-tuning and fold metric calculation.]] - rationale - src/training/finetuning.py
- [[Complete transformer encoder for tabular data]] - rationale - src/transformer.py
- [[DataFrame]] - code
- [[Example function that 1) Splits trainval (e.g., 9010) 2) Generates masked DF…]] - rationale - src/utils.py
- [[Fine-tune and evaluate the classifier using the legacy optimization loop.]] - rationale - src/training/finetuning.py
- [[FinetuningOutcome]] - code - src/training/types.py
- [[FoldSplit]] - code - src/training/types.py
- [[For each row in the DataFrame, generates the resulting embedding 1) Transforms…]] - rationale - src/embedder.py
- [[Given a masked DataFrame, asks the Transformer to reconstruct, only at MASK…]] - rationale - src/models.py
- [[Hyperparameters]] - code - src/training/types.py
- [[Immutable values exchanged between the training workflow stages.]] - rationale - src/training/types.py
- [[Log metrics at an optional training step.]] - rationale - src/training/types.py
- [[Masked reconstruction pre-training stage.]] - rationale - src/training/pretraining.py
- [[Parameters ---------- embedder  TabularEmbedder Responsible for generating…]] - rationale - src/models.py
- [[PreparedDataset]] - code - src/training/types.py
- [[Preprocesses the table to replace null values with the `NULL` token, applies…]] - rationale - src/utils.py
- [[PretrainingOutcome]] - code - src/training/types.py
- [[Protocol]] - code
- [[Separates real numerical values from special tokens `MASK` and `NULL`.…]] - rationale - src/utils.py
- [[TabularEmbedder]] - code - src/embedder.py
- [[TabularTransformerEncoder]] - code - src/transformer.py
- [[Train the masked reconstruction model using the legacy optimization loop.]] - rationale - src/training/pretraining.py
- [[TrainingTracker]] - code - src/training/types.py
- [[TridentModel]] - code - src/models.py
- [[TridentPretrainer]] - code - src/models.py
- [[Unified model for the classification task 1) Generates tabular embeddings…]] - rationale - src/models.py
- [[_stub_training_runtime()]] - code - tests/unit/test_training_runtime.py
- [[create_pretrain_datasets()]] - code - src/utils.py
- [[device]] - code
- [[device_1]] - code
- [[df  pd.DataFrame Input DataFrame (possibly maskednull, but in fine-tuning…]] - rationale - src/models.py
- [[embedder.py]] - code - src/embedder.py
- [[finetuning.py]] - code - src/training/finetuning.py
- [[models.py]] - code - src/models.py
- [[preprocess_table()]] - code - src/utils.py
- [[pretraining.py]] - code - src/training/pretraining.py
- [[split_numeric_and_special()]] - code - src/utils.py
- [[test_runner_replays_single_split_into_parent_without_cv_artifacts_or_children()]] - code - tests/unit/test_training_runtime.py
- [[test_runner_uses_fold_buffers_and_finalizes_cross_validation_once()]] - code - tests/unit/test_training_runtime.py
- [[test_training_runtime.py]] - code - tests/unit/test_training_runtime.py
- [[train_and_evaluate_classifier()]] - code - src/training/finetuning.py
- [[train_pretrainer()]] - code - src/training/pretraining.py
- [[transformer.py]] - code - src/transformer.py
- [[types.py]] - code - src/training/types.py
- [[utils.py]] - code - src/utils.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Model__Training_Core
SORT file.name ASC
```

## Connections to other communities
- 35 edges to [[_COMMUNITY_Training Config & Data Pipeline]]
- 27 edges to [[_COMMUNITY_MLflow Fold Tracking]]
- 8 edges to [[_COMMUNITY_Artifact Writing & Persistence]]
- 8 edges to [[_COMMUNITY_Cross-Validation Summary Stats]]
- 5 edges to [[_COMMUNITY_MLflow Tracking Tests]]
- 5 edges to [[_COMMUNITY_Transformer Encoder Layers]]
- 3 edges to [[_COMMUNITY_Fold Metrics Computation]]
- 1 edge to [[_COMMUNITY_CLI Entry & Hyperparameter Search]]

## Top bridge nodes
- [[types.py]] - degree 33, connects to 5 communities
- [[PreparedDataset]] - degree 27, connects to 4 communities
- [[test_training_runtime.py]] - degree 22, connects to 4 communities
- [[finetuning.py]] - degree 17, connects to 3 communities
- [[train_and_evaluate_classifier()]] - degree 15, connects to 3 communities