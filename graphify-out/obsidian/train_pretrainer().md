---
source_file: "src/training/pretraining.py"
type: "code"
community: "Model & Training Core"
location: "L16"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Model__Training_Core
---

# train_pretrainer()

## Connections
- [[FoldSplit]] - `uses` [INFERRED]
- [[Hyperparameters]] - `uses` [INFERRED]
- [[PreparedDataset]] - `uses` [INFERRED]
- [[PretrainingOutcome]] - `calls` [EXTRACTED]
- [[TabularEmbedder]] - `calls` [EXTRACTED]
- [[TabularTransformerEncoder]] - `calls` [EXTRACTED]
- [[Train the masked reconstruction model using the legacy optimization loop.]] - `rationale_for` [EXTRACTED]
- [[TrainingTracker]] - `uses` [INFERRED]
- [[TridentPretrainer]] - `calls` [EXTRACTED]
- [[_stub_training_runtime()]] - `indirect_call` [INFERRED]
- [[device_1]] - `references` [EXTRACTED]
- [[preprocess_table()]] - `calls` [EXTRACTED]
- [[pretraining.py]] - `contains` [EXTRACTED]
- [[run_training()]] - `calls` [EXTRACTED]
- [[runner.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Model__Training_Core