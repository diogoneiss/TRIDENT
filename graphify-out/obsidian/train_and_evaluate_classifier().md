---
source_file: "src/training/finetuning.py"
type: "code"
community: "Model & Training Core"
location: "L64"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Model__Training_Core
---

# train_and_evaluate_classifier()

## Connections
- [[Fine-tune and evaluate the classifier using the legacy optimization loop.]] - `rationale_for` [EXTRACTED]
- [[FinetuningOutcome]] - `calls` [EXTRACTED]
- [[FoldResult]] - `uses` [INFERRED]
- [[FoldSplit]] - `uses` [INFERRED]
- [[Hyperparameters]] - `uses` [INFERRED]
- [[PreparedDataset]] - `uses` [INFERRED]
- [[PretrainingOutcome]] - `uses` [INFERRED]
- [[TrainingTracker]] - `uses` [INFERRED]
- [[TridentModel]] - `calls` [EXTRACTED]
- [[build_fold_result()]] - `calls` [EXTRACTED]
- [[device]] - `references` [EXTRACTED]
- [[finetuning.py]] - `contains` [EXTRACTED]
- [[preprocess_table()]] - `calls` [EXTRACTED]
- [[run_training()]] - `calls` [EXTRACTED]
- [[runner.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Model__Training_Core