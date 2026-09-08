---
source_file: "src/training/config.py"
type: "code"
community: "Training Config & Data Pipeline"
location: "L27"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Training_Config__Data_Pipeline
---

# resolve_training_request()

## Connections
- [[dot-from_name()]] - `calls` [EXTRACTED]
- [[dot-save_model()]] - `indirect_call` [INFERRED]
- [[DatasetSpec]] - `uses` [INFERRED]
- [[Namespace_1]] - `references` [EXTRACTED]
- [[RuntimeOptions]] - `calls` [EXTRACTED]
- [[TrainingRequest]] - `calls` [EXTRACTED]
- [[cli.py]] - `imports` [EXTRACTED]
- [[config.py]] - `contains` [EXTRACTED]
- [[load_hyperparameters()]] - `calls` [EXTRACTED]
- [[run_from_namespace()]] - `calls` [EXTRACTED]
- [[test_legacy_namespace_accepts_hyperparameter_override()]] - `calls` [EXTRACTED]
- [[test_legacy_namespace_uses_legacy_runtime_defaults()]] - `calls` [EXTRACTED]
- [[test_training_config.py]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Training_Config__Data_Pipeline