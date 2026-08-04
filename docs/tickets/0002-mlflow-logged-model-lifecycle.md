# Add MLflow logged-model lifecycle for deployable TRIDENT classifiers

## Goal

Extend the curated cross-validation tracking layout so that the selected
`best_fold` classifier is logged as an MLflow model with a stable inference
interface, signature, input example, and optional Model Registry version.

## Why this is deferred

The current `TridentModel` accepts a pandas DataFrame that must already have
TRIDENT's preprocessing conventions, while its embedder owns fitted categorical
encoders and numerical special-token behavior. Logging the raw PyTorch module
alone would not establish a safe serving contract. The tracking reorganization
can therefore proceed independently; model lifecycle support needs a dedicated
inference adapter and end-to-end serving verification.

## Proposed work

1. Define the public prediction input as a pandas DataFrame containing raw
   feature columns in training schema order, without the label column. Decide
   and document how missing values, unknown categorical values, and extra or
   missing columns are handled.
2. Implement a focused MLflow `pyfunc` prediction adapter that restores the
   selected `TridentModel`, owns the preprocessing required for that contract,
   returns class labels and (if supported) class probabilities, and validates
   the input schema before inference.
3. Log the adapter or native PyTorch model only for the `best_fold` diagnostic
   run using `mlflow.pyfunc.log_model()` or `mlflow.pytorch.log_model()` with a
   real input example and explicit/inferred signature. Include the fitted
   preprocessing state, label mapping, feature schema, and required source
   code/dependencies in the model package.
4. Tag the logged model with the parent run ID, dataset variant, fold-ranking
   metric, score, and training code version. Keep `worst_fold` as a diagnostic
   artifact only unless a comparison use case requires a second logged model.
5. Evaluate whether to register successful models in a registry named
   `TRIDENT.<base_dataset>` (or document why logged models alone are enough).
   If registering, use explicit aliases/tags rather than treating the latest
   version as production-ready.
6. Add an isolated test that trains a small classifier, logs the selected
   model to a temporary tracking URI, reloads it through MLflow, and verifies
   predictions and schema validation against the original model.

## Acceptance criteria

- The model page in MLflow displays a valid signature, input example,
  dependencies, parent-run link, and `f1_macro` metadata.
- The reloaded model predicts the same labels as the in-memory selected model
  for a representative held-out batch.
- Unknown categories, absent required columns, and invalid column order fail
  with clear errors or follow a documented supported policy.
- Existing tracking behavior remains available when model saving is disabled.
- A database-backed tracking URI is used for any Model Registry test.

## References

- MLflow documents `mlflow.pytorch.log_model()` and recommends a signature for
  PyTorch serving correctness: https://mlflow.org/docs/latest/ml/deep-learning/pytorch/
- Signatures and input examples define the model's inference contract:
  https://mlflow.org/docs/latest/ml/model/signatures/
- Model Registry concepts and versioning guidance:
  https://mlflow.org/docs/latest/ml/model-registry/
