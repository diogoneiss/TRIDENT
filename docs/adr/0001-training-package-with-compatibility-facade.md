# Use a training package behind a compatibility facade

The training workflow will live in `src/training/`, while `train.py` retains its public function and CLI behavior. This separates configuration, data preparation, training, artifacts, and orchestration without forcing current callers such as `main.py` and `opt.py` to migrate during a behavior-preserving refactor.
