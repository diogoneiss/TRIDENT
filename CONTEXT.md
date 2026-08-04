# TRIDENT Experiment Tracking

This context defines the terms used to compare and diagnose TRIDENT training
experiments.

## Language

**Fold-ranking metric**:
The classification metric that orders cross-validation folds for diagnostic
retention. In TRIDENT, it is `f1_macro`.
_Avoid_: primary score, best metric

**Internal CV interval**:
A two-sided 95% Student-t interval calculated from the metric values of the
cross-validation folds. It describes within-run variability, not independent
test-set generalization uncertainty.
_Avoid_: confidence interval, generalization guarantee

**Loss band**:
The mean and internal CV interval for one loss series at a given training
epoch. TRIDENT has bands for pre-training and fine-tuning, each with train and
validation loss.
_Avoid_: loss confidence interval, loss summary

**Single-split run**:
A training execution evaluated with the dataset's predefined split rather than
cross-validation. It is represented by one top-level MLflow run.
_Avoid_: single-fold child run, nested single split

**Diagnostic fold run**:
One of the two nested MLflow runs retained from a cross-validation execution:
the `best_fold` or `worst_fold` according to the fold-ranking metric. It holds
the fold's detailed histories and selected artifacts.
_Avoid_: subrun, fold run

When every fold has the same fold-ranking metric value, one diagnostic fold
run is retained with the role `best_and_worst`.

**Dataset provenance**:
The MLflow dataset lineage and companion artifact that identify a prepared
dataset's source, transformations, schema, split strategy, and fold count.
_Avoid_: dataset metadata, data log

**Experiment family**:
The MLflow experiment containing every variant of one base dataset, named
`TRIDENT/<base_dataset>`. Tags identify a particular dataset variant and its
evaluation mode.
_Avoid_: global experiment, per-variant experiment

**CV summary**:
The parent-run metrics derived from all folds: mean, internal CV bounds, sample
standard deviation, observed minimum and maximum, and fold count. Loss series
include only mean and internal CV bounds at each epoch.
_Avoid_: aggregate run, average fold
