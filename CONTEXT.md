# TRIDENT Experiment Tracking

This context defines the terms used to compare and diagnose TRIDENT training
experiments.

## Language

**Fold-ranking metric**:
The metric that orders cross-validation folds for diagnostic retention, paired
with the direction that makes a fold the best one. Each task declares its own:
the classification task ranks by `f1_macro`, higher being better; the imputation
task ranks by `impute_score`, lower being better. The pair travels together,
because the selection inverts without the direction.
_Avoid_: primary score, best metric

**Internal CV interval**:
A two-sided 95% Student-t interval calculated from the metric values of the
cross-validation folds. It describes within-run variability, not independent
test-set generalization uncertainty.
_Avoid_: confidence interval, generalization guarantee

**Loss band**:
The mean and internal CV interval for one loss series at a given training
epoch. TRIDENT has bands for pre-training and for the task's second stage
(fine-tuning or the decode stage), each with train and validation loss.
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
evaluation mode. It is the home of every run that trains on the dataset and the
source of truth for it; a mirror experiment only reflects it.
_Avoid_: global experiment, per-variant experiment

**CV summary**:
The parent-run metrics derived from all folds: mean, internal CV bounds, sample
standard deviation, observed minimum and maximum, and fold count. Loss series
include only mean and internal CV bounds at each epoch.
_Avoid_: aggregate run, average fold

**Stage timing**:
The wall-clock seconds one fold spent in pre-training or in the task's second
stage (fine-tuning or the decode stage), measured in the runner around each
stage call. It excludes data preparation, plotting,
and artifact writes, and is only comparable between runs that share the same
`device`, `gpu_name`, `torch_version`, and `cuda_version` tags. Timings are
tracked metrics, never fold metrics, so they stay out of the regression
fixture.
_Avoid_: run duration, epoch time

**Optuna trial run**:
One nested MLflow run per hyper-parameter search trial under its study run.
The trainer logs into it a lightweight record: parameters, the structured
tags, and final metrics only. It is never a comparison parent and never has
diagnostic fold runs; `--retrain_best` produces the full parent record for
the chosen configuration.
_Avoid_: trial parent, nested training run

**Component-ablation benchmark**:
A reproducible experimental protocol that estimates each selected TRIDENT
component's contribution while holding the dataset, data split, seed schedule,
training budget, and reporting rules fixed. It is not a comparison with
external methods.
_Avoid_: model leaderboard, baseline benchmark

**Full-factorial ablation**:
The eight configurations formed by independently enabling or disabling the
three selected TRIDENT mechanisms. It estimates main effects and interactions;
it is not a leave-one-out comparison.
_Avoid_: one-factor-at-a-time ablation

**Paired single-seed CV protocol**:
All ablation configurations use one shared seed and the same five
cross-validation folds for a data condition. Fold-level comparisons are paired,
but conclusions do not quantify initialization or resampling variability.
_Avoid_: multi-seed robustness study

**Primary endpoint**:
Macro F1 is the pre-specified measure used to assess ablation effects.
Accuracy, micro F1, precision, recall, and confusion-derived measures are
secondary diagnostics and do not independently determine the study conclusion.
_Avoid_: primary score, co-primary metric

**Self-masked cell**:
An observed feature cell that TRIDENT's dynamic masking hides behind the mask
token at run time. Training re-rolls them every epoch; imputation evaluation
draws them once per fold at the evaluation mask rate. They exist on every
dataset variant, including `_00nan`.
_Avoid_: artificial NaN, corrupted cell, dynamic-mask cell

**Induced-missing cell**:
A feature cell that is missing in a `_20nan` to `_80nan` dataset variant but
observed in the row-aligned `_00nan` variant of the same base dataset. The
dataset generator injected it, so its true value is known. Native missingness
present in `_00nan` itself is never an induced-missing cell.
_Avoid_: true missing, injected NaN, artificial NaN

**Imputation ground truth**:
The true value of a scored cell: the variant's own value for a self-masked cell,
and the `_00nan` sibling's value for an induced-missing cell. Imputation error
compares the decoder's reconstruction against it on the test fold only; training
never reads the sibling, and induced-missing cells are presented to the model as
the mask token when they are scored.
_Avoid_: original value, clean dataset, target table

**Baseline imputer**:
A fixed imputer scored on exactly the cells the decoder is scored on, so that its
error sits beside the model's on every scored population: the mean/mode baseline,
which the fold-ranking score divides by, and three learned baselines: KNN at two
neighbourhood sizes and one gradient-boosting model per column. All learn from the
training fold alone and never rank a fold.

**Baseline bar**:
The lowest score any baseline imputer reached on one population of one run: the bar a
claim that the decoder beats the baselines has to clear. Never above 1.0, since the
mean/mode baseline sits there. Runs log it as their best baseline.
_Avoid_: KNN bar, best baseline, reference score
_Avoid_: reference imputer, naive imputer, external imputer, benchmark imputer

**Imputation task**:
The training task in which the model reconstructs hidden feature values rather
than predicting a label. It shares pre-training with the classification task and
replaces classifier fine-tuning with the decode stage. It is chosen per
invocation, never per dataset, and its runs carry the `task` tag.
_Avoid_: decode mode, reconstruction mode, decode task

**Decode stage**:
The second training stage of the imputation task, in which the decoder is
trained on self-masked cells with masks re-rolled every epoch. It mirrors
classifier fine-tuning in schedule, checkpoint selection and stage timing, and
never re-initialises the pretrained encoder.
_Avoid_: decoder fine-tuning, stage two, imputation fine-tuning

**Decoder**:
The per-column output heads that map the transformer's output at a hidden cell
back to a value: a distribution over the column's real categories, or a scalar
in scaled space. It can never emit a mask, null or placeholder token.
_Avoid_: reconstruction head, output layer, imputer head

**Search objective**:
The metric and direction a hyperparameter search ranks its trials by, together
with the split it is scored on. It is not the fold-ranking metric: the
imputation task scores its search on the validation split of the trial's
predefined split, so the test split never chooses hyperparameters and stays
untouched until the chosen configuration is retrained. The classification
search scores the test split, as it always has.
_Avoid_: objective value, trial score, best metric

**Search-space profile**:
A named choice of which hyperparameters a search samples and which it holds.
`full` samples every hyperparameter the task uses; `reduced` samples only the
ones that govern the task's own stage and the corruption it learns from.
Chosen per study and recorded on the study and its trials.
_Avoid_: reduced space, tuning mode, search preset

**Held hyperparameter**:
A hyperparameter a search-space profile does not sample. Every trial runs it at
the task default rather than at any stored configuration's value, so trials
differ only in what was sampled.
_Avoid_: frozen (that word belongs to encoder weights), fixed, constant

**Promoted configuration**:
A search's best hyperparameters, published where later runs of the same task
and dataset variant find them without being told. Promotion is a deliberate
act: finishing a study never promotes on its own, and a promoted imputation
configuration never changes what a classification run reads.
_Avoid_: best params file, saved hyperparameters

**Mirror experiment**:
A derived MLflow experiment that gathers, for one task, a mirror run of every
run from every experiment family, so that datasets can be read side by side in
one table. Every run is mirrored, diagnostic fold runs and Optuna runs included,
and nesting is preserved: a mirror run's parent is the mirror run of its
source's parent. There is one per task. It is never a run's home: nothing
trains into it, and the experiment family stays the source of truth.
_Avoid_: global experiment, aggregate experiment, all-datasets experiment

**Mirror run**:
A copy of one source run inside a mirror experiment, carrying the source's
parameters, tags, metrics and dataset lineage but not its artifacts, and
pointing back at the source by its run id. It is derived: nothing logs into it
directly, it is rewritten whenever the source changes, and it goes when the
source is deleted. Every run states whether it is one, so a query across
experiments can keep source runs and mirror runs apart.
_Avoid_: replica, copy, duplicate run, shadow run
