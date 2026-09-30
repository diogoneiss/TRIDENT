# Ticket 0005: Batched MLflow Metric Writes (Behavior-Preserving)

## Status

Done.

## Problem

A run buffers every epoch metric in memory (`BufferedFoldTracker`) and writes the lot to
MLflow when it finishes. The writing is what cost: `MlflowTracker._replay_record` called
`mlflow.log_metric` once per value, `_log_loss_bands` called `mlflow.log_metrics` once per
epoch, and `_log_summary_metrics` once per metric group. Each call is its own SQLite
transaction, and a commit is paid in fsyncs, not in rows.

On gorgona8, with the store still on the hard disk in rollback mode, a commit cost about
206 ms (400 `log_metric` calls on a scratch store on the same disk). Ablation run
`614714f8` (2026-09-29) spent about 760 of its 1207 s writing: 1388 metrics per diagnostic
child at about 200 ms each (291 s and 275 s), plus 196 s of parent loss bands. The same kind of
run on the Windows SSD spent about 40 s. [ADR 0010](../adr/0010-server-store-on-ssd-with-hdd-replica.md)
has since moved the server store to the SSD in WAL mode, which cuts each commit to a few
milliseconds. What is left after that is the number of commits, which is what this ticket
removes.

## What changed

1. **`_log_metric_events`** (`src/training/tracking.py`) writes a sequence of
   `LoggedMetric` into the active run with `MlflowClient.log_batch`, a thousand per call
   (`METRICS_PER_BATCH`, now shared with `src/training/mirroring.py`). Every former
   per-value or per-group write goes through it: the diagnostic children's and the
   single-split parent's replay, the loss bands, the CV summary shared with Optuna trials,
   and the Optuna trial's step-less finals. A run now commits its metrics in a handful of
   transactions instead of one per value.
2. **Timestamps are claimed, one millisecond per metric, in logged order.**
   `get_metric_history` sorts by `(timestamp, step, value)`, so each metric takes the next
   millisecond from `_MetricClock`, which never hands out the same or an earlier
   millisecond twice in one process: not when two flushes land in the same millisecond, and
   not when the wall clock steps back. A step-less metric is stored at step 0, as
   `mlflow.log_metric` stored it.
3. `MlflowTracker.log_metrics`, the params, `opt.py` and the mirroring are unchanged. The
   mirror copies each row with its timestamp, so a mirror keeps its source's order.

**What a metric timestamp means now.** It marks when the run was written, plus the
metric's position in the flush, not the epoch that produced the value. That was already
true of the replayed diagnostic children, whose timestamps recorded the replay; the
difference is that ties are gone. Before, the three statistics of each loss band shared
one timestamp, and on a fast disk consecutive commits could share a millisecond. No flag
or tag: no metric value, key, step or run changes, and timestamps are not a comparison
axis in the MLflow table.

## Exactness standard and result

The same three scenarios ran with the old code (`51096b6`) and the new, each into its own
store, through `train.main` as the ablation kit calls it: seed 7, 60 pre-training and 40
fine-tuning or decode epochs, `plot_losses` on, on the CPU (`CUDA_VISIBLE_DEVICES=`) with
4 threads in both. The scenarios were `vehicle_00nan` classification with `cv_folds=3`,
`credit-g_20nan` imputation with `cv_folds=3` (baselines and `best_baseline/*` tags), and
`vehicle_00nan` single split. That makes 14 runs per store: parents, best and worst
children, and every mirror.

Between the two stores, on the SSD in WAL and on the hard disk in rollback mode alike:

- **Identical:** the run tree, status, params, tags (all but `mlflow.runName`,
  `mlflow.parentRunId` and `source_run_id`), every metric history as
  `get_metric_history` orders it (1586 keys, 7346 rows, `(step, value, is_nan)`),
  `latest_metrics`, and the artifact files (30, in 7 runs).
- **Differing only by wall clock:** the values of 108 `time/*` and `cv/time/*` keys, with
  the same steps.
- **Rows sharing a timestamp:** 2596 in the old store, 0 in the new.
- **The comparison catches small changes:** a planted change of 1e-9 to one value and one
  deleted row were both reported.

`pytest -m "not integration"` (240, with the typing gate) and `pytest -m integration`
(the `vehicle_00nan` and `credit-g_20nan` fixtures, unmodified) pass.

## Measured speedup

Seconds spent writing the run to MLflow (`finalize_cross_validation` or
`log_single_split_record`: metrics, artifacts and dataset input), in the runs above. The
mirroring, already batched, is timed apart and did not change.

| Scenario | Hard disk, rollback: old → new | Whole run on the hard disk | SSD, WAL: old → new |
|---|---|---|---|
| `vehicle_00nan`, 3 folds | 210.3 → 1.9 | 251.9 → 43.0 | 2.12 → 0.11 |
| `credit-g_20nan` imputation, 3 folds | 219.0 → 2.5 | 303.8 → 88.7 | 2.11 → 0.33 |
| `vehicle_00nan`, single split | 85.3 → 0.5 | 105.5 → 20.6 | 0.77 → 0.02 |

On the SSD store of ADR 0010 the saving per real run (300 and 150 epochs, 5 folds) is
seconds, not minutes. On a store on a hard disk it was most of the run.

## Tests

- `test_finalize_cross_validation_stores_every_epoch_of_every_history` and
  `test_log_single_split_record_stores_every_epoch_of_every_history`: characterisation,
  written and passing on the old code first. Multi-epoch histories logged the way the
  trainer logs them must come back key by key, step by step, with the run's shown value
  being the last one logged, and every mirror must hold its source's rows exactly.
- `test_every_run_stores_its_metrics_in_the_order_they_were_logged`: no two rows of a run
  share a timestamp, and a replayed run reads back in the exact logged order. Failed on
  the old code (the CV parent had 91 rows over 23 timestamps). Parametrised over a clock
  that steps back 10 ms per reading; with `_MetricClock` reduced to the bare wall clock,
  both cases fail.
- The disabled-tracker guard also forbids `MlflowClient.log_batch`, the new write path.
