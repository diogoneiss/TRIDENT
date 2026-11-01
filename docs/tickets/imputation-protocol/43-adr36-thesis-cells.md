# The thesis tables' cells that ADR 0036 changed, rerun on today's default (2026-11-01 16:44 GMT-3)

Not a hypothesis test: a rerun so the thesis tables report today's default. Written before the
first cell.

- **Why:** ADR 0036 (accepted 2026-10-31) changed the default of pendigits and letter at 20 and
  40nan (BATCH 1024, LR_PRE and LR_DECODE x4, EPOCHS_DECODE 600). The tables still read those
  variants from runs on the old configuration.
- **The user's go:** "ok, va e faça o item 2 então (isso da adr 36)" (2026-11-01, after 16:20 GMT-3).
- **Which cells:** the development tables (`thesis_table.py`) read pendigits_20nan and letter_20nan
  at 101/202/303 and 42/7/13 (404/505/606 does not include them); the confirmation summary
  (`tabpfn_report.py`, `confirmation_report.py`) reads pendigits_40nan and letter_40nan at
  707/808/909. 18 cells.
- **Each cell:** today's default through the promoted files, five folds, `--score_checkpoints
  --score_calibrated` (every readout the tables use), MLflow tag
  `experiment=adr36-thesis-2026-11-01` (`scripts/experiments/adr36_thesis_runs.py`,
  `run_adr36_thesis_queue.sh`; 6 cells on gorgona8, 12 on medusa1).
- **Then:** `report_common.development_source` and `confirmation_report` read these runs for
  those variants, the tables are rebuilt, and the before/after counts are written here. The offline
  baselines (E41's, E42's, E55's TabPFN) depend only on the folds and are reused.

## Outcome

(open)
