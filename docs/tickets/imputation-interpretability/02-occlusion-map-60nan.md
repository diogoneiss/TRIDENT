# Does the occlusion map hold at 60nan? (E79; pre-registered 2026-10-26)

This ticket was written and committed before the study's first cell, and before any result of
E78 ([01](01-occlusion-map.md)) was read.

- **Why:** E78 tests the map at 40nan only. ADR 0030's scope runs to 60nan, and at 60nan a target
  has fewer observed sources in each row, so a map could stop being faithful there.
- **The user's go:** "voce pode executar coisas até 7h30, estarei fora até lá, aproveite o tempo
  extra de gpu para rodar mais variantes por exemplo" (2026-10-26, before 23:22 GMT-3).
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e79` runs and reads this study with E78's code; E78's default path is
    unchanged.
  - `generate_synthblock.py` now also writes `synthblock_60nan` with seed 42 + 60. The 40nan file
    is rewritten byte for byte (same md5).
  - `run_occlusion_queue.sh` takes `STUDY=e79` and a `DEADLINE`.

## Design (fixed before the first cell)

- **Cells:** E78's ten tables at 60nan, 30 cells. Each table runs at the seed set whose earlier
  runs K is checked against:
  - `synthblock_60nan`, pendigits, letter and electricity at 707, 808 and 909 (E44 for the real
    ones);
  - vehicle, biodeg, kc2, kr-vs-kp, spambase and credit-g at 101, 202 and 303 (the thesis table,
    `development_source`).
  - Today's default configuration, five folds, `--score_remask`, read in place exactly as E78.
- **Running:** four GPU queues (`STUDY=e79`). Each starts when the E78 queue of the same number
  completes. MLflow tag `experiment=occlusion-map-60nan-2026-10-27`; state in
  `/scratch2/diogoneiss/trident-2026-10-27-occlusion-60/`.
- **The deadline, fixed now and by the clock alone:** no queue starts a cell after
  2026-10-27, 07:00 GMT-3, so the GPU is free when the user returns at 07:30.
  - Cells not started by then run later, when the user decides; the reading waits for all 30.
  - The deadline never depends on a result: no cell is read before all 30 have run.

## Measures and reading

E78's, unchanged, through `occlusion_map.py --study e79 --tally`:
- **B, primary:** D with the top 3 sources ranked on the validation gaps, 20 random draws, the 1%
  SESOI. Supported if faithful on 7 or more of the 9 real variants and not faithful on none; refuted
  if faithful on 4 or fewer; mixed otherwise.
- **C, gate:** the block AUROC on `synthblock_60nan`. It recovers the blocks if the interval lies
  above 0.8 with every seed above 0.8. It is inconclusive (floor) if K is not below `mean_mode`.
- **Descriptive:** the same list as E78.

**What each reading leads to:**
- **E78 and E79 both supported:** the map's faithfulness holds over the scope ADR 0030 uses (40 and
  60nan, MCAR).
- **E79 weaker than E78:** the claim is limited to 40nan, and the tables where it fails are named.
- Neither study changes a default.
