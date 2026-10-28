# Does the occlusion map hold at 20nan and 80nan? (E85; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** E78 (40nan, supported) and E79 (60nan, supported at the bar) leave the grid's two ends
  untested. The thesis's claim is stated per rate.
- **The user's go:** "de acordo em fazer 1 depois o 2, com 4 em paralelo" (2026-10-27, before
  20:40 GMT-3). This is item 2.
- **Code:** no change to `src/`.
  - `occlusion_map.py --study e85` runs and reads this study with E78's code.
  - The reader now applies the rule to each rate on its own and gives one truth call per synthetic
    variant. E79's full printout is byte for byte identical after the change, checked with diff.
  - `generate_synthblock.py` also writes `synthblock_20nan` and `synthblock_80nan`. The 40/60nan
    files are unchanged (md5).
  - `run_occlusion_queue.sh` takes `STUDY=e85`.

## Design (fixed before the first cell)

- **Cells:** the ten tables at 20nan and 80nan, 60 cells. Each runs at the seed set whose earlier
  runs K is checked against:
  - **20nan:** the nine real tables at 101/202/303 (the thesis table, `development_source`;
    pendigits and letter there use ADR 0018's bigger encoder);
  - **80nan:** pendigits, letter, electricity, vehicle, biodeg and kc2 at 707/808/909 (E44), and
    kr-vs-kp, spambase and credit-g at 101/202/303;
  - **synthblock** at 707/808/909 at both rates.
  - Today's default, five folds, `--score_remask`, read in place exactly as E78.
- **Running:** four GPU queues (`STUDY=e85`). Each starts when E84's queue of the same number
  completes. MLflow tag `experiment=occlusion-map-grid-2026-10-27`; state in
  `/scratch2/diogoneiss/trident-2026-10-27-occlusion-grid/`. About 8 h.

## Measures and reading

E78's, unchanged, **applied to each rate separately** (`occlusion_map.py --study e85 --tally`):
- **B, primary, per rate:** supported if faithful on 7 or more of 9 and not faithful on none;
  refuted if faithful on 4 or fewer; mixed otherwise.
- **C, gate, per rate:** the AUROC on `synthblock_<rate>` lies above 0.8 with every seed above 0.8.
  It is inconclusive (floor) if K is not below `mean_mode`.
- **Expected thin support at 80nan (fixed now, not a reason to stop):**
  - A pair needs 10 cells. On the small tables at 80nan, the validation map may leave a target
    with fewer than 3 ranked sources, and that target is skipped.
  - A variant with no usable target reads inconclusive.
- **Descriptive:** as in E78.

**What it leads to:**
- The claim gains its 20nan and 80nan entries as each rate reads.
- No default changes.

## Amendment (2026-10-27, 20:44 GMT-3): deferred before any cell

- The user asked for only part of E84 and a database migration at about 23:00.
- The chained waiters were stopped; no E85 cell started.
- The design and the rule are unchanged, and the study waits for the user's go.

## Second amendment (2026-10-28, 00:30 GMT-3): running, 80nan first, on torch 2.11.0, deadline 08:30

- **The user's go:** "voce pode voltar ao que tinha planejado antes: o resto do 1, 2 e 4 em
  paralelo. vá" (2026-10-28, before 00:30 GMT-3).
- **Torch:** every E85 cell runs under torch 2.11.0+cu128 (installed by the ADR 0033 preparation,
  2026-10-27). E78 and E79 ran under 2.5.1.
  - The study is consistent within itself.
  - The descriptive "K against the earlier run of the same cell" line will no longer read 0.0,
    because of the torch change; that is not a fault of the in-place reading.
- **Order and deadline, fixed now and by the clock only:**
  - Each queue deals its 80nan cells first, then its 20nan cells (`occlusion_map.todo`; E78 and
    E79's single-rate dealing is unchanged).
  - No cell starts after 2026-10-28, 08:30 GMT-3, since the user stops experiments after 09:00.
- **If a rate does not complete:** it stays **partial, not complete**, and gets no verdict. Its
  cells finish later on the user's go, and its rule is read only then.
