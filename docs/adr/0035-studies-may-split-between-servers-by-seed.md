# ADR 0035: A study may split its cells between the servers by seed

- **Status:** **accepted on 2026-10-29, 20:22 GMT-3** by the user: "sobre o e96, podemos dividir em
  qualquer maquina, vimos que nao ha diferença relevante e nao precisamos testar em mais datasets".
- **Builds on:**
  - ADR 0033, a second server, medusa1, that trains beside gorgona8;
  - ADR 0034, the `host` tag on every run.
  - E86, E94 and E96 ([cross-machine/01](../tickets/cross-machine/01-gorgona8-medusa1.md),
    [02](../tickets/cross-machine/02-equivalence.md),
    [03](../tickets/cross-machine/03-equivalence-without-kc2.md)).
- **It supersedes** the per-table scope that E96's ticket fixed for imputation.

## Context

- gorgona8 (RTX 3090 Ti) and medusa1 (two RTX 5090) are each deterministic run to run. They give
  different fold scores for the same cell, because GPU training differs between architectures. A
  fold can move as much as a seed change moves it.
- **What matters for a study is a systematic shift, and none was found:**
  - **E94, classification:** medusa − gorgona8 macro-F1 +0.0009 [−0.0056, +0.0073] over 150 fold
    pairs. Equivalent within ±0.01.
  - **E94, imputation:** +0.54% [−0.06%, +1.13%] over 350 pairs, no verdict. The tail came from
    kc2_20nan's folds.
  - **E96, imputation without kc2, fresh seeds:** +0.04% [−0.30%, +0.38%] over 250 pairs.
    Equivalent within ±1%.
- E96's ticket scoped a split to credit-g, vehicle and biodeg. The user judged the evidence enough
  for every table: no relevant difference was seen, and no more tables need testing.

## Decision

1. **A study, classification or imputation, may split its cells between gorgona8 and medusa1 by
   seed, on any table.**
2. **Both arms of a pair stay on one machine.** The A/B (or method/opponent) difference of a cell
   is always taken within one machine. The machine then acts like another seed draw, and the
   pairing removes it from each fold's difference.
3. **Each run records where it ran,** through the `host` tag (ADR 0034). A study's report states
   the split.
4. **A study reads the stores and files of both machines:**
   - runs through the leader's store (ADR 0033);
   - offline files copied to one machine before the tally, as E94 and E96 did.

## Consequences

- Parallel work roughly doubles: one study can use both servers.
- **The known limit, kept on record:** kc2's folds are heavy-tailed across machines (E94: folds up
  to +0.51). On kc2 a split adds that noise to a study's fold differences, not a bias. Its
  verdicts were already the widest of the tables.
- A later study that finds a machine effect in its own data (for example, a split study whose
  per-machine halves disagree) reports it, and the user may revisit this decision.
