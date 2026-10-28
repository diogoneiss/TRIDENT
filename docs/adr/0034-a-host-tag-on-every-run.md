# ADR 0034: A host tag on every run

- **Status:** **accepted on 2026-10-28** by the user. The user asked whether to mark the machine
  that ran each run: windows, gorgona8 or medusa1. On the proposal to add the tag and backfill it,
  they answered "sim". On Windows's value they said: "prefiro que o hostname quando windows seja so
  windows".
- **Builds on:** ADR 0006 (mirror runs carry their source's tags), ADR 0009 (the store hand-off)
  and ADR 0033 (a second server, whose follower logs into the leader's store).

## Context

- Since ADR 0033, runs from two Linux servers land in one store. Runs from the Windows PC already
  sit beside them. A study split across machines (E86 checks whether that is sound) has to say
  where each cell ran.
- `gpu_name` (`src/training/environment.py`) already tells the machines apart, because each has its
  own GPU model: RTX 3050 Laptop (Windows), RTX 3090 Ti (gorgona8), RTX 5090 (medusa1). It says
  `none` on a CPU-only run, though, and it ties the reading to hardware that can change.

## Decision

1. **Every run records `host` next to the other environment tags.**
   - On Windows the value is `windows`.
   - Elsewhere it is `socket.gethostname()`: `gorgona8`, and `medusa` on medusa1 (its hostname;
     `medusa1` is the ssh alias).
   - The tag is written on the machine that trains. A follower's run logged into the leader's
     store therefore names the follower.
2. **The tag does not change training.** It is a descriptor, as `gpu_name` is. No flag gates it,
   and the classification and imputation fixtures do not record tags.
3. **Backfill from `gpu_name`.** `scripts/backfill_host_tag.py` maps the three GPU models to the
   three values, sources and mirrors alike, and marks each stamped run `host_backfilled=true`.
   - Runs with no `gpu_name` keep no `host`. They were recorded before the environment tags
     existed, and their machine cannot be told.
   - The backfill writes through the leader's server, after a checked backup of the leader's store.

## Consequences

- A query can filter or group by `tags.host`. It must still add `tags.is_mirror = false`, because
  mirrors carry the same tag.
- A new GPU on a known machine changes nothing for new runs. Only the backfill's map has to know
  the old ones.
