# 06. The hyperparameter JSON schema and the three naming layers

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-11 (charting session, round 3)
Blocked by: 03 (the enum decision changes what the schema's typed fields are)

## Question

The same hyperparameter is spelled three different ways depending on which layer you are
looking at:

| Layer | Spelling | Example |
|---|---|---|
| CLI flag / argparse `dest` | lowercase, snake | `--lr_scheduler` |
| Dataclass field | descriptive, snake | `pretraining_epochs` |
| Hyperparameter JSON key | SCREAMING, abbreviated, part Portuguese | `EPOCHS_PRE`, `PROB_MASCARA` |

`Hyperparameters.from_mapping` (`src/training/types.py:134`) bridges the JSON layer to the
dataclass layer by hand, twenty-odd times:

```python
dimension=int(values.get("DIM", values.get("dimension", 128))),
pretraining_epochs=int(values.get("EPOCHS_PRE", values.get("pretraining_epochs", 300))),
mask_probability=float(values.get("PROB_MASCARA", values.get("mask_probability", 0.5))),
```

Three things are unchecked here and each can fail silently:

1. **Every default is written twice** — once as the dataclass field default, once again as
   the third argument to `.get()`. Nothing makes them agree. If they ever disagree, the
   value you get depends on whether the key was present, which is the worst possible
   failure mode.
2. **An unknown key is ignored.** A JSON file with `EPOCH_PRE` (typo) trains happily at
   the default, and the run looks legitimate in MLflow.
3. **Nothing types the JSON document.** `Mapping[str, Any]` is the whole contract.

Ticket 04 of the previous map established that a promoted configuration lives at
`datasets/hiperparams/<base>/<dataset>.imputation.json`, so this file format is a real
persisted surface that Optuna writes and training reads.

How should this schema be typed?

## Scope, already fixed

Settled during charting and recorded on the map:

- **Typing the mapping is in scope.**
- **Renaming any JSON key is out of scope.** `EPOCHS_PRE` and `PROB_MASCARA` are a
  user-facing persisted contract, exactly like the MLflow tag values. A rename routes
  through the flag-plus-backfill pattern as its own effort. This ticket makes the mapping
  *checked*, not *prettier*.

## What to decide

1. **The representation.** A `TypedDict` with `total=False` for the JSON document? An
   explicit field-name-to-JSON-key mapping table that `from_mapping` iterates? A
   `dataclasses.fields()`-driven loop with the JSON key carried in each field's `metadata`?
   The third is interesting because it puts the JSON key *next to* the field it names,
   which is the same "single source" principle ticket 03 applied to the name tuples.
2. **Whether the duplicated defaults collapse.** If the JSON key lives in field metadata,
   `.get(key, <default>)` can fall back to the dataclass default rather than restating it,
   and failure mode 1 disappears by construction. Is that in scope, or is it a behaviour
   change dressed as typing?

   **This was verified during charting rather than left as an assumption**, by walking the
   `from_mapping` AST and comparing every innermost `.get()` default against the
   corresponding `dataclasses.fields()` default:

   > **22 fields are read by `from_mapping`, and every field with a default is read — none
   > is missed. 21 of them restate the default as a literal, and all 21 agree with the
   > dataclass. The 22nd, `lr_scheduler` (`types.py:160`), does not restate anything: it
   > passes `DEFAULT_LR_SCHEDULER`, the constant.**

   Two consequences for this ticket. First, **collapsing the duplication is behaviour-
   neutral today** — that is now a measured fact, not a hope, so the decision is about
   design rather than risk. Second, **the codebase already contains one instance of the
   right pattern**, and `lr_scheduler` is the worked example to generalise from rather
   than a design to invent. Re-run the check before acting if the tree has moved; the
   script is trivial to reconstruct from this description.
3. **Unknown keys.** Ignore (today's behaviour), warn, or reject? This is a genuine
   behaviour change whichever way it goes, so if anything but "ignore" is chosen it needs
   the flag-plus-default treatment. Note that ADR 0005 established that
   `hyperparams_override` **replaces** the base configuration rather than merging into it,
   so an ignored typo means a default, not a stale value — which makes this less dangerous
   than it first appears, but no less silent.
4. **Where it lives.** `types.py` beside `Hyperparameters`, or its own module.

## Answer

**The JSON key moves into each field's `metadata`, and `from_mapping` becomes a
`fields()` loop.**

```python
pretraining_epochs: int = field(
    default=300, metadata={"json_key": "EPOCHS_PRE"}
)
```

Chosen over a `TypedDict` describing the document and over a module-level key-to-field
table. Both alternatives type the mapping; neither removes the duplication, and the table
adds a *third* place the field list is written down — the same multiple-source structure
that produced the half-migration ticket 03 diagnosed. **This is ticket 03's principle
applied to the second naming layer**: the JSON key lives next to the field it names, so it
cannot drift from it, and the default is stated once instead of twice.

**The duplicated defaults collapse, and the charting measurement is what makes that safe.**
Of the 22 fields `from_mapping` reads, 21 restate their default as a literal and all 21
agree with the dataclass; the 22nd, `lr_scheduler` at `types.py:160`, already passes
`DEFAULT_LR_SCHEDULER` rather than restating it. So the collapse is behaviour-neutral
today — measured, not assumed — and `lr_scheduler` is the worked example to generalise
from rather than a pattern to invent.

**Unknown keys keep today's behaviour: ignored.** Changing that is a behaviour change, and
this effort is behaviour-neutral by construction. ADR 0005 established that
`hyperparams_override` *replaces* the base configuration rather than merging into it, so an
ignored typo yields the default rather than a stale value — which is silent, but bounded.
If rejecting unknown keys is wanted later it needs the flag-plus-default treatment and its
own effort. Note it in `docs/BACKLOG.md` when ticket 09 runs.

**Both naming layers stay exactly as spelled.** `EPOCHS_PRE`, `PROB_MASCARA` and the
lowercase aliases are a persisted contract; this ticket makes the mapping checked, not
prettier. The `metadata` dict must reproduce today's key strings character for character,
including the dual lookup where a field accepts both the SCREAMING key and its lowercase
alias.

### For the execution ticket

- `dataclasses.field(metadata=...)` takes a `MappingProxyType`; read it with
  `f.metadata["json_key"]` and let a missing key raise rather than defaulting, so a field
  added without a key fails loudly at import.
- The dual-key lookup (`values.get("DIM", values.get("dimension", …))`) means metadata
  needs both spellings, not one. Decide whether the lowercase alias is derivable from
  `f.name` — it appears to be, for all 22 — and if so derive it rather than storing it.
  **Verify that on all 22 before relying on it.**
- `f.default` is `dataclasses.MISSING` for a field without one. All 22 currently have
  defaults, but the loop should handle `MISSING` rather than assume.
- `lr_scheduler`'s coercion to the `LrScheduler` enum (ticket 03, caution 3) lands in this
  same loop. Sequence the two tickets so one of them is not rewritten by the other.

## Notes

Call `mattpocock-skills:grilling` and `mattpocock-skills:domain-modeling`. HITL.

Read before starting: `src/training/types.py:134` onward (`from_mapping` in full),
`src/training/config.py` (`_load_base_hyperparameters`, `logged_hyperparameters`,
`resolve_training_request`), README's hyperparameter config schema section, and
[ADR 0005 decision 4](../../../adr/0005-reduced-optuna-search-for-imputation.md) for the
promoted-configuration lookup.

Watch for the trap in point 2: verifying that the ~20 duplicated default pairs currently
agree is cheap and mechanical, and doing it first turns a judgement call into a fact. If
any pair disagrees, that is a bug find, and it belongs in `docs/BACKLOG.md` regardless of
what this ticket decides.
