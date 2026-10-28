"""Three-way merge of an old leader's store into the new leader's (ADR 0033, decision 8).

``base`` is the fork base, the state both sides last shared; ``theirs`` is the old leader's
store as it stands; ``ours`` is the new leader's live store, which the merge writes one run
per transaction, so no transaction holds the write lock near MLflow's 5 s busy timeout.
Before each write, one JSON line saying how to undo it goes to the journal; ``undo`` replays
the journal backwards. Rules:

- a run or experiment new in theirs is inserted (an experiment whose name ours already has
  refuses the whole merge); its datasets are matched to ours by (experiment, name, digest);
- on a base run, a field or tag that theirs changed is applied when ours left it as in the
  base; changed on both sides to different values, ours wins (the higher epoch) and the
  lost value is logged; params and metrics theirs added are inserted, and latest_metrics
  recomputed;
- rows in any other table (logged models, traces, ...) that theirs added refuse the merge.

Standard library only.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scripts.sync_remote import StoreError

RUN_FIELDS = ("status", "end_time", "lifecycle_stage", "deleted_time", "name")
EXPERIMENT_FIELDS = ("name", "lifecycle_stage")
# Tables the rules above cover; theirs adding rows to any other table refuses the merge.
HANDLED = {
    "runs", "params", "tags", "metrics", "latest_metrics", "inputs", "input_tags", "datasets",
    "experiments", "experiment_tags", "alembic_version", "workspaces",
}
Say = Callable[..., None]


@dataclass
class Plan:
    new_experiments: list[int] = field(default_factory=list)
    changed_experiments: list[int] = field(default_factory=list)
    new_runs: list[str] = field(default_factory=list)
    changed_runs: list[str] = field(default_factory=list)
    missing_runs: list[str] = field(default_factory=list)


def connect(ours: Path, base: Path, theirs: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{ours.as_uri()}?mode=rw", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    for alias, path in (("b", base), ("t", theirs)):
        connection.execute(f"ATTACH DATABASE ? AS {alias}", (f"{path.as_uri()}?mode=ro",))
    return connection


def columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f'PRAGMA main.table_info("{table}")')]


def _ids(connection: sqlite3.Connection, sql: str, *args: Any) -> list[Any]:
    return [row[0] for row in connection.execute(sql, args)]


def plan(connection: sqlite3.Connection) -> Plan:
    """What a merge would do; raises StoreError for what it refuses to do at all."""
    versions = {_ids(connection, f"SELECT version_num FROM {db}.alembic_version")[0] for db in ("main", "b", "t")}
    if len(versions) != 1:
        raise StoreError(f"the three stores are at different MLflow schema versions: {sorted(versions)}")
    for table in _ids(connection, "SELECT name FROM t.sqlite_master WHERE type = 'table'"):
        if table not in HANDLED and table != "sqlite_sequence":
            added = connection.execute(f'SELECT (SELECT count(*) FROM t."{table}") - (SELECT count(*) FROM b."{table}")').fetchone()[0]
            if added:
                raise StoreError(f"theirs has {added} more row(s) in {table}, for which no merge rule is written")
    result = Plan()
    result.new_experiments = _ids(connection, "SELECT experiment_id FROM t.experiments EXCEPT SELECT experiment_id FROM b.experiments ORDER BY 1")
    for experiment in result.new_experiments:
        name, workspace = connection.execute("SELECT name, workspace FROM t.experiments WHERE experiment_id = ?", (experiment,)).fetchone()
        clash = connection.execute(
            "SELECT experiment_id FROM main.experiments WHERE experiment_id = ? OR (name = ? AND workspace = ?)", (experiment, name, workspace)
        ).fetchone()
        if clash:
            raise StoreError(f"theirs' new experiment {experiment} ({name!r}) clashes with ours' experiment {clash[0]}; rename it on the old leader and run again")
    picked = ", ".join(EXPERIMENT_FIELDS)
    result.changed_experiments = _ids(
        connection, f"SELECT experiment_id FROM (SELECT experiment_id, {picked} FROM t.experiments EXCEPT SELECT experiment_id, {picked} FROM b.experiments) "
        "WHERE experiment_id IN (SELECT experiment_id FROM b.experiments) ORDER BY 1",
    )
    result.new_runs = _ids(connection, "SELECT run_uuid FROM t.runs EXCEPT SELECT run_uuid FROM b.runs ORDER BY 1")
    result.missing_runs = _ids(connection, "SELECT run_uuid FROM b.runs EXCEPT SELECT run_uuid FROM t.runs ORDER BY 1")
    fields = ", ".join(RUN_FIELDS)
    changed: set[str] = set(_ids(connection, f"SELECT run_uuid FROM (SELECT run_uuid, {fields} FROM t.runs EXCEPT SELECT run_uuid, {fields} FROM b.runs)"))
    for table in ("tags", "params", "latest_metrics"):
        changed.update(_ids(connection, f"SELECT run_uuid FROM (SELECT * FROM t.{table} EXCEPT SELECT * FROM b.{table})"))
    changed.update(_ids(connection, "SELECT run_uuid FROM (SELECT key, run_uuid FROM b.tags EXCEPT SELECT key, run_uuid FROM t.tags)"))
    changed.update(_ids(connection, "SELECT run_uuid FROM b.runs WHERE status = 'RUNNING'"))
    result.changed_runs = sorted(changed - set(result.new_runs) - set(result.missing_runs))
    return result


def _rows(connection: sqlite3.Connection, db: str, table: str, where: str, *args: Any) -> list[dict[str, Any]]:
    names = ", ".join(f'"{c}"' for c in columns(connection, table))
    return [dict(row) for row in connection.execute(f'SELECT {names} FROM {db}."{table}" WHERE {where}', args)]


def _insert(connection: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        names = ", ".join(f'"{c}"' for c in row)
        connection.execute(f'INSERT OR IGNORE INTO main."{table}" ({names}) VALUES ({", ".join("?" * len(row))})', tuple(row.values()))


def _journal(journal: Path | None, entry: dict[str, Any]) -> None:
    if journal is not None:
        with open(journal, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()


def _three_way(base: Any, theirs: Any, ours: Any) -> str:
    """``apply`` theirs, ``keep`` ours (nothing to do), or ``conflict`` (ours wins)."""
    if theirs == base or ours == theirs:
        return "keep"
    return "apply" if ours == base else "conflict"


def _insert_experiment(connection: sqlite3.Connection, experiment: int, journal: Path | None, say: Say) -> None:
    _journal(journal, {"kind": "experiment_inserted", "experiment_id": experiment})
    with connection:
        _insert(connection, "experiments", _rows(connection, "t", "experiments", "experiment_id = ?", experiment))
        _insert(connection, "experiment_tags", _rows(connection, "t", "experiment_tags", "experiment_id = ?", experiment))
    say("reconcile_experiment_inserted", experiment_id=experiment)


def _change_experiment(connection: sqlite3.Connection, experiment: int, journal: Path | None, say: Say) -> None:
    rows = {db: _rows(connection, db, "experiments", "experiment_id = ?", experiment) for db in ("b", "t", "main")}
    if not rows["main"]:
        return
    before = rows["main"][0]
    updates = {}
    for name in EXPERIMENT_FIELDS:
        verdict = _three_way(rows["b"][0][name], rows["t"][0][name], before[name])
        if verdict == "apply":
            updates[name] = rows["t"][0][name]
        elif verdict == "conflict":
            say("reconcile_conflict", experiment_id=experiment, field=name, kept=before[name], lost=rows["t"][0][name])
    if updates:
        _journal(journal, {"kind": "experiment_changed", "experiment_id": experiment, "before": {k: before[k] for k in updates}})
        with connection:
            connection.execute(
                f"UPDATE main.experiments SET {', '.join(f'{k} = ?' for k in updates)} WHERE experiment_id = ?", (*updates.values(), experiment)
            )
        say("reconcile_experiment_changed", experiment_id=experiment, fields=updates)


def _insert_run(connection: sqlite3.Connection, run: str, journal: Path | None, say: Say) -> None:
    if _rows(connection, "main", "runs", "run_uuid = ?", run):
        say("reconcile_run_skipped", run=run, reason="already in ours")
        return
    inputs = _rows(connection, "t", "inputs", "destination_type = 'RUN' AND destination_id = ?", run)
    new_datasets: list[str] = []
    mapped: dict[str, str] = {}
    for item in inputs:
        if item["source_type"] != "DATASET":
            continue
        dataset = _rows(connection, "t", "datasets", "dataset_uuid = ?", item["source_id"])[0]
        same = connection.execute(
            "SELECT dataset_uuid FROM main.datasets WHERE experiment_id = ? AND name = ? AND digest = ?",
            (dataset["experiment_id"], dataset["name"], dataset["digest"]),
        ).fetchone()
        if same:
            mapped[item["source_id"]] = same[0]
        elif item["source_id"] not in new_datasets:
            new_datasets.append(item["source_id"])
    _journal(journal, {"kind": "run_inserted", "run": run, "datasets": new_datasets})
    with connection:
        for table in ("runs", "params", "tags", "metrics", "latest_metrics"):
            _insert(connection, table, _rows(connection, "t", table, "run_uuid = ?", run))
        for dataset_uuid in new_datasets:
            _insert(connection, "datasets", _rows(connection, "t", "datasets", "dataset_uuid = ?", dataset_uuid))
        for item in inputs:
            _insert(connection, "input_tags", _rows(connection, "t", "input_tags", "input_uuid = ?", item["input_uuid"]))
            _insert(connection, "inputs", [dict(item, source_id=mapped.get(item["source_id"], item["source_id"]))])
    say("reconcile_run_inserted", run=run, datasets_added=len(new_datasets), datasets_matched=len(mapped))


def _recompute_latest(connection: sqlite3.Connection, run: str) -> None:
    connection.execute("DELETE FROM main.latest_metrics WHERE run_uuid = ?", (run,))
    connection.execute(
        "INSERT INTO main.latest_metrics (key, value, timestamp, step, is_nan, run_uuid) "
        "SELECT key, value, timestamp, step, is_nan, run_uuid FROM (SELECT *, row_number() OVER "
        "(PARTITION BY key ORDER BY step DESC, timestamp DESC, value DESC) AS rank FROM main.metrics WHERE run_uuid = ?) WHERE rank = 1",
        (run,),
    )


def _change_run(connection: sqlite3.Connection, run: str, journal: Path | None, say: Say) -> None:
    ours = _rows(connection, "main", "runs", "run_uuid = ?", run)
    if not ours:
        say("reconcile_run_skipped", run=run, reason="not in ours")
        return
    base, theirs = (_rows(connection, db, "runs", "run_uuid = ?", run)[0] for db in ("b", "t"))
    updates = {}
    for name in RUN_FIELDS:
        verdict = _three_way(base[name], theirs[name], ours[0][name])
        if verdict == "apply":
            updates[name] = theirs[name]
        elif verdict == "conflict":
            say("reconcile_conflict", run=run, field=name, kept=ours[0][name], lost=theirs[name])
    tag = {db: {r["key"]: r["value"] for r in _rows(connection, db, "tags", "run_uuid = ?", run)} for db in ("b", "t", "main")}
    tag_changes: dict[str, str | None] = {}
    for key in sorted(set(tag["b"]) | set(tag["t"]) | set(tag["main"])):
        verdict = _three_way(tag["b"].get(key), tag["t"].get(key), tag["main"].get(key))
        if verdict == "apply":
            tag_changes[key] = tag["t"].get(key)
        elif verdict == "conflict":
            say("reconcile_conflict", run=run, tag=key, kept=tag["main"].get(key), lost=tag["t"].get(key))
    names = ", ".join(columns(connection, "metrics"))
    new_metrics = [dict(r) for r in connection.execute(
        f"SELECT {names} FROM t.metrics WHERE run_uuid = ? EXCEPT SELECT {names} FROM main.metrics WHERE run_uuid = ?", (run, run)
    )]
    new_params = [dict(r) for r in connection.execute(
        "SELECT key, value, run_uuid FROM t.params WHERE run_uuid = ? AND key NOT IN (SELECT key FROM main.params WHERE run_uuid = ?)", (run, run)
    )]
    if not (updates or tag_changes or new_metrics or new_params):
        return
    _journal(journal, {
        "kind": "run_changed", "run": run, "before": {k: ours[0][k] for k in updates},
        "tags_before": {k: tag["main"].get(k) for k in tag_changes},
        "params_added": [p["key"] for p in new_params], "metrics_added": new_metrics,
        "latest_before": _rows(connection, "main", "latest_metrics", "run_uuid = ?", run) if new_metrics else None,
    })
    with connection:
        if updates:
            connection.execute(f"UPDATE main.runs SET {', '.join(f'{k} = ?' for k in updates)} WHERE run_uuid = ?", (*updates.values(), run))
        for key, value in tag_changes.items():
            connection.execute("DELETE FROM main.tags WHERE run_uuid = ? AND key = ?", (run, key))
            if value is not None:
                connection.execute("INSERT INTO main.tags (key, value, run_uuid) VALUES (?, ?, ?)", (key, value, run))
        _insert(connection, "params", new_params)
        if new_metrics:
            _insert(connection, "metrics", new_metrics)
            _recompute_latest(connection, run)
    for key, value in tag_changes.items():
        say("reconcile_tag_applied", run=run, tag=key, value=value)
    if updates or new_metrics or new_params:
        say("reconcile_run_updated", run=run, fields=updates, metrics_added=len(new_metrics), params_added=len(new_params))


def merge(ours: Path, base: Path, theirs: Path, journal: Path | None, say: Say) -> Plan:
    """Plan, and with a ``journal`` apply, the merge; without one it is a dry run."""
    connection = connect(ours, base, theirs)
    try:
        result = plan(connection)
        for run in result.missing_runs:
            say("reconcile_run_missing", run=run, note="in the base, gone from theirs (mlflow gc?); left as is")
        if journal is None:
            return result
        for experiment in result.new_experiments:
            _insert_experiment(connection, experiment, journal, say)
        for experiment in result.changed_experiments:
            _change_experiment(connection, experiment, journal, say)
        for run in result.new_runs:
            _insert_run(connection, run, journal, say)
        for run in result.changed_runs:
            _change_run(connection, run, journal, say)
        return result
    finally:
        connection.close()


def undo(store: Path, journal: Path, say: Say) -> int:
    """Revert a merge from its journal, newest entry first; returns how many entries."""
    entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    connection = sqlite3.connect(f"{store.as_uri()}?mode=rw", uri=True, timeout=30)
    try:
        for entry in reversed(entries):
            with connection:
                if entry["kind"] == "run_inserted":
                    run = entry["run"]
                    connection.execute(
                        "DELETE FROM input_tags WHERE input_uuid IN (SELECT input_uuid FROM inputs WHERE destination_type = 'RUN' AND destination_id = ?)", (run,)
                    )
                    connection.execute("DELETE FROM inputs WHERE destination_type = 'RUN' AND destination_id = ?", (run,))
                    for table in ("latest_metrics", "metrics", "params", "tags", "runs"):
                        connection.execute(f"DELETE FROM {table} WHERE run_uuid = ?", (run,))
                    for dataset in entry["datasets"]:
                        connection.execute(
                            "DELETE FROM datasets WHERE dataset_uuid = ? AND NOT EXISTS (SELECT 1 FROM inputs WHERE source_id = ?)", (dataset, dataset)
                        )
                elif entry["kind"] == "run_changed":
                    run = entry["run"]
                    for name, value in entry["before"].items():
                        connection.execute(f"UPDATE runs SET {name} = ? WHERE run_uuid = ?", (value, run))
                    for key, value in entry["tags_before"].items():
                        connection.execute("DELETE FROM tags WHERE run_uuid = ? AND key = ?", (run, key))
                        if value is not None:
                            connection.execute("INSERT INTO tags (key, value, run_uuid) VALUES (?, ?, ?)", (key, value, run))
                    for key in entry["params_added"]:
                        connection.execute("DELETE FROM params WHERE run_uuid = ? AND key = ?", (run, key))
                    for row in entry["metrics_added"]:
                        connection.execute(f"DELETE FROM metrics WHERE {' AND '.join(f'{k} IS ?' for k in row)}", tuple(row.values()))
                    if entry["latest_before"] is not None:
                        connection.execute("DELETE FROM latest_metrics WHERE run_uuid = ?", (run,))
                        for row in entry["latest_before"]:
                            names = ", ".join(row)
                            connection.execute(f"INSERT INTO latest_metrics ({names}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
                elif entry["kind"] == "experiment_changed":
                    for name, value in entry["before"].items():
                        connection.execute(f"UPDATE experiments SET {name} = ? WHERE experiment_id = ?", (value, entry["experiment_id"]))
                elif entry["kind"] == "experiment_inserted":
                    connection.execute("DELETE FROM experiment_tags WHERE experiment_id = ?", (entry["experiment_id"],))
                    connection.execute("DELETE FROM experiments WHERE experiment_id = ?", (entry["experiment_id"],))
            say("reconcile_undone", kind=entry["kind"], target=entry.get("run", entry.get("experiment_id")))
    finally:
        connection.close()
    return len(entries)
