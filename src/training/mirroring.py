"""Mirror run trees into the derived per-task experiments (ADR 0006).

A *mirror run* is a copy of one source run inside ``TRIDENT/mirror/<task>``: params,
tags, complete metric histories, dataset inputs, run name, status and timestamps, but no
artifacts. ``mlflow.parentRunId`` is the only tag rewritten, to the mirror of the
source's parent, so the UI nests mirrors as it nests sources. Every mirror run says
``is_mirror=true`` and names its source in ``source_run_id``.

The trainer and ``scripts/mirror_runs.py`` both call :func:`mirror_run_tree`, so the two
paths cannot produce different mirrors. Everything goes through the public
``MlflowClient`` API; the store deduplicates replayed metrics, params and inputs, which
is what makes a second call an upsert rather than a duplicate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Sequence, TypeVar

from mlflow.entities import Dataset, DatasetInput, InputTag, Metric, Param, Run, RunTag, ViewType
from mlflow.tracking import MlflowClient

from src.mlflow_utils import (
    IS_MIRROR_TAG,
    SOURCE_RUN_ID_TAG,
    TASK_TAG,
    get_or_create_mirror_experiment,
    is_mirror_experiment,
    mirror_experiment_name,
)

logger = logging.getLogger(__name__)

_Entry = TypeVar("_Entry", Param, Metric, RunTag)

PARENT_RUN_ID_TAG = "mlflow.parentRunId"
RUN_NAME_TAG = "mlflow.runName"

# ``log_batch`` accepts at most 1000 metrics, 100 params, 100 tags, and 1000 entries in
# total per call; each kind is sent on its own, so the per-kind limits are the binding ones.
METRICS_PER_BATCH = 1000
_PARAMS_PER_BATCH = 100
_TAGS_PER_BATCH = 100
_SEARCH_PAGE = 1000

# Tags a mirror run owns; never removed because the source lacks them.
_MIRROR_OWN_TAGS = frozenset({IS_MIRROR_TAG, SOURCE_RUN_ID_TAG, PARENT_RUN_ID_TAG, RUN_NAME_TAG})


def _param(key: str, value: str) -> Param:
    # mlflow's entity constructors are untyped; one typed door for the strict gate.
    return Param(key, value)  # type: ignore[no-untyped-call]


def _tag(key: str, value: str) -> RunTag:
    return RunTag(key, value)  # type: ignore[no-untyped-call]


def metric(key: str, value: float, timestamp: int, step: int) -> Metric:
    return Metric(key, value, timestamp, step)


@dataclass
class MirrorReport:
    """What one mirroring pass did, as source run ids."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    # Sources still ``RUNNING``: no mirror until they terminate.
    skipped: list[str] = field(default_factory=list)
    # ``<source run id>:<param>`` where the mirror already held a different value.
    param_conflicts: list[str] = field(default_factory=list)
    # ``<source run id>: <reason>`` for sources that could not be mirrored.
    failed: list[str] = field(default_factory=list)
    # Sources stamped ``is_mirror=false`` because they predate the tag (store sync only).
    stamped: list[str] = field(default_factory=list)
    # Mirror run ids deleted because their source no longer exists at all (store sync only).
    orphans: list[str] = field(default_factory=list)


class MirrorIndex:
    """Mirror runs by source run id, one scan per mirror experiment."""

    def __init__(self, client: MlflowClient) -> None:
        self._client = client
        self._experiments: dict[str, str] = {}
        self._mirrors: dict[str, dict[str, Run]] = {}

    def experiment_for(self, task: str, create: bool = True) -> str | None:
        """The task's mirror experiment id; None when it does not exist and ``create`` is off."""
        if task not in self._experiments:
            if create:
                self._experiments[task] = get_or_create_mirror_experiment(task)
            else:
                experiment = self._client.get_experiment_by_name(mirror_experiment_name(task))
                if experiment is None:
                    return None
                self._experiments[task] = str(experiment.experiment_id)
        return self._experiments[task]

    def mirror_of(self, experiment_id: str, source_run_id: str) -> Run | None:
        return self._mirrors_in(experiment_id).get(source_run_id)

    def remember(self, experiment_id: str, source_run_id: str, mirror: Run) -> None:
        self._mirrors_in(experiment_id)[source_run_id] = mirror

    def forget(self, experiment_id: str, source_run_id: str) -> None:
        self._mirrors_in(experiment_id).pop(source_run_id, None)

    def _mirrors_in(self, experiment_id: str) -> dict[str, Run]:
        if experiment_id not in self._mirrors:
            self._mirrors[experiment_id] = {
                run.data.tags[SOURCE_RUN_ID_TAG]: run
                for run in iter_runs(self._client, experiment_id)
                if SOURCE_RUN_ID_TAG in run.data.tags
            }
        return self._mirrors[experiment_id]


def iter_runs(
    client: MlflowClient, experiment_id: str, filter_string: str = ""
) -> Iterator[Run]:
    """Every run of an experiment, deleted ones included, across result pages."""
    page_token: str | None = None
    while True:
        page = client.search_runs(
            [experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ALL,
            max_results=_SEARCH_PAGE,
            page_token=page_token,
        )
        yield from page
        page_token = page.token
        if not page_token:
            break


def children_of(client: MlflowClient, run: Run) -> list[Run]:
    """The runs nested directly under ``run``, in its own experiment."""
    return list(
        iter_runs(
            client,
            run.info.experiment_id,
            filter_string=f"tags.`{PARENT_RUN_ID_TAG}` = '{run.info.run_id}'",
        )
    )


def mirror_run_tree(
    client: MlflowClient,
    root_run_id: str,
    report: MirrorReport | None = None,
    index: MirrorIndex | None = None,
    apply: bool = True,
) -> MirrorReport:
    """Mirror a root run and every run nested under it, parents before children.

    Parents go first because a child's ``mlflow.parentRunId`` is rewritten to the parent's
    mirror, which has to exist by then. A ``RUNNING`` source is skipped with its subtree;
    a deleted or ``FAILED`` source is skipped with its subtree and has its mirror deleted. With ``apply`` off nothing is written and the
    report says what would have been.
    """
    report = report if report is not None else MirrorReport()
    index = index if index is not None else MirrorIndex(client)
    _mirror_subtree(client, client.get_run(root_run_id), None, index, report, apply)
    return report


def _mirror_subtree(
    client: MlflowClient,
    source: Run,
    parent_mirror_id: str | None,
    index: MirrorIndex,
    report: MirrorReport,
    apply: bool,
) -> None:
    mirror_id = mirror_run(client, source, parent_mirror_id, index, report, apply)
    if mirror_id is None:
        return
    for child in children_of(client, source):
        # ``search_runs`` pages carry no inputs; the full record does.
        _mirror_subtree(client, client.get_run(child.info.run_id), mirror_id, index, report, apply)


def mirror_run(
    client: MlflowClient,
    source: Run,
    parent_mirror_id: str | None,
    index: MirrorIndex,
    report: MirrorReport,
    apply: bool = True,
) -> str | None:
    """Create or refresh the mirror of one run; return its id, or None when there is none.

    In a dry run the source's own id stands in for the mirror id, so the walk continues.
    """
    source_id = source.info.run_id
    task = source.data.tags.get(TASK_TAG)
    if task is None:
        report.failed.append(f"{source_id}: no {TASK_TAG} tag, so no mirror experiment")
        return None
    experiment_id = index.experiment_for(task, create=apply)
    existing = None if experiment_id is None else index.mirror_of(experiment_id, source_id)

    # A deleted source loses its mirror, and so does one that failed: a mirror experiment
    # holds only runs that finished (ADR 0006 decision 5, amended 2026-09-30).
    if source.info.lifecycle_stage != "active" or source.info.status == "FAILED":
        if existing is not None and existing.info.lifecycle_stage == "active":
            if apply:
                client.delete_run(existing.info.run_id)
            report.deleted.append(source_id)
        return None
    if source.info.status == "RUNNING":
        report.skipped.append(source_id)
        return None
    if not apply:
        (report.created if existing is None else report.updated).append(source_id)
        return str(source_id)
    assert experiment_id is not None  # created above when applying

    tags = {
        key: value
        for key, value in source.data.tags.items()
        if key not in (PARENT_RUN_ID_TAG, RUN_NAME_TAG)
    }
    tags[IS_MIRROR_TAG] = "true"
    tags[SOURCE_RUN_ID_TAG] = source_id
    if parent_mirror_id is not None:
        tags[PARENT_RUN_ID_TAG] = parent_mirror_id

    params: list[Param]
    metrics: list[Metric]
    if existing is None:
        mirror = client.create_run(
            experiment_id,
            start_time=source.info.start_time,
            tags=tags,
            run_name=source.info.run_name,
        )
        mirror_id = mirror.info.run_id
        params = [_param(key, value) for key, value in source.data.params.items()]
        metrics = list(_metric_history(client, source))
        report.created.append(source_id)
    else:
        mirror_id = existing.info.run_id
        if existing.info.lifecycle_stage != "active":
            client.restore_run(mirror_id)
        if existing.info.run_name != source.info.run_name:
            client.update_run(mirror_id, name=source.info.run_name)
        for key in existing.data.tags:
            if key not in tags and key not in _MIRROR_OWN_TAGS:
                client.delete_tag(mirror_id, key)
        if existing.data.tags.get(PARENT_RUN_ID_TAG) is not None and parent_mirror_id is None:
            client.delete_tag(mirror_id, PARENT_RUN_ID_TAG)
        params = []
        for key, value in source.data.params.items():
            held = existing.data.params.get(key)
            if held is None:
                params.append(_param(key, value))
            elif held != value:
                report.param_conflicts.append(f"{source_id}:{key}")
        already = {
            (m.key, m.value, m.step, m.timestamp) for m in _metric_history(client, existing)
        }
        metrics = [
            m for m in _metric_history(client, source)
            if (m.key, m.value, m.step, m.timestamp) not in already
        ]
        report.updated.append(source_id)

    for tag_batch in chunks([_tag(key, value) for key, value in tags.items()], _TAGS_PER_BATCH):
        client.log_batch(mirror_id, tags=tag_batch)
    for param_batch in chunks(params, _PARAMS_PER_BATCH):
        client.log_batch(mirror_id, params=param_batch)
    for metric_batch in chunks(metrics, METRICS_PER_BATCH):
        client.log_batch(mirror_id, metrics=metric_batch)
    inputs = [
        DatasetInput(
            dataset=Dataset(
                name=entry.dataset.name,
                digest=entry.dataset.digest,
                source_type=entry.dataset.source_type,
                source=entry.dataset.source,
                schema=entry.dataset.schema,
                profile=entry.dataset.profile,
            ),
            tags=[InputTag(tag.key, tag.value) for tag in entry.tags],
        )
        for entry in source.inputs.dataset_inputs
    ]
    if inputs:
        client.log_inputs(mirror_id, datasets=inputs)
    client.set_terminated(mirror_id, status=source.info.status, end_time=source.info.end_time)
    index.remember(experiment_id, source_id, client.get_run(mirror_id))
    return str(mirror_id)


def _metric_history(client: MlflowClient, run: Run) -> Iterator[Metric]:
    for key in run.data.metrics:
        yield from client.get_metric_history(run.info.run_id, key)


def chunks(items: Sequence[_Entry], size: int) -> Iterator[Sequence[_Entry]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sync_store(
    client: MlflowClient,
    apply: bool,
    tasks: frozenset[str] | None = None,
    experiments: frozenset[str] | None = None,
) -> MirrorReport:
    """Mirror every root of every experiment family, and reconcile what is already there.

    Every active source without ``is_mirror`` is stamped ``false`` on the way. Mirror
    experiments are never scanned as sources. When the scan is not restricted to named
    experiments, mirrors whose source no longer exists anywhere are deleted too.
    """
    report = MirrorReport()
    index = MirrorIndex(client)
    seen: set[str] = set()
    for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
        if is_mirror_experiment(experiment.name):
            continue
        if experiments is not None and experiment.name not in experiments:
            continue
        roots: list[Run] = []
        for run in iter_runs(client, str(experiment.experiment_id)):
            seen.add(run.info.run_id)
            if run.info.lifecycle_stage == "active" and IS_MIRROR_TAG not in run.data.tags:
                if apply:
                    client.set_tag(run.info.run_id, IS_MIRROR_TAG, "false")
                report.stamped.append(run.info.run_id)
            if PARENT_RUN_ID_TAG not in run.data.tags:
                roots.append(run)
        for root in roots:
            if tasks is not None and root.data.tags.get(TASK_TAG) not in tasks:
                continue
            mirror_run_tree(client, root.info.run_id, report, index, apply)
    if experiments is None:
        wanted = None if tasks is None else {mirror_experiment_name(task) for task in tasks}
        for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
            if not is_mirror_experiment(experiment.name):
                continue
            if wanted is not None and experiment.name not in wanted:
                continue
            for mirror in iter_runs(client, str(experiment.experiment_id)):
                source_id = mirror.data.tags.get(SOURCE_RUN_ID_TAG)
                if mirror.info.lifecycle_stage != "active" or source_id in seen:
                    continue
                if apply:
                    client.delete_run(mirror.info.run_id)
                report.orphans.append(mirror.info.run_id)
    return report
