"""Keeping a result cache under its size limit by dropping what was used longest ago."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import prune_cache  # noqa: E402

DAY = 86_400.0


def _entry(directory: Path, name: str, kilobytes: int, days_ago: float, now: float) -> Path:
    path = directory / f"{name}.json"
    path.write_bytes(b"x" * kilobytes * 1024)
    os.utime(path, (now - days_ago * DAY, now - days_ago * DAY))
    return path


def test_the_least_recently_used_entries_go_first_until_the_cache_fits(tmp_path) -> None:
    now = 1_000 * DAY
    old = _entry(tmp_path, "old", 400, 30, now)
    middle = _entry(tmp_path, "middle", 400, 10, now)
    recent = _entry(tmp_path, "recent", 400, 1, now)

    dry = prune_cache.prune(tmp_path, max_bytes=900 * 1024, max_age_days=None, apply=False, now=now)
    assert dry.removed == [old] and old.exists()

    report = prune_cache.prune(tmp_path, max_bytes=900 * 1024, max_age_days=None, apply=True, now=now)
    assert report.removed == [old]
    assert not old.exists() and middle.exists() and recent.exists()
    assert report.kept_bytes == 800 * 1024


def test_entries_older_than_the_age_limit_go_whatever_the_size(tmp_path) -> None:
    now = 1_000 * DAY
    stale = _entry(tmp_path, "stale", 1, 40, now)
    fresh = _entry(tmp_path, "fresh", 1, 2, now)

    report = prune_cache.prune(tmp_path, max_bytes=10**9, max_age_days=30, apply=True, now=now)

    assert report.removed == [stale]
    assert fresh.exists()


def test_only_cache_entries_are_ever_touched(tmp_path) -> None:
    """Anything that is not an entry (a note, a half-written file of a live run) stays."""
    now = 1_000 * DAY
    note = tmp_path / "README.txt"
    note.write_text("keep me")
    partial = tmp_path / "abc.123.tmp"
    partial.write_text("{")
    os.utime(note, (0, 0))
    os.utime(partial, (0, 0))

    report = prune_cache.prune(tmp_path, max_bytes=0, max_age_days=1, apply=True, now=now)

    assert report.removed == []
    assert note.exists() and partial.exists()
