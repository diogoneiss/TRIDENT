#!/usr/bin/env python3
"""Keep a result cache under a size limit, dropping the entries used longest ago.

The baseline cache (``results/baseline_cache/``, ADR 0007 decision 10) refreshes an entry's
modification time on every hit, so the oldest modification time is the entry used longest
ago. This script removes entries older than ``--max_age_days`` and then, oldest first, as many
more as it takes to bring the cache under ``--max_mb``. Only ``*.json`` entries are touched: a
half-written ``.tmp`` of a live run and anything else in the folder stay. Removing an entry
only costs a recomputation the next time it is needed.

Usage (dry run by default):

    python3 scripts/prune_cache.py                      # results/baseline_cache, 1024 MB
    python3 scripts/prune_cache.py --max_mb 500 --max_age_days 30 --apply
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path("results") / "baseline_cache"
DEFAULT_MAX_MB = 1024


@dataclass
class PruneReport:
    removed: list[Path] = field(default_factory=list)
    removed_bytes: int = 0
    kept_bytes: int = 0


def prune(
    cache_dir: Path, max_bytes: int, max_age_days: float | None, apply: bool, now: float | None = None
) -> PruneReport:
    now = time.time() if now is None else now
    entries = sorted(
        ((path.stat().st_mtime, path.stat().st_size, path) for path in cache_dir.glob("*.json")),
        key=lambda entry: entry[0],
    )
    total = sum(size for _, size, _ in entries)
    report = PruneReport()
    for modified, size, path in entries:
        too_old = max_age_days is not None and now - modified > max_age_days * 86_400
        if not too_old and total <= max_bytes:
            continue
        report.removed.append(path)
        report.removed_bytes += size
        total -= size
        if apply:
            path.unlink(missing_ok=True)
    report.kept_bytes = total
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max_mb", type=float, default=DEFAULT_MAX_MB)
    parser.add_argument("--max_age_days", type=float, default=None)
    parser.add_argument("--apply", action="store_true", help="Delete. Without it, only report.")
    args = parser.parse_args(argv)
    if not args.cache_dir.is_dir():
        print(f"no cache at {args.cache_dir}; nothing to do")
        return 0
    report = prune(args.cache_dir, int(args.max_mb * 1024 * 1024), args.max_age_days, args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(
        f"[{mode}] {args.cache_dir}: {len(report.removed)} entr(ies) removed "
        f"({report.removed_bytes / 2**20:.1f} MB), {report.kept_bytes / 2**20:.1f} MB kept"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
