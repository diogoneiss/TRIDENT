"""Every experiment in docs/EXPERIMENTATION_LOG.md has one Index row, and every Index row an entry.

The thesis wiki (its `.sync/fork-delta.sh`) finds new and finished experiments by parsing only the
`| Enn |` Index rows; from E118 to E174 entries were written without rows and it saw none of them.
"""

import re
from pathlib import Path

LOG = Path(__file__).parents[2] / "docs" / "EXPERIMENTATION_LOG.md"
TEXT = LOG.read_text(encoding="utf-8")
# id -> the row's status cell: | Enn | question | dates | status | headline |
ROWS = re.findall(r"^\| (E\d+) \|[^|]*\|[^|]*\| ([^|]*) \|", TEXT, re.M)
# id -> the first line of the entry's Status field
ENTRIES = re.findall(r"^### (E\d+) · .*\n\n- \*\*Status:\*\* *(.*)", TEXT, re.M)


def test_every_entry_heading_has_an_index_row_and_back() -> None:
    headings = re.findall(r"^### (E\d+) · ", TEXT, re.M)
    rows = [e for e, _ in ROWS]
    assert len(rows) == len(set(rows)), "an Index row is duplicated"
    assert sorted(set(headings) - set(rows)) == [], "entries without an Index row"
    assert sorted(set(rows) - set(headings)) == [], "Index rows without an entry"


def test_an_index_row_is_in_progress_exactly_while_its_entry_is() -> None:
    row_status = dict(ROWS)
    stale = [
        e for e, status in ENTRIES
        if status.startswith("in progress") != ("in progress" in row_status.get(e, "in progress"))
    ]
    assert len(ENTRIES) > 100, "the Status line pattern no longer matches the entries"
    assert stale == [], "Index status disagrees with the entry's Status on in progress"
