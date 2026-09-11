"""The type checker is a test, so it runs where this repo already looks.

`AGENTS.md` makes `pytest -m "not integration"` the ritual for every agent and every
human. A checker wired anywhere else -- a pre-commit hook, a documented command -- is a
checker that has to be remembered, and this codebase has already shown what happens to a
convention nobody enforces: `TRACKING_RUN_ROLES` sat declared in `types.py` while
`tracking.py` went on writing `"run_role"` as a bare literal. See ADR 0006.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything the standard names as checked. `tests/` is deliberately absent: ticket 02
# scoped this gate to source, and typing 4,000-odd lines of test bodies pays least.
CHECKED_PATHS = ("src", "main.py", "opt.py", "train.py", "scripts")


def test_source_tree_passes_the_type_checker() -> None:
    # `sys.executable -m mypy` rather than a bare `mypy`, so the pinned interpreter's
    # mypy runs and not whatever happens to be first on PATH.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(REPO_ROOT / "pyproject.toml"),
            *CHECKED_PATHS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "mypy reported type errors.\n\n"
        f"{completed.stdout}\n{completed.stderr}\n"
        "Fix the error, or -- if the module is one the strictness ramp has not reached "
        "yet -- widen its [[tool.mypy.overrides]] block in pyproject.toml and say so in "
        "the commit message."
    )
