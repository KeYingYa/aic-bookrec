"""Every tracked Python file must at least parse.

This exists because a broken string literal in ``cli/submit.py`` passed the rest
of the suite unnoticed: ``pytest tests/`` imports the library and the tests, but
never the CLI entry points, so a syntax error in one of them is invisible until
someone runs it. The smoke tests catch behaviour; this catches typos.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "legacy", "MyModule"}


def _python_files() -> list[Path]:
    return sorted(
        path for path in PROJECT_ROOT.rglob("*.py")
        if not (SKIP_PARTS & set(path.parts))
    )


def test_python_files_discovered():
    """Guard against the discovery itself silently finding nothing."""
    assert len(_python_files()) >= 20


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_file_parses(path: Path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - only on a real defect
        pytest.fail(f"{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}")
