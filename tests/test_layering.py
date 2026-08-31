"""The engine layer must stay free of Qt.

Keeping the engine UI-agnostic is what lets it be tested headlessly and run on
a worker thread. That boundary erodes silently — one convenient ``QObject``
import and the whole layer needs a running application to import — so it is
asserted rather than trusted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import doelab.engine

ENGINE_DIR = Path(doelab.engine.__file__).parent
FORBIDDEN_ROOTS = {"PySide6", "PyQt5", "PyQt6", "shiboken6", "pyqtgraph", "matplotlib"}


def _imported_roots(source: str) -> set[str]:
    """Top-level package names imported by a module."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no external package root.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "module_path",
    sorted(ENGINE_DIR.glob("*.py")),
    ids=lambda p: p.name,
)
def test_engine_module_imports_no_gui_package(module_path: Path) -> None:
    offenders = _imported_roots(module_path.read_text(encoding="utf-8")) & FORBIDDEN_ROOTS
    assert not offenders, (
        f"{module_path.name} imports GUI package(s) {sorted(offenders)}; "
        "the engine layer must stay importable without a UI"
    )
