"""Application entry point."""

from __future__ import annotations

import os
import sys

# matplotlib picks its Qt binding from QT_API. Pinning it before the backend is
# imported keeps it from selecting a different (or absent) binding than the one
# the app itself runs on.
os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from .ui.main_window import MainWindow  # noqa: E402
from .ui.theme import STYLESHEET  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DOE Lab")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
