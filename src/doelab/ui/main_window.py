"""The application shell: navigation rail, stage gating, and project I/O."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from ..engine.project import FILE_SUFFIX, Project, Stage
from .pages.analyze_page import AnalyzePage
from .pages.base import Page
from .pages.design_page import DesignPage
from .pages.metamodel_page import MetamodelPage
from .pages.optimize_page import OptimizePage
from .pages.problem_page import ProblemPage
from .pages.run_page import RunPage
from .workers import wait_for_all


class MainWindow(QMainWindow):
    """Hosts the ordered workflow pages.

    The rail enforces the pipeline: a stage is selectable only once its inputs
    exist. Rather than letting the user open a page that cannot work and
    discover the problem through an error, unreachable stages are disabled and
    say what is missing when clicked.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DOE Lab")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        self.project = Project.from_problem("gasoline_engine")

        self.pages: list[Page] = [
            ProblemPage(),
            DesignPage(),
            RunPage(),
            AnalyzePage(),
            MetamodelPage(),
            OptimizePage(),
        ]

        self._build_layout()
        self._build_menu()

        for page in self.pages:
            page.project_changed.connect(self._on_project_changed)
            page.status_message.connect(self._show_status)
            page.busy_changed.connect(self._on_busy_changed)
            page.bind(self.project)

        self.rail.setCurrentRow(0)
        self._refresh_rail()
        self._update_title()

    # -- construction -------------------------------------------------------

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = QListWidget()
        self.rail.setObjectName("navRail")
        self.rail.setFixedWidth(210)
        self.rail.setIconSize(QSize(0, 0))
        for index, page in enumerate(self.pages, start=1):
            item = QListWidgetItem(f"{index}.  {page.title}")
            item.setSizeHint(QSize(0, 40))
            self.rail.addItem(item)
        self.rail.currentRowChanged.connect(self._on_rail_changed)

        self.stack = QStackedWidget()
        for page in self.pages:
            self.stack.addWidget(page)

        layout.addWidget(self.rail)
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.path_label = QLabel("Unsaved project")
        self.statusBar().addPermanentWidget(self.path_label)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # -- navigation ---------------------------------------------------------

    def _on_rail_changed(self, row: int) -> None:
        if row < 0:
            return
        page = self.pages[row]
        blocker = self.project.blocker(page.stage)
        if blocker is not None:
            # Bounce back rather than showing a page that cannot function.
            self._show_status(blocker)
            current = self.stack.currentIndex()
            self.rail.blockSignals(True)
            self.rail.setCurrentRow(current)
            self.rail.blockSignals(False)
            return
        self.stack.setCurrentIndex(row)
        page.refresh()

    def _refresh_rail(self) -> None:
        for row, page in enumerate(self.pages):
            item = self.rail.item(row)
            reachable = self.project.reached(page.stage)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsEnabled
                if reachable
                else item.flags() & ~Qt.ItemFlag.ItemIsEnabled
            )
            item.setToolTip("" if reachable else (self.project.blocker(page.stage) or ""))

    def _on_project_changed(self) -> None:
        self._refresh_rail()
        # A change upstream can invalidate the page the user is on, so re-render
        # it to reflect what actually survived. The page that raised the change
        # renders itself, so skip it here rather than drawing it twice.
        current = self.stack.currentWidget()
        if isinstance(current, Page) and current is not self.sender():
            current.refresh()
        self._update_title()

    def _on_busy_changed(self, _busy: bool) -> None:
        """Lock navigation while any stage is recomputing.

        A stage that finishes after the user has moved on invalidates whatever
        depended on its previous output, which would quietly discard downstream
        work the user had since done — metamodels fitted against experiments
        that the late-arriving design just replaced. Holding the user on the
        page that is working keeps the pipeline honest.
        """
        busy = any(page.is_busy for page in self.pages)
        self.rail.setEnabled(not busy)

    def _show_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _update_title(self) -> None:
        name = self.project.path.name if self.project.path else "Unsaved project"
        self.path_label.setText(str(self.project.path) if self.project.path else "Unsaved project")
        self.setWindowTitle(f"DOE Lab — {name}")

    # -- project I/O --------------------------------------------------------

    def _adopt(self, project: Project) -> None:
        self.project = project
        for page in self.pages:
            page.bind(project)
        self._refresh_rail()
        self._update_title()

        # Land on the furthest stage the loaded project supports, so reopening
        # finished work does not start the user back at step one.
        target = 0
        for row, page in enumerate(self.pages):
            if project.reached(page.stage):
                target = row
        self.rail.setCurrentRow(target)

    def new_project(self) -> None:
        self._adopt(Project.from_problem("gasoline_engine"))
        self._show_status("Started a new project")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", f"DOE Lab project (*{FILE_SUFFIX});;JSON (*.json)"
        )
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open project", f"{type(exc).__name__}: {exc}")
            return
        self._adopt(project)
        self._show_status(f"Opened {Path(path).name}")

    def save_project(self) -> None:
        if self.project.path is None:
            self.save_project_as()
            return
        self._write(self.project.path)

    def save_project_as(self) -> None:
        suggested = self.project.path.name if self.project.path else f"study{FILE_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", suggested, f"DOE Lab project (*{FILE_SUFFIX});;JSON (*.json)"
        )
        if not path:
            return
        self._write(Path(path))

    def _write(self, path: Path) -> None:
        try:
            self.project.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save project", f"{type(exc).__name__}: {exc}")
            return
        self._update_title()
        self._show_status(f"Saved {path.name}")

    # -- shutdown -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop outstanding work before the widgets go away.

        Qt aborts the process if a ``QThread`` is destroyed while still
        running, and a worker mid-flight holds references to widgets that
        closing is about to destroy. Signalling the long-running loops to stop
        and then waiting is what makes closing during an optimization safe.
        """
        for page in self.pages:
            page.shutdown()
        wait_for_all(15_000)
        super().closeEvent(event)
