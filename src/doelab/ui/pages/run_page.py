"""Stage 3 — evaluate the design with the solver."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...engine import analysis
from ...engine.project import Stage
from ..widgets.tables import DataFrameView
from ..workers import run_async
from .base import Card, FieldRow, Page


class MetricTile(QWidget):
    """A single headline number with its caption."""

    def __init__(self, label: str, value: str = "—"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)

        self._value = QLabel(value)
        self._value.setObjectName("metricValue")
        self._caption = QLabel(label)
        self._caption.setObjectName("metricLabel")

        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


class RunPage(Page):
    stage = Stage.RESULTS
    title = "Run"
    subtitle = "Evaluate every experiment in the design and collect the responses."

    def __init__(self):
        super().__init__()
        self._task = None

        self.body.addWidget(self._build_controls())
        self.body.addWidget(self._build_tiles())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_results_card())
        splitter.addWidget(self._build_summary_card())
        splitter.setSizes([620, 220])
        self.body.addWidget(splitter, stretch=1)

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> Card:
        card = Card("Execution")

        row = FieldRow()
        self.status_label = QLabel("No experiments run yet.")
        self.status_label.setObjectName("pageSubtitle")
        row.addWidget(self.status_label)
        row.spacer()

        self.run_button = QPushButton("Run experiments")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        row.addWidget(self.run_button)
        card.add_layout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        card.add(self.progress)
        return card

    def _build_tiles(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.tiles = {
            "experiments": MetricTile("Experiments"),
            "responses": MetricTile("Responses"),
            "design": MetricTile("Design type"),
            "noise": MetricTile("Noise"),
        }
        for tile in self.tiles.values():
            card = Card()
            card.add(tile)
            layout.addWidget(card)
        layout.addStretch(1)
        return container

    def _build_results_card(self) -> Card:
        card = Card("Results")
        self.results_view = DataFrameView(decimals=3)
        card.add(self.results_view, stretch=1)
        return card

    def _build_summary_card(self) -> Card:
        card = Card("Response summary")
        self.summary_view = DataFrameView(decimals=3, show_index=True)
        card.add(self.summary_view, stretch=1)
        return card

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None:
            return

        design_rows = len(self.project.design) if self.project.has_design else 0
        self.tiles["experiments"].set_value(str(design_rows))
        self.tiles["responses"].set_value(str(len(self.project.responses)))

        labels = dict(
            [("full_factorial", "Full factorial"), ("latin_hypercube", "Latin hypercube"),
             ("d_optimal", "D-optimal")]
        )
        self.tiles["design"].set_value(labels.get(self.project.design_spec.kind, "—"))
        self.tiles["noise"].set_value(
            f"σ {self.project.noise.sigma:g}" if self.project.noise.enabled else "off"
        )

        self.run_button.setEnabled(self.project.has_design)

        if self.project.has_results:
            self._show_results()
            self.status_label.setText(
                f"{len(self.project.results)} experiments evaluated."
            )
        else:
            self.results_view.set_frame(pd.DataFrame())
            self.summary_view.set_frame(pd.DataFrame())
            self.status_label.setText(
                "No experiments run yet." if self.project.has_design
                else "Generate a design first."
            )

    def _show_results(self) -> None:
        assert self.project is not None
        combined = pd.concat(
            [self.project.design.reset_index(drop=True), self.project.results.reset_index(drop=True)],
            axis=1,
        )
        combined.insert(0, "Run", range(1, len(combined) + 1))
        self.results_view.set_frame(combined)
        self.summary_view.set_frame(
            analysis.summary_statistics(self.project.results).reset_index(names="Response")
        )

    # -- execution ----------------------------------------------------------

    def _run(self) -> None:
        if self.project is None or not self.project.has_design:
            return

        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.set_busy(True)
        self.notify("Running experiments...")

        problem = self.project.problem()
        design = self.project.design
        noise = self.project.noise

        self._task = run_async(
            lambda: problem.evaluate(design, noise),
            on_finished=self._on_finished,
            on_failed=self._on_failed,
        )

    def _on_finished(self, results: pd.DataFrame) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.set_busy(False)
        if self.project is None:
            return

        self.project.results = results
        # Fitted metamodels describe the previous results, not these.
        self.project.invalidate_from(Stage.METAMODELS)

        self.notify(f"Ran {len(results)} experiments")
        # Publish before rendering (see DesignPage).
        self.project_changed.emit()
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.set_busy(False)
        self.notify("Run failed")
        QMessageBox.critical(self, "Could not run experiments", message.split("\n\n")[0])
