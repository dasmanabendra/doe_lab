"""Stage 6 — search the surrogates, and watch the front form.

The dashboard updates every generation while the run is in flight. That means
the engine's per-generation callback fires on the worker thread, so it is
routed through a Qt signal (:class:`GenerationBridge`) before anything touches
a widget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...engine import optimize as opt
from ...engine import parallel
from ...engine.factors import ResponseRole
from ...engine.project import Stage
from ..widgets.parallel import ParallelCoordinatesPlot
from ..widgets.plots import LivePlot, MplPanel, SERIES_COLOURS, style_axes
from ..widgets.tables import DataFrameView
from ..workers import run_async
from .base import Card, FieldRow, Page
from .run_page import MetricTile


class GenerationBridge(QObject):
    """Carries per-generation updates from the worker thread to the UI thread."""

    updated = Signal(object)

    def emit_update(self, update: opt.GenerationUpdate) -> None:
        self.updated.emit(update)


class OptimizePage(Page):
    stage = Stage.OPTIMIZE
    title = "Optimize"
    subtitle = (
        "NSGA-II over the fitted metamodels. The front shown is the surrogate's — "
        "validate it against the solver before trusting any design on it."
    )

    def __init__(self):
        super().__init__()
        self._task = None
        self._run: opt.OptimizationRun | None = None
        self._result: opt.OptimizationResult | None = None
        self._bridge = GenerationBridge()
        self._bridge.updated.connect(self._on_generation)
        self._history: list[tuple[int, float]] = []

        self.body.addWidget(self._build_controls())
        self.body.addWidget(self._build_tiles())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_dashboard())
        splitter.addWidget(self._build_lower_tabs())
        # A little taller than the dashboard alone would want, because the
        # lower pane now holds the front explorer as well as the tables, and
        # the axis labels need height before they read.
        splitter.setSizes([420, 360])
        self.body.addWidget(splitter, stretch=1)

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> Card:
        card = Card("Run")

        row = FieldRow()
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(8, 500)
        self.pop_spin.setValue(50)
        row.field("Population", self.pop_spin, width=90)

        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 2000)
        self.gen_spin.setValue(40)
        row.field("Generations", self.gen_spin, width=90)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10_000)
        self.seed_spin.setValue(1)
        row.field("Seed", self.seed_spin, width=80)

        row.spacer()

        self.start_button = QPushButton("Run")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.pause_button.setEnabled(False)
        row.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)

        self.extend_spin = QSpinBox()
        self.extend_spin.setRange(1, 500)
        self.extend_spin.setValue(20)
        self.extend_spin.setFixedWidth(70)
        row.addWidget(self.extend_spin)

        self.extend_button = QPushButton("Extend")
        self.extend_button.clicked.connect(self._extend)
        self.extend_button.setEnabled(False)
        row.addWidget(self.extend_button)

        card.add_layout(row)

        self.objective_label = QLabel()
        self.objective_label.setObjectName("pageSubtitle")
        self.objective_label.setWordWrap(True)
        card.add(self.objective_label)
        return card

    def _build_tiles(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.tiles = {
            "generation": MetricTile("Generation"),
            "evaluations": MetricTile("Evaluations"),
            "indicator": MetricTile("Hypervolume"),
            "front": MetricTile("Front size"),
            "feasible": MetricTile("Feasible"),
            "elapsed": MetricTile("Elapsed"),
        }
        for tile in self.tiles.values():
            card = Card()
            card.add(tile)
            layout.addWidget(card)
        layout.addStretch(1)
        return container

    def _build_dashboard(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        front_card = Card("Pareto front")
        self.front_plot = LivePlot(x_label="", y_label="")
        self._front_all = self.front_plot.plot(
            [], [], pen=None, symbol="o", symbolSize=5,
            symbolBrush=pg.mkBrush(200, 205, 212, 130),
            symbolPen=None, name="evaluated",
        )
        self._front_best = self.front_plot.plot(
            [], [], pen=pg.mkPen("#2f6fb0", width=2), symbol="o", symbolSize=8,
            symbolBrush=pg.mkBrush("#2f6fb0"), symbolPen=pg.mkPen("w", width=1),
            name="front",
        )
        front_card.add(self.front_plot, stretch=1)
        splitter.addWidget(front_card)

        convergence_card = Card("Convergence")
        self.convergence_plot = LivePlot(x_label="generation", y_label="indicator")
        self._convergence_curve = self.convergence_plot.plot(
            [], [], pen=pg.mkPen("#3f8f5b", width=2),
        )
        convergence_card.add(self.convergence_plot, stretch=1)
        splitter.addWidget(convergence_card)

        splitter.setSizes([700, 500])
        return splitter

    def _build_lower_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        designs_page = QWidget()
        designs_layout = QVBoxLayout(designs_page)
        designs_layout.setContentsMargins(8, 8, 8, 8)
        self.front_view = DataFrameView(decimals=3)
        designs_layout.addWidget(self.front_view)
        tabs.addTab(designs_page, "Front designs")

        explorer_page = QWidget()
        explorer_layout = QVBoxLayout(explorer_page)
        explorer_layout.setContentsMargins(8, 8, 8, 8)
        explorer_layout.setSpacing(8)
        self.front_explorer = ParallelCoordinatesPlot()
        explorer_layout.addWidget(self.front_explorer, stretch=1)
        explorer_note = QLabel(
            "The front, one line per design. Brush an objective to the range you can "
            "live with and read off what it costs on the others — which is the choice "
            "a Pareto front leaves you to make."
        )
        explorer_note.setObjectName("pageSubtitle")
        explorer_note.setWordWrap(True)
        explorer_layout.addWidget(explorer_note)
        tabs.addTab(explorer_page, "Front explorer")

        validation_page = QWidget()
        validation_layout = QVBoxLayout(validation_page)
        validation_layout.setContentsMargins(8, 8, 8, 8)
        validation_layout.setSpacing(8)

        row = FieldRow()
        note = QLabel(
            "Re-runs the true solver on the front. Cross-validation cannot catch a "
            "surrogate extrapolating into a corner the design never sampled — this can."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        row.addWidget(note, stretch=1)
        self.validate_button = QPushButton("Validate front")
        self.validate_button.clicked.connect(self._validate)
        self.validate_button.setEnabled(False)
        row.addWidget(self.validate_button)
        validation_layout.addLayout(row)

        self.validation_view = DataFrameView(decimals=4)
        validation_layout.addWidget(self.validation_view, stretch=1)
        self.validation_plot = MplPanel(height=2.6)
        validation_layout.addWidget(self.validation_plot, stretch=1)
        tabs.addTab(validation_page, "Validation")

        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log)
        tabs.addTab(log_page, "Log")

        return tabs

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None:
            return

        objectives, constraints = opt.split_roles(self.project.responses)
        parts = []
        for response in objectives:
            verb = "minimize" if response.role is ResponseRole.OBJECTIVE_MIN else "maximize"
            parts.append(f"{verb} {response.name}")
        for response in constraints:
            bounds = []
            if response.lower is not None:
                bounds.append(f"≥ {response.lower:g}")
            if response.upper is not None:
                bounds.append(f"≤ {response.upper:g}")
            parts.append(f"{response.name} {' and '.join(bounds)}")
        self.objective_label.setText("   ·   ".join(parts) if parts else "No objectives set.")

        self._configure_front_axes(objectives)

        # Report a missing prerequisite on the page itself rather than through
        # the shared status bar: every page refreshes when the project is bound,
        # so a status message posted here would surface while the user is
        # looking at some entirely different stage.
        missing = self._missing_models()
        self.start_button.setEnabled(not missing and bool(objectives))
        if missing:
            self.objective_label.setText(
                "Needs a metamodel for: " + ", ".join(missing)
            )

    def _missing_models(self) -> list[str]:
        if self.project is None:
            return []
        objectives, constraints = opt.split_roles(self.project.responses)
        needed = {r.name for r in objectives} | {r.name for r in constraints}
        return sorted(needed - set(self.project.metamodels))

    def _configure_front_axes(self, objectives) -> None:
        if len(objectives) >= 2:
            self.front_plot.setLabel("bottom", objectives[0].name)
            self.front_plot.setLabel("left", objectives[1].name)
            self.tiles["indicator"]._caption.setText("Hypervolume")
        elif objectives:
            self.front_plot.setLabel("bottom", "design")
            self.front_plot.setLabel("left", objectives[0].name)
            self.tiles["indicator"]._caption.setText(f"Best {objectives[0].name}")

    # -- run control --------------------------------------------------------

    def _start(self) -> None:
        if self.project is None:
            return
        missing = self._missing_models()
        if missing:
            QMessageBox.information(
                self, "Metamodels needed",
                "Fit a metamodel for each of: " + ", ".join(missing),
            )
            return

        config = opt.OptimizationConfig(
            pop_size=self.pop_spin.value(),
            n_generations=self.gen_spin.value(),
            seed=self.seed_spin.value(),
        )
        self.project.optimization = config

        try:
            self._run = opt.OptimizationRun(
                self.project.factors, self.project.responses, self.project.metamodels, config
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot start optimization", str(exc))
            return

        self._history = []
        self._front_all.setData([], [])
        self._front_best.setData([], [])
        self._convergence_curve.setData([], [])
        self.front_explorer.clear()
        self.log.clear()
        self._append_log(
            f"Starting NSGA-II: population {config.pop_size}, "
            f"{config.n_generations} generations, seed {config.seed}"
        )

        self._set_running(True)
        self._execute()

    def _execute(self) -> None:
        run = self._run
        assert run is not None
        self._task = run_async(
            lambda: run.execute(on_generation=self._bridge.emit_update),
            on_finished=self._on_finished,
            on_failed=self._on_failed,
        )

    def _set_running(self, running: bool) -> None:
        self.set_busy(running)
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.extend_button.setEnabled(not running and self._run is not None)
        self.validate_button.setEnabled(not running and self._result is not None)
        for widget in (self.pop_spin, self.gen_spin, self.seed_spin):
            widget.setEnabled(not running)
        if not running:
            self.pause_button.setText("Pause")

    def shutdown(self) -> None:
        """Break out of the generation loop so the worker thread can end.

        An optimization can be mid-run, or paused, when the window closes.
        ``stop()`` covers both: it sets the stop flag and releases the pause
        gate, so the loop reaches its next check and returns.
        """
        if self._run is not None:
            self._run.stop()

    def _toggle_pause(self) -> None:
        if self._run is None:
            return
        if self._run.is_paused:
            self._run.resume()
            self.pause_button.setText("Pause")
            self._append_log("Resumed")
            self.notify("Resumed")
        else:
            self._run.pause()
            self.pause_button.setText("Resume")
            self._append_log("Paused")
            self.notify("Paused")

    def _stop(self) -> None:
        if self._run is None:
            return
        self._run.stop()
        self._append_log("Stop requested — finishing the current generation")
        self.notify("Stopping...")

    def _extend(self) -> None:
        if self._run is None:
            return
        extra = self.extend_spin.value()
        self._run.extend(extra)
        self._append_log(f"Extending by {extra} generations")
        self._set_running(True)
        self._execute()

    # -- live updates -------------------------------------------------------

    def _on_generation(self, update: opt.GenerationUpdate) -> None:
        self.tiles["generation"].set_value(f"{update.generation} / {update.n_generations}")
        self.tiles["evaluations"].set_value(str(update.n_evaluations))
        self.tiles["indicator"].set_value(f"{update.indicator:.4g}")
        self.tiles["front"].set_value(str(len(update.pareto)))
        self.tiles["feasible"].set_value(str(update.n_feasible))
        self.tiles["elapsed"].set_value(f"{update.elapsed:.1f}s")

        self._history.append((update.generation, update.indicator))
        generations = [g for g, _ in self._history]
        indicators = [v for _, v in self._history]
        self._convergence_curve.setData(generations, indicators)

        assert self.project is not None
        objectives, _ = opt.split_roles(self.project.responses)

        if len(objectives) >= 2:
            population = update.population
            feasible = population[population["Feasible"]]
            self._front_all.setData(
                feasible[objectives[0].name].to_numpy(dtype=float),
                feasible[objectives[1].name].to_numpy(dtype=float),
            )
            front = update.pareto.sort_values(objectives[0].name)
            self._front_best.setData(
                front[objectives[0].name].to_numpy(dtype=float),
                front[objectives[1].name].to_numpy(dtype=float),
            )
        elif objectives:
            values = update.population[objectives[0].name].to_numpy(dtype=float)
            self._front_all.setData(np.arange(len(values), dtype=float), values)

        if update.generation % 5 == 0 or update.generation == 1:
            self._append_log(
                f"gen {update.generation:4d}   {update.indicator_name} {update.indicator:.6g}   "
                f"front {len(update.pareto):3d}   feasible {update.n_feasible}"
            )

    def _on_finished(self, result: opt.OptimizationResult) -> None:
        self._result = result
        self._set_running(False)

        verb = "stopped" if result.stopped_early else "completed"
        self._append_log(
            f"Run {verb} after {result.generations_completed} generations "
            f"({len(result.designs)} designs evaluated)"
        )
        self.notify(f"Optimization {verb} — {len(result.pareto)} designs on the front")

        self._show_front(result.pareto)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._append_log("FAILED: " + message.split("\n")[0])
        self.notify("Optimization failed")
        QMessageBox.critical(self, "Optimization failed", message.split("\n\n")[0])

    def _show_front(self, front: pd.DataFrame) -> None:
        if front.empty or self.project is None:
            self.front_view.set_frame(pd.DataFrame())
            self.front_explorer.clear()
            return
        objectives, constraints = opt.split_roles(self.project.responses)
        columns = (
            [r.name for r in objectives]
            + [r.name for r in constraints]
            + self.project.factors.names
        )
        available = [c for c in columns if c in front.columns]
        self.front_view.set_frame(front[available].reset_index(drop=True))

        # Filled in once the run ends rather than every generation. The live
        # dashboard already owns the per-generation redraw, and a front moving
        # under the user's brushes would make them impossible to aim.
        frame = front.reset_index(drop=True)
        axes = parallel.build_axes(self.project.factors, self.project.responses, frame)
        self.front_explorer.set_data(
            frame, axes, parallel.feasibility(frame, self.project.responses)
        )

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    # -- validation ---------------------------------------------------------

    def _validate(self) -> None:
        if self.project is None or self._result is None or self._result.pareto.empty:
            return

        self.validate_button.setEnabled(False)
        self.notify("Validating the front against the solver...")

        problem = self.project.problem()
        space = self.project.factors
        responses = self.project.responses
        front = self._result.pareto

        self._task = run_async(
            lambda: opt.validate_pareto(problem, space, front, responses),
            on_finished=self._on_validated,
            on_failed=self._on_validation_failed,
        )

    def _on_validated(self, validated: pd.DataFrame) -> None:
        self.validate_button.setEnabled(True)
        assert self.project is not None

        summary = opt.validation_summary(validated, self.project.responses)
        self.validation_view.set_frame(summary)
        self._draw_validation(validated)

        worst = summary["RMSE % of range"].max() if len(summary) else float("nan")
        if np.isfinite(worst):
            self.notify(f"Validated — worst response RMSE is {worst:.1f}% of its range")
            self._append_log(f"Validation: worst RMSE {worst:.2f}% of range")

    def _on_validation_failed(self, message: str) -> None:
        self.validate_button.setEnabled(True)
        QMessageBox.critical(self, "Validation failed", message.split("\n\n")[0])

    def _draw_validation(self, validated: pd.DataFrame) -> None:
        if self.project is None:
            return
        objectives, _ = opt.split_roles(self.project.responses)
        plotted = [r for r in objectives if f"{r.name}_actual" in validated]
        if not plotted:
            self.validation_plot.message("Nothing to compare.")
            return

        self.validation_plot.clear()
        figure = self.validation_plot.figure

        for index, response in enumerate(plotted, start=1):
            axes = figure.add_subplot(1, len(plotted), index)
            predicted = validated[f"{response.name}_predicted"].to_numpy(dtype=float)
            actual = validated[f"{response.name}_actual"].to_numpy(dtype=float)

            axes.scatter(predicted, actual, s=18, alpha=0.7,
                         color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
                         edgecolors="white", linewidths=0.4)
            low = float(min(predicted.min(), actual.min()))
            high = float(max(predicted.max(), actual.max()))
            axes.plot([low, high], [low, high], "--", color="#9aa0a6", linewidth=1)

            style_axes(axes, title=response.name,
                       x_label="surrogate", y_label="solver")

        self.validation_plot.draw()
