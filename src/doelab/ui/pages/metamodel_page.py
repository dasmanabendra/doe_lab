"""Stage 5 — fit surrogates and explore what they predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...engine import metamodel as mm
from ...engine.factors import ResponseRole
from ...engine.project import Stage
from ..widgets.controls import FactorControlPanel
from ..widgets.plots import MplPanel, SERIES_COLOURS, style_axes
from ..widgets.tables import DataFrameView
from ..workers import run_async
from .base import Card, FieldRow, Page

# Wide enough for the longest fit-type heading, narrow enough that the radio
# stays under it. The columns are fixed at this width so they do not reflow.
CHOICE_COLUMN_WIDTH = 96

# Room for a long response name, plus a gap before the first radio column.
NAME_COLUMN_WIDTH = 200


class MetamodelPage(Page):
    stage = Stage.METAMODELS
    title = "Metamodels"
    subtitle = (
        "Fit a fast surrogate to each response. The optimizer searches these, not the "
        "solver, so their cross-validated accuracy sets a ceiling on how much any "
        "optimum found later can be trusted."
    )

    def __init__(self):
        super().__init__()
        self._task = None
        self._fit_buttons: dict[str, QButtonGroup] = {}

        # Dragging a slider fires continuously, and a contour redraw evaluates
        # the surrogate over a few thousand grid points. Coalesce the bursts so
        # only the position the user settles on is actually rendered.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self._draw_explorer)

        self.body.addWidget(self._build_setup_card())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_quality_tab(), "Fit quality")
        self.tabs.addTab(self._build_explorer_tab(), "Prediction explorer")
        self.body.addWidget(self.tabs, stretch=1)

    # -- construction -------------------------------------------------------

    def _build_setup_card(self) -> Card:
        card = Card("Fit types")

        self.selection_container = QWidget()
        self.selection_layout = QGridLayout(self.selection_container)
        self.selection_layout.setContentsMargins(0, 0, 0, 0)
        self.selection_layout.setHorizontalSpacing(18)
        self.selection_layout.setVerticalSpacing(4)
        card.add(self.selection_container)

        row = FieldRow()
        self.warning_label = QLabel()
        self.warning_label.setObjectName("warningLabel")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        row.addWidget(self.warning_label)
        row.spacer()

        self.fit_button = QPushButton("Fit metamodels")
        self.fit_button.setObjectName("primary")
        self.fit_button.clicked.connect(self._fit)
        row.addWidget(self.fit_button)
        card.add_layout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        card.add(self.progress)
        return card

    def _build_quality_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        metrics_card = Card("Cross-validated metrics")
        self.metrics_view = DataFrameView(decimals=4)
        metrics_card.add(self.metrics_view, stretch=1)

        note = QLabel(
            "A high R² beside a poor CV R² means the surrogate has memorized the design "
            "rather than learned the response — any optimum found on it would be fictional."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        metrics_card.add(note)
        layout.addWidget(metrics_card, stretch=1)

        plot_card = Card("Cross-validated prediction vs actual")
        self.parity_plot = MplPanel(height=3.5)
        plot_card.add(self.parity_plot, stretch=1)
        layout.addWidget(plot_card, stretch=2)
        return page

    def _build_explorer_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 6, 10)
        control_layout.setSpacing(10)

        axis_card = Card("Plot")
        row1 = FieldRow()
        self.response_combo = QComboBox()
        self.response_combo.currentIndexChanged.connect(self._draw_explorer)
        row1.field("Response", self.response_combo, width=170)
        axis_card.add_layout(row1)

        row2 = FieldRow()
        self.x_combo = QComboBox()
        self.x_combo.currentIndexChanged.connect(self._on_axis_changed)
        row2.field("X axis", self.x_combo, width=170)
        axis_card.add_layout(row2)

        row3 = FieldRow()
        self.y_combo = QComboBox()
        self.y_combo.addItem("(none — 1-D sweep)", None)
        self.y_combo.currentIndexChanged.connect(self._on_axis_changed)
        row3.field("Y axis", self.y_combo, width=170)
        axis_card.add_layout(row3)
        control_layout.addWidget(axis_card)

        held_card = Card("Held at")
        self.control_panel = FactorControlPanel()
        self.control_panel.changed.connect(self._schedule_redraw)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.control_panel)
        held_card.add(scroll, stretch=1)

        reset_button = QPushButton("Reset to centre")
        reset_button.clicked.connect(self._reset_controls)
        held_card.add(reset_button)
        control_layout.addWidget(held_card, stretch=1)

        self.explorer_plot = MplPanel()

        splitter.addWidget(control_panel)
        splitter.addWidget(self.explorer_plot)
        splitter.setSizes([330, 870])
        return splitter

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None or not self.project.has_results:
            return

        self._build_selection_grid()
        self._populate_explorer_combos()

        if self.project.has_metamodels:
            self._show_fit_results()
        else:
            self.metrics_view.set_frame(pd.DataFrame())
            self.parity_plot.message("Fit metamodels to see how well they predict.")
            self.explorer_plot.message("Fit metamodels to explore their predictions.")
            self.warning_label.setVisible(False)

    def _build_selection_grid(self) -> None:
        """One row per response, with a fit type per row."""
        while self.selection_layout.count():
            item = self.selection_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Unparent first: deleteLater alone leaves the widget visible
                # until the event loop runs, overlapping its replacement.
                widget.setParent(None)
                widget.deleteLater()
        # The groups are parented to the page, not the grid, so clearing the
        # grid leaves them behind to accumulate on every rebuild.
        for group in self._fit_buttons.values():
            group.deleteLater()
        self._fit_buttons = {}

        assert self.project is not None
        header = QLabel("Response")
        header.setObjectName("cardTitle")
        self.selection_layout.addWidget(header, 0, 0)
        choices = [label for _, label in mm.FIT_TYPES] + ["Skip"]
        for column, label in enumerate(choices, start=1):
            title = QLabel(label)
            title.setObjectName("cardTitle")
            # Centred, because the radio below it is centred. Left-aligned, a
            # heading starts a long way from the control it names.
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.selection_layout.addWidget(title, 0, column)

        # Spare width goes to an empty trailing column. Left at their default
        # stretch of zero the columns share it equally instead, which parks
        # each radio in the middle of a cell hundreds of pixels wide -- nowhere
        # near the heading it belongs to, and sliding right across the card on
        # every resize. Pinning the choices next to the names also keeps the
        # eye-travel along a row short.
        self.selection_layout.setColumnStretch(0, 0)
        self.selection_layout.setColumnMinimumWidth(0, NAME_COLUMN_WIDTH)
        for column in range(1, len(choices) + 1):
            self.selection_layout.setColumnStretch(column, 0)
            self.selection_layout.setColumnMinimumWidth(column, CHOICE_COLUMN_WIDTH)
        self.selection_layout.setColumnStretch(len(choices) + 1, 1)

        existing = {spec.response: spec.fit_type for spec in self.project.metamodel_specs}

        for row, response in enumerate(self.project.responses, start=1):
            needed = response.role is not ResponseRole.IGNORED
            name_label = QLabel(response.name)
            if needed:
                name_label.setToolTip("Required by the optimizer for this role.")
            self.selection_layout.addWidget(name_label, row, 0)

            group = QButtonGroup(self)
            for column, (key, _) in enumerate(mm.FIT_TYPES, start=1):
                button = QRadioButton()
                button.setObjectName("matrixChoice")
                group.addButton(button, column - 1)
                self.selection_layout.addWidget(button, row, column, Qt.AlignmentFlag.AlignCenter)

            skip = QRadioButton()
            skip.setObjectName("matrixChoice")
            group.addButton(skip, len(mm.FIT_TYPES))
            self.selection_layout.addWidget(
                skip, row, len(mm.FIT_TYPES) + 1, Qt.AlignmentFlag.AlignCenter
            )

            chosen = existing.get(response.name)
            if chosen is not None:
                index = [k for k, _ in mm.FIT_TYPES].index(chosen)
                group.button(index).setChecked(True)
            elif needed:
                group.button(1).setChecked(True)  # quadratic default
            else:
                skip.setChecked(True)

            self._fit_buttons[response.name] = group

    def _selected_specs(self) -> list[mm.MetamodelSpec]:
        keys = [key for key, _ in mm.FIT_TYPES]
        specs = []
        for name, group in self._fit_buttons.items():
            index = group.checkedId()
            if 0 <= index < len(keys):
                specs.append(mm.MetamodelSpec(name, keys[index]))
        return specs

    def _populate_explorer_combos(self) -> None:
        assert self.project is not None
        names = self.project.factors.names

        for combo, include_none, default in (
            (self.x_combo, False, 0),
            (self.y_combo, True, 1),
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if include_none:
                combo.addItem("(none — 1-D sweep)", None)
            for name in names:
                combo.addItem(name, name)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else default)
            combo.blockSignals(False)

        self.control_panel.build(self.project.factors)

    def _schedule_redraw(self) -> None:
        """Restart the debounce window; the last change in a burst wins."""
        self._redraw_timer.start()

    def _reset_controls(self) -> None:
        if self.project is not None:
            self.control_panel.reset(self.project.factors)
            self._draw_explorer()

    def _on_axis_changed(self) -> None:
        axes = {self.x_combo.currentData(), self.y_combo.currentData()} - {None}
        self.control_panel.set_axis_factors(axes)
        self._draw_explorer()

    # -- fitting ------------------------------------------------------------

    def _fit(self) -> None:
        if self.project is None or not self.project.has_results:
            return
        specs = self._selected_specs()
        if not specs:
            QMessageBox.information(
                self, "Nothing to fit", "Choose a fit type for at least one response."
            )
            return

        self.fit_button.setEnabled(False)
        self.progress.setVisible(True)
        self.set_busy(True)
        self.notify(f"Fitting {len(specs)} metamodels...")

        project = self.project
        self._task = run_async(
            lambda: project.build_metamodels(specs),
            on_finished=self._on_fitted,
            on_failed=self._on_failed,
        )

    def _on_fitted(self, built: tuple) -> None:
        self.progress.setVisible(False)
        self.fit_button.setEnabled(True)
        self.set_busy(False)
        if self.project is None:
            return

        # Commit here, on the UI thread, rather than inside the worker.
        specs, models = built
        self.project.adopt_metamodels(specs, models)

        self.notify(f"Fitted {len(models)} metamodels")
        # Publish before rendering (see DesignPage).
        self.project_changed.emit()
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.fit_button.setEnabled(True)
        self.set_busy(False)
        self.notify("Fitting failed")
        QMessageBox.critical(self, "Could not fit metamodels", message.split("\n\n")[0])

    def _show_fit_results(self) -> None:
        assert self.project is not None
        models = self.project.metamodels

        table = mm.metrics_table({m.label: m for m in models.values()})
        self.metrics_view.set_frame(table)
        self._draw_parity(models)
        self._show_warnings(models)

        self.response_combo.blockSignals(True)
        current = self.response_combo.currentText()
        self.response_combo.clear()
        self.response_combo.addItems(sorted(models))
        index = self.response_combo.findText(current)
        self.response_combo.setCurrentIndex(max(0, index))
        self.response_combo.blockSignals(False)

        self._on_axis_changed()

    def _show_warnings(self, models: dict) -> None:
        """Warn about surrogates that are actually unfit for use.

        The kernel optimizer's own complaints are filtered out by the engine:
        they fire constantly against a deterministic solver and say nothing
        about accuracy, and a banner that appears on every fit is one the user
        learns to ignore. What matters is whether a surrogate generalizes,
        which cross-validation measures directly.
        """
        notes: list[str] = []

        weak = sorted(m.label for m in models.values() if m.is_weak)
        if weak:
            notes.append(
                f"Weak fit (cross-validated R² below {mm.WEAK_FIT_THRESHOLD:.2f}): "
                + ", ".join(weak)
                + ". Optimizing against these would chase the surrogate, not the response — "
                "try a richer fit type or more experiments."
            )

        unexpected = sorted({w for m in models.values() for w in m.notable_warnings})
        if unexpected:
            shown = "  ·  ".join(w.rstrip(".") for w in unexpected[:2])
            if len(unexpected) > 2:
                shown += f"  (+{len(unexpected) - 2} more)"
            notes.append("Fit warnings: " + shown)

        self.warning_label.setText("\n".join(notes))
        self.warning_label.setVisible(bool(notes))

    # -- plots --------------------------------------------------------------

    def _draw_parity(self, models: dict) -> None:
        if not models:
            self.parity_plot.message("No metamodels fitted.")
            return
        self.parity_plot.clear()
        figure = self.parity_plot.figure

        count = len(models)
        columns = min(count, 3)
        rows = (count + columns - 1) // columns

        for index, (name, model) in enumerate(sorted(models.items()), start=1):
            axes = figure.add_subplot(rows, columns, index)
            actual = model.actual
            predicted = model.cv_predicted

            if not np.isfinite(predicted).all():
                axes.text(0.5, 0.5, "too few runs\nto cross-validate",
                          ha="center", va="center", transform=axes.transAxes,
                          fontsize=8, color="#6b7280")
                axes.set_axis_off()
                continue

            axes.scatter(actual, predicted, s=14, alpha=0.65,
                         color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
                         edgecolors="white", linewidths=0.4)
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            axes.plot([low, high], [low, high], "--", color="#9aa0a6", linewidth=1)

            style_axes(axes, title=f"{name}  (CV R² {model.metrics.cv_r2:.4f})",
                       x_label="actual", y_label="predicted")

        self.parity_plot.draw()

    def _draw_explorer(self) -> None:
        if self.project is None or not self.project.has_metamodels:
            return
        response = self.response_combo.currentText()
        if not response or response not in self.project.metamodels:
            return

        model = self.project.metamodels[response]
        space = self.project.factors
        x_name = self.x_combo.currentData()
        y_name = self.y_combo.currentData()
        if x_name is None:
            return

        fixed = self.control_panel.values()
        self.explorer_plot.clear()
        figure = self.explorer_plot.figure
        axes = figure.add_subplot(111)

        unit = next((r.unit for r in self.project.responses if r.name == response), "")
        label = f"{response} [{unit}]" if unit else response

        if y_name is None or y_name == x_name:
            self._draw_sweep(axes, model, space, x_name, fixed, label)
        else:
            self._draw_contour(figure, axes, model, space, x_name, y_name, fixed, label)

        self.explorer_plot.draw()

    def _draw_sweep(self, axes, model, space, x_name, fixed, label) -> None:
        xs, values = mm.predict_sweep(model, space, x_name, fixed, resolution=120)
        factor = space[x_name]

        if factor.is_categorical:
            positions = np.arange(len(xs))
            axes.bar(positions, values, color=SERIES_COLOURS[0], width=0.55)
            axes.set_xticks(positions)
            axes.set_xticklabels([str(v) for v in xs], fontsize=8)
        else:
            axes.plot(xs, values, color=SERIES_COLOURS[0], linewidth=2)

        style_axes(axes, title=f"{label} vs {x_name}",
                   x_label=self._factor_label(factor), y_label=label)

    def _draw_contour(self, figure, axes, model, space, x_name, y_name, fixed, label) -> None:
        xs, ys, z = mm.predict_grid(model, space, x_name, y_name, fixed, resolution=60)
        x_factor = space[x_name]
        y_factor = space[y_name]

        x_positions = np.arange(len(xs)) if x_factor.is_categorical else xs.astype(float)
        y_positions = np.arange(len(ys)) if y_factor.is_categorical else ys.astype(float)

        filled = axes.contourf(x_positions, y_positions, z, levels=18, cmap="viridis")
        lines = axes.contour(x_positions, y_positions, z, levels=8,
                             colors="white", linewidths=0.5, alpha=0.6)
        axes.clabel(lines, inline=True, fontsize=7, fmt="%.3g")

        bar = figure.colorbar(filled, ax=axes)
        bar.set_label(label, fontsize=9)
        bar.ax.tick_params(labelsize=8)

        if x_factor.is_categorical:
            axes.set_xticks(x_positions)
            axes.set_xticklabels([str(v) for v in xs], fontsize=8)
        if y_factor.is_categorical:
            axes.set_yticks(y_positions)
            axes.set_yticklabels([str(v) for v in ys], fontsize=8)

        style_axes(axes, title=f"{label}",
                   x_label=self._factor_label(x_factor),
                   y_label=self._factor_label(y_factor))

    @staticmethod
    def _factor_label(factor) -> str:
        return f"{factor.name} [{factor.unit}]" if factor.unit else factor.name
