"""Stage 2 — generate a design and inspect how it covers the space."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from ...engine import doe
from ...engine.project import Stage
from ..widgets.plots import MplPanel, SERIES_COLOURS, style_axes
from ..widgets.tables import DataFrameView
from ..workers import run_async
from .base import Card, FieldRow, Page


class DesignPage(Page):
    stage = Stage.DESIGN
    title = "Design"
    subtitle = (
        "Choose how to sample the factor space. The scatter shows where the runs "
        "actually land, which is the quickest way to see what a design type buys you."
    )

    def __init__(self):
        super().__init__()
        self._task = None
        self._loading = False

        self.body.addWidget(self._build_controls())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_matrix_card())
        splitter.addWidget(self._build_plot_card())
        splitter.setSizes([560, 640])
        self.body.addWidget(splitter, stretch=1)

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> Card:
        card = Card("Design setup")

        row = FieldRow()
        self.kind_combo = QComboBox()
        for key, label in doe.DESIGN_KINDS:
            self.kind_combo.addItem(label, key)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        row.field("Type", self.kind_combo, width=180)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, 20_000)
        self.n_spin.setValue(60)
        row.field("Experiments", self.n_spin, width=110)

        self.order_combo = QComboBox()
        self.order_combo.addItem("Linear", "linear")
        self.order_combo.addItem("Quadratic", "quadratic")
        self.order_combo.setCurrentIndex(1)
        row.field("Model order", self.order_combo, width=130)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10_000)
        row.field("Seed", self.seed_spin, width=90)

        row.spacer()

        self.generate_button = QPushButton("Generate design")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        row.addWidget(self.generate_button)
        card.add_layout(row)

        self.hint_label = QLabel()
        self.hint_label.setObjectName("pageSubtitle")
        self.hint_label.setWordWrap(True)
        card.add(self.hint_label)
        return card

    def _build_matrix_card(self) -> Card:
        card = Card("Experiment matrix")
        self.summary_label = QLabel("No design generated yet.")
        self.summary_label.setObjectName("pageSubtitle")
        card.add(self.summary_label)

        self.matrix_view = DataFrameView(decimals=3)
        card.add(self.matrix_view, stretch=1)
        return card

    def _build_plot_card(self) -> Card:
        card = Card("Design space coverage")

        row = FieldRow()
        self.x_combo = QComboBox()
        self.x_combo.currentIndexChanged.connect(self._draw_plot)
        row.field("X", self.x_combo, width=150)
        self.y_combo = QComboBox()
        self.y_combo.currentIndexChanged.connect(self._draw_plot)
        row.field("Y", self.y_combo, width=150)
        self.colour_combo = QComboBox()
        self.colour_combo.currentIndexChanged.connect(self._draw_plot)
        row.field("Colour by", self.colour_combo, width=150)
        row.spacer()
        card.add_layout(row)

        self.plot = MplPanel()
        card.add(self.plot, stretch=1)
        return card

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None:
            return
        self._loading = True
        try:
            spec = self.project.design_spec
            index = self.kind_combo.findData(spec.kind)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
            self.n_spin.setValue(spec.n_experiments)
            order_index = self.order_combo.findData(spec.model_order)
            if order_index >= 0:
                self.order_combo.setCurrentIndex(order_index)
            self.seed_spin.setValue(spec.seed or 0)

            self._sync_controls()
            self._populate_axis_combos()
        finally:
            self._loading = False

        if self.project.has_design:
            self._show_design(self.project.design)
        else:
            self.matrix_view.set_frame(pd.DataFrame())
            self.summary_label.setText("No design generated yet.")
            self.plot.message("Generate a design to see how it covers the space.")

    def _sync_controls(self) -> None:
        """Only show the inputs the selected design type actually uses."""
        kind = self.kind_combo.currentData()
        is_factorial = kind == "full_factorial"
        is_d_optimal = kind == "d_optimal"

        self.n_spin.setEnabled(not is_factorial)
        self.order_combo.setEnabled(is_d_optimal)
        self.seed_spin.setEnabled(not is_factorial)

        if self.project is None or self.project.factors is None:
            return

        if is_factorial:
            size = doe.full_factorial_size(self.project.factors)
            self.hint_label.setText(
                f"Every combination of every level: {size} runs. "
                "Set level counts per factor on the Problem page — the total is their product, "
                "so it grows quickly."
            )
        elif is_d_optimal:
            self.hint_label.setText(
                "Picks the runs that maximize information for the chosen model form. "
                "The design to use when the run budget is fixed and factors are mixed."
            )
        else:
            self.hint_label.setText(
                "Space-filling and stratified in every factor at once, including "
                "categorical ones. Scales to many factors where a factorial cannot."
            )

    def _populate_axis_combos(self) -> None:
        if self.project is None or self.project.factors is None:
            return
        names = self.project.factors.names
        for combo, default in (
            (self.x_combo, 0),
            (self.y_combo, 1 if len(names) > 1 else 0),
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(names.index(current) if current in names else default)
            combo.blockSignals(False)

        current = self.colour_combo.currentText()
        self.colour_combo.blockSignals(True)
        self.colour_combo.clear()
        self.colour_combo.addItem("None")
        self.colour_combo.addItems(self.project.factors.categorical_names)
        if current:
            index = self.colour_combo.findText(current)
            self.colour_combo.setCurrentIndex(max(0, index))
        elif self.project.factors.categorical_names:
            self.colour_combo.setCurrentIndex(1)
        self.colour_combo.blockSignals(False)

    def _on_kind_changed(self) -> None:
        self._sync_controls()

    # -- generation ---------------------------------------------------------

    def _generate(self) -> None:
        if self.project is None or self.project.factors is None:
            return

        spec = doe.DesignSpec(
            kind=self.kind_combo.currentData(),
            n_experiments=self.n_spin.value(),
            model_order=self.order_combo.currentData(),
            seed=self.seed_spin.value(),
        )

        if spec.kind == "full_factorial":
            size = doe.full_factorial_size(self.project.factors)
            if size > 5000:
                confirm = QMessageBox.question(
                    self,
                    "Large design",
                    f"A full factorial over these factors is {size} runs. Generate it anyway?",
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return

        self.generate_button.setEnabled(False)
        self.set_busy(True)
        self.notify("Generating design...")

        space = self.project.factors
        self._task = run_async(
            lambda: doe.generate(space, spec),
            on_finished=lambda design: self._on_generated(design, spec),
            on_failed=self._on_failed,
        )

    def _on_generated(self, design: pd.DataFrame, spec: doe.DesignSpec) -> None:
        self.generate_button.setEnabled(True)
        self.set_busy(False)
        if self.project is None:
            return

        self.project.design_spec = spec
        self.project.design = design
        # A new design means the previous results describe runs that no longer exist.
        self.project.invalidate_from(Stage.RESULTS)

        self.notify(f"Generated {len(design)} experiments")
        # Publish before rendering. The project state is already committed, so
        # the navigation rail must learn about it even if drawing the scatter
        # then fails.
        self.project_changed.emit()
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self.generate_button.setEnabled(True)
        self.set_busy(False)
        self.notify("Design generation failed")
        QMessageBox.critical(self, "Could not generate design", message.split("\n\n")[0])

    def _show_design(self, design: pd.DataFrame) -> None:
        self.matrix_view.set_frame(design)

        parts = [f"{len(design)} experiments"]
        if self.project is not None and self.project.factors is not None:
            try:
                efficiency = doe.d_efficiency(
                    self.project.factors, design, self.project.design_spec.model_order
                )
                parts.append(f"D-efficiency {efficiency:.4f}")
            except Exception:
                # Efficiency is informational; a rank-deficient design should
                # not stop the matrix from being shown.
                pass
        self.summary_label.setText("   ·   ".join(parts))
        self._draw_plot()

    # -- plotting -----------------------------------------------------------

    def _draw_plot(self) -> None:
        if self._loading or self.project is None or not self.project.has_design:
            return
        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        if not x_name or not y_name:
            return

        design = self.project.design
        space = self.project.factors
        self.plot.clear()
        axes = self.plot.figure.add_subplot(111)

        colour_by = self.colour_combo.currentText()
        groups = (
            [(label, design[design[colour_by] == label]) for label in sorted(design[colour_by].unique())]
            if colour_by and colour_by != "None" and colour_by in design
            else [(None, design)]
        )

        for index, (label, subset) in enumerate(groups):
            axes.scatter(
                self._axis_values(subset, x_name, space),
                self._axis_values(subset, y_name, space),
                s=32,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.6,
                color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
                label=str(label) if label is not None else None,
            )

        self._apply_categorical_ticks(axes, design, x_name, space, "x")
        self._apply_categorical_ticks(axes, design, y_name, space, "y")

        style_axes(
            axes,
            title=f"{len(design)} experiments",
            x_label=self._axis_label(x_name, space),
            y_label=self._axis_label(y_name, space),
        )
        if groups[0][0] is not None:
            axes.legend(fontsize=8, frameon=False, title=colour_by, title_fontsize=8)
        self.plot.draw()

    @staticmethod
    def _axis_values(subset: pd.DataFrame, name: str, space) -> np.ndarray:
        factor = space[name]
        if factor.is_categorical:
            # Jitter category indices so overlapping points stay countable.
            lookup = {c: i for i, c in enumerate(factor.categories)}
            base = subset[name].map(lookup).to_numpy(dtype=float)
            rng = np.random.default_rng(0)
            return base + rng.uniform(-0.12, 0.12, size=len(base))
        return subset[name].to_numpy(dtype=float)

    @staticmethod
    def _apply_categorical_ticks(axes, design, name: str, space, axis: str) -> None:
        factor = space[name]
        if not factor.is_categorical:
            return
        positions = list(range(len(factor.categories)))
        if axis == "x":
            axes.set_xticks(positions)
            axes.set_xticklabels(factor.categories, fontsize=8)
        else:
            axes.set_yticks(positions)
            axes.set_yticklabels(factor.categories, fontsize=8)

    @staticmethod
    def _axis_label(name: str, space) -> str:
        factor = space[name]
        return f"{name} [{factor.unit}]" if factor.unit else name
