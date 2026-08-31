"""Per-factor input controls for exploring a fitted surface.

A factor's control follows its type: continuous factors get a slider paired
with a spin box, categorical factors get a combo box. Forcing a category onto a
slider would imply an ordering and a midpoint that a nominal factor does not
have.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSlider,
    QWidget,
)

from ...engine.factors import Factor, FactorSpace

_SLIDER_STEPS = 1000


class FactorControl(QWidget):
    """One factor's input, typed to the factor."""

    changed = Signal()

    def __init__(self, factor: Factor):
        super().__init__()
        self.factor = factor
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        label = QLabel(f"{factor.name} [{factor.unit}]" if factor.unit else factor.name)
        label.setObjectName("fieldLabel")
        layout.addWidget(label, 0, 0, 1, 2)

        if factor.is_categorical:
            self.combo = QComboBox()
            self.combo.addItems(factor.categories)
            self.combo.currentIndexChanged.connect(self._emit)
            layout.addWidget(self.combo, 1, 0, 1, 2)
            self.slider = None
            self.spin = None
        else:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, _SLIDER_STEPS)
            self.slider.setValue(_SLIDER_STEPS // 2)
            self.slider.valueChanged.connect(self._on_slider)

            self.spin = QDoubleSpinBox()
            self.spin.setRange(factor.low, factor.high)
            self.spin.setDecimals(3)
            self.spin.setSingleStep((factor.high - factor.low) / 100)
            self.spin.setValue((factor.low + factor.high) / 2)
            self.spin.valueChanged.connect(self._on_spin)

            layout.addWidget(self.slider, 1, 0)
            layout.addWidget(self.spin, 1, 1)
            layout.setColumnStretch(0, 1)
            self.spin.setFixedWidth(96)
            self.combo = None

    # -- value ---------------------------------------------------------------

    def value(self) -> Any:
        if self.factor.is_categorical:
            return self.combo.currentText()
        return float(self.spin.value())

    def set_value(self, value: Any) -> None:
        self._updating = True
        try:
            if self.factor.is_categorical:
                self.combo.setCurrentText(str(value))
            else:
                self.spin.setValue(float(value))
                self.slider.setValue(self._to_slider(float(value)))
        finally:
            self._updating = False

    def set_enabled(self, enabled: bool) -> None:
        """Disabled when this factor is one of the plotted axes."""
        for widget in (self.slider, self.spin, self.combo):
            if widget is not None:
                widget.setEnabled(enabled)

    # -- internals -----------------------------------------------------------

    def _to_slider(self, value: float) -> int:
        span = self.factor.high - self.factor.low
        return int(round((value - self.factor.low) / span * _SLIDER_STEPS))

    def _from_slider(self, position: int) -> float:
        span = self.factor.high - self.factor.low
        return self.factor.low + position / _SLIDER_STEPS * span

    def _on_slider(self, position: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.spin.setValue(self._from_slider(position))
        finally:
            self._updating = False
        self.changed.emit()

    def _on_spin(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.slider.setValue(self._to_slider(value))
        finally:
            self._updating = False
        self.changed.emit()

    def _emit(self) -> None:
        if not self._updating:
            self.changed.emit()


class FactorControlPanel(QWidget):
    """A stack of factor controls representing one point in the space."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setVerticalSpacing(6)
        self.controls: dict[str, FactorControl] = {}

    def build(self, space: FactorSpace) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Unparent before scheduling deletion. deleteLater defers to the
                # event loop, so a widget that is merely removed from the layout
                # stays visible and paints on top of its replacement until then.
                widget.setParent(None)
                widget.deleteLater()
        self.controls = {}

        for row, factor in enumerate(space):
            control = FactorControl(factor)
            control.changed.connect(self.changed)
            self._layout.addWidget(control, row, 0)
            self.controls[factor.name] = control

        self.reset(space)

    def reset(self, space: FactorSpace) -> None:
        for name, value in space.midpoint().items():
            if name in self.controls:
                self.controls[name].set_value(value)

    def values(self) -> dict[str, Any]:
        return {name: control.value() for name, control in self.controls.items()}

    def set_axis_factors(self, names: set[str]) -> None:
        """Grey out factors currently used as plot axes."""
        for name, control in self.controls.items():
            control.set_enabled(name not in names)
