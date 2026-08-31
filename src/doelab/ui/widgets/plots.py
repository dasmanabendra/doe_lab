"""Plotting surfaces.

Two libraries, for two different jobs:

* **matplotlib** for static analysis plots — contours, scatter matrices,
  predicted-versus-actual. Its contouring and colorbars are far ahead of the
  alternatives, and these plots are drawn once per user action.
* **pyqtgraph** for the optimization dashboard, which redraws every generation
  while the run is in flight. matplotlib redraws are too slow to keep that
  smooth.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# A qualitative palette used for series that have no inherent ordering.
SERIES_COLOURS = [
    "#3b6fb6", "#c9522c", "#3f8f5b", "#8659a8",
    "#c99b2c", "#4aa3a8", "#a8536f", "#6b7280",
]


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas sized to fill its container.

    Redraws here are **synchronous**. ``draw_idle`` defers the render to the
    event loop, which opens a window in which the figure can be cleared and
    repopulated before the queued draw runs. The constrained-layout engine then
    walks artists that ``clear()`` has already detached — they have no axes left
    — and raises deep inside the layout pass. Every plot in this app is redrawn
    in response to a discrete user action rather than continuously, so drawing
    immediately costs nothing and removes the race entirely.
    """

    def __init__(self, width: float = 5.0, height: float = 4.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def reset_figure(self) -> None:
        """Clear the figure and give it a fresh layout engine.

        Re-seating the engine drops any layout state cached against the axes
        that were just removed.
        """
        self.figure.clear()
        self.figure.set_layout_engine("constrained")

    def clear(self) -> None:
        self.reset_figure()

    def message(self, text: str) -> None:
        """Replace the plot with a centred note.

        Used for the empty and not-yet-computed states, so a blank panel never
        leaves the user wondering whether something failed.
        """
        self.reset_figure()
        axes = self.figure.add_subplot(111)
        axes.text(
            0.5, 0.5, text,
            ha="center", va="center", transform=axes.transAxes,
            fontsize=10, color="#6b7280", wrap=True,
        )
        axes.set_axis_off()
        self.draw()


class MplPanel(QWidget):
    """A canvas in a layout, for dropping straight into a page."""

    def __init__(self, width: float = 5.0, height: float = 4.0):
        super().__init__()
        self.canvas = MplCanvas(width, height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    @property
    def figure(self) -> Figure:
        return self.canvas.figure

    def clear(self) -> None:
        """Start a fresh figure. Call this before rebuilding a plot."""
        self.canvas.reset_figure()

    def draw(self) -> None:
        self.canvas.draw()

    def message(self, text: str) -> None:
        self.canvas.message(text)


class LivePlot(pg.PlotWidget):
    """A pyqtgraph plot for data that updates while a run is in flight."""

    def __init__(self, title: str = "", x_label: str = "", y_label: str = ""):
        super().__init__(title=title)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", x_label)
        self.setLabel("left", y_label)
        # Bottom-right: a front that minimizes one objective while maximizing
        # another hugs the opposite corner, so this is the region least likely
        # to have points under it.
        self.addLegend(offset=(-10, -10))


def scatter_item(colour: str, size: int = 7, symbol: str = "o") -> pg.ScatterPlotItem:
    """A scatter series in one of the palette colours."""
    brush = pg.mkBrush(colour)
    return pg.ScatterPlotItem(size=size, symbol=symbol, brush=brush, pen=pg.mkPen(None))


def style_axes(axes, title: str = "", x_label: str = "", y_label: str = "") -> None:
    """Consistent matplotlib axis styling across the app."""
    if title:
        axes.set_title(title, fontsize=10, pad=8)
    if x_label:
        axes.set_xlabel(x_label, fontsize=9)
    if y_label:
        axes.set_ylabel(y_label, fontsize=9)
    axes.tick_params(labelsize=8)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def categorical_ticks(axes, values: np.ndarray, axis: str = "x") -> None:
    """Label an axis with category names instead of indices."""
    positions = np.arange(len(values))
    labels = [str(v) for v in values]
    if axis == "x":
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=8)
    else:
        axes.set_yticks(positions)
        axes.set_yticklabels(labels, fontsize=8)
