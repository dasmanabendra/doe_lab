"""Parallel coordinates: every design, every variable, one picture.

The plot gives each variable its own vertical axis and draws every design as a
polyline crossing them all. Where the analysis tables collapse the study to one
number per factor-response pair, this keeps the individual runs -- which is
what makes the runs that disagree with a trend findable.

Two implementation choices carry most of the weight.

**One item per group, not per design.** All the polylines in a group are drawn
as a single :class:`pyqtgraph.PlotDataItem` using its ``connect`` array, with a
break marked after each design's last point. Giving every design its own item
is the usual way this plot ends up unusable: brushing redraws on each mouse
move, and the per-item overhead makes that cost grow with the design count.
Here a redraw is three ``setData`` calls no matter how many designs there are.

**Filtering fades rather than hides.** A design outside the brushed bands stays
on the plot in pale grey. The excluded designs are the context that gives the
surviving ones meaning -- "these twenty" says little without the hundred they
were chosen from -- and it keeps widening a band an obviously reversible act
rather than something that looks like it recovers deleted data.

pyqtgraph rather than matplotlib, following the split set out in
:mod:`doelab.ui.widgets.plots`: this redraws continuously while a handle is
dragged, which is the live path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...engine import parallel
from ...engine.parallel import Axis
from ..theme import ACCENT, MUTED, TEXT

MUTED_LINE = (170, 176, 186, 90)
# Semi-transparent, because a hundred opaque lines over nine axes is a solid
# block. Overlap density then reads as tone: where many designs follow the same
# path the bundle darkens, which is most of what the plot has to say.
PASSING_LINE = (47, 111, 176, 150)
VIOLATING_LINE = (201, 82, 44, 165)
VIOLATING = "#c9522c"

# Categorical levels are single points, so without a nudge every design sharing
# a level lands on the same pixel and the bundle reads as one line.
JITTER = 0.018

# How close, in axis heights, the cursor must come before a line is picked up.
HOVER_REACH = 0.02

_TOP = 1.0
_BOTTOM = 0.0
_NAME_Y = 1.10
_GROUP_Y = 1.28
_GROUP_RULE_Y = 1.20

# Below this the group band is dropped: three rows of labels above the axes
# will not fit legibly in a short splitter pane, and the parameters/responses
# split is the one of them the axis order already implies.
_GROUP_BAND_MIN_HEIGHT = 210


class AxisBrush(pg.GraphicsObject):
    """The draggable lower and upper bound on one axis.

    Drawn as an item rather than assembled from :class:`pyqtgraph.LinearRegionItem`
    because a region item spans the entire plot width. Confining one to a single
    axis would mean giving each axis its own ``ViewBox``, and the polylines have
    to cross those boundaries -- they are the whole point of the plot.
    """

    changed = Signal()
    menu_requested = Signal(object)  # screen position

    def __init__(self, x: float):
        super().__init__()
        self._x = x
        self._low = 0.0
        self._high = 1.0
        self._grabbed: str | None = None
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setZValue(20)

    # -- geometry -----------------------------------------------------------

    def boundingRect(self):
        # Constant, spanning the axis column: the handles are drawn at a pixel
        # size that varies with the widget, but a rect that changed with it
        # would need a geometry update on every resize for no benefit.
        return pg.QtCore.QRectF(self._x - 0.35, -0.05, 0.7, 1.10)

    @property
    def band(self) -> tuple[float, float]:
        return (self._low, self._high)

    @property
    def is_narrowed(self) -> bool:
        return self._low > parallel.TOLERANCE or self._high < 1.0 - parallel.TOLERANCE

    def set_band(self, low: float, high: float) -> None:
        self._low, self._high = float(low), float(high)
        self.update()

    def reset(self) -> None:
        self.set_band(0.0, 1.0)

    # -- painting -----------------------------------------------------------

    def paint(self, painter, *_args) -> None:
        vx = abs(self.pixelLength(pg.Point(1, 0)) or 0.02)
        vy = abs(self.pixelLength(pg.Point(0, 1)) or 0.005)

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(QColor(0, 0, 0, 30)))
        half = 7.0 * vx
        if self._low > parallel.TOLERANCE:
            painter.drawRect(pg.QtCore.QRectF(self._x - half, _BOTTOM, 2 * half, self._low))
        if self._high < 1.0 - parallel.TOLERANCE:
            painter.drawRect(
                pg.QtCore.QRectF(self._x - half, self._high, 2 * half, _TOP - self._high)
            )

        active = self.is_narrowed
        painter.setPen(QPen(QColor(ACCENT if active else "#9aa1ab"), 0))
        painter.setBrush(QBrush(QColor(ACCENT if active else "#e4e7ec")))
        grip_w, grip_h = 8.0 * vx, 3.5 * vy
        for position in (self._low, self._high):
            painter.drawRoundedRect(
                pg.QtCore.QRectF(
                    self._x - grip_w, position - grip_h, 2 * grip_w, 2 * grip_h
                ),
                grip_h,
                grip_h,
            )

    # -- interaction --------------------------------------------------------

    def mouseDragEvent(self, ev) -> None:
        if ev.button() is not Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()

        if ev.isStart():
            grabbed_at = ev.buttonDownPos().y()
            self._grabbed = (
                "low" if abs(grabbed_at - self._low) <= abs(grabbed_at - self._high) else "high"
            )

        target = float(np.clip(ev.pos().y(), 0.0, 1.0))
        if self._grabbed == "low":
            self._low = min(target, self._high)
        else:
            self._high = max(target, self._low)

        self.update()
        self.changed.emit()

        if ev.isFinish():
            self._grabbed = None

    def mouseClickEvent(self, ev) -> None:
        if ev.button() is Qt.MouseButton.RightButton:
            ev.accept()
            self.menu_requested.emit(ev.screenPos())
        else:
            ev.ignore()


class ParallelCoordinatesPlot(QWidget):
    """A brushable parallel coordinates view over a design table."""

    design_hovered = Signal(int)  # row index, or -1 when nothing is under the cursor

    def __init__(self):
        super().__init__()
        self._frame = pd.DataFrame()
        self._axes: list[Axis] = []
        self._order: list[int] = []
        self._values = np.empty((0, 0))
        self._display = np.empty((0, 0))
        self._feasible = np.ones(0, dtype=bool)
        self._brushes: list[AxisBrush] = []
        self._decorations: list[pg.GraphicsObject] = []
        self._hovered = -1
        self._with_group_band = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.count_label = QLabel()
        self.count_label.setObjectName("cardTitle")
        header.addWidget(self.count_label)
        self.hint_label = QLabel()
        self.hint_label.setObjectName("pageSubtitle")
        header.addWidget(self.hint_label)
        header.addStretch(1)
        self.reset_button = QPushButton("Reset filters")
        self.reset_button.clicked.connect(self.reset_filters)
        self.reset_button.setEnabled(False)
        header.addWidget(self.reset_button)
        layout.addLayout(header)

        self._plot = pg.PlotWidget()
        # Stated here rather than relying on the global set in ``plots.py``.
        # That call runs as an import side effect, so a page that reaches this
        # widget without touching that module gets a black plot.
        self._plot.setBackground("w")
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        view = self._plot.getPlotItem().getViewBox()
        view.setMouseEnabled(x=False, y=False)
        view.setMenuEnabled(False)
        view.setDefaultPadding(0.0)
        # A floor, but a low one. A splitter can overrule a minimum when the
        # window has no space left to give, so the layout has to stay legible
        # by needing little rather than by demanding a lot -- which is what the
        # group band dropping out and the inline end labels are for.
        self._plot.setMinimumHeight(150)
        layout.addWidget(self._plot, stretch=1)

        self._muted = self._line_item(pg.mkPen(MUTED_LINE, width=1), z=1)
        self._passing = self._line_item(pg.mkPen(PASSING_LINE, width=1.3), z=3)
        self._violating = self._line_item(pg.mkPen(VIOLATING_LINE, width=1.3), z=4)
        self._highlight = self._line_item(pg.mkPen(TEXT, width=2.4), z=6)
        self._spines = self._line_item(pg.mkPen("#9aa1ab", width=1), z=2)
        self._rails = self._line_item(
            pg.mkPen(VIOLATING, width=1.6, style=Qt.PenStyle.DashLine), z=5
        )

        self._readout = pg.TextItem(color=TEXT, anchor=(0, 0), fill=(255, 255, 255, 225))
        self._readout.setFont(QFont("Segoe UI", 8))
        self._readout.setZValue(30)
        self._readout.setVisible(False)
        view.addItem(self._readout, ignoreBounds=True)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._show_message("Run the experiments to explore the designs.")

    def _line_item(self, pen, z: int) -> pg.PlotDataItem:
        item = pg.PlotDataItem(pen=pen, antialias=True)
        item.setZValue(z)
        self._plot.addItem(item)
        return item

    # -- data ---------------------------------------------------------------

    def set_data(
        self, frame: pd.DataFrame, axes: list[Axis], feasible: np.ndarray | None = None
    ) -> None:
        """Show a table of designs against a prepared set of axes."""
        self._frame = frame.reset_index(drop=True)
        self._axes = list(axes)
        self._order = list(range(len(axes)))
        self._hovered = -1

        if len(axes) < 2 or self._frame.empty:
            self._values = np.empty((len(self._frame), len(axes)))
            self._display = self._values
            self._clear_axis_furniture()
            self._show_message(
                "Two or more variables are needed to draw parallel coordinates."
                if not self._frame.empty
                else "Run the experiments to explore the designs."
            )
            return

        self._values = parallel.normalize(self._frame, self._axes)
        self._display = self._jittered(self._values)
        self._feasible = (
            np.ones(len(self._frame), dtype=bool) if feasible is None
            else np.asarray(feasible, dtype=bool)
        )

        self._build_axis_furniture()
        self._redraw()

    def clear(self) -> None:
        self.set_data(pd.DataFrame(), [])

    def resizeEvent(self, event) -> None:
        """Rebuild only when the plot crosses the height the group band needs.

        Rebuilding on every resize step would throw away the user's brushed
        bands mid-drag of a splitter; the verdict flipping is the only thing
        that actually changes what has to be drawn.
        """
        super().resizeEvent(event)
        if not self._brushes:
            return
        if (self._plot.height() >= _GROUP_BAND_MIN_HEIGHT) != self._with_group_band:
            self._reorder(self._order)

    def _jittered(self, values: np.ndarray) -> np.ndarray:
        """Spread the lines sharing a categorical level far enough to count them.

        Applied to a copy used only for drawing. Filtering reads the exact
        values, so a band snapped to a category still admits every design on
        that level rather than whichever ones happened to be nudged inside it.
        """
        display = values.copy()
        categorical = [i for i, axis in enumerate(self._axes) if axis.is_categorical]
        if not categorical:
            return display
        # Seeded, so the same study draws the same way every time it is opened.
        noise = np.random.default_rng(0).uniform(-JITTER, JITTER, size=values.shape)
        for index in categorical:
            display[:, index] = np.clip(display[:, index] + noise[:, index], 0.0, 1.0)
        return display

    # -- axis furniture -----------------------------------------------------

    def _clear_axis_furniture(self) -> None:
        view = self._plot.getPlotItem().getViewBox()
        for item in self._decorations:
            view.removeItem(item)
        self._decorations = []
        for brush in self._brushes:
            view.removeItem(brush)
        self._brushes = []
        for item in (
            self._muted, self._passing, self._violating,
            self._highlight, self._spines, self._rails,
        ):
            item.setData([], [])
        self._readout.setVisible(False)

    def _build_axis_furniture(self) -> None:
        """Axis lines, labels, group bands, constraint rails and brush handles."""
        self._clear_axis_furniture()
        view = self._plot.getPlotItem().getViewBox()
        shown = [self._axes[i] for i in self._order]
        count = len(shown)

        self._with_group_band = self._plot.height() >= _GROUP_BAND_MIN_HEIGHT
        top = (_GROUP_Y if self._with_group_band else _NAME_Y) + 0.06
        view.setRange(xRange=(-0.6, count - 0.4), yRange=(-0.05, top), padding=0)

        xs, ys, connect = [], [], []
        for column in range(count):
            xs += [column, column]
            ys += [_BOTTOM, _TOP]
            connect += [1, 0]
        self._spines.setData(
            np.array(xs, dtype=float), np.array(ys, dtype=float),
            connect=np.array(connect, dtype=np.uint8),
        )

        for column, axis in enumerate(shown):
            arrow = {"min": "  ↓", "max": "  ↑"}.get(axis.direction or "", "")
            self._text(f"{axis.name}{arrow}", column, _NAME_Y, TEXT, bold=True)
            if axis.is_categorical:
                for position, label in axis.ticks():
                    self._text(
                        label, column + 0.14, position, MUTED,
                        anchor=(0, 0.5), over_lines=True,
                    )
            else:
                # Tucked inside the ends rather than given rows of their own.
                # Two more label rows is height the axes need more, and the
                # extremes of an axis are where the fewest lines run.
                low, high = axis.ticks()
                self._text(high[1], column + 0.05, _TOP, MUTED, anchor=(0, 0), over_lines=True)
                self._text(low[1], column + 0.05, _BOTTOM, MUTED, anchor=(0, 1), over_lines=True)

            brush = AxisBrush(float(column))
            brush.changed.connect(self._redraw)
            brush.menu_requested.connect(
                lambda position, index=column: self._show_axis_menu(index, position)
            )
            view.addItem(brush, ignoreBounds=True)
            self._brushes.append(brush)

        self._build_group_bands(shown)
        self._build_rails(shown)

    def _build_group_bands(self, shown: list[Axis]) -> None:
        """Label the factor and response runs, and rule a line under each.

        With nine axes an unlabelled strip gives no clue where the variables
        you set end and the ones you measured begin -- the distinction the rest
        of the workflow is built around. Dropped entirely when the plot is too
        short to carry it; see :data:`_GROUP_BAND_MIN_HEIGHT`.
        """
        if not self._with_group_band:
            return

        xs, ys, connect = [], [], []
        start = 0
        while start < len(shown):
            stop = start
            while stop + 1 < len(shown) and shown[stop + 1].group == shown[start].group:
                stop += 1
            label = "Parameters" if shown[start].group == "factor" else "Responses"
            self._text(label, (start + stop) / 2.0, _GROUP_Y, MUTED)
            xs += [start - 0.35, stop + 0.35]
            ys += [_GROUP_RULE_Y, _GROUP_RULE_Y]
            connect += [1, 0]
            start = stop + 1

        if not xs:
            return
        rule = pg.PlotDataItem(
            np.array(xs, dtype=float), np.array(ys, dtype=float),
            connect=np.array(connect, dtype=np.uint8),
            pen=pg.mkPen("#c8ccd3", width=1),
        )
        rule.setZValue(2)
        self._plot.getPlotItem().getViewBox().addItem(rule, ignoreBounds=True)
        self._decorations.append(rule)

    def _build_rails(self, shown: list[Axis]) -> None:
        rails = parallel.rail_positions(shown)
        if not rails:
            self._rails.setData([], [])
            return
        xs, ys, connect = [], [], []
        for column, bounds in rails.items():
            for position, value in bounds:
                xs += [column - 0.2, column + 0.2]
                ys += [position, position]
                connect += [1, 0]
                self._text(
                    f"{value:g}", column + 0.22, position, VIOLATING,
                    anchor=(0, 0.5), over_lines=True,
                )
        self._rails.setData(
            np.array(xs, dtype=float), np.array(ys, dtype=float),
            connect=np.array(connect, dtype=np.uint8),
        )

    def _text(
        self, text: str, x: float, y: float, colour: str, bold: bool = False,
        anchor: tuple[float, float] = (0.5, 0.5), over_lines: bool = False,
    ) -> None:
        # Labels sitting inside the plotting area are knocked out against the
        # polylines behind them. Grey text on a dense bundle is unreadable
        # exactly where the bundle is most interesting.
        item = pg.TextItem(
            text, color=colour, anchor=anchor,
            fill=(255, 255, 255, 200) if over_lines else None,
        )
        font = QFont("Segoe UI", 8)
        font.setBold(bold)
        item.setFont(font)
        item.setPos(x, y)
        item.setZValue(10)
        self._plot.getPlotItem().getViewBox().addItem(item, ignoreBounds=True)
        self._decorations.append(item)

    def _show_message(self, text: str) -> None:
        self.count_label.setText(text)
        self.hint_label.setText("")
        self.reset_button.setEnabled(False)

    # -- drawing ------------------------------------------------------------

    def _bands(self) -> np.ndarray:
        return np.array([brush.band for brush in self._brushes], dtype=float)

    def _redraw(self) -> None:
        if not self._brushes:
            return
        columns = self._order
        passing = parallel.filter_mask(self._values[:, columns], self._bands())

        feasible = self._feasible
        self._muted.setData(**self._polylines(~passing))
        self._passing.setData(**self._polylines(passing & feasible))
        self._violating.setData(**self._polylines(passing & ~feasible))
        if self._hovered >= 0 and not passing[self._hovered]:
            self._set_hover(-1)

        self._update_header(passing, feasible)

    def _polylines(self, selected: np.ndarray) -> dict:
        """One design per polyline, all of them in a single item.

        The ``connect`` array is what allows that: a zero after each design's
        last point tells the renderer to lift the pen rather than run on into
        the next design's first axis.

        Returned as ``setData`` keyword arguments, since an empty selection has
        to clear ``connect`` as well as the data -- a stale connect array left
        behind would be the wrong length for whatever is drawn next.
        """
        rows = np.flatnonzero(selected)
        width = len(self._order)
        if rows.size == 0 or width == 0:
            return {"x": np.array([]), "y": np.array([])}

        heights = self._display[np.ix_(rows, self._order)]
        xs = np.tile(np.arange(width, dtype=float), rows.size)
        ys = heights.ravel()

        connect = np.ones(ys.size, dtype=np.uint8)
        connect[width - 1 :: width] = 0
        # A segment needs both ends placed; an axis that could not place a
        # design leaves a gap instead of a line drawn to nowhere.
        finite = np.isfinite(ys)
        connect &= finite
        connect[:-1] &= finite[1:]

        return {"x": xs, "y": np.nan_to_num(ys, nan=0.5), "connect": connect}

    def _update_header(self, passing: np.ndarray, feasible: np.ndarray) -> None:
        total = len(self._frame)
        kept = int(passing.sum())
        filtered = any(brush.is_narrowed for brush in self._brushes)

        parts = [
            f"{kept} of {total} designs pass the filters" if filtered
            else f"{total} designs"
        ]
        violations = int((~feasible).sum())
        if violations:
            parts.append(f"{violations} infeasible")
        self.count_label.setText("   ·   ".join(parts))
        self.hint_label.setText("Drag a handle to filter · right-click an axis to reorder")
        self.reset_button.setEnabled(filtered)

    def reset_filters(self) -> None:
        for brush in self._brushes:
            brush.reset()
        self._redraw()

    # -- axis ordering ------------------------------------------------------

    def _show_axis_menu(self, column: int, screen_position) -> None:
        menu = QMenu(self)
        axis = self._axes[self._order[column]]

        move_left = menu.addAction("Move left")
        move_left.setEnabled(column > 0)
        move_right = menu.addAction("Move right")
        move_right.setEnabled(column < len(self._order) - 1)
        menu.addSeparator()
        hide = menu.addAction(f"Hide {axis.name}")
        hide.setEnabled(len(self._order) > 2)
        restore = menu.addAction("Show all, original order")
        restore.setEnabled(len(self._order) != len(self._axes) or self._order != sorted(self._order))

        chosen = menu.exec(screen_position.toPoint())
        if chosen is None:
            return
        if chosen is move_left:
            self._swap(column, column - 1)
        elif chosen is move_right:
            self._swap(column, column + 1)
        elif chosen is hide:
            self._reorder([i for n, i in enumerate(self._order) if n != column])
        elif chosen is restore:
            self._reorder(list(range(len(self._axes))))

    def _swap(self, a: int, b: int) -> None:
        order = list(self._order)
        order[a], order[b] = order[b], order[a]
        self._reorder(order)

    def _reorder(self, order: list[int]) -> None:
        """Rebuild against a new axis order, carrying the brushed bands over.

        Reordering answers a question about adjacency, not about which designs
        matter: a plot that silently dropped the filters when an axis moved
        would make the two impossible to use together.
        """
        bands = {self._order[n]: brush.band for n, brush in enumerate(self._brushes)}
        self._order = order
        self._build_axis_furniture()
        for column, index in enumerate(self._order):
            if index in bands:
                self._brushes[column].set_band(*bands[index])
        self._redraw()

    # -- hover --------------------------------------------------------------

    def _on_mouse_moved(self, position) -> None:
        if not self._brushes or self._display.size == 0:
            return
        view = self._plot.getPlotItem().getViewBox()
        if not self._plot.sceneBoundingRect().contains(position):
            self._set_hover(-1)
            return

        point = view.mapSceneToView(position)
        x, y = point.x(), point.y()
        width = len(self._order)
        if not (_BOTTOM - 0.02 <= y <= _TOP + 0.02) or not (0.0 <= x <= width - 1):
            self._set_hover(-1)
            return

        left = min(int(np.floor(x)), width - 2)
        fraction = x - left
        heights = self._display[:, self._order]
        interpolated = (
            heights[:, left] + fraction * (heights[:, left + 1] - heights[:, left])
        )

        visible = parallel.filter_mask(self._values[:, self._order], self._bands())
        distance = np.where(visible, np.abs(interpolated - y), np.inf)
        nearest = int(np.argmin(distance))
        self._set_hover(
            nearest if distance[nearest] <= HOVER_REACH else -1,
            side="right" if x < (width - 1) / 2.0 else "left",
        )

    def _set_hover(self, row: int, side: str = "right") -> None:
        if row == self._hovered:
            return
        self._hovered = row

        if row < 0:
            self._highlight.setData([], [])
            self._readout.setVisible(False)
            self.design_hovered.emit(-1)
            return

        selected = np.zeros(len(self._frame), dtype=bool)
        selected[row] = True
        self._highlight.setData(**self._polylines(selected))

        lines = [f"Run {row + 1}"]
        for index in self._order:
            axis = self._axes[index]
            value = self._frame.at[row, axis.name]
            shown = value if isinstance(value, str) else f"{float(value):.4g}"
            lines.append(f"{axis.name}  {shown}{(' ' + axis.unit) if axis.unit else ''}")
        # Park the readout on the far side of the plot from the cursor. Nine
        # lines of values cover an axis wherever they land, so the one to cover
        # is the one furthest from what the user is currently pointing at.
        self._readout.setText("\n".join(lines))
        if side == "right":
            self._readout.setAnchor(QPointF(1, 0))
            self._readout.setPos(len(self._order) - 1 + 0.45, _TOP)
        else:
            self._readout.setAnchor(QPointF(0, 0))
            self._readout.setPos(-0.45, _TOP)
        self._readout.setVisible(True)
        self.design_hovered.emit(row)
