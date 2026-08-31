"""Table models and views for showing DataFrames.

The workflow puts a lot of numbers on screen — experiment matrices, results,
correlation blocks, design populations. These share one read-only model rather
than converting each frame into ``QTableWidgetItem``s, which would copy every
cell and fall over on the thousands of rows an optimization produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QHeaderView, QTableView


def _diverging_colour(value: float) -> QColor:
    """Blue for negative, red for positive, white at zero.

    Used for correlation tables, where sign carries as much meaning as
    magnitude, so a single-hue ramp would hide half the information.
    """
    v = max(-1.0, min(1.0, float(value)))
    if v >= 0:
        return QColor(255, int(255 * (1 - v * 0.75)), int(255 * (1 - v * 0.75)))
    return QColor(int(255 * (1 + v * 0.75)), int(255 * (1 + v * 0.75)), 255)


def _sequential_colour(fraction: float) -> QColor:
    """White to blue ramp for magnitudes that have no meaningful sign."""
    f = max(0.0, min(1.0, float(fraction)))
    return QColor(int(255 - 90 * f), int(255 - 40 * f), 255)


class DataFrameModel(QAbstractTableModel):
    """Read-only view of a DataFrame, with optional cell shading.

    ``shading`` selects how numeric cells are coloured:

    ``None``
        No shading.
    ``"diverging"``
        Signed scale over ``[-1, 1]`` — for correlations.
    ``"sequential"``
        Magnitude scale over each column's own range — for sensitivities.
    ``"fraction"``
        Magnitude scale over a fixed ``[0, 1]``. For cells that are already
        shares of a whole: rescaling those to each column's own range would
        force a white and a full-blue cell into every column and destroy the
        comparison *between* columns, which is the only reason to shade them.
    """

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        shading: str | None = None,
        decimals: int = 4,
        show_index: bool = False,
    ):
        super().__init__()
        self._frame = frame if frame is not None else pd.DataFrame()
        self._shading = shading
        self._decimals = decimals
        self._show_index = show_index
        self._column_ranges: dict[str, tuple[float, float]] = {}
        self._recompute_ranges()

    # -- data management ----------------------------------------------------

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame if frame is not None else pd.DataFrame()
        self._recompute_ranges()
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def _recompute_ranges(self) -> None:
        self._column_ranges = {}
        if self._shading != "sequential" or self._frame.empty:
            return
        for name in self._frame.columns:
            column = self._frame[name]
            if pd.api.types.is_numeric_dtype(column):
                values = column.to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    self._column_ranges[str(name)] = (float(finite.min()), float(finite.max()))

    # -- QAbstractTableModel ------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._format(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole:
            return self._background(index, value)

        return None

    def _format(self, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            magnitude = abs(float(value))
            # Very large and very small magnitudes are unreadable in fixed
            # notation; switch to scientific rather than printing a wall of
            # zeros or truncating to nothing.
            if magnitude != 0 and (magnitude >= 1e6 or magnitude < 1e-4):
                return f"{value:.{self._decimals}e}"
            return f"{value:.{self._decimals}f}"
        return str(value)

    def _background(self, index: QModelIndex, value):
        if self._shading is None:
            return None
        if not isinstance(value, (float, int, np.number)) or isinstance(value, bool):
            return None
        if not np.isfinite(float(value)):
            return None

        if self._shading == "diverging":
            return QBrush(_diverging_colour(float(value)))

        if self._shading == "fraction":
            return QBrush(_sequential_colour(float(value)))

        name = str(self._frame.columns[index.column()])
        low, high = self._column_ranges.get(name, (0.0, 1.0))
        span = high - low
        fraction = 0.0 if span <= 0 else (float(value) - low) / span
        return QBrush(_sequential_colour(fraction))

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            font = QFont()
            font.setBold(True)
            return font
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[section]) if self._show_index else str(section + 1)


class DataFrameView(QTableView):
    """A compact table view wired to a :class:`DataFrameModel`."""

    def __init__(
        self,
        shading: str | None = None,
        decimals: int = 4,
        show_index: bool = False,
        stretch_last: bool = False,
    ):
        super().__init__()
        self._model = DataFrameModel(shading=shading, decimals=decimals, show_index=show_index)
        self.setModel(self._model)

        self.setAlternatingRowColors(shading is None)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSortingEnabled(False)
        self.setWordWrap(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.verticalHeader().setVisible(show_index)
        self.horizontalHeader().setStretchLastSection(stretch_last)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self._model.set_frame(frame)
        self.resizeColumnsToContents()

    @property
    def frame(self) -> pd.DataFrame:
        return self._model.frame
