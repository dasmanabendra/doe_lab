"""Editable tables for factor ranges and response roles.

Both tables edit the engine's dataclasses in place. They are deliberately
*shape*-preserving: factors cannot be added, removed, or retyped, because the
analytic problem defines its own inputs and outputs — a factor the solver has
never heard of could not be evaluated. What the user does control is the part
that matters to a study: how wide each factor's range is, how many levels it
gets, and what role each response plays in the optimization.

The factor table is type-aware. A continuous row edits min/max/levels; a
categorical row edits its level list, and its numeric cells are inert rather
than merely ignored, so the table never invites an edit it will discard.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QStyledItemDelegate,
    QTableView,
    QWidget,
)

from ...engine.factors import CategoricalFactor, FactorSpace, Response, ResponseRole

_INERT_BACKGROUND = QColor(245, 245, 247)

ROLE_LABELS: list[tuple[ResponseRole, str]] = [
    (ResponseRole.OBJECTIVE_MIN, "Minimize"),
    (ResponseRole.OBJECTIVE_MAX, "Maximize"),
    (ResponseRole.CONSTRAINT, "Constrain"),
    (ResponseRole.IGNORED, "Ignore"),
]
_LABEL_TO_ROLE = {label: role for role, label in ROLE_LABELS}
_ROLE_TO_LABEL = dict(ROLE_LABELS)


class FactorTableModel(QAbstractTableModel):
    """Edits factor ranges, levels and categories in place."""

    COLUMNS = ["Factor", "Type", "Unit", "Min", "Max", "Levels", "Categories"]
    COL_NAME, COL_TYPE, COL_UNIT, COL_MIN, COL_MAX, COL_LEVELS, COL_CATEGORIES = range(7)

    changed = Signal()

    def __init__(self, space: FactorSpace | None = None):
        super().__init__()
        self._space = space

    def set_space(self, space: FactorSpace | None) -> None:
        self.beginResetModel()
        self._space = space
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid() or self._space is None:
            return 0
        return len(self._space)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def _is_editable(self, factor: Any, column: int) -> bool:
        if column in (self.COL_NAME, self.COL_TYPE, self.COL_UNIT):
            return False
        if factor.is_categorical:
            return column == self.COL_CATEGORIES
        return column in (self.COL_MIN, self.COL_MAX, self.COL_LEVELS)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid() or self._space is None:
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        factor = self._space[index.row()]
        if self._is_editable(factor, index.column()):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._space is None:
            return None
        factor = self._space[index.row()]
        column = index.column()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._value(factor, column, role)

        if role == Qt.ItemDataRole.BackgroundRole and not self._is_editable(factor, column):
            if column not in (self.COL_NAME, self.COL_TYPE, self.COL_UNIT):
                return _INERT_BACKGROUND

        if role == Qt.ItemDataRole.ToolTipRole:
            if factor.is_categorical and column in (self.COL_MIN, self.COL_MAX):
                return "Categorical factors have levels, not numeric bounds."
            if not factor.is_categorical and column == self.COL_CATEGORIES:
                return "Continuous factors are defined by a range, not a level list."
            return factor.description or None

        if role == Qt.ItemDataRole.TextAlignmentRole and column in (
            self.COL_MIN, self.COL_MAX, self.COL_LEVELS
        ):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return None

    def _value(self, factor: Any, column: int, role: int):
        if column == self.COL_NAME:
            return factor.name
        if column == self.COL_TYPE:
            return "categorical" if factor.is_categorical else "continuous"
        if column == self.COL_UNIT:
            return factor.unit
        if column == self.COL_LEVELS:
            return factor.n_levels
        if factor.is_categorical:
            if column == self.COL_CATEGORIES:
                return ", ".join(factor.categories)
            return "" if role == Qt.ItemDataRole.DisplayRole else None
        if column == self.COL_MIN:
            return factor.low if role == Qt.ItemDataRole.EditRole else f"{factor.low:g}"
        if column == self.COL_MAX:
            return factor.high if role == Qt.ItemDataRole.EditRole else f"{factor.high:g}"
        return ""

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or self._space is None:
            return False
        factor = self._space[index.row()]
        column = index.column()

        try:
            if column == self.COL_CATEGORIES and factor.is_categorical:
                categories = [part.strip() for part in str(value).split(",") if part.strip()]
                # Rebuild through the dataclass so its own validation runs.
                replacement = CategoricalFactor(
                    factor.name, categories, factor.unit, factor.description
                )
                factor.categories = replacement.categories
            elif column == self.COL_MIN:
                new_low = float(value)
                if new_low >= factor.high:
                    return False
                factor.low = new_low
            elif column == self.COL_MAX:
                new_high = float(value)
                if new_high <= factor.low:
                    return False
                factor.high = new_high
            elif column == self.COL_LEVELS:
                levels = int(value)
                if levels < 2:
                    return False
                factor.levels = levels
            else:
                return False
        except (ValueError, TypeError):
            return False

        self.dataChanged.emit(index, index)
        self.changed.emit()
        return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            font = QFont()
            font.setBold(True)
            return font
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return str(section + 1)


class RoleDelegate(QStyledItemDelegate):
    """Presents the response role as a combo box of readable verbs."""

    def createEditor(self, parent: QWidget, option, index: QModelIndex) -> QWidget:
        editor = QComboBox(parent)
        for _, label in ROLE_LABELS:
            editor.addItem(label)
        return editor

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:
        editor.setCurrentText(str(index.data(Qt.ItemDataRole.EditRole)))

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class ResponseTableModel(QAbstractTableModel):
    """Edits each response's optimization role and its constraint bounds."""

    COLUMNS = ["Response", "Unit", "Role", "Lower", "Upper", "Description"]
    COL_NAME, COL_UNIT, COL_ROLE, COL_LOWER, COL_UPPER, COL_DESCRIPTION = range(6)

    changed = Signal()

    def __init__(self, responses: list[Response] | None = None):
        super().__init__()
        self._responses = responses or []

    def set_responses(self, responses: list[Response]) -> None:
        self.beginResetModel()
        self._responses = responses
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._responses)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def _is_editable(self, response: Response, column: int) -> bool:
        if column == self.COL_ROLE:
            return True
        if column in (self.COL_LOWER, self.COL_UPPER):
            # Bounds only mean something for a constraint.
            return response.role is ResponseRole.CONSTRAINT
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if self._is_editable(self._responses[index.row()], index.column()):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        response = self._responses[index.row()]
        column = index.column()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if column == self.COL_NAME:
                return response.name
            if column == self.COL_UNIT:
                return response.unit
            if column == self.COL_ROLE:
                return _ROLE_TO_LABEL[response.role]
            if column == self.COL_DESCRIPTION:
                return response.description
            bound = response.lower if column == self.COL_LOWER else response.upper
            if response.role is not ResponseRole.CONSTRAINT:
                return ""
            if bound is None:
                return "" if role == Qt.ItemDataRole.DisplayRole else None
            return bound if role == Qt.ItemDataRole.EditRole else f"{bound:g}"

        if role == Qt.ItemDataRole.BackgroundRole:
            if column in (self.COL_LOWER, self.COL_UPPER) and not self._is_editable(response, column):
                return _INERT_BACKGROUND

        if role == Qt.ItemDataRole.ToolTipRole:
            if column in (self.COL_LOWER, self.COL_UPPER) and response.role is not ResponseRole.CONSTRAINT:
                return "Set the role to Constrain to give this response a bound."
            return response.description or None

        if role == Qt.ItemDataRole.TextAlignmentRole and column in (self.COL_LOWER, self.COL_UPPER):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole:
            return False
        response = self._responses[index.row()]
        column = index.column()

        if column == self.COL_ROLE:
            new_role = _LABEL_TO_ROLE.get(str(value))
            if new_role is None:
                return False
            # A constraint must carry a bound. Seed one rather than leaving an
            # invalid response the optimizer would later reject.
            if new_role is ResponseRole.CONSTRAINT and response.lower is None and response.upper is None:
                response.upper = 0.0
            response.role = new_role
            # Bound cells change editability with the role.
            self.dataChanged.emit(
                index.siblingAtColumn(self.COL_LOWER), index.siblingAtColumn(self.COL_UPPER)
            )
        elif column in (self.COL_LOWER, self.COL_UPPER):
            text = str(value).strip()
            if text == "":
                bound = None
            else:
                try:
                    bound = float(text)
                except ValueError:
                    return False
            other = response.upper if column == self.COL_LOWER else response.lower
            if bound is None and other is None:
                return False  # would leave a constraint with no bound at all
            if column == self.COL_LOWER:
                response.lower = bound
            else:
                response.upper = bound
        else:
            return False

        self.dataChanged.emit(index, index)
        self.changed.emit()
        return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            font = QFont()
            font.setBold(True)
            return font
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return str(section + 1)


def make_table_view(model: QAbstractTableModel, stretch_column: int | None = None) -> QTableView:
    """A compact editable table view."""
    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
    view.verticalHeader().setDefaultSectionSize(24)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_column is not None:
        view.horizontalHeader().setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)
    return view
