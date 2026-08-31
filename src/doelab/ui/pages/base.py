"""Shared page scaffolding.

Every workflow stage is a full-window page with the same anatomy: a heading, a
one-line explanation of what the stage is for, and its content. Pages never
reach for the project themselves — the main window binds it and calls
:meth:`Page.refresh`, so a page's displayed state is always a function of the
project as it stands when the page is shown.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...engine.project import Project, Stage


class Page(QWidget):
    """Base class for the workflow pages.

    ``project_changed`` tells the main window that this page mutated the
    project, so the navigation rail can re-evaluate which later stages are now
    reachable (or no longer are).
    """

    project_changed = Signal()
    status_message = Signal(str)
    busy_changed = Signal(bool)

    stage: Stage = Stage.PROBLEM
    title: str = ""
    subtitle: str = ""

    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self._busy = False

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(20, 16, 20, 16)
        self._outer.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        heading = QLabel(self.title)
        heading.setObjectName("pageHeading")
        header.addWidget(heading)
        if self.subtitle:
            caption = QLabel(self.subtitle)
            caption.setObjectName("pageSubtitle")
            caption.setWordWrap(True)
            header.addWidget(caption)
        self._outer.addLayout(header)

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        self._outer.addLayout(self.body, stretch=1)

    def bind(self, project: Project) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the page from the current project state."""

    def shutdown(self) -> None:
        """Ask any work this page started to stop.

        Called before the window closes. Pages that only dispatch short tasks
        need do nothing; pages driving a long loop must signal it to finish, or
        the application will sit waiting on a thread that has no reason to end.
        """

    def set_busy(self, busy: bool) -> None:
        """Announce that this page has started or finished background work.

        The window locks navigation while any page is busy. Without that, a
        stage still recomputing in the background can finish *after* the user
        has moved on and built something downstream from the previous result —
        the late completion then invalidates work that was already done, which
        surfaces as metamodels fitted to experiments that no longer exist.
        """
        if busy == self._busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)

    @property
    def is_busy(self) -> bool:
        return self._busy

    def notify(self, message: str) -> None:
        self.status_message.emit(message)


class Card(QFrame):
    """A titled panel grouping related controls."""

    def __init__(self, title: str = "", spacing: int = 8):
        super().__init__()
        self.setObjectName("card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 12)
        self._layout.setSpacing(spacing)

        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            self._layout.addWidget(label)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout, stretch: int = 0):
        self._layout.addLayout(layout, stretch)
        return layout

    @property
    def layout_(self) -> QVBoxLayout:
        return self._layout


class FieldRow(QHBoxLayout):
    """A horizontal row of labelled controls."""

    def __init__(self, spacing: int = 10):
        super().__init__()
        self.setSpacing(spacing)

    def field(self, label: str, widget: QWidget, width: int | None = None) -> QWidget:
        text = QLabel(label)
        text.setObjectName("fieldLabel")
        self.addWidget(text)
        if width:
            widget.setFixedWidth(width)
        self.addWidget(widget)
        return widget

    def spacer(self) -> None:
        """Push everything after this to the right.

        Stretch rather than a filler widget: a bare ``QWidget`` inherits the
        window background, which reads as a stray grey panel against a card.
        """
        self.addStretch(1)


class BlockedNotice(QWidget):
    """Shown in place of a page whose prerequisites are not met.

    Better than disabling the content silently: it names the missing step, so
    the pipeline's ordering teaches itself.
    """

    def __init__(self, message: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel(message)
        self._label.setObjectName("blockedNotice")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

    def set_message(self, message: str) -> None:
        self._label.setText(message)
