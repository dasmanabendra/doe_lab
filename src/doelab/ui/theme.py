"""Application styling.

A single light palette, stated explicitly rather than inherited from the
platform, so the dense tables and heat-mapped cells keep the contrast they were
designed against. Cell shading in the analysis tables assumes a light
background; letting the OS supply a dark one would make that shading unreadable.
"""

from __future__ import annotations

from pathlib import Path

# Qt resolves stylesheet url() against the filesystem, and CSS wants forward
# slashes even on Windows.
CHECK_GLYPH = (Path(__file__).parent / "assets" / "check.svg").as_posix()

ACCENT = "#2f6fb0"
BACKGROUND = "#f4f5f7"
SURFACE = "#ffffff"
BORDER = "#d8dbe0"
TEXT = "#1f2430"
MUTED = "#6b7280"

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-size: 12px;
}}

QLabel#pageHeading {{
    font-size: 18px;
    font-weight: 600;
    color: {TEXT};
}}

QLabel#pageSubtitle {{
    font-size: 12px;
    color: {MUTED};
}}

QLabel#cardTitle {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT};
    padding-bottom: 2px;
}}

QLabel#fieldLabel {{
    color: {MUTED};
}}

QLabel#blockedNotice {{
    font-size: 13px;
    color: {MUTED};
    padding: 40px;
}}

QLabel#metricValue {{
    font-size: 16px;
    font-weight: 600;
    color: {TEXT};
}}

QLabel#metricLabel {{
    font-size: 11px;
    color: {MUTED};
}}

QLabel#warningLabel {{
    color: #92400e;
    background: #fef3c7;
    border: 1px solid #fde68a;
    border-radius: 4px;
    padding: 6px 8px;
}}

QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QFrame#card QLabel {{
    background: transparent;
}}

QListWidget#navRail {{
    background: {SURFACE};
    border: none;
    border-right: 1px solid {BORDER};
    outline: none;
    padding-top: 8px;
}}

QListWidget#navRail::item {{
    padding: 10px 14px;
    margin: 1px 6px;
    border-radius: 5px;
    color: {TEXT};
}}

QListWidget#navRail::item:selected {{
    background: {ACCENT};
    color: white;
    font-weight: 600;
}}

QListWidget#navRail::item:disabled {{
    color: #b6bac1;
}}

QListWidget#navRail::item:hover:!selected:!disabled {{
    background: #e8eef6;
}}

QTableView {{
    background: {SURFACE};
    alternate-background-color: #fafbfc;
    border: 1px solid {BORDER};
    border-radius: 4px;
    gridline-color: #eceef1;
    selection-background-color: #cfe0f2;
    selection-color: {TEXT};
}}

QHeaderView::section {{
    background: #eef0f3;
    color: {TEXT};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 7px;
}}

QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
}}

QPushButton:hover:!disabled {{
    background: #eef2f7;
    border-color: #b9c2cc;
}}

QPushButton:disabled {{
    color: #b6bac1;
    background: #f7f8f9;
}}

QPushButton#primary {{
    background: {ACCENT};
    color: white;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}

QPushButton#primary:hover:!disabled {{
    background: #275e97;
}}

QPushButton#primary:disabled {{
    background: #a9c2dd;
    border-color: #a9c2dd;
    color: #eef2f7;
}}

QPushButton#danger {{
    background: #b93b3b;
    color: white;
    border: 1px solid #b93b3b;
    font-weight: 600;
}}

QPushButton#danger:hover:!disabled {{
    background: #9e3131;
}}

QPushButton#danger:disabled {{
    background: #e0b4b4;
    border-color: #e0b4b4;
    color: #f7eaea;
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 18px;
}}

QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: #f2f3f5;
    color: #a8adb5;
}}

QProgressBar {{
    background: #e9ebee;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 8px;
    background: {SURFACE};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {MUTED};
}}

QPlainTextEdit, QTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 11px;
}}

QSplitter::handle {{
    background: {BORDER};
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE};
}}

QTabBar::tab {{
    background: transparent;
    padding: 6px 14px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    color: {MUTED};
}}

QTabBar::tab:selected {{
    background: {SURFACE};
    border-color: {BORDER};
    border-bottom-color: {SURFACE};
    color: {TEXT};
    font-weight: 600;
}}

QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {MUTED};
}}

/* Tick and radio indicators are drawn here in full, deliberately.
   The blanket ``QWidget`` background rule above matches these widgets too,
   which hands their painting to Qt's stylesheet engine and takes the native
   style out of play. With no rule describing the indicator, the engine has
   nothing to draw the *checked* state with, so a selected radio or checkbox
   renders as empty space -- clicking one appears to make it vanish. */

QRadioButton, QCheckBox {{
    spacing: 6px;
    background: transparent;
}}

/* Matrix cells carry no label of their own -- the column heading names them --
   so the gap reserved for text would push the indicator off the column centre. */
QRadioButton#matrixChoice {{
    spacing: 0px;
}}

QRadioButton::indicator, QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid #9aa1ab;
    background: {SURFACE};
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator:hover, QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

QRadioButton::indicator:checked {{
    border-color: {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 {ACCENT}, stop:0.45 {ACCENT},
                                stop:0.5 {SURFACE}, stop:1 {SURFACE});
}}

QCheckBox::indicator:checked {{
    border-color: {ACCENT};
    background: {ACCENT};
    image: url("{CHECK_GLYPH}");
}}

QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {{
    border-color: #ccd0d6;
    background: #f2f3f5;
}}

QScrollArea {{
    border: none;
    background: {BACKGROUND};
}}
"""
