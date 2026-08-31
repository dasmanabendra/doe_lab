"""Stage 4 — sensitivity and correlation over the completed experiments."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ...engine import analysis, parallel
from ...engine.project import Stage
from ..widgets.parallel import ParallelCoordinatesPlot
from ..widgets.plots import SERIES_COLOURS, MplPanel, style_axes
from ..widgets.tables import DataFrameView
from .base import Card, Page


class AnalyzePage(Page):
    stage = Stage.METAMODELS  # available once results exist
    title = "Analyze"
    subtitle = (
        "Which factors drive which responses, and how strongly. Read this before "
        "fitting metamodels — it tells you what the surrogate will need to capture."
    )

    def __init__(self):
        super().__init__()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sensitivity_tab(), "Factor sensitivity")
        self.tabs.addTab(self._build_partial_tab(), "Partial R²")
        self.tabs.addTab(self._build_correlation_tab(), "Correlations")
        self.tabs.addTab(self._build_explorer_tab(), "Design explorer")
        self.body.addWidget(self.tabs, stretch=1)

    # -- construction -------------------------------------------------------

    def _build_sensitivity_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        card = Card("Share of variation explained")
        self.sensitivity_view = DataFrameView(shading="sequential", decimals=4, show_index=True)
        card.add(self.sensitivity_view, stretch=1)

        note = QLabel(
            "Squared standardized regression coefficients, normalized so each row sums to 1. "
            "Because it is fitted linearly, a factor whose effect is purely curved and "
            "symmetric can score near zero — cross-check against the correlations and the "
            "fitted metamodels."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        card.add(note)
        layout.addWidget(card, stretch=2)

        plot_card = Card("Sensitivity by response")
        self.sensitivity_plot = MplPanel(height=3.0)
        plot_card.add(self.sensitivity_plot, stretch=1)
        layout.addWidget(plot_card, stretch=3)
        return page

    def _build_partial_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        card = Card("Partial R² — what each factor explains on its own")
        # Fixed 0-1 shading, not the sensitivity tab's per-column scale: these
        # numbers are absolute, so a dark cell must mean "large" everywhere in
        # the table rather than "largest in this column".
        self.partial_view = DataFrameView(shading="fraction", decimals=3, show_index=True)
        card.add(self.partial_view, stretch=1)

        note = QLabel(
            "Each factor is dropped from the model and the model refitted; what the fit "
            "loses is what that factor was uniquely providing. Read a cell as: of the "
            "variation the <i>other</i> factors cannot account for, this fraction is "
            "recovered by adding this one back. Unlike the sensitivity shares these do "
            "not sum to anything in particular — two factors that move together in the "
            "design can both score low while jointly explaining a great deal."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        card.add(note)
        layout.addWidget(card, stretch=1)

        plot_card = Card("Where each response's variation goes")
        self.variance_plot = MplPanel(height=3.0)
        plot_card.add(self.variance_plot, stretch=1)

        plot_note = QLabel(
            "Each bar is one whole response. <b>Shared</b> is variation the model explains "
            "but cannot attribute to any one factor, because the design moved those factors "
            "together; <b>unexplained</b> is what a linear model does not reach at all — the "
            "part to go looking for in a quadratic or Kriging surrogate. A bar running past "
            "1 means the unique contributions overlap to more than the model explains, which "
            "is a stronger warning about the same thing."
        )
        plot_note.setObjectName("pageSubtitle")
        plot_note.setWordWrap(True)
        plot_card.add(plot_note)

        layout.addWidget(plot_card, stretch=3)
        return page

    def _build_correlation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        pearson_card = Card("Pearson — linear association")
        self.pearson_view = DataFrameView(shading="diverging", decimals=3, show_index=True)
        pearson_card.add(self.pearson_view, stretch=1)
        layout.addWidget(pearson_card, stretch=1)

        spearman_card = Card("Spearman — monotonic association")
        self.spearman_view = DataFrameView(shading="diverging", decimals=3, show_index=True)
        spearman_card.add(self.spearman_view, stretch=1)
        layout.addWidget(spearman_card, stretch=1)

        note = QLabel(
            "Where Spearman is strong but Pearson is weak, the relationship is monotonic "
            "but curved — a sign that a linear metamodel will not be enough."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def _build_explorer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        card = Card("Every design, every variable")
        self.explorer = ParallelCoordinatesPlot()
        card.add(self.explorer, stretch=1)

        note = QLabel(
            "One line per experiment. Lines that stay parallel between two neighbouring "
            "axes move together; lines that cross in an X trade off against each other. "
            "Only adjacent axes say anything, so right-click an axis to move it beside "
            "the one you want to compare it with."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        card.add(note)

        layout.addWidget(card, stretch=1)
        return page

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None or not self.project.has_results:
            self._clear()
            return

        space = self.project.factors
        design = self.project.design
        results = self.project.results

        try:
            sensitivity = analysis.factor_sensitivity(space, design, results)
            decomposition = analysis.variance_decomposition(space, design, results)
        except ValueError as exc:
            self._clear()
            self.notify(str(exc))
            return

        self.sensitivity_view.set_frame(sensitivity.reset_index(names="Response"))
        self._draw_sensitivity(sensitivity)

        self.partial_view.set_frame(decomposition.partial.reset_index(names="Response"))
        self._draw_variance(decomposition)

        for method, view in (("pearson", self.pearson_view), ("spearman", self.spearman_view)):
            matrix = analysis.correlation_matrix(space, design, results, method)
            factor_labels, response_labels = analysis.split_labels(space, matrix)
            # Show the factors-versus-responses block rather than the full square:
            # the factor-factor corner is design structure, not a finding.
            block = matrix.loc[factor_labels, response_labels]
            view.set_frame(block.reset_index(names="Factor"))

        self._draw_explorer(space, design, results)

    def _draw_explorer(self, space, design: pd.DataFrame, results: pd.DataFrame) -> None:
        """Factors and responses side by side, one row per experiment.

        Built the same way the results table is, because it is the same table —
        the explorer just draws it instead of listing it.
        """
        assert self.project is not None
        frame = pd.concat(
            [design.reset_index(drop=True), results.reset_index(drop=True)], axis=1
        )
        axes = parallel.build_axes(space, self.project.responses, frame)
        self.explorer.set_data(
            frame, axes, parallel.feasibility(frame, self.project.responses)
        )

    def _clear(self) -> None:
        empty = pd.DataFrame()
        self.sensitivity_view.set_frame(empty)
        self.partial_view.set_frame(empty)
        self.pearson_view.set_frame(empty)
        self.spearman_view.set_frame(empty)
        self.sensitivity_plot.message("Run the experiments to see the analysis.")
        self.variance_plot.message("Run the experiments to see the analysis.")
        self.explorer.clear()

    def _draw_sensitivity(self, sensitivity: pd.DataFrame) -> None:
        self.sensitivity_plot.clear()
        axes = self.sensitivity_plot.figure.add_subplot(111)

        responses = list(sensitivity.index)
        factors = list(sensitivity.columns)
        bottom = [0.0] * len(responses)

        for index, factor in enumerate(factors):
            values = sensitivity[factor].to_numpy(dtype=float)
            axes.barh(
                responses,
                values,
                left=bottom,
                label=factor,
                color=SERIES_COLOURS[index % len(SERIES_COLOURS)],
                height=0.6,
            )
            bottom = [b + v for b, v in zip(bottom, values)]

        axes.set_xlim(0, 1)
        style_axes(axes, x_label="Share of variation explained")
        axes.legend(
            fontsize=8, frameon=False, ncol=min(len(factors), 4),
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
        )
        self.sensitivity_plot.draw()

    def _draw_variance(self, decomposition: analysis.VarianceDecomposition) -> None:
        """One bar per response, partitioned into unique / shared / unexplained.

        The three parts sum to 1, so the bar *is* the whole response. Left to
        right: what each factor accounts for on its own, what the model
        explains but cannot book to any single factor, and what a linear model
        does not reach at all. The sensitivity chart next door renormalizes the
        first group to fill the bar, which is what makes this one worth drawing
        — the grey is the part that view cannot show.
        """
        self.variance_plot.clear()
        axes = self.variance_plot.figure.add_subplot(111)

        unique = decomposition.semi_partial
        model = decomposition.model
        factors = list(unique.columns)
        labels = [self._fit_label(model, name) for name in unique.index]

        segments = [
            (factor, unique[factor].to_numpy(dtype=float), colour, None, "white")
            for factor, colour in zip(factors, itertools.cycle(SERIES_COLOURS))
        ]
        # Confounded factors can push the shared part a hair below zero. It is
        # not a length that can be drawn, so clamp it; the totals absorb it.
        segments.append(
            ("shared", model["shared"].to_numpy(dtype=float), "#9ca3af", None, "white")
        )
        segments.append(
            (
                "unexplained",
                model["unexplained"].to_numpy(dtype=float),
                "#eef1f5",
                "#c8cfd9",
                "#4b5563",
            )
        )

        bottom = np.zeros(len(labels))
        for name, values, colour, edge, text_colour in segments:
            widths = np.clip(np.nan_to_num(values), 0.0, None)
            bars = axes.barh(
                labels, widths, left=bottom, label=name,
                color=colour, edgecolor=edge, height=0.6,
            )
            axes.bar_label(
                bars,
                # Only segments wide enough to hold the number get one.
                labels=[f"{w:.2f}" if w >= 0.08 else "" for w in widths],
                label_type="center", fontsize=7, color=text_colour,
            )
            bottom = bottom + widths

        # A bar normally ends at exactly 1. It runs past only where ``shared``
        # came out negative — the unique contributions overlapping to more than
        # the model as a whole explains. Widen the axis rather than clipping
        # it, and mark where 1 was, so the overflow reads as the finding it is.
        axes.set_xlim(0, max(1.0, float(bottom.max(initial=1.0))))
        if bottom.max(initial=0.0) > 1.0 + 1e-9:
            axes.axvline(1.0, color="#94a3b8", linewidth=0.8, linestyle="--")

        # No x label: every tick is already a fraction of one response, and at
        # the window's minimum height the row it would cost is the difference
        # between the two-line tick labels clearing each other and colliding.
        style_axes(axes)
        axes.tick_params(axis="y", labelsize=7.5)
        axes.legend(
            fontsize=8, frameon=False, ncol=min(len(segments), 4),
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
        )
        self.variance_plot.draw()

    @staticmethod
    def _fit_label(model: pd.DataFrame, response: str) -> str:
        """A response name with its overall fit underneath.

        The fit belongs on the axis rather than in a separate table: a bar of
        unique contributions means something quite different at R² 0.9 than at
        R² 0.2, and splitting the two apart invites reading the first without
        the second.
        """
        r_squared = model.at[response, "r_squared"]
        adjusted = model.at[response, "adjusted_r_squared"]
        if not np.isfinite(r_squared):
            return f"{response}\nno variation"
        fit = f"R² {r_squared:.3f}"
        if np.isfinite(adjusted):
            fit += f"   adj {adjusted:.3f}"
        return f"{response}\n{fit}"
