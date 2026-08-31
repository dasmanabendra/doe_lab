"""Stage 1 — choose the problem, and set factor ranges and response roles."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ...engine.factors import FactorSpace
from ...engine.project import Project, Stage
from ...engine.solver import list_problems
from ..widgets.editors import (
    FactorTableModel,
    ResponseTableModel,
    RoleDelegate,
    make_table_view,
)
from .base import Card, FieldRow, Page


class ProblemPage(Page):
    stage = Stage.PROBLEM
    title = "Problem"
    subtitle = (
        "Pick the response surface to study, then set how wide each factor ranges "
        "and what role each response plays in the optimization."
    )

    def __init__(self):
        super().__init__()
        self._loading = False

        self.body.addWidget(self._build_source_card())
        self.body.addWidget(self._build_factors_card(), stretch=3)
        self.body.addWidget(self._build_responses_card(), stretch=2)

    # -- construction -------------------------------------------------------

    def _build_source_card(self) -> Card:
        card = Card("Source")

        row = FieldRow()
        self.problem_combo = QComboBox()
        for name, title in list_problems():
            self.problem_combo.addItem(title, name)
        self.problem_combo.currentIndexChanged.connect(self._on_problem_selected)
        row.field("Problem", self.problem_combo, width=340)
        row.spacer()
        card.add_layout(row)

        self.description_label = QLabel()
        self.description_label.setObjectName("pageSubtitle")
        self.description_label.setWordWrap(True)
        card.add(self.description_label)

        noise_row = FieldRow()
        self.noise_check = QCheckBox("Add measurement noise")
        self.noise_check.toggled.connect(self._on_noise_changed)
        noise_row.addWidget(self.noise_check)

        self.noise_sigma = QDoubleSpinBox()
        self.noise_sigma.setDecimals(3)
        self.noise_sigma.setRange(0.0, 0.5)
        self.noise_sigma.setSingleStep(0.005)
        self.noise_sigma.setSuffix(" relative")
        self.noise_sigma.valueChanged.connect(self._on_noise_changed)
        noise_row.field("Sigma", self.noise_sigma, width=130)

        self.noise_seed = QSpinBox()
        self.noise_seed.setRange(0, 10_000)
        self.noise_seed.valueChanged.connect(self._on_noise_changed)
        noise_row.field("Seed", self.noise_seed, width=90)
        noise_row.spacer()
        card.add_layout(noise_row)

        hint = QLabel(
            "Noise is what makes a design of experiments worth running at all — "
            "with a perfectly repeatable solver, every replicate would agree exactly."
        )
        hint.setObjectName("pageSubtitle")
        hint.setWordWrap(True)
        card.add(hint)

        return card

    def _build_factors_card(self) -> Card:
        card = Card("Factors")
        self.factor_model = FactorTableModel()
        self.factor_model.changed.connect(self._on_factors_edited)
        self.factor_table = make_table_view(self.factor_model, stretch_column=FactorTableModel.COL_CATEGORIES)
        card.add(self.factor_table, stretch=1)

        note = QLabel(
            "Ranges, level counts and category lists are editable. The set of factors "
            "itself is fixed by the problem — the solver can only evaluate the inputs it defines."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        card.add(note)
        return card

    def _build_responses_card(self) -> Card:
        card = Card("Responses")
        self.response_model = ResponseTableModel()
        self.response_model.changed.connect(self._on_responses_edited)
        self.response_table = make_table_view(
            self.response_model, stretch_column=ResponseTableModel.COL_DESCRIPTION
        )
        self.response_table.setItemDelegateForColumn(
            ResponseTableModel.COL_ROLE, RoleDelegate(self.response_table)
        )
        card.add(self.response_table, stretch=1)

        note = QLabel(
            "Objectives are what the optimizer trades off; constraints bound the search; "
            "ignored responses are still computed and plotted, just not optimized."
        )
        note.setObjectName("pageSubtitle")
        note.setWordWrap(True)
        card.add(note)
        return card

    # -- state --------------------------------------------------------------

    def refresh(self) -> None:
        if self.project is None:
            return
        self._loading = True
        try:
            index = self.problem_combo.findData(self.project.problem_name)
            if index >= 0:
                self.problem_combo.setCurrentIndex(index)
            self.description_label.setText(self.project.problem().description)

            self.noise_check.setChecked(self.project.noise.enabled)
            self.noise_sigma.setValue(self.project.noise.sigma)
            self.noise_seed.setValue(self.project.noise.seed or 0)
            self.noise_sigma.setEnabled(self.project.noise.enabled)
            self.noise_seed.setEnabled(self.project.noise.enabled)

            self.factor_model.set_space(self.project.factors)
            self.response_model.set_responses(self.project.responses)
        finally:
            self._loading = False

    # -- edits --------------------------------------------------------------

    def _on_problem_selected(self) -> None:
        if self._loading or self.project is None:
            return
        name = self.problem_combo.currentData()
        if name == self.project.problem_name:
            return

        # A different problem means different factors and responses entirely,
        # so nothing downstream can survive.
        fresh = Project.from_problem(name)
        self.project.problem_name = name
        self.project.factors = fresh.factors
        self.project.responses = fresh.responses
        self.project.metamodel_specs = []
        self.project.invalidate_from(Stage.DESIGN)

        self.refresh()
        self.notify(f"Switched to {self.problem_combo.currentText()}")
        self.project_changed.emit()

    def _on_noise_changed(self) -> None:
        if self._loading or self.project is None:
            return
        self.project.noise.enabled = self.noise_check.isChecked()
        self.project.noise.sigma = self.noise_sigma.value()
        self.project.noise.seed = self.noise_seed.value()
        self.noise_sigma.setEnabled(self.project.noise.enabled)
        self.noise_seed.setEnabled(self.project.noise.enabled)

        # The design is still valid, but its results were produced under the
        # previous noise setting.
        self.project.invalidate_from(Stage.RESULTS)
        self.notify("Noise settings changed — re-run the experiments")
        self.project_changed.emit()

    def _on_factors_edited(self) -> None:
        if self._loading or self.project is None:
            return
        self.project.invalidate_from(Stage.DESIGN)
        self.notify("Factors changed — the design needs regenerating")
        self.project_changed.emit()

    def _on_responses_edited(self) -> None:
        if self._loading or self.project is None:
            return
        # Roles and bounds only affect the optimization, so results and fitted
        # metamodels remain valid.
        self.notify("Response roles updated")
        self.project_changed.emit()
