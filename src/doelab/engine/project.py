"""The study document: everything a session accumulates, and its file format.

A project is saved as a single JSON file. Fitted metamodels are deliberately
*not* serialized — only their specs are, and they are refit from the stored
experiments on load. Refitting costs a second or two, whereas pickling
scikit-learn estimators would tie saved files to the exact library version that
wrote them.

The stage pipeline is strictly ordered: factors -> design -> results ->
metamodels -> optimization. Each stage's output is the next one's input, so
editing an upstream stage invalidates everything downstream. That invalidation
is enforced here in :meth:`Project.invalidate_from` rather than left to the UI,
because silently keeping results that no longer correspond to the current
factors is the kind of bug that produces confident, wrong answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import pandas as pd

from .doe import DesignSpec
from .factors import FactorSpace, Response
from .metamodel import Metamodel, MetamodelSpec, fit_metamodel
from .optimize import OptimizationConfig
from .solver import NoiseConfig, Problem, get_problem

FILE_VERSION = 1
FILE_SUFFIX = ".doelab.json"


class Stage(IntEnum):
    """Pipeline stages, in dependency order."""

    PROBLEM = 0
    DESIGN = 1
    RESULTS = 2
    METAMODELS = 3
    OPTIMIZE = 4


def _df_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """JSON-friendly form of a DataFrame.

    Per-column ``tolist()`` is what converts numpy scalars into native Python
    types; dumping ``values`` wholesale would leave numpy types the JSON
    encoder rejects.
    """
    return {
        "columns": [str(c) for c in df.columns],
        "data": {str(c): df[c].tolist() for c in df.columns},
    }


def _df_from_dict(payload: dict[str, Any] | None) -> pd.DataFrame | None:
    if not payload:
        return None
    return pd.DataFrame(payload["data"], columns=payload["columns"])


@dataclass
class Project:
    """A complete study."""

    problem_name: str = "gasoline_engine"
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    factors: FactorSpace | None = None
    responses: list[Response] = field(default_factory=list)
    design_spec: DesignSpec = field(default_factory=DesignSpec)
    design: pd.DataFrame | None = None
    results: pd.DataFrame | None = None
    metamodel_specs: list[MetamodelSpec] = field(default_factory=list)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Runtime only: rebuilt on load, never written to disk.
    metamodels: dict[str, Metamodel] = field(default_factory=dict, repr=False)
    path: Path | None = field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_problem(cls, problem_name: str) -> Project:
        """Start a fresh study from one of the analytic problems."""
        problem = get_problem(problem_name)
        return cls(
            problem_name=problem_name,
            factors=FactorSpace(problem.make_factors()),
            responses=problem.make_responses(),
        )

    def problem(self) -> Problem:
        return get_problem(self.problem_name)

    # -- stage state --------------------------------------------------------

    @property
    def has_design(self) -> bool:
        return self.design is not None and len(self.design) > 0

    @property
    def has_results(self) -> bool:
        return self.results is not None and len(self.results) > 0

    @property
    def has_metamodels(self) -> bool:
        return bool(self.metamodels)

    def reached(self, stage: Stage) -> bool:
        """Whether a stage's prerequisites are satisfied."""
        if stage <= Stage.PROBLEM:
            return self.factors is not None
        if stage is Stage.DESIGN:
            return self.factors is not None
        if stage is Stage.RESULTS:
            return self.has_design
        if stage is Stage.METAMODELS:
            return self.has_results
        if stage is Stage.OPTIMIZE:
            return self.has_metamodels
        return False

    def blocker(self, stage: Stage) -> str | None:
        """Why a stage is not yet available, phrased for the user."""
        if self.reached(stage):
            return None
        return {
            Stage.DESIGN: "Choose a problem and its factors first.",
            Stage.RESULTS: "Generate a design first.",
            Stage.METAMODELS: "Run the experiments first.",
            Stage.OPTIMIZE: "Fit a metamodel for each objective and constraint first.",
        }.get(stage, "Earlier steps are incomplete.")

    def invalidate_from(self, stage: Stage) -> None:
        """Discard everything that depends on a stage that just changed.

        Called when factors are edited, a design is regenerated, and so on.
        Keeping downstream artifacts alive would leave results attributed to
        factor definitions that no longer exist.
        """
        if stage <= Stage.DESIGN:
            self.design = None
        if stage <= Stage.RESULTS:
            self.results = None
        if stage <= Stage.METAMODELS:
            self.metamodels = {}

    # -- metamodels ---------------------------------------------------------

    def build_metamodels(
        self, specs: list[MetamodelSpec] | None = None
    ) -> tuple[list[MetamodelSpec], dict[str, Metamodel]]:
        """Fit the requested metamodels **without** storing them.

        Deliberately free of side effects so it is safe to call from a worker
        thread: the caller commits the result with :meth:`adopt_metamodels`
        once back on the thread that owns the project. Fitting and storing in
        one step would leave the project mutated from another thread before any
        observer had been told, so a reader could see models attached to a
        study state that had not been published yet.

        Keyed by response name, so the optimizer can look each one up. When a
        response appears more than once the last spec wins, which matches the
        UI offering a single fit type per response.
        """
        if not self.has_results or self.factors is None:
            raise ValueError("run the experiments before fitting metamodels")

        specs = specs if specs is not None else self.metamodel_specs
        if not specs:
            raise ValueError("no metamodels requested")

        assert self.design is not None and self.results is not None
        models: dict[str, Metamodel] = {}
        for spec in specs:
            models[spec.response] = fit_metamodel(
                self.factors, self.design, self.results, spec
            )
        return list(specs), models

    def adopt_metamodels(
        self, specs: list[MetamodelSpec], models: dict[str, Metamodel]
    ) -> dict[str, Metamodel]:
        """Attach a fitted set of metamodels to the project."""
        self.metamodel_specs = list(specs)
        self.metamodels = models
        return models

    def fit_metamodels(self, specs: list[MetamodelSpec] | None = None) -> dict[str, Metamodel]:
        """Fit and store in one step, for callers on the owning thread."""
        return self.adopt_metamodels(*self.build_metamodels(specs))

    def default_metamodel_specs(self, fit_type: str = "quadratic") -> list[MetamodelSpec]:
        """One spec per response that the optimizer will need."""
        from .factors import ResponseRole

        return [
            MetamodelSpec(r.name, fit_type)
            for r in self.responses
            if r.role is not ResponseRole.IGNORED
        ]

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FILE_VERSION,
            "problem_name": self.problem_name,
            "noise": self.noise.to_dict(),
            "factors": self.factors.to_list() if self.factors else [],
            "responses": [r.to_dict() for r in self.responses],
            "design_spec": self.design_spec.to_dict(),
            "design": _df_to_dict(self.design) if self.design is not None else None,
            "results": _df_to_dict(self.results) if self.results is not None else None,
            "metamodel_specs": [s.to_dict() for s in self.metamodel_specs],
            "optimization": self.optimization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], refit: bool = True) -> Project:
        version = data.get("version", FILE_VERSION)
        if version > FILE_VERSION:
            raise ValueError(
                f"project file version {version} is newer than this build supports "
                f"(v{FILE_VERSION})"
            )

        project = cls(
            problem_name=data["problem_name"],
            noise=NoiseConfig.from_dict(data["noise"]),
            factors=FactorSpace.from_list(data["factors"]) if data["factors"] else None,
            responses=[Response.from_dict(r) for r in data["responses"]],
            design_spec=DesignSpec.from_dict(data["design_spec"]),
            design=_df_from_dict(data.get("design")),
            results=_df_from_dict(data.get("results")),
            metamodel_specs=[
                MetamodelSpec.from_dict(s) for s in data.get("metamodel_specs", [])
            ],
            optimization=OptimizationConfig.from_dict(data["optimization"]),
        )

        if refit and project.metamodel_specs and project.has_results:
            project.fit_metamodels()
        return project

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path, refit: bool = True) -> Project:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        project = cls.from_dict(data, refit=refit)
        project.path = path
        return project
