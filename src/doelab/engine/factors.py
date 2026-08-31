"""Factor and response definitions, and the sample space they span.

A *factor* is an input that a design varies. Two kinds are supported:

``ContinuousFactor``
    A real interval ``[low, high]``. ``levels`` only matters to designs that
    discretize (full factorial, and the D-optimal candidate set).

``CategoricalFactor``
    An unordered (nominal) set of named levels.

Operating conditions such as engine speed are modelled as ordinary continuous
factors. Tools that wrap an expensive physics solver keep a separate "case"
dimension because each operating point is another costly run; with an analytic
solver that distinction is a cost artifact rather than a mathematical one.

A *response* is an output the solver computes. Its ``role`` is what turns a
study into an optimization problem: objectives are minimized or maximized,
constraints must satisfy their bounds, and ignored responses are recorded but
not optimized against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


class ResponseRole(str, Enum):
    """How a response participates in the optimization problem."""

    OBJECTIVE_MIN = "objective_min"
    OBJECTIVE_MAX = "objective_max"
    CONSTRAINT = "constraint"
    IGNORED = "ignored"

    @property
    def is_objective(self) -> bool:
        return self in (ResponseRole.OBJECTIVE_MIN, ResponseRole.OBJECTIVE_MAX)


@dataclass
class ContinuousFactor:
    """A factor spanning the real interval ``[low, high]``."""

    name: str
    low: float
    high: float
    unit: str = ""
    levels: int = 3
    description: str = ""

    is_categorical: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(
                f"factor {self.name!r}: high ({self.high}) must exceed low ({self.low})"
            )
        if self.levels < 2:
            raise ValueError(f"factor {self.name!r}: needs at least 2 levels")

    @property
    def n_levels(self) -> int:
        return self.levels

    def level_values(self) -> list[float]:
        """Evenly spaced levels, used by discretizing designs."""
        return [float(v) for v in np.linspace(self.low, self.high, self.levels)]

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map values in ``[0, 1]`` onto ``[low, high]``."""
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)

    def to_coded(self, values: np.ndarray) -> np.ndarray:
        """Map real values onto ``[-1, 1]``, the usual coding for design theory."""
        v = np.asarray(values, dtype=float)
        return 2.0 * (v - self.low) / (self.high - self.low) - 1.0

    def clip(self, value: float) -> float:
        return float(min(max(value, self.low), self.high))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "continuous",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "levels": self.levels,
            "description": self.description,
        }


@dataclass
class CategoricalFactor:
    """A factor taking one of a fixed set of unordered levels."""

    name: str
    categories: list[str]
    unit: str = ""
    description: str = ""

    is_categorical: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self.categories = [str(c) for c in self.categories]
        if len(self.categories) < 2:
            raise ValueError(f"factor {self.name!r}: needs at least 2 categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"factor {self.name!r}: categories must be unique")

    @property
    def n_levels(self) -> int:
        return len(self.categories)

    def level_values(self) -> list[str]:
        return list(self.categories)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Bin ``[0, 1]`` into one equal stratum per category.

        Doing it this way rather than sampling categories at random is what
        lets Latin hypercube sampling keep its stratification guarantee on
        categorical columns: each category claims an equal share of the unit
        interval, so proportional coverage falls out of the LHS permutation.
        """
        k = len(self.categories)
        idx = np.clip((np.asarray(u, dtype=float) * k).astype(int), 0, k - 1)
        return np.asarray(self.categories, dtype=object)[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "categorical",
            "name": self.name,
            "categories": list(self.categories),
            "unit": self.unit,
            "description": self.description,
        }


Factor = ContinuousFactor | CategoricalFactor


@dataclass(frozen=True)
class CodedColumn:
    """One column of a coded design matrix, and where it came from."""

    label: str
    factor: str
    is_indicator: bool


def factor_from_dict(data: dict[str, Any]) -> Factor:
    """Rebuild a factor from its ``to_dict`` form."""
    payload = {k: v for k, v in data.items() if k != "kind"}
    if data["kind"] == "continuous":
        return ContinuousFactor(**payload)
    if data["kind"] == "categorical":
        return CategoricalFactor(**payload)
    raise ValueError(f"unknown factor kind: {data['kind']!r}")


@dataclass
class Response:
    """A solver output, and how the optimizer should treat it."""

    name: str
    unit: str = ""
    role: ResponseRole = ResponseRole.IGNORED
    lower: float | None = None
    upper: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        self.role = ResponseRole(self.role)
        if self.role is ResponseRole.CONSTRAINT and self.lower is None and self.upper is None:
            raise ValueError(
                f"response {self.name!r}: a constraint needs a lower or upper bound"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "role": self.role.value,
            "lower": self.lower,
            "upper": self.upper,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        return cls(**data)


class FactorSpace:
    """An ordered collection of factors, and the mappings into it.

    This owns the translation between the abstractions the rest of the engine
    works in: unit-hypercube samples (what samplers produce), tabular factor
    values (what the solver and metamodels consume), and numeric coded matrices
    (what design-optimality criteria need).
    """

    def __init__(self, factors: Sequence[Factor]):
        self.factors: list[Factor] = list(factors)
        if not self.factors:
            raise ValueError("a factor space needs at least one factor")
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")

    def __len__(self) -> int:
        return len(self.factors)

    def __iter__(self) -> Iterable[Factor]:
        return iter(self.factors)

    def __getitem__(self, key: int | str) -> Factor:
        if isinstance(key, str):
            for f in self.factors:
                if f.name == key:
                    return f
            raise KeyError(key)
        return self.factors[key]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def continuous(self) -> list[ContinuousFactor]:
        return [f for f in self.factors if not f.is_categorical]

    @property
    def categorical(self) -> list[CategoricalFactor]:
        return [f for f in self.factors if f.is_categorical]

    @property
    def continuous_names(self) -> list[str]:
        return [f.name for f in self.continuous]

    @property
    def categorical_names(self) -> list[str]:
        return [f.name for f in self.categorical]

    def from_unit_cube(self, u: np.ndarray) -> pd.DataFrame:
        """Map an ``(n, d)`` array of unit-cube samples to factor values."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        if u.shape[1] != len(self):
            raise ValueError(
                f"expected {len(self)} columns for this factor space, got {u.shape[1]}"
            )
        columns = {f.name: f.from_unit(u[:, i]) for i, f in enumerate(self.factors)}
        return pd.DataFrame(columns)

    def coded_matrix(
        self, df: pd.DataFrame, drop_first: bool = True
    ) -> tuple[np.ndarray, list[CodedColumn]]:
        """Encode factor values numerically.

        Continuous factors are coded to ``[-1, 1]``; categorical factors are
        expanded to indicator columns. ``drop_first`` uses reference coding,
        which keeps ``XᵀX`` non-singular once an intercept is added — the form
        D-optimal exchange needs.

        Each column carries the factor it came from, because callers building
        higher-order terms must know which indicator columns are siblings:
        indicators of one categorical factor are mutually exclusive, so their
        product is identically zero and would silently make ``XᵀX`` singular.
        """
        blocks: list[np.ndarray] = []
        columns: list[CodedColumn] = []
        for f in self.factors:
            if f.is_categorical:
                cats = f.categories[1:] if drop_first else f.categories
                for cat in cats:
                    blocks.append((df[f.name].to_numpy() == cat).astype(float))
                    columns.append(CodedColumn(f"{f.name}={cat}", f.name, True))
            else:
                blocks.append(f.to_coded(df[f.name].to_numpy()))
                columns.append(CodedColumn(f.name, f.name, False))
        if not blocks:
            return np.empty((len(df), 0)), []
        return np.column_stack(blocks), columns

    def midpoint(self) -> dict[str, float | str]:
        """A neutral sample: interval centres, and the first categorical level."""
        out: dict[str, float | str] = {}
        for f in self.factors:
            out[f.name] = f.categories[0] if f.is_categorical else (f.low + f.high) / 2.0
        return out

    def to_frame(self, samples: Sequence[dict[str, Any]]) -> pd.DataFrame:
        """Build a factor-ordered DataFrame from a list of sample dicts."""
        return pd.DataFrame(list(samples), columns=self.names)

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.factors]

    @classmethod
    def from_list(cls, data: Sequence[dict[str, Any]]) -> FactorSpace:
        return cls([factor_from_dict(d) for d in data])
