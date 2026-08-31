"""Shared fixtures for the engine tests."""

from __future__ import annotations

import pandas as pd
import pytest

from doelab.engine import doe
from doelab.engine.factors import (
    CategoricalFactor,
    ContinuousFactor,
    FactorSpace,
    Response,
    ResponseRole,
)
from doelab.engine.solver import get_problem


@pytest.fixture
def mixed_space() -> FactorSpace:
    """A small space with both factor kinds and a 3-level categorical.

    Three levels matter: with two, the mutually-exclusive-indicator problem
    that breaks quadratic model matrices cannot arise.
    """
    return FactorSpace(
        [
            ContinuousFactor("x", 0.0, 10.0, "mm", levels=3),
            ContinuousFactor("y", -5.0, 5.0, "deg", levels=4),
            CategoricalFactor("mat", ["alu", "steel", "iron"]),
        ]
    )


@pytest.fixture
def engine_problem():
    return get_problem("gasoline_engine")


@pytest.fixture
def engine_space(engine_problem) -> FactorSpace:
    return FactorSpace(engine_problem.make_factors())


@pytest.fixture
def engine_study(engine_problem, engine_space) -> tuple[FactorSpace, pd.DataFrame, pd.DataFrame, list[Response]]:
    """A ready-to-analyze study: LHS design plus its noiseless results."""
    design = doe.latin_hypercube(engine_space, 120, seed=42)
    results = engine_problem.evaluate(design)
    return engine_space, design, results, engine_problem.make_responses()
