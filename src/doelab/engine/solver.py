"""Analytic stand-ins for an expensive physics solver.

Each :class:`Problem` declares the factors it accepts and the responses it
produces, and evaluates a whole design at once. Selecting a problem is what
populates the factor and response tables in the UI.

Responses are built from polynomial and trigonometric terms so that the
surfaces have genuine interior optima, factor interactions, and competing
objectives — the structure a DOE workflow exists to uncover — while still
evaluating in microseconds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .factors import (
    CategoricalFactor,
    ContinuousFactor,
    Factor,
    Response,
    ResponseRole,
)


@dataclass
class NoiseConfig:
    """Multiplicative Gaussian noise applied to solver outputs.

    Noise is relative (``value * (1 + N(0, sigma))``) so a single setting is
    meaningful across responses with wildly different magnitudes. Real DOE
    practice exists largely *because* observations are noisy, so the toggle is
    worth having even against an analytic solver.

    A fixed ``seed`` makes a given design reproducible; the generator is
    created once per ``evaluate`` call.
    """

    enabled: bool = False
    sigma: float = 0.01
    seed: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "sigma": self.sigma, "seed": self.seed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseConfig:
        return cls(**data)


class Problem(ABC):
    """A named analytic response surface over a declared factor space."""

    name: str = "problem"
    title: str = "Problem"
    description: str = ""

    @abstractmethod
    def make_factors(self) -> list[Factor]:
        """Default factor definitions; the user may narrow ranges afterwards."""

    @abstractmethod
    def make_responses(self) -> list[Response]:
        """Default response definitions, including their optimization roles."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Evaluate every response over a factor-valued DataFrame."""

    def evaluate(self, df: pd.DataFrame, noise: NoiseConfig | None = None) -> pd.DataFrame:
        """Evaluate the design and optionally corrupt it with noise."""
        raw = self.compute(df)
        out = pd.DataFrame(raw, index=df.index)
        if noise is not None and noise.enabled and noise.sigma > 0:
            rng = np.random.default_rng(noise.seed)
            for col in out.columns:
                factor = 1.0 + rng.normal(0.0, noise.sigma, size=len(out))
                out[col] = out[col].to_numpy() * factor
        return out


def _unit(df: pd.DataFrame, name: str, low: float, high: float) -> np.ndarray:
    """Normalize a factor column onto ``[0, 1]``."""
    return (df[name].to_numpy(dtype=float) - low) / (high - low)


# --------------------------------------------------------------------------
# Pseudo-engine problem
# --------------------------------------------------------------------------

# Relative lower heating value and knock resistance per fuel. E85 carries less
# energy per unit mass (so needs more fuel for the same work, raising BSFC) but
# resists knock well enough to allow more spark advance.
_FUEL_ENERGY = {"Regular": 1.00, "Premium": 1.01, "E85": 0.72}
_FUEL_OCTANE = {"Regular": 0.0, "Premium": 1.0, "E85": 1.35}


class GasolineEngineProblem(Problem):
    """A four-response engine-like surface with a categorical fuel choice.

    Structure worth knowing when reading the results:

    * **Spark timing** has a best-torque value (MBT) that advances with fuel
      octane — this is the interaction that makes ``fuel_type`` matter.
    * **Runner length** and **RPM** interact through an intake-resonance ridge
      (the cosine term), so the torque contour over that pair is a diagonal
      band rather than a simple bowl.
    * **BSFC and Torque compete**, and E85 trades fuel economy for output,
      which is what gives the optimizer a non-trivial Pareto front.
    """

    name = "gasoline_engine"
    title = "Gasoline Engine (4 responses, mixed factors)"
    description = (
        "Engine-like surface over spark timing, valve timing, runner length, RPM "
        "and fuel type. Minimize BSFC and maximize torque, subject to a peak "
        "cylinder pressure limit."
    )

    def make_factors(self) -> list[Factor]:
        return [
            ContinuousFactor("Spark_Timing", -30.0, -10.0, "deg", levels=3,
                             description="Ignition advance before TDC"),
            ContinuousFactor("Valve_Timing", 210.0, 250.0, "deg", levels=3,
                             description="Intake cam phasing"),
            ContinuousFactor("Runner_Length", 100.0, 500.0, "mm", levels=3,
                             description="Intake runner length"),
            ContinuousFactor("RPM", 1000.0, 5000.0, "rev/min", levels=3,
                             description="Engine speed"),
            CategoricalFactor("Fuel_Type", ["Regular", "Premium", "E85"],
                              description="Fuel grade"),
        ]

    def make_responses(self) -> list[Response]:
        return [
            Response("BSFC", "g/kW-h", ResponseRole.OBJECTIVE_MIN,
                     description="Brake specific fuel consumption"),
            Response("Torque", "N-m", ResponseRole.OBJECTIVE_MAX,
                     description="Brake torque"),
            Response("Power", "kW", ResponseRole.IGNORED,
                     description="Brake power, derived from torque and speed"),
            # Set where it actually binds: peak pressure rises with spark
            # advance almost in lockstep with torque, so this limit cuts the
            # high-output end of the front rather than sitting inactive.
            Response("Max_Cyl_Pressure", "bar", ResponseRole.CONSTRAINT, upper=64.0,
                     description="Peak cylinder pressure"),
        ]

    def compute(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        s = _unit(df, "Spark_Timing", -30.0, -10.0)   # 0 = most advanced
        v = _unit(df, "Valve_Timing", 210.0, 250.0)
        ell = _unit(df, "Runner_Length", 100.0, 500.0)
        n = _unit(df, "RPM", 1000.0, 5000.0)

        fuel = df["Fuel_Type"].astype(str).to_numpy()
        energy = np.vectorize(_FUEL_ENERGY.get)(fuel).astype(float)
        octane = np.vectorize(_FUEL_OCTANE.get)(fuel).astype(float)

        # Advance is measured before TDC, so a *smaller* normalized value means
        # more advance. Flip it so higher = more advance, which reads better in
        # the expressions below.
        advance = 1.0 - s

        # Best-torque spark advance (MBT), pushed further with knock-resistant
        # fuel. Best *economy* sits a little retarded of MBT, which is what puts
        # the two objectives genuinely in tension rather than having them peak
        # together.
        mbt = 0.45 + 0.13 * octane
        best_economy = mbt - 0.14
        spark_shape = 1.0 - 2.4 * (advance - mbt) ** 2

        # Cam phasing optimum drifts later with engine speed.
        vt_opt = 0.40 + 0.30 * n
        vt_shape = 1.0 - 1.7 * (v - vt_opt) ** 2

        # Intake resonance: short runners suit high speed, long runners low.
        # The cosine puts a tuned ridge across the (runner length, RPM) plane.
        tune = np.cos(2.0 * np.pi * (ell - 0.55 * n - 0.20))

        # Volumetric-efficiency style torque curve, peaking mid-range.
        rpm_shape = 1.0 - 2.0 * (n - 0.45) ** 2

        torque = (
            340.0
            * np.clip(rpm_shape, 0.15, None)
            * np.clip(vt_shape, 0.30, None)
            * np.clip(spark_shape, 0.25, None)
            * (1.0 + 0.11 * tune)
            * (1.0 + 0.04 * octane)
        )

        power = torque * df["RPM"].to_numpy(dtype=float) * 2.0 * np.pi / 60.0 / 1000.0

        # BSFC as a base rate plus bounded penalties for operating away from the
        # efficiency island, divided by the fuel's relative energy content.
        # Expressed additively rather than as 1/efficiency: a reciprocal blows
        # up wherever the efficiency terms stack at their floor, producing
        # corner values no engine would show and a surface dominated by that
        # corner rather than by the physics.
        bsfc = (
            238.0
            + 150.0 * (advance - best_economy) ** 2
            + 90.0 * (n - 0.40) ** 2
            + 25.0 * (v - vt_opt) ** 2
            - 12.0 * tune
        ) / energy

        # Peak pressure climbs with advance and speed; knock-limited fuels sit
        # a little lower for the same advance.
        max_cyl_pressure = (
            38.0
            + 46.0 * advance
            + 18.0 * n
            + 9.0 * advance * n
            - 6.0 * octane
        )

        return {
            "BSFC": bsfc,
            "Torque": torque,
            "Power": power,
            "Max_Cyl_Pressure": max_cyl_pressure,
        }


# --------------------------------------------------------------------------
# Standard benchmarks
# --------------------------------------------------------------------------


class BraninProblem(Problem):
    """The Branin-Hoo function: two continuous factors, three global minima."""

    name = "branin"
    title = "Branin-Hoo (2 factors, 1 objective)"
    description = "Classic two-dimensional test function with three equal global minima."

    def make_factors(self) -> list[Factor]:
        return [
            ContinuousFactor("x1", -5.0, 10.0, levels=5),
            ContinuousFactor("x2", 0.0, 15.0, levels=5),
        ]

    def make_responses(self) -> list[Response]:
        return [Response("f", "", ResponseRole.OBJECTIVE_MIN)]

    def compute(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x1 = df["x1"].to_numpy(dtype=float)
        x2 = df["x2"].to_numpy(dtype=float)
        a, b = 1.0, 5.1 / (4.0 * np.pi**2)
        c, r = 5.0 / np.pi, 6.0
        s, t = 10.0, 1.0 / (8.0 * np.pi)
        f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return {"f": f}


class RosenbrockProblem(Problem):
    """The two-dimensional Rosenbrock valley, minimized at ``(1, 1)``."""

    name = "rosenbrock"
    title = "Rosenbrock (2 factors, 1 objective)"
    description = "Narrow curved valley; easy to find, hard to converge in."

    def make_factors(self) -> list[Factor]:
        return [
            ContinuousFactor("x1", -2.0, 2.0, levels=5),
            ContinuousFactor("x2", -1.0, 3.0, levels=5),
        ]

    def make_responses(self) -> list[Response]:
        return [Response("f", "", ResponseRole.OBJECTIVE_MIN)]

    def compute(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x1 = df["x1"].to_numpy(dtype=float)
        x2 = df["x2"].to_numpy(dtype=float)
        return {"f": (1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2}


class ZDT1Problem(Problem):
    """ZDT1, whose true Pareto front is ``f2 = 1 - sqrt(f1)`` on ``[0, 1]``.

    The analytic front is what makes this the right problem for asserting that
    the optimizer actually converges.
    """

    name = "zdt1"
    title = "ZDT1 (6 factors, 2 objectives)"
    description = "Two-objective benchmark with a known convex Pareto front."

    n_vars = 6

    def make_factors(self) -> list[Factor]:
        return [
            ContinuousFactor(f"x{i + 1}", 0.0, 1.0, levels=3)
            for i in range(self.n_vars)
        ]

    def make_responses(self) -> list[Response]:
        return [
            Response("f1", "", ResponseRole.OBJECTIVE_MIN),
            Response("f2", "", ResponseRole.OBJECTIVE_MIN),
        ]

    def compute(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x = df[[f"x{i + 1}" for i in range(self.n_vars)]].to_numpy(dtype=float)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * x[:, 1:].sum(axis=1) / (self.n_vars - 1)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return {"f1": f1, "f2": f2}


PROBLEMS: dict[str, type[Problem]] = {
    cls.name: cls
    for cls in (
        GasolineEngineProblem,
        BraninProblem,
        RosenbrockProblem,
        ZDT1Problem,
    )
}


def list_problems() -> list[tuple[str, str]]:
    """``(name, title)`` for every registered problem, for populating a combo box."""
    return [(name, cls.title) for name, cls in PROBLEMS.items()]


def get_problem(name: str) -> Problem:
    """Instantiate a registered problem by name."""
    try:
        return PROBLEMS[name]()
    except KeyError:
        raise KeyError(
            f"unknown problem {name!r}; available: {sorted(PROBLEMS)}"
        ) from None
