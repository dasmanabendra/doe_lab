"""Multi-objective optimization over fitted metamodels.

The optimizer searches the *surrogate*, not the solver. That is the point of
building metamodels: a genetic algorithm needs tens of thousands of
evaluations, which is only tractable against something that predicts in
microseconds.

The generation loop is driven explicitly rather than through pymoo's
``minimize()`` helper. Owning the loop is what makes Pause, Stop and Extend
real controls instead of cosmetic ones: the run checks its control flags
between generations, and extending simply raises the target and keeps going
from the population already in hand.

Concurrency note: this module is deliberately Qt-free. Control flags are
``threading.Event`` objects, and progress is delivered by callback, so a UI can
run this on a worker thread and marshal updates however it likes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination
from pymoo.core.variable import Choice, Real
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .factors import FactorSpace, Response, ResponseRole
from .metamodel import Metamodel


@dataclass
class OptimizationConfig:
    """Algorithm settings for a run."""

    pop_size: int = 40
    n_generations: int = 40
    seed: int | None = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "pop_size": self.pop_size,
            "n_generations": self.n_generations,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationConfig:
        return cls(**data)


@dataclass
class GenerationUpdate:
    """Snapshot emitted after each generation, for live display."""

    generation: int
    n_generations: int
    n_evaluations: int
    elapsed: float
    indicator: float
    indicator_name: str
    n_feasible: int
    population: pd.DataFrame = field(repr=False)
    pareto: pd.DataFrame = field(repr=False)


@dataclass
class OptimizationResult:
    """Everything a finished (or interrupted) run produced."""

    designs: pd.DataFrame
    pareto: pd.DataFrame
    history: pd.DataFrame
    generations_completed: int
    stopped_early: bool
    indicator_name: str


class _SurrogateProblem(PymooProblem):
    """Wraps the fitted metamodels as a pymoo problem.

    Mixed variables are declared through pymoo's ``vars`` interface, so
    continuous factors become ``Real`` and categorical factors become
    ``Choice`` — the genetic operators then handle each type natively instead
    of forcing categories through a continuous encoding.

    Evaluation is batched rather than elementwise: pymoo hands over the whole
    population, which becomes a single DataFrame and a single vectorized
    ``predict`` per metamodel.
    """

    def __init__(
        self,
        space: FactorSpace,
        objectives: list[Response],
        constraints: list[Response],
        models: dict[str, Metamodel],
    ):
        variables: dict[str, Any] = {}
        for f in space:
            if f.is_categorical:
                variables[f.name] = Choice(options=list(f.categories))
            else:
                variables[f.name] = Real(bounds=(f.low, f.high))

        super().__init__(vars=variables, n_obj=len(objectives), n_ieq_constr=len(constraints))
        self.space = space
        self.objectives = objectives
        self.constraints = constraints
        self.models = models

    def _to_frame(self, X: Any) -> pd.DataFrame:
        """Turn pymoo's population representation into a factor DataFrame."""
        rows = list(X) if not isinstance(X, np.ndarray) else list(X.ravel())
        frame = pd.DataFrame(list(rows))
        for f in self.space:
            frame[f.name] = (
                frame[f.name].astype(str)
                if f.is_categorical
                else frame[f.name].astype(float)
            )
        return frame[self.space.names]

    def _evaluate(self, X: Any, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        frame = self._to_frame(X)

        objective_columns = []
        for response in self.objectives:
            prediction = self.models[response.name].predict(frame)
            # pymoo minimizes; maximizing is the same search on the negation.
            if response.role is ResponseRole.OBJECTIVE_MAX:
                prediction = -prediction
            objective_columns.append(prediction)
        out["F"] = np.column_stack(objective_columns)

        if self.constraints:
            constraint_columns = []
            for response in self.constraints:
                prediction = self.models[response.name].predict(frame)
                # pymoo's convention is g(x) <= 0 for a satisfied constraint.
                if response.upper is not None:
                    constraint_columns.append(prediction - response.upper)
                if response.lower is not None:
                    constraint_columns.append(response.lower - prediction)
            out["G"] = np.column_stack(constraint_columns)


def split_roles(responses: list[Response]) -> tuple[list[Response], list[Response]]:
    """Partition responses into objectives and constraints by their role."""
    objectives = [r for r in responses if r.role.is_objective]
    constraints = [r for r in responses if r.role is ResponseRole.CONSTRAINT]
    return objectives, constraints


def count_constraint_terms(constraints: list[Response]) -> int:
    """A two-sided constraint contributes two inequalities, not one."""
    return sum(
        (1 if r.upper is not None else 0) + (1 if r.lower is not None else 0)
        for r in constraints
    )


class OptimizationRun:
    """A controllable NSGA-II run against the metamodels.

    Typical use from a worker thread::

        run = OptimizationRun(space, responses, models, config)
        result = run.execute(on_generation=emit)

    while another thread calls :meth:`pause`, :meth:`resume`, :meth:`stop` or
    :meth:`extend`.
    """

    def __init__(
        self,
        space: FactorSpace,
        responses: list[Response],
        models: dict[str, Metamodel],
        config: OptimizationConfig | None = None,
    ):
        self.space = space
        self.config = config or OptimizationConfig()
        self.objectives, self.constraints = split_roles(responses)

        if not self.objectives:
            raise ValueError(
                "optimization needs at least one response with an objective role"
            )

        needed = {r.name for r in self.objectives} | {r.name for r in self.constraints}
        missing = sorted(needed - set(models))
        if missing:
            raise ValueError(
                "no metamodel fitted for: " + ", ".join(missing) +
                ". Fit a metamodel for every objective and constraint first."
            )
        self.models = {name: models[name] for name in needed}

        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()  # cleared while paused

        self._target_generations = self.config.n_generations
        self._generation = 0
        self._n_evaluations = 0
        self._designs: list[pd.DataFrame] = []
        self._history: list[dict[str, Any]] = []
        self._reference_point: np.ndarray | None = None
        self._started: float | None = None

        self._problem = _SurrogateProblem(
            space, self.objectives, self.constraints, self.models
        )
        self._algorithm = NSGA2(
            pop_size=self.config.pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )
        self._algorithm.setup(
            self._problem,
            termination=NoTermination(),
            seed=self.config.seed,
            verbose=False,
        )

    # -- controls -----------------------------------------------------------

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        """Ask the run to finish after the current generation."""
        self._stop.set()
        self._running.set()  # so a paused run can observe the stop and exit

    def extend(self, extra_generations: int) -> None:
        """Run this many further generations from wherever the run now stands.

        The budget is measured from the current generation, not added to the
        original target. After an early stop those differ: a run halted at
        generation 12 of 200 and then extended by 5 should do 5 more
        generations, not 193.
        """
        if extra_generations < 1:
            raise ValueError("extend needs a positive number of generations")
        self._target_generations = self._generation + extra_generations
        self._stop.clear()

    @property
    def is_paused(self) -> bool:
        return not self._running.is_set()

    @property
    def generation(self) -> int:
        return self._generation

    # -- execution ----------------------------------------------------------

    def execute(
        self, on_generation: Callable[[GenerationUpdate], None] | None = None
    ) -> OptimizationResult:
        """Run generations until the budget is met or the run is stopped."""
        if self._started is None:
            self._started = time.perf_counter()

        while self._generation < self._target_generations:
            # Blocks while paused; stop() releases it so the flag is observed.
            self._running.wait()
            if self._stop.is_set():
                break

            self._algorithm.next()
            self._generation += 1

            update = self._snapshot()
            if on_generation is not None:
                on_generation(update)

        return self._result()

    def _snapshot(self) -> GenerationUpdate:
        population = self._population_frame()
        self._designs.append(population)
        self._n_evaluations += len(population)

        all_designs = pd.concat(self._designs, ignore_index=True)
        pareto = pareto_front(all_designs, self.objectives)
        indicator, indicator_name = self._indicator(all_designs)

        elapsed = time.perf_counter() - (self._started or time.perf_counter())
        n_feasible = int(all_designs["Feasible"].sum())

        self._history.append(
            {
                "Generation": self._generation,
                "Evaluations": self._n_evaluations,
                indicator_name: indicator,
                "Feasible": n_feasible,
                "Elapsed": elapsed,
            }
        )

        return GenerationUpdate(
            generation=self._generation,
            n_generations=self._target_generations,
            n_evaluations=self._n_evaluations,
            elapsed=elapsed,
            indicator=indicator,
            indicator_name=indicator_name,
            n_feasible=n_feasible,
            population=population,
            pareto=pareto,
        )

    def _population_frame(self) -> pd.DataFrame:
        """Current population as factor values, response values and feasibility."""
        pop = self._algorithm.pop
        frame = self._problem._to_frame(np.array([ind.X for ind in pop], dtype=object))

        F = np.atleast_2d(np.array([ind.F for ind in pop], dtype=float))
        for i, response in enumerate(self.objectives):
            # Undo the sign flip used to turn maximization into minimization,
            # so the table shows the response as the user defined it.
            values = F[:, i]
            frame[response.name] = -values if response.role is ResponseRole.OBJECTIVE_MAX else values

        if self.constraints:
            for response in self.constraints:
                frame[response.name] = self.models[response.name].predict(frame)
            G = np.atleast_2d(np.array([ind.G for ind in pop], dtype=float))
            frame["Feasible"] = (G <= 1e-9).all(axis=1)
        else:
            frame["Feasible"] = True

        frame.insert(0, "Generation", self._generation)
        return frame

    def _indicator(self, designs: pd.DataFrame) -> tuple[float, str]:
        """Convergence measure: hypervolume for many objectives, best value for one.

        The hypervolume reference point is fixed from the first generation's
        worst objective values. Recomputing it each generation would let the
        reference drift and make the trace non-monotonic, which would defeat
        its purpose as a convergence signal.
        """
        feasible = designs[designs["Feasible"]]
        source = feasible if len(feasible) else designs

        matrix = self._objective_matrix(source)
        if matrix.size == 0:
            return float("nan"), "Hypervolume"

        if len(self.objectives) == 1:
            best = float(matrix[:, 0].min())
            response = self.objectives[0]
            if response.role is ResponseRole.OBJECTIVE_MAX:
                best = -best
            return best, f"Best {response.name}"

        if self._reference_point is None:
            self._reference_point = matrix.max(axis=0) * 1.1 + 1e-9

        from pymoo.indicators.hv import HV

        hv = HV(ref_point=self._reference_point)
        return float(hv(matrix)), "Hypervolume"

    def _objective_matrix(self, designs: pd.DataFrame) -> np.ndarray:
        """Objectives in minimization form, as pymoo indicators expect."""
        if designs.empty:
            return np.empty((0, len(self.objectives)))
        columns = []
        for response in self.objectives:
            values = designs[response.name].to_numpy(dtype=float)
            columns.append(-values if response.role is ResponseRole.OBJECTIVE_MAX else values)
        return np.column_stack(columns)

    def _result(self) -> OptimizationResult:
        designs = (
            pd.concat(self._designs, ignore_index=True)
            if self._designs
            else pd.DataFrame()
        )
        history = pd.DataFrame(self._history)
        indicator_name = (
            "Hypervolume" if len(self.objectives) > 1
            else f"Best {self.objectives[0].name}"
        )
        return OptimizationResult(
            designs=designs,
            pareto=pareto_front(designs, self.objectives) if len(designs) else pd.DataFrame(),
            history=history,
            generations_completed=self._generation,
            stopped_early=self._stop.is_set(),
            indicator_name=indicator_name,
        )


def validate_pareto(
    problem: Any,
    space: FactorSpace,
    pareto: pd.DataFrame,
    responses: list[Response],
    noise: Any = None,
) -> pd.DataFrame:
    """Re-run the true solver on the Pareto designs and compare.

    An optimizer run against a surrogate finds the optimum *of the surrogate*.
    Where the search pushes into regions the design sampled thinly — typically
    the corners of the space, which is exactly where optima like to sit — the
    metamodel is extrapolating, and its optimum can be some distance from the
    real one. Cross-validation will not reveal this, because it only scores the
    surrogate where training data already exists.

    Re-evaluating the handful of Pareto designs is cheap next to the thousands
    of surrogate evaluations the search consumed, and it is the only honest way
    to know whether the front is real.

    Returns the front with ``<response>_predicted``, ``<response>_actual`` and
    ``<response>_error`` columns added.
    """
    if pareto.empty:
        return pareto

    factors = pareto[space.names].reset_index(drop=True)
    actual = problem.evaluate(factors, noise)

    out = factors.copy()
    for response in responses:
        name = response.name
        if name not in pareto.columns or name not in actual.columns:
            continue
        predicted = pareto[name].to_numpy(dtype=float)
        truth = actual[name].to_numpy(dtype=float)
        out[f"{name}_predicted"] = predicted
        out[f"{name}_actual"] = truth
        out[f"{name}_error"] = truth - predicted
    return out


def validation_summary(validated: pd.DataFrame, responses: list[Response]) -> pd.DataFrame:
    """Per-response error statistics from :func:`validate_pareto`."""
    rows = []
    for response in responses:
        column = f"{response.name}_error"
        if column not in validated:
            continue
        errors = validated[column].to_numpy(dtype=float)
        actual = validated[f"{response.name}_actual"].to_numpy(dtype=float)
        spread = float(np.ptp(actual)) if len(actual) > 1 else 0.0
        rows.append(
            {
                "Response": response.name,
                "Mean error": float(errors.mean()),
                "Mean |error|": float(np.abs(errors).mean()),
                "Max |error|": float(np.abs(errors).max()),
                "RMSE": float(np.sqrt((errors**2).mean())),
                "RMSE % of range": (
                    float(np.sqrt((errors**2).mean()) / spread * 100) if spread > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def pareto_front(designs: pd.DataFrame, objectives: list[Response]) -> pd.DataFrame:
    """Non-dominated, feasible designs, deduplicated.

    Only feasible designs are eligible: a design that violates a constraint is
    not a trade-off worth showing, however good its objectives look.
    """
    if designs.empty:
        return designs

    feasible = designs[designs["Feasible"]] if "Feasible" in designs else designs
    if feasible.empty:
        return feasible

    columns = []
    for response in objectives:
        values = feasible[response.name].to_numpy(dtype=float)
        columns.append(-values if response.role is ResponseRole.OBJECTIVE_MAX else values)
    matrix = np.column_stack(columns)

    indices = NonDominatedSorting().do(matrix, only_non_dominated_front=True)
    front = feasible.iloc[indices]

    names = [r.name for r in objectives]
    return front.drop_duplicates(subset=names).sort_values(names[0]).reset_index(drop=True)
