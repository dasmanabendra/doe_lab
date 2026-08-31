"""NSGA-II driving, run controls, and Pareto handling."""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from doelab.engine import doe, metamodel as mm, optimize as opt
from doelab.engine.factors import FactorSpace, ResponseRole
from doelab.engine.solver import get_problem


class ExactModel:
    """Stands in for a metamodel by calling the true function.

    Optimizer correctness and surrogate accuracy are separate concerns, and
    conflating them makes failures ambiguous: a front that misses the analytic
    optimum could mean either the search is broken or the surrogate is
    extrapolating. Substituting an exact model removes the second explanation,
    so these tests fail only when the search itself is wrong.
    """

    def __init__(self, problem, response: str):
        self.problem = problem
        self.response = response

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.problem.compute(frame)[self.response]


@pytest.fixture
def zdt1_exact():
    problem = get_problem("zdt1")
    space = FactorSpace(problem.make_factors())
    responses = problem.make_responses()
    models = {r.name: ExactModel(problem, r.name) for r in responses}
    return space, responses, models


@pytest.fixture
def engine_surrogates(engine_problem, engine_space):
    """Kriging surrogates for every non-ignored engine response."""
    design = doe.latin_hypercube(engine_space, 150, seed=5)
    results = engine_problem.evaluate(design)
    responses = engine_problem.make_responses()
    models = {
        r.name: mm.fit_metamodel(engine_space, design, results, mm.MetamodelSpec(r.name, "kriging"))
        for r in responses
        if r.role is not ResponseRole.IGNORED
    }
    return engine_space, responses, models


class TestConvergence:
    def test_reaches_the_known_analytic_pareto_front(self, zdt1_exact):
        """ZDT1's front is f2 = 1 - sqrt(f1); the search must land on it."""
        space, responses, models = zdt1_exact
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=60, n_generations=120, seed=1)
        )
        front = run.execute().pareto

        error = np.abs(front["f2"] - (1 - np.sqrt(np.clip(front["f1"], 0, 1))))
        assert error.mean() < 0.05

    def test_spreads_along_the_front_rather_than_clustering(self, zdt1_exact):
        space, responses, models = zdt1_exact
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=60, n_generations=120, seed=1)
        )
        front = run.execute().pareto

        assert front["f1"].min() < 0.15
        assert front["f1"].max() > 0.85

    def test_hypervolume_never_decreases(self, zdt1_exact):
        """Elitist selection plus a fixed reference point implies monotonicity."""
        space, responses, models = zdt1_exact
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=30, n_generations=25, seed=2)
        )
        updates: list[opt.GenerationUpdate] = []
        run.execute(on_generation=updates.append)

        indicators = [u.indicator for u in updates]
        assert all(b >= a - 1e-9 for a, b in zip(indicators, indicators[1:]))
        assert indicators[-1] > indicators[0]

    def test_single_objective_reports_the_best_value(self):
        problem = get_problem("branin")
        space = FactorSpace(problem.make_factors())
        responses = problem.make_responses()
        models = {"f": ExactModel(problem, "f")}

        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=40, n_generations=40, seed=3)
        )
        result = run.execute()

        assert result.indicator_name == "Best f"
        # Branin's three global minima all sit at 0.397887.
        assert result.designs["f"].min() == pytest.approx(0.397887, abs=0.01)


class TestConstraints:
    def test_no_design_on_the_front_violates_a_constraint(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=50, n_generations=30, seed=2)
        )
        front = run.execute().pareto

        limit = next(r.upper for r in responses if r.name == "Max_Cyl_Pressure")
        assert front["Max_Cyl_Pressure"].max() <= limit + 1e-6

    def test_the_constraint_actually_binds(self, engine_surrogates):
        """A constraint that never activates would not be testing anything."""
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=50, n_generations=30, seed=2)
        )
        front = run.execute().pareto

        limit = next(r.upper for r in responses if r.name == "Max_Cyl_Pressure")
        assert (front["Max_Cyl_Pressure"] > limit - 1.0).any()

    def test_counts_two_sided_bounds_as_two_inequalities(self):
        from doelab.engine.factors import Response

        both = Response("p", role=ResponseRole.CONSTRAINT, lower=1.0, upper=2.0)
        one = Response("q", role=ResponseRole.CONSTRAINT, upper=2.0)
        assert opt.count_constraint_terms([both, one]) == 3


class TestMixedVariables:
    def test_categorical_factors_keep_valid_levels_throughout(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=30, n_generations=15, seed=4)
        )
        designs = run.execute().designs

        assert set(designs["Fuel_Type"]) <= set(space["Fuel_Type"].categories)

    def test_continuous_factors_stay_within_bounds(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=30, n_generations=15, seed=4)
        )
        designs = run.execute().designs

        for factor in space.continuous:
            assert designs[factor.name].between(factor.low, factor.high).all()

    def test_maximized_objectives_are_reported_unnegated(self, engine_surrogates):
        """Torque is maximized internally as -Torque; the table must not show that."""
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=30, n_generations=10, seed=5)
        )
        designs = run.execute().designs

        assert (designs["Torque"] > 0).all()


class TestControls:
    def test_stop_takes_effect_at_the_requested_generation(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=20, n_generations=200, seed=6)
        )

        def halt(update: opt.GenerationUpdate) -> None:
            if update.generation == 8:
                run.stop()

        result = run.execute(on_generation=halt)

        assert result.generations_completed == 8
        assert result.stopped_early

    def test_pause_blocks_the_loop_until_resumed(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=20, n_generations=12, seed=7)
        )
        pause_for = 0.35

        def hold(update: opt.GenerationUpdate) -> None:
            if update.generation == 3:
                run.pause()
                threading.Timer(pause_for, run.resume).start()

        started = time.perf_counter()
        run.execute(on_generation=hold)

        assert time.perf_counter() - started >= pause_for

    def test_stop_releases_a_paused_run(self, engine_surrogates):
        """Otherwise stopping a paused run would deadlock the worker thread."""
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=20, n_generations=100, seed=8)
        )

        def pause_then_stop(update: opt.GenerationUpdate) -> None:
            if update.generation == 3:
                run.pause()
                threading.Timer(0.1, run.stop).start()

        finished = threading.Event()

        def drive() -> None:
            run.execute(on_generation=pause_then_stop)
            finished.set()

        threading.Thread(target=drive, daemon=True).start()
        assert finished.wait(timeout=15), "a paused run did not observe stop()"

    def test_extend_continues_from_where_the_run_stopped(self, engine_surrogates):
        """Not from the original budget: stopping at 8 of 200 then extending by 5 means 13."""
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=20, n_generations=200, seed=9)
        )

        def halt(update: opt.GenerationUpdate) -> None:
            if update.generation == 8:
                run.stop()

        first = run.execute(on_generation=halt)
        assert first.generations_completed == 8

        run.extend(5)
        second = run.execute()

        assert second.generations_completed == 13
        assert not second.stopped_early
        assert len(second.designs) > len(first.designs)

    def test_extend_rejects_a_non_positive_budget(self, engine_surrogates):
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(space, responses, models, opt.OptimizationConfig(n_generations=1))
        with pytest.raises(ValueError, match="positive"):
            run.extend(0)


class TestSetupValidation:
    def test_requires_at_least_one_objective(self, engine_surrogates):
        space, responses, models = engine_surrogates
        ignored = [r for r in responses if not r.role.is_objective]

        with pytest.raises(ValueError, match="at least one response"):
            opt.OptimizationRun(space, ignored, models)

    def test_requires_a_metamodel_for_every_optimized_response(self, engine_surrogates):
        space, responses, models = engine_surrogates
        incomplete = {k: v for k, v in models.items() if k != "Torque"}

        with pytest.raises(ValueError, match="Torque"):
            opt.OptimizationRun(space, responses, incomplete)


class TestParetoFront:
    def test_keeps_only_non_dominated_designs(self):
        from doelab.engine.factors import Response

        objectives = [
            Response("a", role=ResponseRole.OBJECTIVE_MIN),
            Response("b", role=ResponseRole.OBJECTIVE_MIN),
        ]
        designs = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 2.0],
                "b": [3.0, 2.0, 1.0, 3.0],  # the last row is dominated by row 1
                "Feasible": [True, True, True, True],
            }
        )
        front = opt.pareto_front(designs, objectives)

        assert len(front) == 3
        assert 2.0 not in front[front["b"] == 3.0]["a"].tolist()

    def test_excludes_infeasible_designs(self):
        from doelab.engine.factors import Response

        objectives = [Response("a", role=ResponseRole.OBJECTIVE_MIN)]
        designs = pd.DataFrame({"a": [0.0, 5.0], "Feasible": [False, True]})

        front = opt.pareto_front(designs, objectives)
        assert front["a"].tolist() == [5.0]


class TestValidation:
    def test_reports_zero_error_where_the_surrogate_is_exact(self, zdt1_exact):
        space, responses, models = zdt1_exact
        problem = get_problem("zdt1")
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=30, n_generations=20, seed=1)
        )
        front = run.execute().pareto

        validated = opt.validate_pareto(problem, space, front, responses)
        summary = opt.validation_summary(validated, responses)

        assert (summary["Max |error|"] < 1e-9).all()

    def test_exposes_surrogate_error_on_a_fitted_model(self, engine_problem, engine_surrogates):
        """The point of validation: surface the gap a metamodel cannot self-report."""
        space, responses, models = engine_surrogates
        run = opt.OptimizationRun(
            space, responses, models, opt.OptimizationConfig(pop_size=40, n_generations=20, seed=2)
        )
        front = run.execute().pareto

        validated = opt.validate_pareto(engine_problem, space, front, responses)

        assert "BSFC_predicted" in validated
        assert "BSFC_actual" in validated
        assert "BSFC_error" in validated
        assert len(validated) == len(front)
