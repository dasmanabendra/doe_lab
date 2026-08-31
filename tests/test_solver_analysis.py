"""The analytic problems, and the statistics computed over their results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doelab.engine import analysis, doe
from doelab.engine.factors import ContinuousFactor, FactorSpace
from doelab.engine.solver import NoiseConfig, get_problem, list_problems


class TestProblems:
    def test_every_registered_problem_evaluates(self):
        for name, _ in list_problems():
            problem = get_problem(name)
            space = FactorSpace(problem.make_factors())
            design = doe.latin_hypercube(space, 20, seed=0)
            results = problem.evaluate(design)

            assert len(results) == 20
            assert np.isfinite(results.to_numpy(dtype=float)).all()
            assert list(results.columns) == [r.name for r in problem.make_responses()]

    def test_rejects_an_unknown_problem(self):
        with pytest.raises(KeyError, match="unknown problem"):
            get_problem("nope")

    def test_branin_matches_its_published_minimum(self):
        """Three global minima, all at f = 0.397887."""
        problem = get_problem("branin")
        known = pd.DataFrame({"x1": [-np.pi, np.pi, 9.42478], "x2": [12.275, 2.275, 2.475]})

        assert problem.evaluate(known)["f"].to_numpy() == pytest.approx(0.397887, abs=1e-4)

    def test_rosenbrock_vanishes_at_its_optimum(self):
        problem = get_problem("rosenbrock")
        assert problem.evaluate(pd.DataFrame({"x1": [1.0], "x2": [1.0]}))["f"][0] == pytest.approx(0.0)

    def test_zdt1_front_satisfies_its_analytic_identity(self):
        """With x2..xn = 0 the front is exactly f2 = 1 - sqrt(f1)."""
        problem = get_problem("zdt1")
        n = problem.n_vars
        front = pd.DataFrame(
            {f"x{i + 1}": ([0.0, 0.25, 0.5, 0.75, 1.0] if i == 0 else [0.0] * 5) for i in range(n)}
        )
        out = problem.evaluate(front)

        assert out["f2"].to_numpy() == pytest.approx(1 - np.sqrt(out["f1"].to_numpy()))


class TestEngineProblem:
    def test_power_is_consistent_with_torque_and_speed(self, engine_study):
        """P = T*omega is a physical identity, not an independent fit."""
        _, design, results, _ = engine_study
        omega = design["RPM"].to_numpy() * 2 * np.pi / 60

        assert results["Power"].to_numpy() == pytest.approx(
            results["Torque"].to_numpy() * omega / 1000
        )

    def test_bsfc_stays_in_a_physically_plausible_band(self, engine_study):
        """A reciprocal formulation used to blow this past 1600 g/kW-h."""
        _, _, results, _ = engine_study
        assert results["BSFC"].min() > 150
        assert results["BSFC"].max() < 600

    def test_e85_costs_fuel_economy(self, engine_study):
        """Lower energy density means more fuel mass for the same work."""
        _, design, results, _ = engine_study
        by_fuel = results.assign(fuel=design["Fuel_Type"]).groupby("fuel")["BSFC"].mean()

        assert by_fuel["E85"] > by_fuel["Regular"]

    def test_torque_peaks_away_from_the_speed_extremes(self, engine_problem, engine_space):
        centre = engine_space.midpoint()
        sweep = pd.DataFrame({**{k: [v] * 30 for k, v in centre.items()}})
        sweep["RPM"] = np.linspace(1000, 5000, 30)
        torque = engine_problem.evaluate(sweep)["Torque"].to_numpy()

        assert torque.argmax() not in (0, len(torque) - 1)


class TestNoise:
    def test_disabled_by_default(self, engine_problem, engine_space):
        design = doe.latin_hypercube(engine_space, 30, seed=0)
        assert engine_problem.evaluate(design).equals(engine_problem.evaluate(design))

    def test_is_reproducible_for_a_seed(self, engine_problem, engine_space):
        design = doe.latin_hypercube(engine_space, 30, seed=0)
        config = NoiseConfig(enabled=True, sigma=0.02, seed=3)

        assert engine_problem.evaluate(design, config).equals(
            engine_problem.evaluate(design, config)
        )

    def test_realized_spread_matches_the_requested_sigma(self, engine_problem, engine_space):
        design = doe.latin_hypercube(engine_space, 2000, seed=0)
        clean = engine_problem.evaluate(design)
        noisy = engine_problem.evaluate(design, NoiseConfig(enabled=True, sigma=0.02, seed=1))

        relative = ((noisy["BSFC"] - clean["BSFC"]) / clean["BSFC"]).std()
        assert relative == pytest.approx(0.02, rel=0.15)


class TestFactorSensitivity:
    def test_shares_match_the_known_coefficients_of_a_linear_response(self):
        """For y = 10a + 2b, shares should be 100/104 and 4/104."""
        space = FactorSpace([ContinuousFactor(n, 0.0, 1.0) for n in ("a", "b", "c")])
        design = doe.latin_hypercube(space, 400, seed=0)
        response = pd.DataFrame({"y": 10 * design["a"] + 2 * design["b"]})

        shares = analysis.factor_sensitivity(space, design, response).loc["y"]

        assert shares["a"] == pytest.approx(100 / 104, abs=0.02)
        assert shares["b"] == pytest.approx(4 / 104, abs=0.02)
        assert shares["c"] == pytest.approx(0.0, abs=0.02)

    def test_shares_sum_to_one(self, engine_study):
        space, design, results, _ = engine_study
        table = analysis.factor_sensitivity(space, design, results)

        assert table.sum(axis=1).to_numpy() == pytest.approx(1.0)

    def test_attributes_cylinder_pressure_mainly_to_spark_timing(self, engine_study):
        space, design, results, _ = engine_study
        shares = analysis.factor_sensitivity(space, design, results).loc["Max_Cyl_Pressure"]

        assert shares.idxmax() == "Spark_Timing"

    def test_categorical_indicators_are_pooled_into_one_factor(self, engine_study):
        space, design, results, _ = engine_study
        table = analysis.factor_sensitivity(space, design, results)

        assert list(table.columns) == space.names
        assert table.loc["BSFC", "Fuel_Type"] > 0.5  # E85 dominates BSFC

    def test_is_blind_to_a_symmetric_quadratic_effect(self):
        """A documented limitation, asserted so it cannot regress silently."""
        space = FactorSpace([ContinuousFactor("a", 0.0, 1.0), ContinuousFactor("b", 0.0, 1.0)])
        design = doe.latin_hypercube(space, 300, seed=0)
        response = pd.DataFrame({"y": (design["a"] - 0.5) ** 2})

        shares = analysis.factor_sensitivity(space, design, response).loc["y"]
        assert shares["a"] < 0.2

    def test_rejects_a_single_experiment(self, engine_space):
        design = doe.latin_hypercube(engine_space, 1, seed=0)
        with pytest.raises(ValueError, match="at least two"):
            analysis.factor_sensitivity(engine_space, design, pd.DataFrame({"y": [1.0]}))


class TestVarianceDecomposition:
    """Partial and semi-partial R² — the un-normalized companion to sensitivity."""

    def test_reports_the_same_r_squared_as_a_direct_least_squares_fit(self, engine_study):
        space, design, results, _ = engine_study
        X, _ = space.coded_matrix(design, drop_first=True)
        A = np.column_stack([np.ones(len(X)), X])

        model = analysis.variance_decomposition(space, design, results).model

        for response in results.columns:
            y = results[response].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            expected = 1.0 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            assert model.at[response, "r_squared"] == pytest.approx(expected)

    def test_the_three_parts_partition_the_response(self, engine_study):
        """unique + shared + unexplained == 1, exactly, for every response."""
        space, design, results, _ = engine_study
        model = analysis.variance_decomposition(space, design, results).model

        totals = model[["unique", "shared", "unexplained"]].sum(axis=1)
        assert totals.to_numpy() == pytest.approx(1.0)

    def test_partial_is_never_below_semi_partial(self, engine_study):
        """Both share a numerator; the partial divides by the smaller total."""
        space, design, results, _ = engine_study
        decomposition = analysis.variance_decomposition(space, design, results)

        difference = decomposition.partial - decomposition.semi_partial
        assert difference.to_numpy().min() >= -1e-12

    def test_an_orthogonal_design_leaves_nothing_shared(self):
        """With uncorrelated factors every factor's contribution is its own."""
        space = FactorSpace([ContinuousFactor(n, 0.0, 1.0) for n in ("a", "b", "c")])
        design = doe.full_factorial(space)
        response = pd.DataFrame(
            {"y": 10 * design["a"] + 2 * design["b"]}, index=design.index
        )

        decomposition = analysis.variance_decomposition(space, design, response)

        assert decomposition.model.at["y", "r_squared"] == pytest.approx(1.0)
        assert decomposition.model.at["y", "shared"] == pytest.approx(0.0, abs=1e-9)
        assert decomposition.partial.at["y", "c"] == pytest.approx(0.0, abs=1e-9)

    def test_confounded_factors_score_low_apart_and_high_together(self):
        """The finding sensitivity cannot report.

        ``b`` is a near-copy of ``a`` and has no effect of its own. Because the
        two carry the same information, neither can claim much *uniquely* — but
        the model still explains the response completely, and the difference
        lands in ``shared``.
        """
        rng = np.random.default_rng(0)
        a = rng.uniform(0.0, 1.0, 200)
        frame = pd.DataFrame({"a": a, "b": a + rng.normal(0.0, 0.02, 200)})
        space = FactorSpace([ContinuousFactor("a", 0.0, 1.0), ContinuousFactor("b", 0.0, 1.0)])
        response = pd.DataFrame({"y": 3 * a})

        decomposition = analysis.variance_decomposition(space, frame, response)

        assert decomposition.model.at["y", "r_squared"] > 0.99
        assert decomposition.model.at["y", "shared"] > 0.9
        assert decomposition.semi_partial.at["y", "a"] < 0.1
        assert decomposition.semi_partial.at["y", "b"] < 0.1

    def test_drops_a_categorical_factor_whole_rather_than_by_level(self, engine_study):
        """Fuel_Type contributes two indicator columns; both must go together.

        Dropping one level at a time would understate the factor, since the
        remaining indicator still carries part of the same contrast.
        """
        space, design, results, _ = engine_study
        decomposition = analysis.variance_decomposition(space, design, results)

        assert list(decomposition.partial.columns) == space.names
        assert decomposition.semi_partial.at["BSFC", "Fuel_Type"] > 0.5  # E85 dominates

    def test_a_constant_response_yields_no_shares_and_no_fit(self, engine_space):
        """R² is undefined without variation — NaN, not a fabricated zero."""
        design = doe.latin_hypercube(engine_space, 30, seed=0)
        response = pd.DataFrame({"flat": np.full(30, 7.0)}, index=design.index)

        decomposition = analysis.variance_decomposition(engine_space, design, response)

        assert decomposition.partial.loc["flat"].to_numpy() == pytest.approx(0.0)
        assert np.isnan(decomposition.model.at["flat", "r_squared"])

    def test_adjusted_r_squared_is_undefined_without_spare_runs(self, mixed_space):
        """Four terms plus an intercept in five runs leaves nothing to penalize."""
        design = doe.latin_hypercube(mixed_space, 5, seed=0)
        response = pd.DataFrame({"y": np.arange(5.0)}, index=design.index)

        model = analysis.variance_decomposition(mixed_space, design, response).model

        assert np.isnan(model.at["y", "adjusted_r_squared"])
        assert np.isfinite(model.at["y", "r_squared"])

    def test_every_figure_is_finite_and_a_fraction(self, engine_study):
        space, design, results, _ = engine_study
        decomposition = analysis.variance_decomposition(space, design, results)

        for table in (decomposition.partial, decomposition.semi_partial):
            values = table.to_numpy()
            assert np.isfinite(values).all()
            assert values.min() >= 0.0
            assert values.max() <= 1.0

    def test_rejects_a_single_experiment(self, engine_space):
        design = doe.latin_hypercube(engine_space, 1, seed=0)
        with pytest.raises(ValueError, match="at least two"):
            analysis.variance_decomposition(engine_space, design, pd.DataFrame({"y": [1.0]}))


class TestCorrelation:
    @pytest.mark.parametrize("method", ["pearson", "spearman"])
    def test_is_symmetric_with_a_unit_diagonal(self, engine_study, method):
        space, design, results, _ = engine_study
        corr = analysis.correlation_matrix(space, design, results, method).to_numpy()

        assert corr == pytest.approx(corr.T)
        assert np.diag(corr) == pytest.approx(1.0)

    def test_contains_no_missing_values(self, engine_study):
        space, design, results, _ = engine_study
        corr = analysis.correlation_matrix(space, design, results)

        assert not corr.isna().to_numpy().any()

    def test_spearman_detects_a_monotonic_relationship_pearson_understates(self):
        """The reason both are reported."""
        space = FactorSpace([ContinuousFactor("a", 0.0, 1.0), ContinuousFactor("b", 0.0, 1.0)])
        design = doe.latin_hypercube(space, 200, seed=0)
        response = pd.DataFrame({"y": np.exp(6 * design["a"])})  # monotonic, strongly curved

        pearson = analysis.correlation_matrix(space, design, response, "pearson").loc["a", "y"]
        spearman = analysis.correlation_matrix(space, design, response, "spearman").loc["a", "y"]

        assert spearman > 0.99
        assert spearman > pearson

    def test_rejects_an_unknown_method(self, engine_study):
        space, design, results, _ = engine_study
        with pytest.raises(ValueError, match="unknown correlation method"):
            analysis.correlation_matrix(space, design, results, "kendall")

    def test_splits_labels_into_factor_and_response_blocks(self, engine_study):
        space, design, results, _ = engine_study
        corr = analysis.correlation_matrix(space, design, results)
        factor_labels, response_labels = analysis.split_labels(space, corr)

        assert "Fuel_Type=E85" in factor_labels
        assert set(response_labels) == set(results.columns)
