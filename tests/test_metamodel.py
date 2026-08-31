"""Surrogate fitting, cross-validation, and prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doelab.engine import doe, metamodel as mm
from doelab.engine.factors import ContinuousFactor, FactorSpace
from doelab.engine.solver import NoiseConfig


@pytest.fixture
def quadratic_study():
    """A response that is exactly quadratic in its two factors."""
    space = FactorSpace([ContinuousFactor("a", -1.0, 1.0), ContinuousFactor("b", -1.0, 1.0)])
    design = doe.latin_hypercube(space, 60, seed=0)
    response = pd.DataFrame(
        {"y": 3 + 2 * design["a"] - 1.5 * design["b"] + 4 * design["a"] ** 2
         + 0.7 * design["a"] * design["b"]}
    )
    return space, design, response


class TestFitQuality:
    def test_quadratic_recovers_a_quadratic_exactly(self, quadratic_study):
        space, design, response = quadratic_study
        model = mm.fit_metamodel(space, design, response, mm.MetamodelSpec("y", "quadratic"))

        assert model.metrics.r2 == pytest.approx(1.0, abs=1e-9)
        assert model.metrics.cv_r2 == pytest.approx(1.0, abs=1e-6)

    def test_linear_cannot_fit_a_curved_surface(self, quadratic_study):
        """The baseline must actually fail, or it says nothing about the others."""
        space, design, response = quadratic_study
        model = mm.fit_metamodel(space, design, response, mm.MetamodelSpec("y", "linear"))

        assert model.metrics.r2 < 0.95

    def test_kriging_recovers_a_quadratic_closely(self, quadratic_study):
        space, design, response = quadratic_study
        model = mm.fit_metamodel(space, design, response, mm.MetamodelSpec("y", "kriging"))

        assert model.metrics.cv_r2 > 0.999

    @pytest.mark.parametrize("fit_type", ["linear", "quadratic", "kriging"])
    def test_every_fit_type_handles_mixed_factors(self, engine_study, fit_type):
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", fit_type))

        assert np.isfinite(model.metrics.r2)
        assert len(model.predict(design)) == len(design)

    def test_kriging_outperforms_quadratic_on_a_strongly_curved_response(self, engine_study):
        space, design, results, _ = engine_study
        kriging = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("Torque", "kriging"))
        quadratic = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("Torque", "quadratic"))

        assert kriging.metrics.cv_r2 > quadratic.metrics.cv_r2

    def test_generalizes_to_points_outside_the_training_design(self, engine_problem, engine_study):
        """In-sample R^2 can be bought by overfitting; held-out data cannot."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "kriging"))

        holdout = doe.latin_hypercube(space, 150, seed=987)
        truth = engine_problem.evaluate(holdout)["BSFC"].to_numpy()
        predicted = model.predict(holdout)

        ss_res = float(((truth - predicted) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        assert 1 - ss_res / ss_tot > 0.95


class TestCrossValidation:
    def test_noise_opens_a_gap_between_fitted_and_cross_validated_scores(
        self, engine_problem, engine_space
    ):
        """The gap is the signal that a surrogate has memorized its design."""
        design = doe.latin_hypercube(engine_space, 120, seed=3)
        noisy = engine_problem.evaluate(design, NoiseConfig(enabled=True, sigma=0.05, seed=1))
        model = mm.fit_metamodel(engine_space, design, noisy, mm.MetamodelSpec("Torque", "kriging"))

        assert model.metrics.r2 > model.metrics.cv_r2

    def test_cross_validation_is_skipped_when_data_is_too_thin(self):
        space = FactorSpace([ContinuousFactor("a", 0.0, 1.0)])
        design = pd.DataFrame({"a": [0.0, 0.5, 1.0]})
        response = pd.DataFrame({"y": [0.0, 1.0, 2.0]})
        model = mm.fit_metamodel(space, design, response, mm.MetamodelSpec("y", "linear"))

        assert np.isnan(model.metrics.cv_r2)

    def test_rejects_a_design_too_small_to_fit(self):
        space = FactorSpace([ContinuousFactor("a", 0.0, 1.0)])
        with pytest.raises(ValueError, match="at least three"):
            mm.fit_metamodel(
                space,
                pd.DataFrame({"a": [0.0, 1.0]}),
                pd.DataFrame({"y": [0.0, 1.0]}),
                mm.MetamodelSpec("y", "linear"),
            )

    def test_rejects_an_unknown_response(self, quadratic_study):
        space, design, response = quadratic_study
        with pytest.raises(KeyError, match="absent"):
            mm.fit_metamodel(space, design, response, mm.MetamodelSpec("absent", "linear"))

    def test_rejects_an_unknown_fit_type(self, quadratic_study):
        space, design, response = quadratic_study
        with pytest.raises(ValueError, match="unknown fit type"):
            mm.fit_metamodel(space, design, response, mm.MetamodelSpec("y", "cubic"))


class TestWarningCapture:
    def test_convergence_warnings_are_returned_rather_than_raised(self, engine_study):
        """They belong beside the fit in the UI, not streaming to stderr."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "kriging"))

        assert isinstance(model.fit_warnings, list)
        assert all(isinstance(w, str) for w in model.fit_warnings)

    def test_noiseless_data_produces_no_notable_warnings(self, engine_study):
        """An analytic solver pins the GP's noise term at its floor every time.

        Reporting that as a problem on every fit would train the user to ignore
        the banner, so it must classify as expected rather than notable.
        """
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "kriging"))

        assert model.notable_warnings == []

    def test_an_unrecognized_warning_stays_notable(self):
        assert not mm.is_expected_warning("something genuinely wrong happened")
        assert mm.is_expected_warning(
            "The optimal value found for dimension 0 of parameter "
            "k2__noise_level is close to the specified lower bound 1e-08."
        )
        assert mm.is_expected_warning("lbfgs failed to converge after 11 iteration(s)")


class TestFitStrength:
    """Whether a surrogate is fit to optimize against, judged on held-out data."""

    def test_a_well_fitted_response_is_not_weak(self, engine_study):
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "kriging"))

        assert model.metrics.cv_r2 > mm.WEAK_FIT_THRESHOLD
        assert not model.is_weak

    def test_a_linear_fit_to_a_curved_response_is_weak(self, engine_study):
        """Torque is strongly non-linear, so a linear surrogate must be flagged."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("Torque", "linear"))

        assert model.is_weak

    def test_weakness_ignores_the_in_sample_score(self, engine_study):
        """Kriging interpolates its training points, so in-sample R^2 is ~1 regardless."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("Torque", "kriging"))

        assert model.metrics.r2 > 0.999
        assert model.is_weak is (model.metrics.cv_r2 < mm.WEAK_FIT_THRESHOLD)

    def test_polynomial_fits_warn_about_nothing(self, engine_study):
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "quadratic"))

        assert model.fit_warnings == []


class TestPredictionHelpers:
    def test_grid_has_one_z_value_per_axis_pair(self, engine_study):
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "quadratic"))

        xs, ys, z = mm.predict_grid(
            model, space, "Spark_Timing", "RPM", space.midpoint(), resolution=15
        )
        assert z.shape == (len(ys), len(xs)) == (15, 15)
        assert np.isfinite(z).all()

    def test_grid_accepts_a_categorical_axis(self, engine_study):
        """A categorical axis is swept over its categories, not a linspace."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("BSFC", "quadratic"))

        xs, ys, z = mm.predict_grid(
            model, space, "Fuel_Type", "RPM", space.midpoint(), resolution=12
        )
        assert list(xs) == ["Regular", "Premium", "E85"]
        assert z.shape == (12, 3)

    def test_sweep_follows_the_known_shape_of_the_torque_curve(self, engine_study):
        """Torque peaks mid-range, so the sweep must not be monotonic."""
        space, design, results, _ = engine_study
        model = mm.fit_metamodel(space, design, results, mm.MetamodelSpec("Torque", "kriging"))

        _, values = mm.predict_sweep(model, space, "RPM", space.midpoint(), resolution=25)
        assert values.argmax() not in (0, len(values) - 1)

    def test_metrics_table_lists_every_model(self, engine_study):
        space, design, results, _ = engine_study
        models = {
            name: mm.fit_metamodel(space, design, results, mm.MetamodelSpec(name, "quadratic"))
            for name in ("BSFC", "Torque")
        }
        table = mm.metrics_table(models)

        assert len(table) == 2
        assert {"r2", "cv_r2", "rmse", "cv_rmse", "n_train"} <= set(table.columns)
