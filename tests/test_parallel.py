"""Axis scaling and brushing for the parallel coordinates view.

Everything asserted here is arithmetic over a table, which is the whole reason
it lives in the engine: the widget that consumes it cannot be exercised without
a display, and the offscreen platform the UI tests run under does not paint the
same way Windows does. Keeping the geometry testable headlessly means only the
painting is left unverified by the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doelab.engine import parallel
from doelab.engine.factors import (
    CategoricalFactor,
    ContinuousFactor,
    FactorSpace,
    Response,
    ResponseRole,
)


@pytest.fixture
def small_space() -> FactorSpace:
    return FactorSpace(
        [
            ContinuousFactor("x", 0.0, 10.0, "mm"),
            CategoricalFactor("mat", ["alu", "steel", "iron"]),
        ]
    )


@pytest.fixture
def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [2.0, 4.0, 6.0, 8.0],
            "mat": ["alu", "steel", "iron", "alu"],
            "flat": [7.0, 7.0, 7.0, 7.0],
            "cost": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def small_responses() -> list[Response]:
    return [
        Response("cost", "USD", ResponseRole.OBJECTIVE_MIN),
        Response("flat", "", ResponseRole.CONSTRAINT, upper=9.0),
    ]


class TestBuildAxes:
    def test_factors_come_before_responses_in_project_order(
        self, small_space, small_frame, small_responses
    ):
        axes = parallel.build_axes(small_space, small_responses, small_frame)

        assert [a.name for a in axes] == ["x", "mat", "cost", "flat"]
        assert [a.group for a in axes] == ["factor", "factor", "response", "response"]

    def test_ignored_responses_are_kept(self, small_space, small_frame):
        """An ignored response is still a measured output worth looking at."""
        responses = [Response("cost", "USD", ResponseRole.IGNORED)]
        axes = parallel.build_axes(small_space, responses, small_frame)

        assert "cost" in [a.name for a in axes]

    def test_columns_the_frame_lacks_are_skipped(self, small_space, small_responses):
        """An optimizer front carries objectives and constraints, not every column."""
        front = pd.DataFrame({"x": [1.0, 2.0], "mat": ["alu", "iron"], "cost": [5.0, 6.0]})
        axes = parallel.build_axes(small_space, small_responses, front)

        assert [a.name for a in axes] == ["x", "mat", "cost"]

    def test_an_empty_frame_yields_no_axes(self, small_space, small_responses):
        axes = parallel.build_axes(small_space, small_responses, pd.DataFrame())
        assert axes == []

    def test_objective_direction_and_constraint_limits_carry_through(
        self, small_space, small_frame, small_responses
    ):
        axes = {a.name: a for a in parallel.build_axes(small_space, small_responses, small_frame)}

        assert axes["cost"].direction == "min"
        assert axes["cost"].limits == (None, None)
        assert axes["flat"].direction is None
        assert axes["flat"].limits == (None, 9.0)

    def test_continuous_axes_span_the_data_not_the_declared_range(
        self, small_space, small_frame, small_responses
    ):
        """x is declared 0..10 but only 2..8 was sampled."""
        axes = {a.name: a for a in parallel.build_axes(small_space, small_responses, small_frame)}

        assert (axes["x"].low, axes["x"].high) == (2.0, 8.0)


class TestNormalize:
    def test_the_extremes_land_on_the_ends(self, small_space, small_frame, small_responses):
        axes = parallel.build_axes(small_space, small_responses, small_frame)
        values = parallel.normalize(small_frame, axes)

        x = values[:, 0]
        assert x.min() == pytest.approx(0.0)
        assert x.max() == pytest.approx(1.0)

    def test_a_constant_column_is_centred_rather_than_dividing_by_zero(
        self, small_space, small_frame, small_responses
    ):
        axes = parallel.build_axes(small_space, small_responses, small_frame)
        values = parallel.normalize(small_frame, axes)

        flat = values[:, [a.name for a in axes].index("flat")]
        assert np.all(flat == 0.5)
        assert np.all(np.isfinite(flat))

    def test_categories_are_evenly_spaced(self, small_space, small_frame, small_responses):
        axes = parallel.build_axes(small_space, small_responses, small_frame)
        mat = axes[1]

        positions = mat.positions(np.array(["alu", "steel", "iron"], dtype=object))
        assert positions == pytest.approx([0.0, 0.5, 1.0])

    def test_an_unknown_category_is_unplaceable_rather_than_guessed(self, small_space):
        mat = parallel.Axis("mat", "factor", 0.0, 2.0, categories=("alu", "steel", "iron"))

        assert np.isnan(mat.positions(np.array(["brass"], dtype=object))[0])

    def test_ticks_label_every_category_but_only_the_numeric_ends(
        self, small_space, small_frame, small_responses
    ):
        axes = parallel.build_axes(small_space, small_responses, small_frame)

        assert [label for _, label in axes[1].ticks()] == ["alu", "steel", "iron"]
        assert len(axes[0].ticks()) == 2

    def test_value_at_inverts_positions(self, small_space, small_frame, small_responses):
        axes = parallel.build_axes(small_space, small_responses, small_frame)

        assert axes[0].value_at(0.5) == pytest.approx(5.0)
        assert axes[1].value_at(0.5) == "steel"


class TestFilterMask:
    def test_an_untouched_plot_keeps_every_design(self):
        values = np.array([[0.1, 0.9], [0.5, 0.2], [1.0, 0.0]])
        bands = np.array([parallel.FULL_BAND, parallel.FULL_BAND])

        assert parallel.filter_mask(values, bands).all()

    def test_a_narrowed_band_excludes_what_falls_outside(self):
        values = np.array([[0.1], [0.5], [0.9]])
        bands = np.array([[0.4, 0.6]])

        assert parallel.filter_mask(values, bands).tolist() == [False, True, False]

    def test_a_design_exactly_on_the_boundary_is_kept(self):
        values = np.array([[0.4], [0.6]])
        bands = np.array([[0.4, 0.6]])

        assert parallel.filter_mask(values, bands).all()

    def test_bands_intersect_across_axes(self):
        """Brushing two axes carves out a region; it does not accumulate rows."""
        values = np.array([[0.5, 0.9], [0.5, 0.5], [0.9, 0.5]])
        bands = np.array([[0.4, 0.6], [0.4, 0.6]])

        assert parallel.filter_mask(values, bands).tolist() == [False, True, False]

    def test_an_unplaceable_design_survives_until_that_axis_is_filtered(self):
        """Otherwise an untouched plot reports fewer designs than it holds."""
        values = np.array([[np.nan, 0.5]])

        wide = np.array([parallel.FULL_BAND, parallel.FULL_BAND])
        assert parallel.filter_mask(values, wide).tolist() == [True]

        narrowed = np.array([[0.2, 0.8], list(parallel.FULL_BAND)])
        assert parallel.filter_mask(values, narrowed).tolist() == [False]


class TestFeasibility:
    def test_an_upper_bound_flags_the_designs_over_it(self, small_frame, small_responses):
        frame = small_frame.assign(flat=[8.0, 9.0, 10.0, 11.0])

        assert parallel.feasibility(frame, small_responses).tolist() == [
            True, True, False, False
        ]

    def test_objectives_never_make_a_design_infeasible(self, small_frame, small_responses):
        """Only constraints bound a design; an objective merely ranks it."""
        frame = small_frame.assign(cost=[1e6, 1e6, 1e6, 1e6])

        assert parallel.feasibility(frame, small_responses).all()

    def test_a_two_sided_constraint_bounds_from_both_directions(self, small_frame):
        responses = [Response("cost", "", ResponseRole.CONSTRAINT, lower=15.0, upper=35.0)]

        assert parallel.feasibility(small_frame, responses).tolist() == [
            False, True, True, False
        ]


class TestRails:
    def test_a_bound_inside_the_data_range_is_placed(self, small_space, small_frame):
        responses = [Response("cost", "", ResponseRole.CONSTRAINT, upper=25.0)]
        axes = parallel.build_axes(small_space, responses, small_frame)
        rails = parallel.rail_positions(axes)

        index = [a.name for a in axes].index("cost")
        (position, bound), = rails[index]
        assert bound == 25.0
        # cost spans 10..40, so 25 sits halfway up.
        assert position == pytest.approx(0.5)

    def test_a_bound_no_design_came_near_is_not_drawn(self, small_space, small_frame):
        """A rail clamped to the axis end would claim designs sit on a boundary."""
        responses = [Response("cost", "", ResponseRole.CONSTRAINT, upper=1000.0)]
        axes = parallel.build_axes(small_space, responses, small_frame)

        assert parallel.rail_positions(axes) == {}


class TestAgainstTheGasolineStudy:
    """The real 120-run study the Analyze page will show."""

    def test_every_factor_and_response_gets_an_axis(self, engine_study):
        space, design, results, responses = engine_study
        frame = pd.concat([design, results], axis=1)

        axes = parallel.build_axes(space, responses, frame)

        assert [a.name for a in axes] == space.names + [r.name for r in responses]
        assert sum(a.is_categorical for a in axes) == 1  # Fuel_Type

    def test_normalized_values_stay_on_the_axis(self, engine_study):
        space, design, results, responses = engine_study
        frame = pd.concat([design, results], axis=1)

        values = parallel.normalize(frame, parallel.build_axes(space, responses, frame))

        assert np.all(values >= -parallel.TOLERANCE)
        assert np.all(values <= 1.0 + parallel.TOLERANCE)

    def test_the_cylinder_pressure_constraint_actually_binds(self, engine_study):
        """If nothing violated it, the infeasible styling would never be seen."""
        space, design, results, responses = engine_study
        frame = pd.concat([design, results], axis=1)

        feasible = parallel.feasibility(frame, responses)

        assert 0 < int((~feasible).sum()) < len(frame)

    def test_the_constraint_rail_is_drawable(self, engine_study):
        space, design, results, responses = engine_study
        frame = pd.concat([design, results], axis=1)

        axes = parallel.build_axes(space, responses, frame)
        index = [a.name for a in axes].index("Max_Cyl_Pressure")

        assert index in parallel.rail_positions(axes)
