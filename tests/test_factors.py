"""Factor definitions and the mappings into the sample space."""

from __future__ import annotations

import numpy as np
import pytest

from doelab.engine.factors import (
    CategoricalFactor,
    ContinuousFactor,
    FactorSpace,
    Response,
    ResponseRole,
    factor_from_dict,
)


class TestContinuousFactor:
    def test_rejects_an_inverted_range(self):
        with pytest.raises(ValueError, match="must exceed"):
            ContinuousFactor("x", 10.0, 1.0)

    def test_rejects_fewer_than_two_levels(self):
        with pytest.raises(ValueError, match="at least 2 levels"):
            ContinuousFactor("x", 0.0, 1.0, levels=1)

    def test_levels_span_the_range_inclusively(self):
        assert ContinuousFactor("x", 0.0, 10.0, levels=3).level_values() == [0.0, 5.0, 10.0]

    def test_coding_maps_the_range_onto_plus_minus_one(self):
        factor = ContinuousFactor("x", 100.0, 500.0)
        coded = factor.to_coded(np.array([100.0, 300.0, 500.0]))
        assert coded == pytest.approx([-1.0, 0.0, 1.0])

    def test_round_trips_through_its_dict_form(self):
        factor = ContinuousFactor("x", -2.0, 3.5, "mm", levels=4, description="d")
        assert factor_from_dict(factor.to_dict()) == factor


class TestCategoricalFactor:
    def test_rejects_duplicate_categories(self):
        with pytest.raises(ValueError, match="unique"):
            CategoricalFactor("c", ["a", "a"])

    def test_rejects_a_single_category(self):
        with pytest.raises(ValueError, match="at least 2"):
            CategoricalFactor("c", ["only"])

    def test_unit_interval_splits_into_equal_strata(self):
        factor = CategoricalFactor("c", ["a", "b", "c"])
        mapped = factor.from_unit(np.array([0.0, 0.32, 0.34, 0.66, 0.67, 0.999]))
        assert list(mapped) == ["a", "a", "b", "b", "c", "c"]

    def test_upper_bound_stays_in_the_last_category(self):
        """u == 1.0 must not index past the end."""
        factor = CategoricalFactor("c", ["a", "b"])
        assert list(factor.from_unit(np.array([1.0]))) == ["b"]

    def test_round_trips_through_its_dict_form(self):
        factor = CategoricalFactor("c", ["x", "y"], description="d")
        assert factor_from_dict(factor.to_dict()) == factor


class TestResponse:
    def test_a_constraint_requires_a_bound(self):
        with pytest.raises(ValueError, match="lower or upper"):
            Response("p", role=ResponseRole.CONSTRAINT)

    def test_objective_roles_report_themselves_as_objectives(self):
        assert ResponseRole.OBJECTIVE_MIN.is_objective
        assert ResponseRole.OBJECTIVE_MAX.is_objective
        assert not ResponseRole.CONSTRAINT.is_objective
        assert not ResponseRole.IGNORED.is_objective

    def test_round_trips_through_its_dict_form(self):
        response = Response("p", "bar", ResponseRole.CONSTRAINT, upper=10.0)
        assert Response.from_dict(response.to_dict()) == response


class TestFactorSpace:
    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            FactorSpace([ContinuousFactor("x", 0, 1), ContinuousFactor("x", 0, 1)])

    def test_rejects_an_empty_space(self):
        with pytest.raises(ValueError, match="at least one factor"):
            FactorSpace([])

    def test_looks_factors_up_by_name(self, mixed_space):
        assert mixed_space["mat"].name == "mat"
        with pytest.raises(KeyError):
            mixed_space["absent"]

    def test_partitions_factors_by_kind(self, mixed_space):
        assert mixed_space.continuous_names == ["x", "y"]
        assert mixed_space.categorical_names == ["mat"]

    def test_unit_cube_mapping_hits_both_bounds(self, mixed_space):
        frame = mixed_space.from_unit_cube(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.99]]))
        assert frame["x"].tolist() == [0.0, 10.0]
        assert frame["y"].tolist() == [-5.0, 5.0]

    def test_unit_cube_rejects_a_width_mismatch(self, mixed_space):
        with pytest.raises(ValueError, match="expected 3 columns"):
            mixed_space.from_unit_cube(np.zeros((4, 2)))

    def test_coded_matrix_uses_reference_coding(self, mixed_space):
        """One indicator fewer than there are categories, so X'X stays invertible."""
        frame = mixed_space.from_unit_cube(np.array([[0.5, 0.5, 0.1]]))
        matrix, columns = mixed_space.coded_matrix(frame)

        assert [c.label for c in columns] == ["x", "y", "mat=steel", "mat=iron"]
        assert matrix.shape == (1, 4)

    def test_coded_columns_record_their_owning_factor(self, mixed_space):
        frame = mixed_space.from_unit_cube(np.array([[0.5, 0.5, 0.1]]))
        _, columns = mixed_space.coded_matrix(frame)

        siblings = [c for c in columns if c.factor == "mat"]
        assert len(siblings) == 2
        assert all(c.is_indicator for c in siblings)

    def test_midpoint_centres_continuous_and_takes_the_first_category(self, mixed_space):
        assert mixed_space.midpoint() == {"x": 5.0, "y": 0.0, "mat": "alu"}

    def test_round_trips_through_its_list_form(self, mixed_space):
        restored = FactorSpace.from_list(mixed_space.to_list())
        assert restored.names == mixed_space.names
        assert restored["mat"].categories == mixed_space["mat"].categories
