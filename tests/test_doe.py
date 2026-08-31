"""Design generation over mixed factor spaces."""

from __future__ import annotations

import numpy as np
import pytest

from doelab.engine import doe
from doelab.engine.factors import CategoricalFactor, ContinuousFactor, FactorSpace


class TestFullFactorial:
    def test_row_count_is_the_product_of_level_counts(self, mixed_space):
        design = doe.full_factorial(mixed_space)
        assert len(design) == 3 * 4 * 3
        assert doe.full_factorial_size(mixed_space) == len(design)

    def test_every_combination_appears_exactly_once(self, mixed_space):
        design = doe.full_factorial(mixed_space)
        assert not design.duplicated().any()

    def test_categorical_levels_are_balanced(self, mixed_space):
        counts = doe.full_factorial(mixed_space)["mat"].value_counts()
        assert set(counts) == {12}

    def test_continuous_factors_span_their_full_range(self, mixed_space):
        design = doe.full_factorial(mixed_space)
        assert design["x"].min() == pytest.approx(0.0)
        assert design["x"].max() == pytest.approx(10.0)


class TestLatinHypercube:
    def test_produces_the_requested_row_count(self, mixed_space):
        assert len(doe.latin_hypercube(mixed_space, 37, seed=0)) == 37

    def test_categorical_coverage_is_proportional(self, mixed_space):
        """The stratified binning should split levels evenly, not randomly."""
        design = doe.latin_hypercube(mixed_space, 300, seed=1)
        counts = design["mat"].value_counts()
        assert set(counts) == {100}

    def test_each_continuous_stratum_is_used_once(self, mixed_space):
        """The defining property of a Latin hypercube."""
        n = 50
        design = doe.latin_hypercube(mixed_space, n, seed=2)
        unit = (design["x"] - 0.0) / 10.0
        strata = np.clip((unit * n).astype(int), 0, n - 1)
        assert len(set(strata)) == n

    def test_stays_inside_the_declared_bounds(self, mixed_space):
        design = doe.latin_hypercube(mixed_space, 100, seed=3)
        assert design["y"].between(-5.0, 5.0).all()
        assert set(design["mat"]) <= {"alu", "steel", "iron"}

    def test_is_reproducible_for_a_seed(self, mixed_space):
        a = doe.latin_hypercube(mixed_space, 20, seed=7)
        b = doe.latin_hypercube(mixed_space, 20, seed=7)
        assert a.equals(b)

    def test_rejects_a_non_positive_size(self, mixed_space):
        with pytest.raises(ValueError):
            doe.latin_hypercube(mixed_space, 0)


class TestDOptimal:
    def test_produces_the_requested_row_count(self, mixed_space):
        assert len(doe.d_optimal(mixed_space, 20, seed=0, n_restarts=2)) == 20

    def test_returns_distinct_runs(self, mixed_space):
        """Replicates would be wasted budget against a deterministic solver."""
        design = doe.d_optimal(mixed_space, 20, seed=0, n_restarts=2)
        assert not design.duplicated().any()

    def test_beats_random_selection_on_its_own_criterion(self, mixed_space):
        rng = np.random.default_rng(0)
        optimal = doe.d_optimal(mixed_space, 25, seed=1, n_restarts=3)
        random_design = mixed_space.from_unit_cube(rng.random((25, len(mixed_space))))

        assert doe.d_efficiency(mixed_space, optimal, "quadratic") > doe.d_efficiency(
            mixed_space, random_design, "quadratic"
        )

    def test_beats_latin_hypercube_on_its_own_criterion(self, mixed_space):
        optimal = doe.d_optimal(mixed_space, 25, seed=1, n_restarts=3)
        lhs = doe.latin_hypercube(mixed_space, 25, seed=1)

        assert doe.d_efficiency(mixed_space, optimal, "quadratic") > doe.d_efficiency(
            mixed_space, lhs, "quadratic"
        )

    def test_quadratic_model_matrix_is_not_singular_with_a_3_level_categorical(
        self, mixed_space
    ):
        """Guards the sibling-indicator trap.

        Two indicators of one categorical factor are mutually exclusive, so
        their product is an all-zero column. Including it would make X'X
        singular for every possible design, and the search would score every
        candidate as -inf.
        """
        design = doe.d_optimal(mixed_space, 25, seed=2, n_restarts=2)
        X = doe._model_matrix(mixed_space, design, "quadratic")

        assert np.linalg.matrix_rank(X) == X.shape[1]
        assert doe._log_det_information(X) > float("-inf")

    def test_rejects_a_budget_too_small_for_the_model(self, mixed_space):
        with pytest.raises(ValueError, match="estimable"):
            doe.d_optimal(mixed_space, 4, model_order="quadratic")

    def test_rejects_more_runs_than_candidates(self):
        tiny = FactorSpace([CategoricalFactor("c", ["a", "b"])])
        with pytest.raises(ValueError, match="candidates"):
            doe.d_optimal(tiny, 50, model_order="linear")

    def test_works_on_a_purely_continuous_space(self):
        space = FactorSpace(
            [ContinuousFactor(f"x{i}", 0.0, 1.0) for i in range(3)]
        )
        assert len(doe.d_optimal(space, 15, seed=0, n_restarts=2)) == 15


class TestGenerate:
    @pytest.mark.parametrize("kind", ["full_factorial", "latin_hypercube", "d_optimal"])
    def test_dispatches_every_registered_kind(self, mixed_space, kind):
        spec = doe.DesignSpec(kind=kind, n_experiments=25, n_restarts=2)
        assert len(doe.generate(mixed_space, spec)) > 0

    def test_spec_survives_a_json_round_trip(self, mixed_space):
        spec = doe.DesignSpec(kind="latin_hypercube", n_experiments=25, seed=11)
        restored = doe.DesignSpec.from_dict(spec.to_dict())
        assert doe.generate(mixed_space, spec).equals(doe.generate(mixed_space, restored))

    def test_rejects_an_unknown_kind(self, mixed_space):
        with pytest.raises(ValueError, match="unknown design kind"):
            doe.generate(mixed_space, doe.DesignSpec(kind="nonsense"))
