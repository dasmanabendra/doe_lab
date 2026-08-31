"""Design-of-experiments generators over mixed continuous/categorical spaces.

Three designs are supported, each answering a different question:

``full_factorial``
    Every combination of every factor's levels. Exhaustive and perfectly
    balanced, but the run count multiplies out fast — especially once
    categorical factors add their levels to the product.

``latin_hypercube``
    A space-filling sample of a chosen size. Stratifies every factor
    independently, so it scales to many factors where a factorial cannot.

``d_optimal``
    Picks the subset of a candidate set that maximizes the information content
    ``det(XᵀX)`` for an assumed model form. This is the design to reach for
    with mixed factor types and a fixed run budget, since it neither requires a
    full product nor assumes a symmetric continuous space.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .factors import FactorSpace

ModelOrder = str  # "linear" | "quadratic"


@dataclass
class DesignSpec:
    """A reproducible description of how to build a design."""

    kind: str = "latin_hypercube"
    n_experiments: int = 50
    model_order: ModelOrder = "quadratic"
    seed: int | None = 0
    n_candidates: int = 2000
    n_restarts: int = 5
    candidate_levels: int = 7

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_experiments": self.n_experiments,
            "model_order": self.model_order,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "n_restarts": self.n_restarts,
            "candidate_levels": self.candidate_levels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignSpec:
        return cls(**data)


def full_factorial(space: FactorSpace) -> pd.DataFrame:
    """Every combination of every factor's levels.

    Continuous factors contribute ``levels`` evenly spaced values; categorical
    factors contribute their categories.
    """
    grids = [f.level_values() for f in space]
    rows = list(itertools.product(*grids))
    return pd.DataFrame(rows, columns=space.names)


def full_factorial_size(space: FactorSpace) -> int:
    """Row count a full factorial would produce, without building it."""
    size = 1
    for f in space:
        size *= f.n_levels
    return size


def latin_hypercube(
    space: FactorSpace, n_experiments: int, seed: int | None = 0
) -> pd.DataFrame:
    """Space-filling sample, stratified in every dimension.

    Categorical factors are handled by binning their LHS coordinate into one
    equal stratum per category (see ``CategoricalFactor.from_unit``), which
    preserves proportional coverage instead of degrading to random assignment.
    """
    if n_experiments < 1:
        raise ValueError("n_experiments must be positive")
    sampler = qmc.LatinHypercube(d=len(space), seed=seed)
    return space.from_unit_cube(sampler.random(n_experiments))


def _model_matrix(
    space: FactorSpace, df: pd.DataFrame, order: ModelOrder
) -> np.ndarray:
    """Build the model matrix ``X`` whose information ``det(XᵀX)`` we maximize.

    Columns are an intercept, the coded main effects (continuous scaled to
    ``[-1, 1]``, categorical reference-coded), and for a quadratic model the
    pairwise interactions plus squared terms for the continuous factors.

    Two families of degenerate column are deliberately excluded, both of which
    would otherwise make ``XᵀX`` singular for *every* design:

    * squares of indicator columns, since ``x² == x`` for a 0/1 column;
    * interactions between indicators of the *same* categorical factor, which
      are mutually exclusive and so multiply to an all-zero column.
    """
    coded, columns = space.coded_matrix(df, drop_first=True)
    n = len(df)
    blocks = [np.ones((n, 1)), coded]

    if order == "quadratic":
        p = coded.shape[1]
        blocks.extend(
            (coded[:, i] * coded[:, j]).reshape(-1, 1)
            for i in range(p)
            for j in range(i + 1, p)
            if not (columns[i].factor == columns[j].factor)
        )
        blocks.extend(
            (coded[:, i] ** 2).reshape(-1, 1)
            for i in range(p)
            if not columns[i].is_indicator
        )
    elif order != "linear":
        raise ValueError(f"unknown model order: {order!r}")

    return np.hstack(blocks)


def _log_det_information(X: np.ndarray) -> float:
    """``log det(XᵀX)``, or ``-inf`` when the design is rank-deficient.

    Working in logs keeps the criterion comparable when the determinant would
    otherwise overflow or underflow for larger models.
    """
    sign, logdet = np.linalg.slogdet(X.T @ X)
    return float(logdet) if sign > 0 else float("-inf")


def _candidate_set(
    space: FactorSpace, n_candidates: int, levels: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Points D-optimal exchange is allowed to choose from.

    A full grid is used when it is small enough to enumerate, since exchanging
    over the exact grid gives the cleanest designs. Otherwise the continuous
    factors are sampled and snapped to a discrete level set, which keeps the
    candidates well spread without enumerating an intractable product.
    """
    grid_size = 1
    for f in space:
        grid_size *= f.n_levels if f.is_categorical else levels
        if grid_size > n_candidates:
            break

    if grid_size <= n_candidates:
        grids = [
            f.level_values()
            if f.is_categorical
            else list(np.linspace(f.low, f.high, levels))
            for f in space
        ]
        rows = list(itertools.product(*grids))
        return pd.DataFrame(rows, columns=space.names)

    u = rng.random((n_candidates, len(space)))
    for i, f in enumerate(space):
        if not f.is_categorical:
            u[:, i] = np.round(u[:, i] * (levels - 1)) / (levels - 1)
    return space.from_unit_cube(u)


def d_optimal(
    space: FactorSpace,
    n_experiments: int,
    model_order: ModelOrder = "quadratic",
    seed: int | None = 0,
    n_candidates: int = 2000,
    n_restarts: int = 5,
    candidate_levels: int = 7,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Select ``n_experiments`` candidates maximizing ``det(XᵀX)``.

    Uses Fedorov point exchange: from a random start, repeatedly swap the
    design point whose replacement most improves the criterion, until no swap
    helps. Several random restarts guard against the local optima this greedy
    scheme can settle into.
    """
    rng = np.random.default_rng(seed)
    candidates = _candidate_set(space, n_candidates, candidate_levels, rng)
    n_cand = len(candidates)

    if n_experiments > n_cand:
        raise ValueError(
            f"cannot choose {n_experiments} distinct runs from {n_cand} candidates"
        )

    X_all = _model_matrix(space, candidates, model_order)
    n_terms = X_all.shape[1]
    if n_experiments < n_terms:
        raise ValueError(
            f"a {model_order} model needs at least {n_terms} experiments "
            f"to be estimable; {n_experiments} requested"
        )

    best_idx: np.ndarray | None = None
    best_score = float("-inf")

    for restart in range(max(1, n_restarts)):
        idx = _exchange(X_all, n_experiments, n_terms, rng)
        score = _log_det_information(X_all[idx])
        if score > best_score:
            best_score, best_idx = score, idx.copy()
        if progress is not None:
            progress(restart + 1, max(1, n_restarts))

    if best_idx is None:
        raise RuntimeError(
            "D-optimal search found no non-singular design; try more experiments, "
            "a linear model order, or a larger candidate set"
        )
    return candidates.iloc[np.sort(best_idx)].reset_index(drop=True)


def _exchange(
    X_all: np.ndarray,
    n: int,
    n_terms: int,
    rng: np.random.Generator,
    max_iter: int = 500,
) -> np.ndarray:
    """One Fedorov point-exchange run from a random start.

    Swapping design row ``i`` for candidate ``j`` changes the determinant by a
    factor ``1 + delta(i, j)`` that follows from a rank-one update of
    ``(XᵀX)⁻¹``::

        delta(i,j) = (1 + v_j)(1 - d_i) + c_ij² - 1

    where ``d_i`` and ``v_j`` are the leverages of the design and candidate
    rows and ``c_ij`` is their cross term. Evaluating the whole exchange
    matrix this way costs one ``(n x n_cand)`` product per iteration, instead
    of refactorizing a determinant for every pair.
    """
    idx = _random_nonsingular_start(X_all, n, n_terms, rng)

    for _ in range(max_iter):
        X = X_all[idx]
        M = X.T @ X
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        A = X @ M_inv
        d_i = np.einsum("ij,ij->i", A, X)
        B = X_all @ M_inv
        v_j = np.einsum("ij,ij->i", B, X_all)
        cross = A @ X_all.T

        delta = (1.0 + v_j[None, :]) * (1.0 - d_i[:, None]) + cross**2 - 1.0

        # Forbid points already in the design. Replicated runs are legitimate
        # in classical D-optimal design, but this solver is deterministic, so a
        # repeated row would return an identical result and simply waste part
        # of the run budget.
        delta[:, idx] = -np.inf

        flat = int(np.argmax(delta))
        i, j = divmod(flat, delta.shape[1])
        if delta[i, j] <= 1e-10:
            break
        idx[i] = j

    return idx


def _random_nonsingular_start(
    X_all: np.ndarray, n: int, n_terms: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a random starting subset that is not rank-deficient.

    Exchange cannot climb away from ``-inf``, so a singular start would stall
    the search immediately.
    """
    for _ in range(200):
        idx = rng.choice(X_all.shape[0], size=n, replace=False)
        if np.linalg.matrix_rank(X_all[idx]) >= n_terms:
            return idx
    return rng.choice(X_all.shape[0], size=n, replace=False)


def d_efficiency(space: FactorSpace, df: pd.DataFrame, model_order: ModelOrder) -> float:
    """Normalized D-efficiency of a design, for comparing designs of one model.

    ``(det(XᵀX)^(1/p)) / n`` — scale-free in the number of model terms ``p``,
    so designs with different run counts remain comparable.
    """
    X = _model_matrix(space, df, model_order)
    n, p = X.shape
    sign, logdet = np.linalg.slogdet(X.T @ X)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p) / n)


def generate(
    space: FactorSpace,
    spec: DesignSpec,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Build a design from its spec."""
    if spec.kind == "full_factorial":
        return full_factorial(space)
    if spec.kind == "latin_hypercube":
        return latin_hypercube(space, spec.n_experiments, spec.seed)
    if spec.kind == "d_optimal":
        return d_optimal(
            space,
            spec.n_experiments,
            model_order=spec.model_order,
            seed=spec.seed,
            n_candidates=spec.n_candidates,
            n_restarts=spec.n_restarts,
            candidate_levels=spec.candidate_levels,
            progress=progress,
        )
    raise ValueError(f"unknown design kind: {spec.kind!r}")


DESIGN_KINDS: list[tuple[str, str]] = [
    ("full_factorial", "Full Factorial"),
    ("latin_hypercube", "Latin Hypercube"),
    ("d_optimal", "D-Optimal"),
]
