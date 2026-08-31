"""Statistical summaries of a completed design.

Four views of the same experiment table, answering different questions:

* **Factor sensitivity** — of the variation a linear model explains, what
  share does each factor take? Normalized, so every row sums to 1.
* **Variance decomposition** — partial and semi-partial R², which do *not*
  normalize, so the model's overall fit and the portion no single factor can
  claim both stay visible.
* **Pearson correlation** — how strong is the *linear* association?
* **Spearman correlation** — how strong is the *monotonic* association?

The pairs are deliberate. Pearson and Spearman disagree in an informative way:
a factor with a strong but curved effect shows a high Spearman and a weak
Pearson, which is a hint that a linear metamodel will not be enough. Likewise
sensitivity and the variance decomposition come from the same linear fit but
differ wherever the design confounds two factors — sensitivity hands both of
them a share, the decomposition books it to ``shared`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FactorSpace

# Below this, a sum of squares is treated as zero. Squared quantities land
# well clear of it, so it only ever catches genuinely degenerate columns.
_EPS = 1e-12


def factor_sensitivity(
    space: FactorSpace, factors_df: pd.DataFrame, responses_df: pd.DataFrame
) -> pd.DataFrame:
    """Share of each response's variation attributable to each factor.

    Defined as **squared standardized regression coefficients, normalized to
    sum to 1 across factors**. A linear model is fitted on the coded factor
    matrix with every column standardized, so each coefficient is on a common
    scale; squaring makes the contributions additive in the same sense as
    variance, and normalizing turns them into shares that are readable as
    percentages.

    Categorical factors contribute several indicator columns, whose squared
    coefficients are summed into a single figure for the factor.

    The measure inherits the linear model's blind spot: a factor whose effect
    is purely quadratic and symmetric about the design centre can score near
    zero despite mattering. Read it alongside the correlation tables and the
    fitted metamodels rather than on its own.

    Returns a DataFrame indexed by response, with one column per factor.
    """
    X, columns = space.coded_matrix(factors_df, drop_first=True)
    if X.shape[0] < 2:
        raise ValueError("factor sensitivity needs at least two experiments")

    # Standardize so coefficients are comparable across columns with different
    # spreads. Columns that never vary carry no information and are zeroed.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    active = sd > 1e-12
    Xs = np.zeros_like(X)
    Xs[:, active] = (X[:, active] - mu[active]) / sd[active]

    rows: dict[str, dict[str, float]] = {}
    for response in responses_df.columns:
        y = responses_df[response].to_numpy(dtype=float)
        y_sd = y.std()
        ys = np.zeros_like(y) if y_sd < 1e-12 else (y - y.mean()) / y_sd

        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)

        per_factor: dict[str, float] = {name: 0.0 for name in space.names}
        for coefficient, column in zip(beta, columns):
            per_factor[column.factor] += float(coefficient) ** 2

        total = sum(per_factor.values())
        if total > 0:
            per_factor = {k: v / total for k, v in per_factor.items()}
        rows[response] = per_factor

    return pd.DataFrame.from_dict(rows, orient="index", columns=space.names)


def correlation_matrix(
    space: FactorSpace,
    factors_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation across all factors and responses together.

    Categorical factors are expanded to indicator columns, since a correlation
    is only defined against a numeric variable; each indicator reads as "this
    level versus the rest".
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown correlation method: {method!r}")

    X, columns = space.coded_matrix(factors_df, drop_first=True)
    numeric = pd.DataFrame(X, columns=[c.label for c in columns], index=factors_df.index)
    combined = pd.concat([numeric, responses_df.set_axis(factors_df.index)], axis=1)

    # Constant columns have undefined correlation; drop them rather than
    # emitting a table full of NaN that reads as a failure.
    varying = combined.loc[:, combined.std(numeric_only=True) > 1e-12]
    return varying.corr(method=method)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column min/mean/max/std, for the results overview."""
    numeric = df.select_dtypes(include=[np.number])
    return pd.DataFrame(
        {
            "min": numeric.min(),
            "mean": numeric.mean(),
            "max": numeric.max(),
            "std": numeric.std(),
        }
    )


def split_labels(space: FactorSpace, corr: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition a correlation table's labels into factor and response groups.

    The UI shades the two blocks differently, and the factor block is the
    coded-column labels rather than the raw factor names.
    """
    factor_labels = [c for c in corr.columns if _owning_factor(space, c) is not None]
    response_labels = [c for c in corr.columns if c not in factor_labels]
    return factor_labels, response_labels


def _owning_factor(space: FactorSpace, label: str) -> str | None:
    base = label.split("=", 1)[0]
    return base if base in space.names else None


@dataclass(frozen=True)
class VarianceDecomposition:
    """Three readings of one linear fit, one row per response.

    ``partial``
        Partial R² per factor: the share of the variation *the other factors
        leave unexplained* that this factor accounts for.
    ``semi_partial``
        Semi-partial R² per factor: the drop in the model's overall R² when
        this factor is removed — its unique contribution to the whole.
    ``model``
        Per response: ``r_squared``, ``adjusted_r_squared``, ``unique``
        (the semi-partials summed), ``shared`` and ``unexplained``. The last
        three partition the response: ``unique + shared + unexplained == 1``.
    """

    partial: pd.DataFrame
    semi_partial: pd.DataFrame
    model: pd.DataFrame


def variance_decomposition(
    space: FactorSpace, factors_df: pd.DataFrame, responses_df: pd.DataFrame
) -> VarianceDecomposition:
    """Apportion each response's variation by refitting without each factor.

    Where :func:`factor_sensitivity` normalizes squared coefficients so every
    row sums to 1, this leaves the totals alone — which is the point. Two
    things become visible that normalizing hides:

    * **How much the model explains at all.** A row of sensitivity shares looks
      identical whether R² is 0.95 or 0.05, because dividing by the total
      removes it by construction.
    * **What no single factor can claim.** When factors are correlated in the
      design, some of the explained variation is attributable only to a group
      of them jointly. It surfaces here as ``shared`` rather than being
      double-counted into every factor's score.

    A factor's contribution is measured by *dropping* it and refitting: the
    increase in residual sum of squares is what it was uniquely providing.
    Categorical factors are dropped whole — every indicator column at once —
    so the figure is per factor, not per level.

    Returns a :class:`VarianceDecomposition`.
    """
    X, columns = space.coded_matrix(factors_df, drop_first=True)
    n_runs = X.shape[0]
    if n_runs < 2:
        raise ValueError("a variance decomposition needs at least two experiments")

    # The intercept goes in column 0 and is never dropped: every reduced model
    # must still be free to fit the response's mean, or the "loss" from
    # removing a factor would include the mean it was absorbing.
    design = np.column_stack([np.ones(n_runs), X])
    owners = [c.factor for c in columns]
    n_terms = X.shape[1]

    partial_rows: dict[str, dict[str, float]] = {}
    unique_rows: dict[str, dict[str, float]] = {}
    model_rows: dict[str, dict[str, float]] = {}

    for response in responses_df.columns:
        y = responses_df[response].to_numpy(dtype=float)
        total = float(((y - y.mean()) ** 2).sum())

        if total <= _EPS:
            # A response that never moved has no variation to apportion, and
            # R² is undefined rather than zero. Report the shares as zero so
            # the table stays numeric, and the fit as NaN so it reads as absent.
            zeros = dict.fromkeys(space.names, 0.0)
            partial_rows[response] = zeros
            unique_rows[response] = dict(zeros)
            model_rows[response] = {
                "r_squared": float("nan"),
                "adjusted_r_squared": float("nan"),
                "unique": 0.0,
                "shared": float("nan"),
                "unexplained": float("nan"),
            }
            continue

        residual_full = _residual_sum_of_squares(design, y)
        r_squared = 1.0 - residual_full / total

        partial: dict[str, float] = {}
        unique: dict[str, float] = {}
        for name in space.names:
            keep = [0] + [i + 1 for i, owner in enumerate(owners) if owner != name]
            residual_reduced = _residual_sum_of_squares(design[:, keep], y)
            # Dropping columns cannot reduce the residual, so the gain is
            # non-negative in exact arithmetic; clamp the float wobble.
            gain = max(residual_reduced - residual_full, 0.0)
            partial[name] = gain / residual_reduced if residual_reduced > _EPS else 0.0
            unique[name] = gain / total

        unique_total = sum(unique.values())
        partial_rows[response] = partial
        unique_rows[response] = unique
        model_rows[response] = {
            "r_squared": r_squared,
            "adjusted_r_squared": _adjusted(r_squared, n_runs, n_terms),
            "unique": unique_total,
            "shared": r_squared - unique_total,
            "unexplained": 1.0 - r_squared,
        }

    return VarianceDecomposition(
        partial=pd.DataFrame.from_dict(partial_rows, orient="index", columns=space.names),
        semi_partial=pd.DataFrame.from_dict(unique_rows, orient="index", columns=space.names),
        model=pd.DataFrame.from_dict(
            model_rows,
            orient="index",
            columns=["r_squared", "adjusted_r_squared", "unique", "shared", "unexplained"],
        ),
    )


def _residual_sum_of_squares(design: np.ndarray, y: np.ndarray) -> float:
    """Least-squares residual sum of squares for ``y ~ design``.

    Computed from the fitted values rather than read off ``lstsq``'s own
    residual return, which comes back empty whenever the matrix is
    rank-deficient — exactly the confounded designs this measure exists to
    expose.
    """
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(residual @ residual)


def _adjusted(r_squared: float, n_runs: int, n_terms: int) -> float:
    """R² penalized for the number of terms fitted.

    Undefined once the design has no spare runs left, which is where an
    unpenalized R² is at its most flattering — hence NaN rather than a number.
    """
    dof = n_runs - n_terms - 1
    if dof <= 0:
        return float("nan")
    return 1.0 - (1.0 - r_squared) * (n_runs - 1) / dof
