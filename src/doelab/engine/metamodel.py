"""Surrogate models fitted to experiment results.

A metamodel replaces the solver with something cheap enough to evaluate
thousands of times per second, which is what makes interactive prediction and
population-based optimization practical.

Three fit types, in increasing order of flexibility and cost:

``linear``
    Main effects only. Fast, and a useful baseline — if it already fits well,
    the response has no meaningful curvature.

``quadratic``
    Adds squares and pairwise interactions. The standard response-surface
    model, and usually enough for a smooth engineering response.

``kriging``
    A Gaussian process. Interpolates the observed points and adapts its own
    smoothness per factor, so it captures shapes a polynomial cannot. It is
    the slowest to fit and the most prone to overfitting a noisy design, which
    is exactly what the cross-validated metrics are there to reveal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .factors import FactorSpace

FIT_TYPES: list[tuple[str, str]] = [
    ("linear", "Linear"),
    ("quadratic", "Quadratic"),
    ("kriging", "Kriging"),
]

# Complaints a Gaussian process routinely makes when fitted to output from a
# deterministic solver. Such data is exactly interpolable, so the learned noise
# term is driven to zero, the signal variance is left free to grow, and the
# hyperparameter search terminates early on a likelihood surface that has gone
# flat because there is nothing left to gain.
#
# None of these says anything about whether the surrogate predicts well — a
# response that is very nearly linear provokes all three while cross-validating
# at R² = 1.0. Fit quality is judged by ``is_weak`` below, measured on held-out
# data, so these are classified as expected and kept out of the user's way.
_EXPECTED_WARNING_FRAGMENTS = (
    "noise_level is close to the specified lower bound",
    "constant_value is close to the specified upper bound",
    "lbfgs failed to converge",
)

# Below this cross-validated R², a surrogate is not trustworthy enough to
# optimize against: the optimizer would be chasing features of the fit rather
# than of the response.
WEAK_FIT_THRESHOLD = 0.90


def is_expected_warning(message: str) -> bool:
    """Whether a fit warning is the ordinary consequence of noiseless data."""
    return any(fragment in message for fragment in _EXPECTED_WARNING_FRAGMENTS)


@dataclass
class MetamodelSpec:
    """Everything needed to rebuild a metamodel from the stored experiments."""

    response: str
    fit_type: str = "quadratic"
    cv_folds: int = 5
    kriging_restarts: int = 2
    kriging_nugget: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "fit_type": self.fit_type,
            "cv_folds": self.cv_folds,
            "kriging_restarts": self.kriging_restarts,
            "kriging_nugget": self.kriging_nugget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetamodelSpec:
        return cls(**data)

    @property
    def label(self) -> str:
        return f"{self.response} ({self.fit_type})"


@dataclass
class MetamodelMetrics:
    """Fit quality, in-sample and cross-validated.

    The gap between the two is the number that matters: a high ``r2`` beside a
    poor ``cv_r2`` means the surrogate has memorized the design rather than
    learned the response, and any optimum found on it will be fictional.
    """

    r2: float
    rmse: float
    cv_r2: float
    cv_rmse: float
    n_train: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "cv_r2": self.cv_r2,
            "cv_rmse": self.cv_rmse,
            "n_train": self.n_train,
        }


def _encoder(space: FactorSpace) -> ColumnTransformer:
    """One-hot the categorical factors, standardize the continuous ones.

    Standardizing matters for kriging, whose kernel measures distance in the
    encoded space, and is harmless for the polynomial fits.
    """
    transformers = []
    if space.continuous_names:
        transformers.append(("num", StandardScaler(), space.continuous_names))
    if space.categorical_names:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                space.categorical_names,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(space: FactorSpace, spec: MetamodelSpec) -> Pipeline:
    """Assemble the encode-then-fit pipeline for a fit type."""
    steps: list[tuple[str, Any]] = [("encode", _encoder(space))]

    if spec.fit_type == "linear":
        steps.append(("model", LinearRegression()))
    elif spec.fit_type == "quadratic":
        # Expanding one-hot columns to degree 2 produces some structurally
        # redundant terms (an indicator squared is itself; two levels of one
        # factor multiply to zero). LinearRegression solves via least squares,
        # which handles the resulting rank deficiency by taking the
        # minimum-norm solution, so these cost a little width but not accuracy.
        steps.append(("expand", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("model", LinearRegression()))
    elif spec.fit_type == "kriging":
        # Signal-variance bounds are kept wide: even with ``normalize_y`` the
        # optimizer can want a scale well away from 1, and pinning it at a
        # bound produces a visibly worse fit.
        #
        # ``alpha`` is fixed jitter on the covariance diagonal, distinct from
        # the WhiteKernel's *learned* noise. An analytic solver is exactly
        # interpolable, which drives the learned noise toward zero and leaves
        # the covariance matrix ill-conditioned; a small jitter keeps the
        # Cholesky factorization stable without pretending the data is noisy.
        kernel = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        steps.append(
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=max(spec.kriging_nugget, 1e-12),
                    normalize_y=True,
                    n_restarts_optimizer=spec.kriging_restarts,
                    random_state=0,
                ),
            )
        )
    else:
        raise ValueError(f"unknown fit type: {spec.fit_type!r}")

    return Pipeline(steps)


@dataclass
class Metamodel:
    """A fitted surrogate for one response."""

    spec: MetamodelSpec
    pipeline: Pipeline
    metrics: MetamodelMetrics
    actual: np.ndarray = field(repr=False)
    cv_predicted: np.ndarray = field(repr=False)
    fit_warnings: list[str] = field(default_factory=list)

    @property
    def notable_warnings(self) -> list[str]:
        """Fit warnings that are not just the signature of noiseless data."""
        return [w for w in self.fit_warnings if not is_expected_warning(w)]

    @property
    def is_weak(self) -> bool:
        """Whether this surrogate is too inaccurate to optimize against.

        Judged on cross-validated R², not the in-sample score: kriging
        interpolates its training points exactly, so in-sample R² is ~1.0 for
        any design and says nothing about generalization.
        """
        return bool(np.isfinite(self.metrics.cv_r2)) and self.metrics.cv_r2 < WEAK_FIT_THRESHOLD

    @property
    def response(self) -> str:
        return self.spec.response

    @property
    def label(self) -> str:
        return self.spec.label

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the response over a factor-valued DataFrame."""
        return np.asarray(self.pipeline.predict(df), dtype=float).ravel()


def fit_metamodel(
    space: FactorSpace,
    factors_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    spec: MetamodelSpec,
) -> Metamodel:
    """Fit one metamodel and cross-validate it."""
    if spec.response not in responses_df.columns:
        raise KeyError(f"response {spec.response!r} is not in the results")

    X = factors_df[space.names]
    y = responses_df[spec.response].to_numpy(dtype=float)

    if len(X) < 3:
        raise ValueError("fitting a metamodel needs at least three experiments")

    # Hyperparameter search can legitimately hit a bound or stop early, which
    # sklearn reports as a warning per fit — and cross-validation multiplies
    # that by the fold count. Collect them as data the caller can surface next
    # to the fit instead of letting them stream to stderr.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        pipeline = build_pipeline(space, spec)
        pipeline.fit(X, y)
        fitted = np.asarray(pipeline.predict(X), dtype=float).ravel()

        # Cross-validation needs at least two samples per fold, and cannot use
        # more folds than there are experiments.
        n_splits = int(min(spec.cv_folds, len(X) // 2))
        if n_splits >= 2:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            cv_pred = np.asarray(
                cross_val_predict(build_pipeline(space, spec), X, y, cv=cv),
                dtype=float,
            ).ravel()
            cv_r2 = float(r2_score(y, cv_pred))
            cv_rmse = float(np.sqrt(mean_squared_error(y, cv_pred)))
        else:
            cv_pred = np.full_like(y, np.nan)
            cv_r2 = float("nan")
            cv_rmse = float("nan")

    fit_warnings = sorted(
        {
            str(w.message).strip().splitlines()[0]
            for w in caught
            if issubclass(w.category, ConvergenceWarning)
        }
    )

    metrics = MetamodelMetrics(
        r2=float(r2_score(y, fitted)),
        rmse=float(np.sqrt(mean_squared_error(y, fitted))),
        cv_r2=cv_r2,
        cv_rmse=cv_rmse,
        n_train=len(X),
    )
    return Metamodel(
        spec=spec,
        pipeline=pipeline,
        metrics=metrics,
        actual=y,
        cv_predicted=cv_pred,
        fit_warnings=fit_warnings,
    )


def metrics_table(models: dict[str, Metamodel]) -> pd.DataFrame:
    """Fit-quality table across metamodels, for the assessment view."""
    rows = []
    for key, model in models.items():
        row = {"Metamodel": key, "Response": model.response, "Fit": model.spec.fit_type}
        row.update(model.metrics.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def predict_grid(
    model: Metamodel,
    space: FactorSpace,
    x_factor: str,
    y_factor: str,
    fixed: dict[str, Any],
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a metamodel over a 2-D slice for contour plotting.

    ``x_factor`` and ``y_factor`` are swept across their ranges (or their
    categories) while every other factor is held at its value in ``fixed``.
    Returns ``(x_values, y_values, Z)`` with ``Z`` shaped ``(len(y), len(x))``
    to match what contour plotting expects.
    """
    xs = _sweep_values(space[x_factor], resolution)
    ys = _sweep_values(space[y_factor], resolution)

    grid_x, grid_y = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    frame = pd.DataFrame(
        {name: [value] * grid_x.size for name, value in fixed.items()}
    )
    frame[x_factor] = np.asarray(xs, dtype=object)[grid_x.ravel()]
    frame[y_factor] = np.asarray(ys, dtype=object)[grid_y.ravel()]
    frame = _restore_dtypes(frame, space)

    z = model.predict(frame[space.names]).reshape(len(ys), len(xs))
    return np.asarray(xs), np.asarray(ys), z


def predict_sweep(
    model: Metamodel,
    space: FactorSpace,
    x_factor: str,
    fixed: dict[str, Any],
    resolution: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a metamodel along one factor, holding the rest fixed."""
    xs = _sweep_values(space[x_factor], resolution)
    frame = pd.DataFrame({name: [value] * len(xs) for name, value in fixed.items()})
    frame[x_factor] = xs
    frame = _restore_dtypes(frame, space)
    return np.asarray(xs), model.predict(frame[space.names])


def _sweep_values(factor: Any, resolution: int) -> list[Any]:
    """Points to sweep a factor across: its categories, or an even grid."""
    if factor.is_categorical:
        return list(factor.categories)
    return [float(v) for v in np.linspace(factor.low, factor.high, resolution)]


def _restore_dtypes(frame: pd.DataFrame, space: FactorSpace) -> pd.DataFrame:
    """Re-impose per-factor dtypes after building a frame from object arrays.

    The grid is assembled through ``object`` arrays so categorical labels
    survive; without this the continuous columns would stay object-typed and
    the scaler would reject them.
    """
    for f in space:
        if not f.is_categorical:
            frame[f.name] = frame[f.name].astype(float)
        else:
            frame[f.name] = frame[f.name].astype(str)
    return frame
