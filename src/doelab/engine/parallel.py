"""Axis geometry for the parallel coordinates view.

A parallel coordinates plot gives every variable its own vertical axis and
draws each design as a polyline crossing them. To do that, values on wildly
different scales -- degrees of spark timing, RPM, grams per kilowatt-hour --
all have to be mapped onto one shared ``[0, 1]`` height. That mapping, the
range brushing built on top of it, and the constraint bounds drawn as rails
are all pure arithmetic over a table, so they live here rather than in the
widget: this module is testable without a display, and the widget is left with
nothing but painting.

**Axes are scaled to the data, not to the declared factor ranges.** An axis
stretched to a factor's full declared interval when the design only occupies
the middle third wastes two thirds of its height and flattens the very
structure the plot exists to show. The end labels print the actual numbers, so
a design that under-fills its space is still visible -- it just does not cost
every other axis its resolution.

Read this alongside :mod:`doelab.engine.analysis`. Those summaries collapse the
design to one number per factor-response pair; this keeps every individual run,
which is what makes the runs that disagree with a trend findable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .factors import FactorSpace, Response, ResponseRole

# Matches the tolerance the optimizer applies to its constraint values, so a
# design this module calls feasible is one ``optimize.py`` would agree with.
TOLERANCE = 1e-9

FULL_BAND = (0.0, 1.0)


@dataclass(frozen=True)
class Axis:
    """One vertical axis: what it shows, and how values map onto its height."""

    name: str
    group: str  # "factor" or "response"
    low: float
    high: float
    categories: tuple[str, ...] | None = None
    unit: str = ""
    direction: str | None = None  # "min" or "max" for objectives, else None
    limits: tuple[float | None, float | None] = (None, None)

    @property
    def is_categorical(self) -> bool:
        return self.categories is not None

    def positions(self, values: np.ndarray | Sequence) -> np.ndarray:
        """Map raw values onto ``[0, 1]``.

        A value this axis cannot place -- a category it does not know, a
        missing number -- becomes NaN rather than being guessed at. Callers
        decide what an unplaceable design means; see :func:`filter_mask`.
        """
        if self.is_categorical:
            assert self.categories is not None
            lookup = {c: i for i, c in enumerate(self.categories)}
            span = max(len(self.categories) - 1, 1)
            return np.array(
                [lookup.get(str(v), np.nan) for v in np.asarray(values, dtype=object)],
                dtype=float,
            ) / span

        column = np.asarray(values, dtype=float)
        span = self.high - self.low
        if span <= TOLERANCE:
            # A constant column has no shape to show. Centring it says so
            # honestly; dividing by the span would produce infinities.
            return np.where(np.isfinite(column), 0.5, np.nan)
        return (column - self.low) / span

    def value_at(self, position: float) -> float | str:
        """Inverse of :meth:`positions`, for reporting a brushed bound."""
        if self.is_categorical:
            assert self.categories is not None
            index = int(round(position * max(len(self.categories) - 1, 1)))
            return self.categories[min(max(index, 0), len(self.categories) - 1)]
        return self.low + position * (self.high - self.low)

    def ticks(self) -> list[tuple[float, str]]:
        """Labelled positions along the axis.

        Continuous axes are labelled at their ends only -- the polylines are
        the content, and intermediate gridlines just add ink. Categorical axes
        get one tick per level, because an unlabelled position on a nominal
        scale means nothing at all.
        """
        if self.is_categorical:
            assert self.categories is not None
            span = max(len(self.categories) - 1, 1)
            return [(i / span, c) for i, c in enumerate(self.categories)]
        return [(0.0, _format(self.low)), (1.0, _format(self.high))]


def _format(value: float) -> str:
    """Compact number for an axis end label."""
    return f"{value:.4g}"


def _data_range(column: np.ndarray) -> tuple[float, float]:
    """Finite min and max, with a usable fallback for a column of nothing."""
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def build_axes(
    space: FactorSpace, responses: Sequence[Response], frame: pd.DataFrame
) -> list[Axis]:
    """One axis per factor then one per response, in project order.

    Responses the optimizer ignores are kept. ``Power`` carries no role in the
    gasoline problem but is still a measured output, and hiding it here would
    make the plot show less than the results table it sits beside.

    Columns the frame does not carry are skipped, which is what lets the same
    call serve both the full experiment table and an optimizer front -- the
    front adds ``Generation`` and ``Feasible`` columns and may omit ignored
    responses entirely.
    """
    if frame.empty:
        return []

    axes: list[Axis] = []

    for factor in space:
        if factor.name not in frame.columns:
            continue
        if factor.is_categorical:
            axes.append(
                Axis(
                    name=factor.name,
                    group="factor",
                    low=0.0,
                    high=float(max(len(factor.categories) - 1, 1)),
                    categories=tuple(factor.categories),
                    unit=factor.unit,
                )
            )
        else:
            low, high = _data_range(frame[factor.name].to_numpy(dtype=float))
            axes.append(
                Axis(name=factor.name, group="factor", low=low, high=high, unit=factor.unit)
            )

    for response in responses:
        if response.name not in frame.columns:
            continue
        low, high = _data_range(frame[response.name].to_numpy(dtype=float))
        direction = (
            "min" if response.role is ResponseRole.OBJECTIVE_MIN
            else "max" if response.role is ResponseRole.OBJECTIVE_MAX
            else None
        )
        limits = (
            (response.lower, response.upper)
            if response.role is ResponseRole.CONSTRAINT
            else (None, None)
        )
        axes.append(
            Axis(
                name=response.name,
                group="response",
                low=low,
                high=high,
                unit=response.unit,
                direction=direction,
                limits=limits,
            )
        )

    return axes


def normalize(frame: pd.DataFrame, axes: Sequence[Axis]) -> np.ndarray:
    """Every design's position on every axis, as ``(n_rows, n_axes)`` in ``[0, 1]``."""
    values = np.empty((len(frame), len(axes)), dtype=float)
    for index, axis in enumerate(axes):
        values[:, index] = axis.positions(frame[axis.name].to_numpy())
    return values


def filter_mask(values: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Which designs fall inside every axis's brushed band.

    ``bands`` is ``(n_axes, 2)`` of normalized lower and upper bounds. A design
    passes when it is inside all of them, so brushing several axes intersects
    rather than unions -- that is what makes stacking filters carve out a
    region instead of accumulating one.

    A design an axis could not place is excluded only if that axis is actually
    being filtered. Excluding it unconditionally would mean an untouched plot
    reported fewer designs than it holds, which reads as data silently going
    missing rather than as a filter doing its job.
    """
    if values.size == 0:
        return np.zeros(len(values), dtype=bool)

    bands = np.asarray(bands, dtype=float)
    lower = bands[:, 0] - TOLERANCE
    upper = bands[:, 1] + TOLERANCE

    inside = (values >= lower) & (values <= upper)
    unplaceable = ~np.isfinite(values)
    untouched = (bands[:, 0] <= TOLERANCE) & (bands[:, 1] >= 1.0 - TOLERANCE)

    return (inside | (unplaceable & untouched)).all(axis=1)


def feasibility(frame: pd.DataFrame, responses: Sequence[Response]) -> np.ndarray:
    """Which designs satisfy every constraint response's bounds.

    Separate from :func:`filter_mask` on purpose: a brush is the user's
    question, a constraint is the problem's own rule. Conflating them would let
    widening a filter appear to make an infeasible design acceptable.
    """
    feasible = np.ones(len(frame), dtype=bool)
    for response in responses:
        if response.role is not ResponseRole.CONSTRAINT:
            continue
        if response.name not in frame.columns:
            continue
        values = frame[response.name].to_numpy(dtype=float)
        if response.upper is not None:
            feasible &= values <= response.upper + TOLERANCE
        if response.lower is not None:
            feasible &= values >= response.lower - TOLERANCE
    return feasible


def rail_positions(axes: Sequence[Axis]) -> dict[int, list[tuple[float, float]]]:
    """Where to draw each constraint bound, keyed by axis index.

    Bounds outside the data range are dropped. A limit no design came near
    cannot be drawn on an axis scaled to the designs, and a rail clamped to the
    axis end would claim designs are sitting on a boundary they are nowhere
    near. The violation count in the header carries that information instead.
    """
    rails: dict[int, list[tuple[float, float]]] = {}
    for index, axis in enumerate(axes):
        placed = []
        for bound in axis.limits:
            if bound is None:
                continue
            position = float(axis.positions(np.array([bound]))[0])
            if np.isfinite(position) and 0.0 <= position <= 1.0:
                placed.append((position, float(bound)))
        if placed:
            rails[index] = placed
    return rails
