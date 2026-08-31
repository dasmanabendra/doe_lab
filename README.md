# DOE Lab

A local desktop application for Design of Experiments, metamodeling, and
multi-objective optimization. Everything is Python, everything runs on your
machine: no server, no database, no browser.

An expensive physics solver is replaced by a library of **analytic response
surfaces** built from polynomial and trigonometric terms, so the whole
workflow — design, run, analyze, fit, optimize — executes in seconds while
still exhibiting the structure a DOE study exists to uncover: interior optima,
factor interactions, and objectives that genuinely compete.

## Documentation

**[Full reference &rarr; `docs/REFERENCE.md`](docs/REFERENCE.md)** — 17 chapters covering the
workflow, every statistical method with worked numbers, the implementation, and the test suite.

New to design of experiments? [Chapter 16](docs/REFERENCE.md#16--the-algorithms-in-plain-terms)
explains every algorithm and term from scratch — Latin hypercube, D-optimal exchange, R²,
cross-validation, kriging, NSGA-II, Pareto fronts, hypervolume — and
[Chapter 17](docs/REFERENCE.md#17--glossary) is a glossary of the vocabulary.

## Running it

```bash
.venv/Scripts/python -m doelab
```

```bash
.venv/Scripts/python -m pytest
```

## The workflow

Six ordered stages down the left rail. Each stage's output is the next one's
input, so a stage unlocks only once its prerequisites exist, and editing
anything upstream discards what depended on it.

1. **Problem** — pick a response surface, set factor ranges and level counts,
   and give each response a role: minimize, maximize, constrain, or ignore.
   Optional measurement noise, since a design of experiments is only
   interesting when observations are not perfectly repeatable.
2. **Design** — Full Factorial, Latin Hypercube, or D-Optimal. The scatter
   shows where runs actually land, which is the fastest way to see what a
   design type buys you.
3. **Run** — evaluate the design and collect responses.
4. **Analyze** — factor sensitivity and **partial R²**, plus Pearson and
   Spearman correlations. Where Spearman is strong but Pearson is weak, the
   relationship is monotonic but curved — a sign a linear metamodel will not be
   enough. The **Design explorer** tab adds a parallel coordinates plot: one
   axis per variable, one line per experiment. The other tabs give a number per
   factor-response pair; this keeps the individual runs, so the ones that
   disagree with a trend stay findable.
5. **Metamodels** — fit Linear, Quadratic, or Kriging surrogates per response,
   with cross-validated metrics and an interactive contour/sweep explorer.
6. **Optimize** — NSGA-II over the surrogates, with a live Pareto front and
   working Pause / Stop / Extend, then **Validate** the front against the real
   solver. The **Front explorer** tab draws the finished front in parallel
   coordinates, for reading what tightening one objective costs on the others.

## Mixed factor types

Continuous and categorical factors are supported throughout, which shapes
several choices:

- **Latin Hypercube** bins each categorical factor's coordinate into one equal
  stratum per level, preserving the stratification guarantee instead of
  degrading to random assignment.
- **D-Optimal** is the design to reach for with mixed factors and a fixed run
  budget, since a full factorial's run count multiplies out with every
  categorical level. It uses Fedorov point exchange with a rank-one update of
  `(XᵀX)⁻¹`, evaluating the whole exchange matrix per iteration rather than
  refactorizing a determinant per candidate pair.
- The quadratic model matrix omits two families of structurally degenerate
  column: squares of indicators (`x² == x`), and interactions between
  indicators of the *same* categorical factor (mutually exclusive, so their
  product is identically zero). Either would make `XᵀX` singular for every
  possible design.
- **NSGA-II** declares each factor as `Real` or `Choice`, so the genetic
  operators handle categories natively.

## Sensitivity versus partial R²

Both come from the same linear fit, and they are both in the app because
normalizing hides two things worth seeing.

**Factor sensitivity** divides every factor's squared standardized coefficient
by their total, so each row sums to 1. That makes factors easy to rank against
each other, at the cost of dividing the model's own R² out by construction: a
row looks identical whether the fit explains 95% of the response or 4%. On
Branin it reports a tidy split between `x1` and `x2` for a model whose R² is
0.04.

**Partial R²** drops one factor, refits, and measures what the fit lost. Nothing
is normalized, so the bar chart can partition each response into three parts
that sum to 1:

- **unique** — what each factor alone accounts for (its semi-partial R²)
- **shared** — explained by the model, but not attributable to any single
  factor, because the design moved those factors together
- **unexplained** — `1 - R²`, the part a *linear* model does not reach, and the
  best single argument for a quadratic or Kriging surrogate

A bar that runs past 1 is the sharper version of the same warning: the unique
contributions overlap to more than the whole model explains. Latin hypercube
sampling stratifies each factor independently but does not orthogonalize them,
so at moderate run counts the columns retain real correlation — a 6-factor,
80-run ZDT1 design leaves pairwise correlations around 0.2, enough to push the
uniques 17% past R². Orthogonalizing the same columns collapses the excess to
exactly zero, which is the identity the engine tests assert.

The tables shade differently for the same reason. Sensitivity shades each
column over its own range, since the rows are shares of different totals.
Partial R² shades over a fixed 0-1, because a dark cell has to mean "large"
everywhere in the table rather than "largest in this column".

## Reading the design explorer

A parallel coordinates plot is read between **adjacent** axes. Lines that stay
parallel from one axis to the next mean the two variables move together; lines
that cross in an X mean they trade off. Non-neighbouring axes say almost
nothing, so right-clicking an axis to move it beside the one you want to
compare against is the main analytic action, not a cosmetic one — the default
factor-then-response ordering exposes only the pairs it happens to put together.

Dragging the handles at either end of an axis brushes it to a range. Designs
that fall outside fade to grey rather than disappearing: the excluded designs
are the context that makes the surviving ones mean anything, and keeping them
on the plot makes widening a band an obviously reversible act. Brushing several
axes intersects, so filters stack into a region of the design space.

Axes are scaled to the data rather than to the declared factor ranges — an axis
stretched to an interval the design only half occupies wastes half its height.
The numbers at each end print the actual extent. Constraint bounds are drawn as
a dashed rail on the response they bound, taken straight from the response
definition, and designs breaking one are drawn in red whether or not they pass
the brushes: a filter is the user's question, a constraint is the problem's own
rule.

## Why validation exists

The optimizer searches the *surrogate*, not the solver — that is the point of
building metamodels, since a genetic algorithm needs tens of thousands of
evaluations. But the search pushes into regions the design sampled thinly,
typically the corners of the space, which is exactly where optima like to sit.
Cross-validation cannot detect this: it only scores the surrogate where
training data already exists.

**Validate front** re-runs the true solver on the handful of Pareto designs —
cheap next to the thousands of surrogate evaluations the search consumed, and
the only honest way to know whether the front is real.

## Architecture

```
src/doelab/
  engine/      pure Python, no Qt imports at all — fully testable headless
  ui/          PySide6, a thin client over the engine
```

The layer boundary is asserted by a test that walks each engine module's import
graph, because it erodes silently: one convenient `QObject` import and the
engine needs a running application to import.

Long operations run on a `QThread` and deliver results through a relay object
owned by the UI thread. Connecting a worker signal straight to a plain callable
gives Qt no receiver, so it cannot determine a target thread and invokes the
callback *on the worker* — which is not safe for code that touches widgets.

Projects save to a single JSON file. Fitted scikit-learn estimators are not
serialized; only their specs are, and they are refit on load. Refitting costs a
second, whereas pickling estimators would tie saved files to the exact library
version that wrote them.
