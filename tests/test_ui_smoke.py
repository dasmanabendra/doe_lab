"""End-to-end exercise of the UI wiring, driven headlessly.

These tests click through the real pages rather than calling the engine, so
they catch the failures unit tests cannot: a signal wired to nothing, a page
that renders before its data exists, a worker whose result never reaches the
widget, stage gating that lets the user open a page that cannot function.

They run under Qt's ``offscreen`` platform, so no display is needed.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("Agg")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from doelab.engine.project import Stage  # noqa: E402
from doelab.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def dialogs(monkeypatch):
    """Record modal dialogs instead of showing them.

    A ``QMessageBox`` blocks until someone dismisses it, and nobody will in a
    headless run — an unexpected error dialog would hang the suite instead of
    failing it. Recording the calls turns that into a visible assertion, and
    lets tests check what the user would have been told.
    """
    recorded: list[tuple[str, str, str]] = []

    def recorder(kind: str, answer):
        def handler(_parent, title="", text="", *args, **kwargs):
            recorded.append((kind, str(title), str(text)))
            return answer

        return staticmethod(handler)

    monkeypatch.setattr(QMessageBox, "critical", recorder("critical", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "warning", recorder("warning", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "information", recorder("information", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", recorder("question", QMessageBox.StandardButton.Yes))
    return recorded


def pump(app: QApplication, predicate, timeout_s: float = 120.0) -> bool:
    """Spin the event loop until ``predicate`` holds or the timeout expires.

    Page actions dispatch to worker threads, so the result only lands once the
    event loop has processed the completion signal.

    After the predicate holds, the loop is pumped a few more times before
    returning. A completion handler commits project state and *then* renders,
    and rendering can itself re-enter the event loop — so a predicate watching
    project state can go true while its handler is still partway through.
    Settling here means callers observe the finished result rather than a
    half-applied one.
    """
    import time

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            for _ in range(5):
                app.processEvents()
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def axis_menu(monkeypatch):
    """Record what the axis context menu offers instead of showing it.

    ``QMenu.exec`` blocks until someone clicks, which nothing will here.
    Swapping the class the widget module looks up is what makes the menu
    testable; patching the method on ``QMenu`` itself does not take, because
    PySide6 binds it from C++.
    """
    from PySide6.QtWidgets import QMenu

    import doelab.ui.widgets.parallel as widget_module

    class Recorder(QMenu):
        offered: dict[str, bool] = {}
        picker = staticmethod(lambda actions: None)

        def exec(self, *_args, **_kwargs):
            listed = [a for a in self.actions() if a.text()]
            Recorder.offered = {a.text(): a.isEnabled() for a in listed}
            return Recorder.picker(listed)

    Recorder.choose = staticmethod(
        lambda prefix: setattr(
            Recorder, "picker",
            staticmethod(
                (lambda actions: None) if prefix is None
                else (lambda actions: next(a for a in actions if a.text().startswith(prefix)))
            ),
        )
    )
    monkeypatch.setattr(widget_module, "QMenu", Recorder)
    return Recorder


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    # close() stops outstanding work and waits for it; draining the event queue
    # afterwards lets the deferred deletions actually run, so the next test
    # starts without a backlog of half-destroyed widgets and threads.
    w.close()
    qapp.processEvents()


class TestShell:
    def test_opens_on_the_first_stage(self, window):
        assert window.rail.currentRow() == 0
        assert window.stack.currentIndex() == 0

    def test_later_stages_start_unreachable(self, window):
        # Design is available immediately; running experiments is not.
        assert window.project.reached(Stage.DESIGN)
        assert not window.project.reached(Stage.RESULTS)

        from PySide6.QtCore import Qt

        run_item = window.rail.item(2)
        assert not (run_item.flags() & Qt.ItemFlag.ItemIsEnabled)

    def test_clicking_a_blocked_stage_does_not_navigate(self, window):
        window.rail.setCurrentRow(5)  # Optimize, with nothing upstream done
        assert window.stack.currentIndex() == 0

    def test_every_page_is_registered_in_the_rail(self, window):
        assert window.rail.count() == len(window.pages) == window.stack.count()


class TestPipeline:
    """Walks the full workflow the way a user would."""

    def test_generates_a_design(self, qapp, window):
        design_page = window.pages[1]
        design_page.n_spin.setValue(40)
        design_page._generate()

        assert pump(qapp, lambda: window.project.has_design)
        assert len(window.project.design) == 40
        assert design_page.matrix_view.frame.shape[0] == 40

    def test_runs_experiments_and_fills_the_results_table(self, qapp, window):
        design_page = window.pages[1]
        design_page.n_spin.setValue(40)
        design_page._generate()
        assert pump(qapp, lambda: window.project.has_design)

        run_page = window.pages[2]
        run_page.refresh()
        run_page._run()

        assert pump(qapp, lambda: window.project.has_results)
        assert len(window.project.results) == 40
        # Results table shows factors and responses side by side.
        assert "BSFC" in run_page.results_view.frame.columns
        assert "Spark_Timing" in run_page.results_view.frame.columns

    def test_analysis_populates_once_results_exist(self, qapp, window):
        _advance_to_results(qapp, window)

        analyze_page = window.pages[3]
        analyze_page.refresh()

        assert not analyze_page.sensitivity_view.frame.empty
        assert not analyze_page.partial_view.frame.empty
        assert not analyze_page.pearson_view.frame.empty
        assert not analyze_page.spearman_view.frame.empty

    def test_partial_r_squared_tab_is_a_fraction_per_factor(self, qapp, window):
        _advance_to_results(qapp, window)

        analyze_page = window.pages[3]
        analyze_page.refresh()

        frame = analyze_page.partial_view.frame
        assert list(frame.columns) == ["Response"] + window.project.factors.names
        shares = frame.drop(columns="Response").to_numpy(dtype=float)
        assert shares.min() >= 0.0 and shares.max() <= 1.0

    def test_partial_r_squared_tab_clears_with_the_rest(self, qapp, window):
        """Every view on the page has to empty together, or a stale table
        reads as analysis of a design that no longer exists."""
        _advance_to_results(qapp, window)

        analyze_page = window.pages[3]
        analyze_page.refresh()
        assert not analyze_page.partial_view.frame.empty

        analyze_page._clear()
        assert analyze_page.partial_view.frame.empty

    def test_fits_metamodels_and_reports_metrics(self, qapp, window):
        _advance_to_results(qapp, window)

        metamodel_page = window.pages[4]
        metamodel_page.refresh()
        metamodel_page._fit()

        assert pump(qapp, lambda: window.project.has_metamodels)
        assert not metamodel_page.metrics_view.frame.empty
        assert {"BSFC", "Torque", "Max_Cyl_Pressure"} <= set(window.project.metamodels)

    def test_optimizes_and_produces_a_front(self, qapp, window):
        _advance_to_metamodels(qapp, window)

        optimize_page = window.pages[5]
        optimize_page.refresh()
        optimize_page.pop_spin.setValue(24)
        optimize_page.gen_spin.setValue(8)
        optimize_page._start()

        assert pump(qapp, lambda: optimize_page._result is not None)
        result = optimize_page._result

        assert result.generations_completed == 8
        assert not result.pareto.empty
        assert not optimize_page.front_view.frame.empty

    def test_validation_compares_the_front_against_the_solver(self, qapp, window):
        _advance_to_metamodels(qapp, window)

        optimize_page = window.pages[5]
        optimize_page.refresh()
        optimize_page.pop_spin.setValue(24)
        optimize_page.gen_spin.setValue(8)
        optimize_page._start()
        assert pump(qapp, lambda: optimize_page._result is not None)

        optimize_page._validate()
        assert pump(qapp, lambda: not optimize_page.validation_view.frame.empty)
        assert "RMSE" in optimize_page.validation_view.frame.columns


class TestInvalidation:
    def test_changing_the_problem_clears_downstream_work(self, qapp, window):
        _advance_to_metamodels(qapp, window)
        assert window.project.has_metamodels

        problem_page = window.pages[0]
        index = problem_page.problem_combo.findData("branin")
        problem_page.problem_combo.setCurrentIndex(index)

        assert not window.project.has_design
        assert not window.project.has_results
        assert not window.project.has_metamodels
        assert window.project.factors.names == ["x1", "x2"]

    def test_regenerating_a_design_clears_stale_results(self, qapp, window):
        _advance_to_results(qapp, window)
        assert window.project.has_results

        design_page = window.pages[1]
        design_page.n_spin.setValue(30)
        design_page._generate()
        assert pump(qapp, lambda: len(window.project.design) == 30)

        assert not window.project.has_results

    def test_editing_noise_invalidates_results_but_keeps_the_design(self, qapp, window):
        _advance_to_results(qapp, window)

        problem_page = window.pages[0]
        problem_page.noise_check.setChecked(True)

        assert window.project.has_design
        assert not window.project.has_results

    def test_the_rail_relocks_stages_after_invalidation(self, qapp, window):
        _advance_to_results(qapp, window)
        assert window.project.reached(Stage.METAMODELS)

        window.pages[0].noise_check.setChecked(True)
        window._refresh_rail()

        assert not window.project.reached(Stage.METAMODELS)


class TestRailGating:
    """The rail's enabled state must track reachability at every step.

    Gating is what stops a user opening a stage whose inputs do not exist, so a
    rail that falls out of step either blocks work that is ready or admits work
    that is not.
    """

    @staticmethod
    def _mismatches(window) -> list[str]:
        from PySide6.QtCore import Qt

        return [
            page.title
            for row, page in enumerate(window.pages)
            if bool(window.rail.item(row).flags() & Qt.ItemFlag.ItemIsEnabled)
            != window.project.reached(page.stage)
        ]

    def test_matches_reachability_on_a_fresh_project(self, window):
        assert self._mismatches(window) == []

    def test_unlocks_the_run_stage_once_a_design_exists(self, qapp, window):
        design_page = window.pages[1]
        design_page.refresh()
        design_page.n_spin.setValue(30)
        design_page._generate()
        assert pump(qapp, lambda: window.project.has_design)

        assert window.project.reached(Stage.RESULTS)
        assert self._mismatches(window) == []

    def test_matches_reachability_after_every_stage(self, qapp, window):
        _advance_to_results(qapp, window)
        assert self._mismatches(window) == []

        _advance_to_metamodels(qapp, window)
        assert self._mismatches(window) == []

    def test_matches_reachability_after_invalidation(self, qapp, window):
        _advance_to_metamodels(qapp, window)
        window.pages[0].noise_check.setChecked(True)
        qapp.processEvents()

        assert self._mismatches(window) == []


class TestBusyLocking:
    """Navigation is locked while a stage recomputes.

    Without this, a slow regeneration can complete after the user has moved on
    and built downstream work from the previous result. The late completion
    invalidates that work, leaving the project in a state its own rules say is
    impossible — metamodels fitted to experiments that no longer exist.
    """

    def test_rail_locks_while_a_stage_is_working(self, qapp, window):
        design_page = window.pages[1]
        design_page.refresh()
        design_page.n_spin.setValue(30)

        assert window.rail.isEnabled()
        design_page._generate()
        assert not window.rail.isEnabled(), "navigation stayed open during work"

        assert pump(qapp, lambda: window.project.has_design)
        assert window.rail.isEnabled(), "navigation stayed locked after work finished"

    def test_busy_clears_even_when_the_work_fails(self, qapp, window, dialogs):
        """A failure must release the lock, or the app is stuck for good."""
        design_page = window.pages[1]
        design_page.refresh()
        design_page.set_busy(True)
        assert not window.rail.isEnabled()

        design_page._on_failed("RuntimeError: boom")

        assert not design_page.is_busy
        assert window.rail.isEnabled()
        assert [kind for kind, *_ in dialogs] == ["critical"]

    def test_a_completed_pipeline_never_holds_impossible_state(self, qapp, window):
        """Metamodels must never outlive the results they were fitted to."""
        _advance_to_metamodels(qapp, window)

        assert window.project.has_metamodels
        assert window.project.has_results, "metamodels exist without their experiments"
        assert window.project.has_design


class TestPersistenceThroughUI:
    def test_saves_and_reopens_a_completed_study(self, qapp, window, tmp_path):
        _advance_to_metamodels(qapp, window)

        path = tmp_path / "study.doelab.json"
        window.project.save(path)

        window.new_project()
        assert not window.project.has_design

        from doelab.engine.project import Project

        window._adopt(Project.load(path))

        assert window.project.has_results
        assert window.project.has_metamodels
        # Reopening lands on the furthest completed stage, not back at step one.
        assert window.rail.currentRow() > 0


# -- helpers ----------------------------------------------------------------


def _advance_to_results(qapp, window, n: int = 40) -> None:
    design_page = window.pages[1]
    design_page.refresh()
    design_page.n_spin.setValue(n)
    design_page._generate()
    assert pump(qapp, lambda: window.project.has_design)

    run_page = window.pages[2]
    run_page.refresh()
    run_page._run()
    assert pump(qapp, lambda: window.project.has_results)


def _advance_to_metamodels(qapp, window, n: int = 40) -> None:
    _advance_to_results(qapp, window, n)
    metamodel_page = window.pages[4]
    metamodel_page.refresh()
    metamodel_page._fit()
    assert pump(qapp, lambda: window.project.has_metamodels)


class TestDesignExplorer:
    """The parallel coordinates tab, on both pages that carry it.

    Only the wiring is asserted here. How the plot actually *renders* cannot be
    checked under the offscreen platform, and the axis arithmetic it depends on
    is covered headlessly in ``test_parallel.py``.
    """

    def test_it_populates_once_results_exist(self, qapp, window):
        _advance_to_results(qapp, window)

        analyze_page = window.pages[3]
        analyze_page.refresh()
        explorer = analyze_page.explorer

        assert len(explorer._brushes) == len(window.project.factors) + len(
            window.project.responses
        )
        assert "40 designs" in explorer.count_label.text()

    def test_brushing_an_axis_fades_designs_rather_than_dropping_them(self, qapp, window):
        """The excluded designs are the context the kept ones are read against."""
        _advance_to_results(qapp, window)

        explorer = window.pages[3].explorer
        window.pages[3].refresh()
        before = len(explorer._frame)

        explorer._brushes[0].set_band(0.4, 0.6)
        explorer._redraw()

        assert "of 40 designs pass the filters" in explorer.count_label.text()
        assert len(explorer._frame) == before, "filtering removed rows from the table"
        assert explorer.reset_button.isEnabled()

        explorer.reset_filters()
        assert "40 designs" in explorer.count_label.text()
        assert not explorer.reset_button.isEnabled()

    def test_reordering_an_axis_keeps_its_filter(self, qapp, window):
        """Moving an axis asks about adjacency, not about which designs matter."""
        _advance_to_results(qapp, window)
        window.pages[3].refresh()
        explorer = window.pages[3].explorer

        explorer._brushes[0].set_band(0.25, 0.75)
        explorer._redraw()
        filtered = explorer.count_label.text()

        explorer._swap(0, 1)

        assert explorer._brushes[1].band == (0.25, 0.75)
        assert explorer._brushes[0].band == (0.0, 1.0)
        assert explorer.count_label.text() == filtered

    def test_the_axis_menu_offers_only_the_moves_that_exist(self, qapp, window, axis_menu):
        _advance_to_results(qapp, window)
        window.pages[3].refresh()
        explorer = window.pages[3].explorer
        last = len(explorer._order) - 1

        explorer._show_axis_menu(0, QPointF(0, 0))
        assert axis_menu.offered["Move left"] is False
        assert axis_menu.offered["Move right"] is True

        explorer._show_axis_menu(last, QPointF(0, 0))
        assert axis_menu.offered["Move left"] is True
        assert axis_menu.offered["Move right"] is False

        # Nothing has been hidden or moved yet, so there is nothing to restore.
        assert axis_menu.offered["Show all, original order"] is False

    def test_dismissing_the_axis_menu_changes_nothing(self, qapp, window, axis_menu):
        _advance_to_results(qapp, window)
        window.pages[3].refresh()
        explorer = window.pages[3].explorer
        before = list(explorer._order)

        axis_menu.choose(None)
        explorer._show_axis_menu(2, QPointF(0, 0))

        assert explorer._order == before

    def test_hiding_an_axis_drops_it_from_the_plot(self, qapp, window):
        _advance_to_results(qapp, window)
        window.pages[3].refresh()
        explorer = window.pages[3].explorer
        total = len(explorer._brushes)

        explorer._reorder([i for i in explorer._order if i != 0])
        assert len(explorer._brushes) == total - 1

        explorer._reorder(list(range(total)))
        assert len(explorer._brushes) == total

    def test_it_clears_when_the_results_go_away(self, qapp, window):
        _advance_to_results(qapp, window)
        window.pages[3].refresh()
        assert window.pages[3].explorer._brushes

        window.pages[0].noise_check.setChecked(True)  # invalidates the results
        window.pages[3].refresh()

        assert window.pages[3].explorer._brushes == []

    def test_the_optimize_page_explores_the_front(self, qapp, window):
        _advance_to_metamodels(qapp, window)

        optimize_page = window.pages[5]
        optimize_page.refresh()
        optimize_page.pop_spin.setValue(24)
        optimize_page.gen_spin.setValue(8)
        optimize_page._start()
        assert pump(qapp, lambda: optimize_page._result is not None)

        explorer = optimize_page.front_explorer
        assert len(explorer._frame) == len(optimize_page._result.pareto)
        # The front carries Generation and Feasible columns that are neither a
        # factor nor a response; they must not become axes.
        assert {a.name for a in explorer._axes}.isdisjoint({"Generation", "Feasible"})


class TestIndicatorStyling:
    """The theme must describe checked indicators, not just uncheck ones.

    Styling ``QWidget``'s background matches radios and checkboxes too, which
    moves their painting from the platform style to Qt's stylesheet engine.
    That engine draws only what the sheet describes, so with no rule for the
    checked state a selected control renders as blank space -- clicking one
    looks like it vanished. The native style never gets a chance to draw it
    back, and the offscreen platform used by these tests picks a different base
    style than Windows does, so this is asserted on the sheet itself rather
    than on rendered pixels.
    """

    def test_checked_states_are_defined(self):
        from doelab.ui.theme import STYLESHEET

        assert "QRadioButton::indicator:checked" in STYLESHEET
        assert "QCheckBox::indicator:checked" in STYLESHEET

    def test_the_checkbox_glyph_is_shipped(self):
        import os

        from doelab.ui.theme import CHECK_GLYPH

        assert os.path.exists(CHECK_GLYPH), CHECK_GLYPH
