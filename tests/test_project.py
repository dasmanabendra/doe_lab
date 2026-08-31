"""Project persistence and stage gating."""

from __future__ import annotations

import json

import numpy as np
import pytest

from doelab.engine import doe
from doelab.engine.metamodel import MetamodelSpec
from doelab.engine.project import Project, Stage
from doelab.engine.solver import NoiseConfig


@pytest.fixture
def run_project(engine_problem):
    """A project carried through design, results and metamodels."""
    project = Project.from_problem("gasoline_engine")
    project.design_spec = doe.DesignSpec(kind="latin_hypercube", n_experiments=60, seed=4)
    project.design = doe.generate(project.factors, project.design_spec)
    project.results = engine_problem.evaluate(project.design, project.noise)
    project.fit_metamodels([MetamodelSpec("BSFC", "quadratic"), MetamodelSpec("Torque", "quadratic")])
    return project


class TestConstruction:
    def test_starts_populated_from_the_chosen_problem(self):
        project = Project.from_problem("gasoline_engine")

        assert project.factors is not None
        assert "Fuel_Type" in project.factors.names
        assert [r.name for r in project.responses][0] == "BSFC"

    def test_rejects_an_unknown_problem(self):
        with pytest.raises(KeyError):
            Project.from_problem("no_such_problem")

    def test_default_specs_skip_ignored_responses(self):
        project = Project.from_problem("gasoline_engine")
        names = [s.response for s in project.default_metamodel_specs()]

        assert "Power" not in names  # Power is role=ignored
        assert {"BSFC", "Torque", "Max_Cyl_Pressure"} == set(names)


class TestStageGating:
    def test_a_fresh_project_can_only_reach_the_design_stage(self):
        project = Project.from_problem("gasoline_engine")

        assert project.reached(Stage.DESIGN)
        assert not project.reached(Stage.RESULTS)
        assert not project.reached(Stage.METAMODELS)
        assert not project.reached(Stage.OPTIMIZE)

    def test_a_completed_project_reaches_every_stage(self, run_project):
        assert all(run_project.reached(s) for s in Stage)

    def test_blocked_stages_explain_themselves(self):
        project = Project.from_problem("gasoline_engine")

        assert project.blocker(Stage.DESIGN) is None
        assert "design" in project.blocker(Stage.RESULTS).lower()
        assert "experiments" in project.blocker(Stage.METAMODELS).lower()

    def test_editing_factors_discards_downstream_artifacts(self, run_project):
        """Stale results would otherwise be attributed to factors that changed."""
        assert run_project.has_results and run_project.has_metamodels

        run_project.invalidate_from(Stage.DESIGN)

        assert not run_project.has_design
        assert not run_project.has_results
        assert not run_project.has_metamodels

    def test_refitting_metamodels_leaves_the_design_intact(self, run_project):
        run_project.invalidate_from(Stage.METAMODELS)

        assert run_project.has_design
        assert run_project.has_results
        assert not run_project.has_metamodels


class TestMetamodelFitting:
    def test_refuses_to_fit_before_the_experiments_run(self):
        project = Project.from_problem("gasoline_engine")
        with pytest.raises(ValueError, match="run the experiments"):
            project.fit_metamodels([MetamodelSpec("BSFC", "linear")])

    def test_refuses_an_empty_request(self, run_project):
        with pytest.raises(ValueError, match="no metamodels requested"):
            run_project.fit_metamodels([])

    def test_keys_models_by_response(self, run_project):
        assert set(run_project.metamodels) == {"BSFC", "Torque"}


class TestPersistence:
    def test_round_trips_through_a_file(self, run_project, tmp_path):
        path = run_project.save(tmp_path / "study.doelab.json")
        loaded = Project.load(path)

        assert loaded.problem_name == run_project.problem_name
        assert loaded.factors.names == run_project.factors.names
        assert loaded.design.equals(run_project.design)
        assert np.allclose(loaded.results.to_numpy(), run_project.results.to_numpy())

    def test_preserves_categorical_values_exactly(self, run_project, tmp_path):
        path = run_project.save(tmp_path / "study.doelab.json")
        loaded = Project.load(path)

        assert loaded.design["Fuel_Type"].tolist() == run_project.design["Fuel_Type"].tolist()

    def test_preserves_response_roles_and_bounds(self, run_project, tmp_path):
        loaded = Project.load(run_project.save(tmp_path / "study.doelab.json"))

        original = {r.name: (r.role, r.upper) for r in run_project.responses}
        restored = {r.name: (r.role, r.upper) for r in loaded.responses}
        assert original == restored

    def test_refits_metamodels_on_load(self, run_project, tmp_path):
        """Fitted estimators are not serialized, so they must come back by refitting."""
        path = run_project.save(tmp_path / "study.doelab.json")
        loaded = Project.load(path)

        assert set(loaded.metamodels) == set(run_project.metamodels)
        sample = loaded.design.head(5)
        assert np.allclose(
            loaded.metamodels["BSFC"].predict(sample),
            run_project.metamodels["BSFC"].predict(sample),
        )

    def test_can_skip_refitting(self, run_project, tmp_path):
        path = run_project.save(tmp_path / "study.doelab.json")
        loaded = Project.load(path, refit=False)

        assert loaded.metamodels == {}
        assert loaded.metamodel_specs  # the specs survive for a later refit

    def test_saved_file_holds_no_pickled_estimator(self, run_project, tmp_path):
        path = run_project.save(tmp_path / "study.doelab.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert "metamodels" not in payload
        assert payload["metamodel_specs"]

    def test_stores_noise_configuration(self, tmp_path, engine_problem):
        project = Project.from_problem("gasoline_engine")
        project.noise = NoiseConfig(enabled=True, sigma=0.03, seed=17)
        loaded = Project.load(project.save(tmp_path / "s.doelab.json"))

        assert loaded.noise.enabled and loaded.noise.sigma == 0.03 and loaded.noise.seed == 17

    def test_rejects_a_file_from_a_newer_build(self, run_project, tmp_path):
        path = tmp_path / "future.doelab.json"
        payload = run_project.to_dict()
        payload["version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="newer than this build"):
            Project.load(path)

    def test_an_empty_project_survives_a_round_trip(self, tmp_path):
        project = Project.from_problem("branin")
        loaded = Project.load(project.save(tmp_path / "empty.doelab.json"))

        assert loaded.design is None
        assert loaded.results is None
        assert not loaded.has_design
