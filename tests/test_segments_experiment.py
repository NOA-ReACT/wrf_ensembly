from importlib import resources
from pathlib import Path

from wrf_ensembly import experiment, segments

SEGMENTS = """
[segments]
enabled = true
max_walltime = "12:00:00"
expected_walltime_per_sim_hour = 1800
safety_factor = 1.0
"""


def make_experiment(
    path: Path, assimilation: str = "", extra: str = SEGMENTS
) -> experiment.Experiment:
    """
    An experiment from the default template (24 cycles of 6 h, 20 members) in restart
    mode, with segments enabled. At 1800 s per simulated hour, a 12 h limit fits 24 h
    (4 cycles).
    """

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    text = template.read_text()
    text = text.replace(
        "[assimilation]\n",
        f'[assimilation]\ncycling_mode = "restart"\n{assimilation}\n',
        1,
    )
    text += extra
    path.mkdir()
    (path / "config.toml").write_text(text)
    exp = experiment.Experiment(path)
    exp.paths.create_directories()
    exp.state.initialize()
    return exp


def add_observations(exp: experiment.Experiment, *cycles: int) -> None:
    for i in cycles:
        exp.paths.obs_seq_path(i).touch()


def test_plan_stops_at_walltime(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")

    plan = exp.plan_segment(0)

    assert (plan.first, plan.last, plan.stop_reason) == (0, 3, segments.STOP_WALLTIME)
    assert plan.checkpoint_interval_min == 24 * 60
    assert exp.state.get_segment_plan(0) == plan


def test_plan_stops_at_observations_and_kept_restart_files(tmp_path: Path):
    exp = make_experiment(
        tmp_path / "exp", assimilation="keep_restart_files_for_cycles = [1]"
    )
    add_observations(exp, 2)

    assert exp.plan_segment(0).last == 1
    assert exp.plan_segment(2).last == 2
    assert exp.plan_segment(3).last == 6


def test_plan_stops_at_run_until(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")

    plan = exp.plan_segment(0, run_until=1)

    assert (plan.last, plan.run_until) == (1, 1)


def test_segment_of(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.plan_segment(0)
    exp.plan_segment(4)

    assert exp.segment_of(0).first == 0
    assert exp.segment_of(3).first == 0
    assert exp.segment_of(4).first == 4
    assert exp.segment_of(8) is None


def test_segment_of_without_segments(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")

    assert exp.segment_of(0) is None


def test_segment_started(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    plan = exp.plan_segment(4)

    # Empty directories (made by the namelist generation) don't count
    exp.paths.scratch_restart_path(4, 3).mkdir(parents=True)
    assert not exp.segment_started(plan)

    (exp.paths.scratch_restart_path(4, 3) / "wrfrst_d01_2021-01-02_06:00:00").touch()
    assert exp.segment_started(plan)


def test_segment_started_by_advanced_member(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    plan = exp.plan_segment(4)

    exp.state.set_member_advanced(6, 0)

    assert exp.segment_started(plan)


def test_reset_forgets_plans(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.plan_segment(0)

    exp.state.reset()

    assert exp.state.get_segment_plans() == []


def test_unreadable_plan_is_ignored(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.paths.segment_plan_path(0).parent.mkdir(parents=True, exist_ok=True)
    exp.paths.segment_plan_path(0).write_text('{"first": 0}')

    assert exp.state.get_segment_plan(0) is None
    assert exp.segment_of(0) is None
