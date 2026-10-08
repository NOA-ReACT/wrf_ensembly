import datetime as dt
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from wrf_ensembly import experiment, jobfiles, segments
from wrf_ensembly.commands import slurm as slurm_commands
from wrf_ensembly.experiment import CycleState

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


def fake_segment_run(exp: experiment.Experiment, member_i: int, first: int, last: int):
    """The files WRF writes for a member running cycles first-last as one run: hourly
    wrfouts and a restart file at the end, all in the first cycle's directories"""

    forecasts = exp.paths.scratch_forecasts_path(first, member_i)
    forecasts.mkdir(parents=True, exist_ok=True)
    t = exp.cycles[first].start
    while t <= exp.cycles[last].end:
        (forecasts / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}").touch()
        t += dt.timedelta(hours=1)
    restarts = exp.paths.scratch_restart_path(first, member_i)
    restarts.mkdir(parents=True, exist_ok=True)
    (restarts / f"wrfrst_d01_{exp.cycles[last].end:%Y-%m-%d_%H:%M:%S}").touch()


def advance_segment(exp: experiment.Experiment, first: int, last: int):
    """What `advance_member` does after wrf.exe, for every member"""

    for m in range(exp.cfg.assimilation.n_members):
        fake_segment_run(exp, m, first, last)
        exp.file_segment_outputs(m, range(first, last + 1))
        for c in reversed(range(first, last + 1)):
            exp.state.set_member_advanced(c, m)


def wrfout_hours(exp: experiment.Experiment, cycle_i: int, member_i: int = 0):
    return sorted(
        int(f.name[-8:-6])
        for f in exp.paths.scratch_forecasts_path(cycle_i, member_i).glob("wrfout_*")
    )


def test_file_segment_outputs(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    fake_segment_run(exp, 0, 4, 6)  # 2021-01-02 00:00 -> 2021-01-02 18:00

    exp.file_segment_outputs(0, range(4, 7))

    # The file at the segment start is gone, the cycle end belongs to the cycle
    assert wrfout_hours(exp, 4) == [1, 2, 3, 4, 5, 6]
    assert wrfout_hours(exp, 5) == [7, 8, 9, 10, 11, 12]
    assert wrfout_hours(exp, 6) == [13, 14, 15, 16, 17, 18]
    end = "wrfrst_d01_2021-01-02_18:00:00"
    assert not (exp.paths.scratch_restart_path(4, 0) / end).exists()
    assert (exp.paths.scratch_restart_path(6, 0) / end).exists()

    # Again, as for a job that died after sorting
    exp.file_segment_outputs(0, range(4, 7))
    assert wrfout_hours(exp, 5) == [7, 8, 9, 10, 11, 12]


def test_file_segment_outputs_without_restart_file(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    fake_segment_run(exp, 0, 4, 6)
    for f in exp.paths.scratch_restart_path(4, 0).iterdir():
        f.unlink()

    with pytest.raises(FileNotFoundError):
        exp.file_segment_outputs(0, range(4, 7))


def test_runnable_segment(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")

    with pytest.raises(experiment.ExperimentStateError, match="plan-segment"):
        exp.runnable_segment()

    exp.plan_segment(0)
    assert exp.runnable_segment() == range(0, 4)


def test_runnable_segment_without_segments(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")
    exp.current_cycle_i = 5

    assert exp.runnable_segment() == range(5, 6)


def test_finish_segment(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.plan_segment(0)

    with pytest.raises(experiment.ExperimentStateError, match="finish-segment"):
        exp.check_segment_finished()
    with pytest.raises(experiment.ExperimentStateError, match="0/20 members"):
        exp.finish_segment()

    advance_segment(exp, 0, 3)
    # The first cycle looks like it could be cycled from, but not before finishing
    assert exp.state_machine.current_cycle.current_state == CycleState.MEMBERS_ADVANCED
    with pytest.raises(experiment.ExperimentStateError):
        exp.check_segment_finished()

    exp.finish_segment()

    assert exp.current_cycle_i == 3
    assert experiment.Experiment(exp.paths.experiment_path).current_cycle_i == 3
    for c in range(3):
        assert exp.state_machine.get_cycle(c).current_state == CycleState.CYCLE_COMPLETE
    assert exp.state_machine.get_cycle(3).current_state == CycleState.MEMBERS_ADVANCED
    exp.check_segment_finished()
    assert exp.state_machine.can_cycle_to_next(3) == (True, True, "")

    # Nothing left to do
    exp.finish_segment()
    assert exp.current_cycle_i == 3


def test_finish_segment_refuses_new_observations_inside(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.plan_segment(0)
    advance_segment(exp, 0, 3)
    add_observations(exp, 1)

    with pytest.raises(experiment.ExperimentStateError, match="Cycle\\(s\\) 1 got"):
        exp.finish_segment()


def test_finish_segment_without_segments_does_nothing(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")

    exp.finish_segment()
    exp.check_segment_finished()

    assert exp.current_cycle_i == 0


def test_reset_segment(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.plan_segment(0)
    advance_segment(exp, 0, 3)
    exp.finish_segment()
    add_observations(exp, 2)
    plan = exp.segment_of(3)

    exp.reset_segment(plan)

    assert exp.current_cycle_i == 0
    for c in range(4):
        assert exp.state.get_advanced_members(c) == set()
        assert wrfout_hours(exp, c) == []
        assert not list(exp.paths.scratch_restart_path(c, 0).glob("wrfrst_*"))
    # Planned again, with the new stop
    assert exp.segment_of(0).last == 2
    assert exp.segment_of(3) is None


def test_reset_segment_refuses_past_segments(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    plan = exp.plan_segment(0)
    exp.current_cycle_i = 4

    with pytest.raises(experiment.ExperimentStateError, match="outside segment"):
        exp.reset_segment(plan)


def test_analysis_jobfile_finishes_segment_and_passes_run_until(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")

    jf = jobfiles.generate_make_analysis_jobfile(exp, 3, True, run_until=9)
    text = jf.read_text()

    assert text.index("ensemble finish-segment") < text.index("ensemble filter")
    assert "cycle_003.obs_seq" in text
    assert "perts_cycle_4.nc" in text
    assert "ensemble cycle --run-until 9" in text
    assert "run-experiment  --run-until 9" in text


def test_analysis_jobfile_without_segments(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")

    text = jobfiles.generate_make_analysis_jobfile(exp, 3).read_text()

    assert "finish-segment" not in text


def test_fit_segment_shortens_unstarted_segment(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    plan = exp.plan_segment(0)

    assert slurm_commands._fit_segment(exp, plan, None) == plan

    shorter = slurm_commands._fit_segment(exp, plan, 2)
    assert (shorter.last, shorter.stop_reason, shorter.run_until) == (2, "run_until", 2)
    assert exp.segment_of(0) == shorter

    add_observations(exp, 1)
    assert slurm_commands._fit_segment(exp, shorter, None).last == 1


def test_fit_segment_refuses_started_segment(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    plan = exp.plan_segment(0)
    exp.state.set_member_advanced(0, 3)

    with pytest.raises(SystemExit):
        slurm_commands._fit_segment(exp, plan, 1)


def record_runs(exp, cycle_i: int, durations: list[int], simulated_h: float | None):
    start = dt.datetime(2026, 1, 1)
    for m, d in enumerate(durations):
        exp.state.set_member_advanced(
            cycle_i,
            m,
            start=start,
            end=start + dt.timedelta(seconds=d),
            duration_s=d,
            simulated_s=None if simulated_h is None else int(simulated_h * 3600),
        )


def test_segment_rate_from_expected_until_there_are_runs(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    record_runs(exp, 0, [3600, 3600], 6)

    assert exp.segment_rate(1) == (1800, "expected_walltime_per_sim_hour")


def test_segment_rate_from_recent_runs(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    # Old runs without simulated_s (the run was the 6 h cycle): 600 s/h
    record_runs(exp, 0, [3600] * 5, None)
    # A segment of 24 h, recorded on its last cycle: 100-190 s/h
    record_runs(exp, 4, [2400 + 240 * k for k in range(10)], 24)

    rate, source = exp.segment_rate(5)

    rates = [600] * 5 + [100 + 10 * k for k in range(10)]
    assert rate == pytest.approx(float(np.percentile(rates, 90)))
    assert source == "p90 of 15 recent runs"
    # Only runs before the cycle being planned count
    assert exp.segment_rate(1)[0] == pytest.approx(600)
    assert exp.segment_rate(0)[1] == "expected_walltime_per_sim_hour"


def test_advance_jobfile_time_limit(tmp_path: Path):
    exp = make_experiment(
        tmp_path / "exp", extra=SEGMENTS.replace("safety_factor = 1.0", "safety_factor = 1.25")
    )
    plan = exp.plan_segment(0)  # 3 cycles: 18 h at 1800 s/h is 9 h, x 1.25 fits 12 h
    assert plan.last == 2

    jf, _ = jobfiles.generate_advance_array_jobfile(exp)
    assert "#SBATCH --time=0-11:15:00" in jf.read_text()

    exp.plan_segment(0, run_until=1)  # 12 h: 6 h x 1.25
    jf, _ = jobfiles.generate_advance_array_jobfile(exp)
    assert "#SBATCH --time=0-07:30:00" in jf.read_text()


def test_advance_jobfile_without_segments_keeps_configured_time(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")

    jf, _ = jobfiles.generate_advance_array_jobfile(exp)

    assert "--time=0-" not in jf.read_text()
