"""
advance-member running segments, with tests/fake_wrf.py standing in for wrf.exe
"""

import sys
from pathlib import Path

import pytest

from wrf_ensembly import checkpoints, experiment

from test_segments_experiment import SEGMENTS, make_experiment

FAKE_WRF = Path(__file__).parent / "fake_wrf.py"


def make_runnable_experiment(
    path: Path, segments: str = SEGMENTS
) -> experiment.Experiment:
    """
    The test experiment (24 cycles of 6 h, 4 cycles per segment), with a fake wrf.exe
    and mpirun, and member 0 set up for cycle 0
    """

    exp = make_experiment(path, extra=segments)
    bin_dir = path / "bin"
    bin_dir.mkdir()
    mpirun = bin_dir / "mpirun"
    mpirun.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')  # drop "-n N"
    mpirun.chmod(0o755)

    config_path = path / "config.toml"
    config_path.write_text(
        config_path.read_text().replace(
            'mpirun_command = "mpirun"', f'mpirun_command = "{mpirun}"'
        )
    )
    exp = experiment.Experiment(path)

    member_dir = exp.paths.member_path(0)
    member_dir.mkdir(parents=True)
    wrf_exe = member_dir / "wrf.exe"
    wrf_exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_WRF}\n")
    wrf_exe.chmod(0o755)
    (member_dir / "wrfinput_d01").touch()
    (member_dir / "wrfbdy_d01").touch()
    return exp


def wrfouts(exp: experiment.Experiment, cycle_i: int) -> list[str]:
    """Times of member 0's wrfouts in a cycle, as "DD_HH" """

    return sorted(
        f.name[-11:-6]
        for f in exp.paths.scratch_forecasts_path(cycle_i, 0).glob("wrfout_*")
    )


def restart_files(exp: experiment.Experiment, cycle_i: int) -> list[str]:
    directory = exp.paths.scratch_restart_path(cycle_i, 0)
    return [f.name[-19:] for _, f in checkpoints.list_checkpoints(directory)]


def test_advance_member_runs_the_segment(tmp_path: Path):
    exp = make_runnable_experiment(
        tmp_path / "exp", SEGMENTS + "checkpoint_interval_hours = 6\n"
    )
    exp.plan_segment(0)

    assert exp.advance_member(0, cores=1)

    assert wrfouts(exp, 0) == [f"01_{h:02d}" for h in range(1, 7)]
    assert wrfouts(exp, 3) == [f"01_{h:02d}" for h in range(19, 24)] + ["02_00"]
    # Checkpoints every 6 h, the newest 2 kept, the last one moved to cycle 3
    assert restart_files(exp, 0) == ["2021-01-01_18:00:00"]
    assert restart_files(exp, 3) == ["2021-01-02_00:00:00"]
    for c in range(4):
        assert exp.state.get_advanced_members(c) == {0}
    runtime = exp.state.get_member(3, 0).runtime
    assert runtime is not None and runtime.simulated_s == 24 * 3600
    assert exp.state.get_member(0, 0).runtime is None


def test_advance_member_resumes_from_confirmed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = make_runnable_experiment(
        tmp_path / "exp", SEGMENTS + "checkpoint_interval_hours = 6\n"
    )
    exp.plan_segment(0)

    # Dies while writing the restart file at 18:00
    monkeypatch.setenv("FAKE_WRF_DIE_AFTER_HOURS", "18")
    assert not exp.advance_member(0, cores=1)
    assert restart_files(exp, 0) == ["2021-01-01_12:00:00", "2021-01-01_18:00:00"]
    assert exp.state.get_advanced_members(0) == set()

    monkeypatch.delenv("FAKE_WRF_DIE_AFTER_HOURS")
    assert exp.find_resume_point(0, range(0, 4))[1].name.endswith("12:00:00")
    assert exp.advance_member(0, cores=1)

    rsl = (exp.paths.member_path(0) / "rsl.out.0000").read_text()
    assert rsl.startswith("start 2021-01-01_12:00:00 restart True")
    assert wrfouts(exp, 1) == [f"01_{h:02d}" for h in range(7, 13)]
    assert restart_files(exp, 3) == ["2021-01-02_00:00:00"]
    assert exp.state.get_member(3, 0).runtime.simulated_s == 12 * 3600
    # The copy it resumed from is the only one in the member directory
    member_restarts = sorted(
        f.name for f in exp.paths.member_path(0).glob("wrfrst_*")
    )
    assert member_restarts == ["wrfrst_d01_2021-01-01_12:00:00"]


def test_advance_member_starts_over_without_confirmed_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = make_runnable_experiment(
        tmp_path / "exp", SEGMENTS + "checkpoint_interval_hours = 6\n"
    )
    exp.plan_segment(0)
    monkeypatch.setenv("FAKE_WRF_DIE_AFTER_HOURS", "18")
    assert not exp.advance_member(0, cores=1)
    (exp.paths.scratch_restart_path(0, 0) / checkpoints.CONFIRMED_FILE).unlink()

    assert exp.find_resume_point(0, range(0, 4)) is None


def test_advance_member_only_files_output_of_a_finished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = make_runnable_experiment(tmp_path / "exp")
    exp.plan_segment(0)
    assert exp.advance_member(0, cores=1)

    # As if the job died before writing the status files
    for c in range(4):
        exp.state.clear_member(c, 0)
    exp = experiment.Experiment(exp.paths.experiment_path)
    monkeypatch.setenv("FAKE_WRF_DIE_AFTER_HOURS", "0")  # Fails if it runs

    assert exp.advance_member(0, cores=1)
    for c in range(4):
        assert exp.state.get_advanced_members(c) == {0}


def test_advance_member_without_segments(tmp_path: Path):
    exp = make_runnable_experiment(tmp_path / "exp", segments="")

    assert exp.advance_member(0, cores=1)

    assert wrfouts(exp, 0) == [f"01_{h:02d}" for h in range(1, 7)]
    assert restart_files(exp, 0) == ["2021-01-01_06:00:00"]
    assert not (exp.paths.scratch_restart_path(0, 0) / checkpoints.CONFIRMED_FILE).exists()
    assert exp.state.get_advanced_members(1) == set()
