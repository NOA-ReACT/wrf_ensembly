"""
The analysis and cycling steps in the `wrfinput` cycling mode, on small stand-ins for
the wrfout, DART and wrfinput files.
"""

from importlib import resources
from pathlib import Path

import netCDF4
import numpy as np
from click.testing import CliRunner

from wrf_ensembly import experiment
from wrf_ensembly.cli import cli
from wrf_ensembly.experiment import StateTransition

N_MEMBERS = 2


def make_experiment(path: Path) -> experiment.Experiment:
    """The default template (wrfinput mode) with two members"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    text = template.read_text().replace("n_members = 20", f"n_members = {N_MEMBERS}")
    path.mkdir()
    (path / "config.toml").write_text(text)
    exp = experiment.Experiment(path)
    exp.paths.create_directories()
    exp.state.initialize()
    return exp


def write_fields(path: Path, value: float, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        for name in names:
            ds.createVariable(name, "f8", ("x",))[:] = value


def read(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        return ds[name][:].filled()


def test_analysis_and_cycle(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    assert exp.cfg.assimilation.cycling_mode == "wrfinput"
    cycle = exp.cycles[0]
    wrfout_name = f"wrfout_d01_{cycle.end:%Y-%m-%d_%H:%M:%S}"
    state = exp.cfg.assimilation.state_variables
    cycled = exp.cfg.assimilation.cycled_variables
    # In the forecast and cycled, but not part of the DART state
    not_in_state = sorted(set(cycled) - set(state))
    assert not_in_state

    # Forecasts (1), what filter wrote (2) and the next cycle's wrfinput (5)
    for member_i in range(N_MEMBERS):
        write_fields(
            exp.paths.scratch_forecasts_path(0, member_i) / wrfout_name,
            1.0,
            sorted(set(state) | set(cycled)),
        )
        write_fields(
            exp.paths.scratch_dart_path(0) / f"dart_member_{member_i:02d}.nc",
            2.0 + member_i,
            state,
        )
        exp.state.set_member_advanced(0, member_i)
    write_fields(exp.paths.ic_path(0, 1), 5.0, sorted(set(cycled) | {"LANDMASK"}))
    exp.paths.bc_path(0, 1).write_text("boundaries of cycle 1")
    exp.load_status()
    exp.state_machine.current_cycle.transition(StateTransition.FILTER_COMPLETE)

    result = CliRunner().invoke(cli, [str(exp.paths.experiment_path), "ensemble", "analysis"])
    assert result.exit_code == 0, result.output

    analysis = exp.paths.scratch_analysis_path(0) / "member_01" / wrfout_name
    np.testing.assert_array_equal(read(analysis, state[0]), 3.0)
    np.testing.assert_array_equal(read(analysis, not_in_state[0]), 1.0)
    with netCDF4.Dataset(analysis) as ds:
        assert ds.experiment_name == exp.cfg.metadata.name
        assert ds.current_cycle == 0
        assert ds.cycle_end == f"{cycle.end:%Y-%m-%d_%H:%M:%S}"
    # The forecast is left alone
    forecast = exp.paths.scratch_forecasts_path(0, 1) / wrfout_name
    np.testing.assert_array_equal(read(forecast, state[0]), 1.0)

    exp = experiment.Experiment(exp.paths.experiment_path)
    assert exp.analysis_run
    member_dir = exp.paths.member_path(1)
    member_dir.mkdir(parents=True, exist_ok=True)
    (member_dir / "rsl.out.0000").touch()
    (member_dir / "wrfout_d01_leftover").touch()

    exp.model.cycle_member(1, use_forecast=False)
    wrfinput = member_dir / "wrfinput_d01"
    in_both = sorted(set(cycled) & set(state))
    np.testing.assert_array_equal(read(wrfinput, in_both[0]), 3.0)  # analysis
    np.testing.assert_array_equal(read(wrfinput, not_in_state[0]), 1.0)  # forecast
    np.testing.assert_array_equal(read(wrfinput, "LANDMASK"), 5.0)  # real.exe
    with netCDF4.Dataset(wrfinput) as ds:
        assert ds.experiment_name == exp.cfg.metadata.name
    assert (member_dir / "wrfbdy_d01").read_text() == "boundaries of cycle 1"
    assert not list(member_dir.glob("rsl*")) and not list(member_dir.glob("wrfout*"))

    # Without an analysis, the forecast is cycled
    exp.model.cycle_member(0, use_forecast=True)
    np.testing.assert_array_equal(
        read(exp.paths.member_path(0) / "wrfinput_d01", in_both[0]), 1.0
    )


def test_setup_copies_initial_and_boundary_conditions(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    write_fields(exp.paths.ic_path(0, 0), 3.0, ["U"])
    exp.paths.bc_path(0, 0).write_text("boundaries of cycle 0")

    result = CliRunner().invoke(
        cli, [str(exp.paths.experiment_path), "ensemble", "setup", "--jobs", "1"]
    )
    assert result.exit_code == 0, result.output
    for member_i in range(N_MEMBERS):
        member_dir = exp.paths.member_path(member_i)
        np.testing.assert_array_equal(read(member_dir / "wrfinput_d01", "U"), 3.0)
        assert (member_dir / "wrfbdy_d01").read_text() == "boundaries of cycle 0"
