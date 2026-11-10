"""
ICON support: namelists, running a member (with tests/fake_icon.py standing in for
the model), the analysis files and cycling.
"""

import datetime as dt
import sys
from importlib import resources
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import config, cycling, experiment
from wrf_ensembly.experiment import StateTransition
from wrf_ensembly.icon import model as icon_model
from wrf_ensembly.icon import namelists

FAKE_ICON = Path(__file__).parent / "fake_icon.py"
UTC = dt.timezone.utc


def template_text() -> str:
    return (
        resources.files("wrf_ensembly.config_templates")
        .joinpath("icon.toml")
        .read_text()
    )


def make_experiment(
    path: Path, extra: str = "", members: int = 2
) -> experiment.Experiment:
    """
    An ICON experiment from the template (4 cycles of 6 h) with a fake model, mpirun
    and input files, and its members set up for cycle 0
    """

    path.mkdir()
    inputs = path / "inputs"
    (inputs / "icon-build" / "bin").mkdir(parents=True)
    (inputs / "icon-build" / "externals" / "ecrad" / "data").mkdir(parents=True)
    icon_exe = inputs / "icon-build" / "bin" / "icon"
    icon_exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_ICON}\n")
    icon_exe.chmod(0o755)
    mpirun = inputs / "mpirun"
    mpirun.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')  # drop "-n N"
    mpirun.chmod(0o755)
    for name in ("grid.nc", "extpar.nc", "ifs.nc"):
        (inputs / name).touch()
    (inputs / "icbc").mkdir()

    text = template_text()
    for old, new in {
        'icon_root = "/path/to/icon-build"': f'icon_root = "{inputs / "icon-build"}"',
        'grid_file   = "/path/to/grid_DOM01.nc"': f'grid_file = "{inputs / "grid.nc"}"',
        'extpar_file = "/path/to/grid_DOM01_external_parameter.nc"': f'extpar_file = "{inputs / "extpar.nc"}"',
        'initial_conditions  = "/path/to/icbc/ifs_initial_DOM01.nc"': f'initial_conditions = "{inputs / "ifs.nc"}"',
        'boundary_conditions = "/path/to/icbc"': f'boundary_conditions = "{inputs / "icbc"}"',
        'mpirun_command = "srun"': f'mpirun_command = "{mpirun}"',
        'dart_root = "/path/to/DART"': f'dart_root = "{inputs / "dart"}"',
        "n_members = 20": f"n_members = {members}",
    }.items():
        assert old in text, old
        text = text.replace(old, new)
    (path / "config.toml").write_text(text + extra)

    exp = experiment.Experiment(path)
    exp.paths.create_directories()
    exp.state.initialize()
    for member_i in range(members):
        icon_model.setup_member(exp, member_i)
    return exp


def fake_filter(exp: experiment.Experiment, value: float = 7.0):
    """Writes what DART's filter would: the state variables of each member"""

    cycle_i = exp.current_cycle_i
    out_dir = exp.paths.scratch_dart_path(cycle_i)
    out_dir.mkdir(parents=True, exist_ok=True)
    for member_i in range(exp.cfg.assimilation.n_members):
        prior = namelists.first_guess_file(
            exp.paths.scratch_forecasts_path(cycle_i, member_i), exp.current_cycle.end
        )
        with (
            netCDF4.Dataset(prior) as fg,
            netCDF4.Dataset(out_dir / f"dart_member_{member_i:02d}.nc", "w") as ds,
        ):
            # DART writes (time, height, cells) too
            for dim in ("time", "height", "ncells_2"):
                ds.createDimension(dim, len(fg.dimensions[dim]))
            for name in exp.cfg.assimilation.state_variables:
                var = ds.createVariable(name, "f8", ("time", "height", "ncells_2"))
                var[:] = value + member_i
    exp.state_machine.current_cycle.transition(StateTransition.FILTER_COMPLETE)


def test_template_is_a_valid_icon_config():
    cfg = config.Config.from_toml(template_text())
    cfg.check()
    assert cfg.is_icon
    assert cfg.icon is not None and cfg.icon.regrid.method == "linear"


@pytest.mark.parametrize(
    "change,message",
    [
        (("analysis_interval = 360", "analysis_interval = 100"), "multiple"),
        (("[icon.namelist.grid_nml]", "[icon.namelist.grid_nml]\ndynamics_grid_filename = 'a.nc'"), "fills in"),
        (('icon_root = "/path/to/icon-build"', ""), "icon_root"),
    ],
)  # fmt: skip
def test_config_check_rejects(change, message):
    text = template_text()
    assert change[0] in text
    cfg = config.Config.from_toml(text.replace(*change))
    with pytest.raises(ValueError, match=message):
        cfg.check()


def test_iso_duration_and_dates():
    assert namelists.iso_duration(360) == "PT06H"
    assert namelists.iso_duration(30) == "PT30M"
    assert namelists.iso_duration(90) == "PT01H30M"
    assert namelists.iso_duration(24 * 60) == "P01D"
    assert namelists.iso_duration(25 * 60 + 5) == "P01DT01H05M"
    with pytest.raises(ValueError):
        namelists.iso_duration(0)

    time = dt.datetime(2026, 9, 13, 6, tzinfo=UTC)
    assert namelists.icon_date(time) == "2026-09-13T06:00:00Z"
    assert namelists.parse_file_date("icon_20260913T060000Z.nc") == time
    assert namelists.forecast_file(Path("/x"), time).name == "icon_20260913T060000Z.nc"


def test_atmosphere_namelist(tmp_path: Path):
    cfg = config.Config.from_toml(
        template_text()
        + """
[[icon.namelist.output_nml]]
output_filename = "pl"
pl_varlist = ["temp"]

[icon.namelist_per_member.member_01.nwp_phy_nml]
inwp_gscp = 1
"""
    )
    cycles = cycling.get_cycle_information(cfg)

    first = namelists.atmosphere_namelist(cfg, cycles[0], 0, tmp_path, False, False)
    assert first["initicon_nml"]["init_mode"] == 2
    assert first["initicon_nml"]["zpbl1"] == 500.0  # from the user's namelist
    assert first["gribout_nml"]["perturbationNumber"] == 1
    assert first["nwp_phy_nml"]["inwp_gscp"] == 2
    assert first["limarea_nml"]["dtime_latbc"] == 10800.0
    assert "pinit_seed" not in first["initicon_nml"]

    # Noise on the initial state: first cycle only, another seed for each member
    assert cfg.icon is not None
    cfg.icon.initial_noise = 0.01
    noisy = namelists.atmosphere_namelist(cfg, cycles[0], 3, tmp_path, False, False)
    assert noisy["initicon_nml"]["pinit_seed"] == 4
    assert noisy["initicon_nml"]["pinit_amplitude"] == 0.01

    later = namelists.atmosphere_namelist(cfg, cycles[1], 1, tmp_path, True, True)
    assert later["initicon_nml"]["init_mode"] == 1
    assert later["initicon_nml"]["lread_ana"] is True
    assert "pinit_seed" not in later["initicon_nml"]
    assert later["gribout_nml"]["perturbationNumber"] == 2
    assert later["nwp_phy_nml"]["inwp_gscp"] == 1

    streams = later["output_nml"]
    assert [Path(s["output_filename"]).name for s in streams] == ["icon", "fg", "pl"]
    assert streams[0]["output_start"] == "2026-09-13T06:00:00Z"
    assert streams[0]["output_interval"] == "PT01H"
    # The first guess is written once, at the end of the cycle
    assert (
        streams[1]["output_start"] == streams[1]["output_end"] == "2026-09-13T12:00:00Z"
    )
    assert streams[2]["output_filename"] == str(tmp_path / "pl")

    path = tmp_path / "NAMELIST_atm"
    namelists.write(later, path)
    text = path.read_text()
    assert text.count("&output_nml") == 3
    assert " lread_ana = .true.\n" in text
    assert " ivadv_tracer = 3, 3, 3, 3, 3\n" in text
    assert " dwdfg_filename = 'first_guess.nc'\n" in text


def test_latbc_map(tmp_path: Path):
    namelists.write_latbc_map({"u": "U", "pres_sfc": "PS"}, tmp_path / "map")
    assert (tmp_path / "map").read_text().split() == ["u", "U", "pres_sfc", "PS"]


def test_missing_boundary_files(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    missing = icon_model.missing_boundary_files(exp, 0)
    # 24 h every 3 h, the one at the start is not needed
    assert len(missing) == 8
    assert missing[0].name == "ifs_20260913_03_lbc_ICON.nc"
    for f in missing:
        f.touch()
    assert icon_model.missing_boundary_files(exp, 0) == []


def test_advance_filter_analysis_cycle(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    member_dir = exp.paths.member_path(0)
    assert (member_dir / "icon").exists()
    assert (member_dir / "map_file.latbc").read_text().startswith("u ")

    for member_i in range(2):
        assert exp.advance_member(member_i, cores=1)
    forecast_dir = exp.paths.scratch_forecasts_path(0, 0)
    # Hourly output from 00 to 06, and the first guess at 06
    assert len(list(forecast_dir.glob("icon_*.nc"))) == 7
    assert [f.name for f in forecast_dir.glob("fg_*.nc")] == ["fg_20260913T060000Z.nc"]
    assert "init_mode = 2" in (member_dir / "NAMELIST_atm").read_text()
    assert exp.state.count_advanced(0) == 2
    status = exp.state.get_advanced_members(0)
    assert status == {0, 1}

    # What the filter gets
    exp.cfg.dart_namelist["model_nml"]["template_file"] = "template.nc"
    exp.paths.dart_work_dir.mkdir(parents=True)
    priors = icon_model.prepare_filter(exp)
    assert [p.name for p in priors] == ["fg_20260913T060000Z.nc"] * 2
    assert (exp.paths.dart_work_dir / "template.nc").resolve() == priors[0].resolve()
    assert (exp.paths.dart_work_dir / "icon_grid.nc").is_symlink()

    exp.load_status()
    fake_filter(exp)
    icon_model.make_analysis(exp, 0)
    icon_model.make_analysis(exp, 1)

    analysis = icon_model.analysis_file(exp, 0, 1)
    with netCDF4.Dataset(analysis) as ds, netCDF4.Dataset(priors[1]) as fg:
        assert set(ds.variables) == {"time", "temp", "pres", "u", "v", "qv"}
        assert ds.uuidOfHGrid == "fake-grid"
        assert ds["time"].units == fg["time"].units
        assert ds["time"][0] == fg["time"][0]
        assert ds["temp"].dimensions == ("time", "height", "ncells_2")
        assert ds["temp"].CDI_grid_type == "unstructured"
        np.testing.assert_array_equal(ds["temp"][:], 8.0)  # member 1
        np.testing.assert_array_equal(ds["qv"][:], 8.0)
    # No state variable lives in the first guess only, so it is not copied
    assert not icon_model.analysis_first_guess_file(exp, 0, 1).exists()
    # The forecast output at the analysis time, with the analysis in it
    with netCDF4.Dataset(
        namelists.forecast_file(analysis.parent, exp.cycles[0].end)
    ) as ds:
        np.testing.assert_array_equal(ds["u"][:], 8.0)
        assert ds.current_cycle == 0

    exp.state_machine.current_cycle.transition(StateTransition.ANALYSIS_COMPLETE)
    for member_i in range(2):
        exp.model.cycle_member(member_i, use_forecast=False)
    assert (member_dir / "first_guess.nc").resolve() == priors[0].resolve()
    assert (member_dir / "analysis.nc").resolve() == icon_model.analysis_file(exp, 0, 0)
    exp.state_machine.current_cycle.transition(StateTransition.CYCLE_COMPLETE)
    exp.set_next_cycle()

    # The next cycle starts from the first guess and the analysis
    assert exp.advance_member(0, cores=1)
    text = (member_dir / "NAMELIST_atm").read_text()
    assert "init_mode = 1" in text and "lread_ana = .true." in text
    assert (
        "experimentStartDate = '2026-09-13T06:00:00Z'"
        in (member_dir / "icon_master.namelist").read_text()
    )
    assert namelists.first_guess_file(
        exp.paths.scratch_forecasts_path(1, 0), exp.cycles[1].end
    ).exists()


def test_cycle_without_analysis_and_cleanup(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", members=1)
    member_dir = exp.paths.member_path(0)
    for cycle_i in range(3):
        assert exp.advance_member(0, cores=1)
        exp.load_status()
        exp.model.cycle_member(0, use_forecast=True)
        assert not (member_dir / "analysis.nc").exists()
        exp.state_machine.current_cycle.transition(StateTransition.CYCLE_COMPLETE)
        exp.set_next_cycle()
        exp.model.clean_old_state()
    assert "lread_ana = .false." in (member_dir / "NAMELIST_atm").read_text()

    def first_guesses(cycle_i: int) -> int:
        return len(list(exp.paths.scratch_forecasts_path(cycle_i, 0).glob("fg_*.nc")))

    # We are on cycle 3, which starts from cycle 2's first guess
    assert [first_guesses(i) for i in range(3)] == [0, 0, 1]
    # The forecast output is left for the postprocessing
    assert len(list(exp.paths.scratch_forecasts_path(0, 0).glob("icon_*.nc"))) == 7


def test_tracers_are_updated_in_the_first_guess(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", members=1)
    exp.cfg.assimilation.state_variables = ["temp", "dust"]
    assert exp.advance_member(0, cores=1)
    exp.load_status()
    fake_filter(exp, value=3.0)
    icon_model.make_analysis(exp, 0)

    with netCDF4.Dataset(icon_model.analysis_file(exp, 0, 0)) as ds:
        # u, v and pres are not analysed but ICON needs them: from the first guess
        assert set(ds.variables) == {"time", "temp", "pres", "u", "v"}
        np.testing.assert_array_equal(ds["temp"][:], 3.0)
        assert float(ds["u"][:].max()) < 1.0
    updated = icon_model.analysis_first_guess_file(exp, 0, 0)
    with netCDF4.Dataset(updated) as ds:
        np.testing.assert_array_equal(ds["dust"][:], 3.0)
        assert float(ds["rho"][:].max()) < 1.0

    exp.state_machine.current_cycle.transition(StateTransition.ANALYSIS_COMPLETE)
    exp.model.cycle_member(0, use_forecast=False)
    assert (exp.paths.member_path(0) / "first_guess.nc").resolve() == updated.resolve()

    exp.cfg.assimilation.state_variables = ["temp", "not_there"]
    with pytest.raises(ValueError, match="not_there"):
        icon_model.make_analysis(exp, 0)


def test_failed_run_is_not_advanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exp = make_experiment(tmp_path / "exp", members=1)
    monkeypatch.setenv("FAKE_ICON_FAIL", "1")
    assert not exp.advance_member(0, cores=1)
    assert exp.state.count_advanced(0) == 0
    monkeypatch.delenv("FAKE_ICON_FAIL")
    assert exp.advance_member(0, cores=1)


def test_file_names(tmp_path: Path):
    time = dt.datetime(2026, 9, 13, 6, tzinfo=UTC)
    icon = make_experiment(tmp_path / "icon").model
    assert icon.name == "icon"
    assert icon.forecast_file_name(time) == "icon_20260913T060000Z.nc"
    assert icon.cycle_end_file_name(time) == "fg_20260913T060000Z.nc"
    assert icon.forecast_glob == "icon_*.nc"

    (tmp_path / "wrf").mkdir()
    (tmp_path / "wrf" / "config.toml").write_text(
        resources.files("wrf_ensembly.config_templates")
        .joinpath("default.toml")
        .read_text()
    )
    wrf = experiment.Experiment(tmp_path / "wrf").model
    assert wrf.name == "wrf"
    assert wrf.forecast_file_name(time) == "wrfout_d01_2026-09-13_06:00:00"
    assert wrf.cycle_end_file_name(time) == "wrfout_d01_2026-09-13_06:00:00"
    assert wrf.forecast_glob == "wrfout_d01_*"
