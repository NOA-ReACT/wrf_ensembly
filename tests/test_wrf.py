import datetime as dt
from importlib import resources
from pathlib import Path

from wrf_ensembly import config, cycling, wrf
from wrf_ensembly.experiment.paths import ExperimentPaths


def make_config() -> config.Config:
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template, inject_environment=False)
    cfg.time_control.start = dt.datetime(2026, 3, 1, 12, tzinfo=dt.timezone.utc)
    cfg.time_control.end = dt.datetime(2026, 3, 3, tzinfo=dt.timezone.utc)
    cfg.time_control.boundary_update_interval = 360
    cfg.time_control.runtime_io = []
    return cfg


def namelist_text(cfg: config.Config, tmp_path: Path) -> str:
    path = tmp_path / "namelist.input"
    wrf.generate_wrf_namelist(cfg, cycling.get_cycle_information(cfg)[0], False, path)
    return path.read_text()


def test_sst_update_sets_lower_boundary_input(tmp_path: Path):
    cfg = make_config()
    cfg.wrf_namelist.setdefault("physics", {})["sst_update"] = 1

    text = namelist_text(cfg, tmp_path)

    assert "io_form_auxinput4 = 2" in text
    assert "auxinput4_inname = 'wrflowinp_d<domain>'" in text
    assert "auxinput4_interval = 360" in text


def test_lower_boundary_input_keeps_user_settings(tmp_path: Path):
    cfg = make_config()
    cfg.wrf_namelist.setdefault("physics", {})["sst_update"] = 1
    cfg.wrf_namelist["time_control"]["auxinput4_interval"] = 60

    assert "auxinput4_interval = 60" in namelist_text(cfg, tmp_path)


def test_no_lower_boundary_input_without_sst_update(tmp_path: Path):
    cfg = make_config()
    cfg.wrf_namelist.setdefault("physics", {})["sst_update"] = 0

    assert "auxinput4" not in namelist_text(cfg, tmp_path)


def test_full_period_spans_the_experiment():
    cfg = make_config()
    period = cycling.get_full_period(cfg)

    assert period.start == cfg.time_control.start
    assert period.end == cfg.time_control.end
    assert period.forecast_end == cfg.time_control.end


def test_full_period_icbc_paths(tmp_path: Path):
    cfg = make_config()
    paths = ExperimentPaths(tmp_path, cfg)

    assert paths.bc_path(0, None) == paths.data_icbc / "wrfbdy_d01"
    assert paths.bc_path(0, 3) == paths.data_icbc / "wrfbdy_d01_cycle_3"
    assert paths.lowinp_path(1, None) == paths.data_icbc / "wrflowinp_d01"

    member_file = paths.icbc_file_path("wrflowinp_d01", 1, None)
    assert member_file == paths.data_icbc / "member_01" / "wrflowinp_d01_member_01"
    member_file.parent.mkdir(parents=True)
    member_file.touch()
    assert paths.lowinp_path(1, None) == member_file


def member_namelist_text(cfg: config.Config, tmp_path: Path, cycle: int) -> str:
    """Namelist for member 0 at `cycle`, as advance-member writes it"""

    paths = ExperimentPaths(tmp_path / "experiment", cfg)
    path = tmp_path / "namelist.input"
    wrf.generate_wrf_namelist(
        cfg,
        cycling.get_cycle_information(cfg)[cycle],
        True,
        path,
        member=0,
        paths=paths,
        add_iofields=False,
    )
    return path.read_text()


def make_restart_config() -> config.Config:
    cfg = make_config()
    cfg.assimilation.cycling_mode = "restart"
    cfg.assimilation.cycled_variables = []
    cfg.time_control.analysis_interval = 360
    return cfg


def test_restart_mode_first_cycle_is_a_cold_start(tmp_path: Path):
    text = member_namelist_text(make_restart_config(), tmp_path, 0)

    assert "restart = .false." in text
    assert "restart_interval = 360" in text
    assert "override_restart_timers = .true." in text
    rst_dir = tmp_path / "experiment/scratch/restart/cycle_000/member_00"
    assert f"rst_outname = '{rst_dir}/wrfrst_d<domain>_<date>'" in text
    assert rst_dir.is_dir()


def test_restart_mode_later_cycles_restart(tmp_path: Path):
    text = member_namelist_text(make_restart_config(), tmp_path, 1)

    assert "restart = .true." in text
    assert "scratch/restart/cycle_001/member_00/wrfrst_d<domain>_<date>" in text


def test_restart_interval_follows_the_cycle_duration(tmp_path: Path):
    cfg = make_restart_config()
    cfg.time_control.cycles = {1: config.CycleConfig(duration=180)}

    assert "restart_interval = 180" in member_namelist_text(cfg, tmp_path, 1)
    assert "restart_interval = 360" in member_namelist_text(cfg, tmp_path, 2)


def test_real_never_restarts(tmp_path: Path):
    cfg = make_restart_config()
    path = tmp_path / "namelist.input"
    wrf.generate_wrf_namelist(cfg, cycling.get_full_period(cfg), False, path)

    text = path.read_text()
    assert "restart = .false." in text
    assert "rst_outname" not in text


def test_wrfinput_mode_keeps_restart_settings(tmp_path: Path):
    cfg = make_config()
    cfg.wrf_namelist["time_control"]["restart_interval"] = 1234

    text = member_namelist_text(cfg, tmp_path, 1)
    assert "restart_interval = 1234" in text
    assert "rst_outname" not in text
    assert "override_restart_timers" not in text
