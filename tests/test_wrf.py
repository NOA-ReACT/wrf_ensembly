import datetime as dt
from importlib import resources
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import config, cycling, update_bc, wrf
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


def make_long_wrfbdy(
    path: Path, n_records: int, interval_h: int = 6, record_chunk: int = 1
) -> list[dt.datetime]:
    """A minimal wrfbdy with `n_records` records starting at 2026-03-01 12:00"""

    start = dt.datetime(2026, 3, 1, 12, tzinfo=dt.timezone.utc)
    times = [start + dt.timedelta(hours=interval_h * i) for i in range(n_records + 1)]
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("bdy_width", 2)
        ds.START_DATE = "2026-03-01_12:00:00"
        for name, values in (
            ("Times", times[:-1]),
            (update_bc.THIS_BDY_TIME, times[:-1]),
            (update_bc.NEXT_BDY_TIME, times[1:]),
        ):
            var = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
            for i, t in enumerate(values):
                update_bc.write_wrf_time(var, i, t)
        mu = ds.createVariable(
            "MU_BXS",
            "f4",
            ("Time", "bdy_width"),
            zlib=True,
            chunksizes=(record_chunk, 1),
        )
        mu.units = "Pa"
        mu[:] = np.arange(n_records * 2).reshape(n_records, 2)
    return times


def test_extract_boundary_records_for_a_cycle(tmp_path: Path):
    times = make_long_wrfbdy(tmp_path / "long", 6)

    n = wrf.extract_boundary_records(
        tmp_path / "long", tmp_path / "out", times[2], times[4]
    )

    assert n == 2
    with netCDF4.Dataset(tmp_path / "out") as ds:
        assert update_bc.parse_wrf_times(ds[update_bc.THIS_BDY_TIME]) == times[2:4]
        np.testing.assert_array_equal(ds["MU_BXS"][:], [[4, 5], [6, 7]])
        assert ds["MU_BXS"].units == "Pa"
        assert ds["MU_BXS"].filters()["zlib"]
        assert ds.START_DATE == "2026-03-02_00:00:00"


@pytest.mark.parametrize("record_chunk", [1, 4])
def test_extract_boundary_records_copies_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_chunk: int
):
    # One record per chunk: the compressed chunks are copied. Otherwise the records are
    # recompressed, with the same result.
    times = make_long_wrfbdy(tmp_path / "long", 6, record_chunk=record_chunk)
    copied = []
    copy_record_chunks = wrf._copy_record_chunks
    monkeypatch.setattr(
        wrf,
        "_copy_record_chunks",
        lambda *args: copied.append(copy_record_chunks(*args)) or copied[-1],
    )

    n = wrf.extract_boundary_records(
        tmp_path / "long", tmp_path / "out", times[1], times[4]
    )

    assert n == 3
    assert copied == [record_chunk == 1]
    with netCDF4.Dataset(tmp_path / "out") as ds:
        assert len(ds.dimensions["Time"]) == 3
        assert update_bc.parse_wrf_times(ds[update_bc.THIS_BDY_TIME]) == times[1:4]
        np.testing.assert_array_equal(ds["MU_BXS"][:], [[2, 3], [4, 5], [6, 7]])
        assert ds["MU_BXS"].chunking() == [record_chunk, 1]


def test_extract_boundary_records_partial_intervals(tmp_path: Path):
    # A cycle starting and ending between boundary times needs both records it touches
    times = make_long_wrfbdy(tmp_path / "long", 6)
    start = times[1] + dt.timedelta(hours=3)

    n = wrf.extract_boundary_records(
        tmp_path / "long", tmp_path / "out", start, start + dt.timedelta(hours=6)
    )

    assert n == 2
    with netCDF4.Dataset(tmp_path / "out") as ds:
        assert update_bc.parse_wrf_times(ds[update_bc.THIS_BDY_TIME]) == times[1:3]


def test_extract_boundary_records_all_records_copies_the_file(tmp_path: Path):
    times = make_long_wrfbdy(tmp_path / "long", 4)

    n = wrf.extract_boundary_records(
        tmp_path / "long", tmp_path / "out", times[0], times[-1]
    )

    assert n == 4
    assert (tmp_path / "out").read_bytes() == (tmp_path / "long").read_bytes()


def test_extract_boundary_records_outside_the_file(tmp_path: Path):
    times = make_long_wrfbdy(tmp_path / "long", 2)

    with pytest.raises(ValueError, match="does not cover"):
        wrf.extract_boundary_records(
            tmp_path / "long",
            tmp_path / "out",
            times[1],
            times[2] + dt.timedelta(hours=1),
        )


def test_segment_namelist(tmp_path: Path):
    cfg = make_restart_config()
    cycles = cycling.get_cycle_information(cfg)
    period = cycling.get_segment_period(cycles, 2, 5)
    paths = ExperimentPaths(tmp_path / "experiment", cfg)
    path = tmp_path / "namelist.input"

    wrf.generate_wrf_namelist(
        cfg, period, True, path, member=0, paths=paths, restart_interval=720
    )
    text = path.read_text()

    # 2026-03-02 00:00 (cycle 2 start) -> 2026-03-03 00:00 (cycle 5 end)
    assert "start_day = 2" in text and "start_hour = 0" in text
    assert "end_day = 3" in text and "end_hour = 0" in text
    assert "run_hours = 0" in text or "run_days = 1" in text
    assert "restart = .true." in text
    assert "restart_interval = 720" in text
    assert "scratch/restart/cycle_002/member_00/wrfrst_d<domain>_<date>" in text
    assert "scratch/forecasts/cycle_002/member_00/wrfout_d<domain>_<date>" in text


def test_wrfout_time():
    assert wrf.wrfout_time("wrfout_d01_2021-01-02_06:00:00") == dt.datetime(
        2021, 1, 2, 6, tzinfo=dt.timezone.utc
    )
    assert wrf.wrfout_time("wrfrst_d01_2021-01-02_06:30:00").minute == 30
