"""IFS to ICON initial and boundary conditions (`wrf_ensembly.icon.preprocess`)"""

import datetime as dt
from pathlib import Path

import pytest

from wrf_ensembly import obs_sequence
from wrf_ensembly.icon import preprocess

UTC = dt.timezone.utc
START = dt.datetime(2026, 9, 13, 0, tzinfo=UTC)


def test_boundary_file_name():
    time = dt.datetime(2026, 9, 14, 3, 30, tzinfo=UTC)
    assert (
        preprocess.boundary_file_name("ifs_<y><m><d>_<h>_lbc_ICON.nc", time)
        == "ifs_20260914_03_lbc_ICON.nc"
    )
    assert preprocess.boundary_file_name("lbc_<y><m><d><h><min><sec>.nc", time) == (
        "lbc_20260914033000.nc"
    )
    with pytest.raises(preprocess.PreprocessError, match="keyword"):
        preprocess.boundary_file_name("lbc_<ddhhmmss>.nc", time)


def test_find_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    grib = tmp_path / "grib"
    grib.mkdir()
    valid = {
        "ifs_2026091300.grb": START,
        "ifs_2026091200.grb": START - dt.timedelta(days=1),  # another analysis
        "ifs_20260913_0_lbc.grb": START,
        "ifs_20260913_3_lbc.grb": START + dt.timedelta(hours=3),
        "ifs_20260913_6_lbc.grb": START + dt.timedelta(hours=6),
        "ifs_20260913_9_lbc.grb": START + dt.timedelta(hours=9),  # after the end
        "ifs_2026091300.grb.tmp_smi": START,  # left behind by the MARS job
        "notes.txt": START,
    }
    for name in valid:
        (grib / name).touch()
    monkeypatch.setattr(preprocess, "valid_time", lambda path, env: valid[path.name])

    end = START + dt.timedelta(hours=6)
    tasks = preprocess.find_tasks(
        grib, tmp_path / "ic.nc", tmp_path / "lbc", "lbc_<d>_<h>.nc", START, end, {}
    )
    assert [(t.kind, t.source.name, t.target.name) for t in tasks] == [
        ("IC", "ifs_2026091300.grb", "ic.nc"),
        ("LBC", "ifs_20260913_0_lbc.grb", "lbc_13_00.nc"),
        ("LBC", "ifs_20260913_3_lbc.grb", "lbc_13_03.nc"),
        ("LBC", "ifs_20260913_6_lbc.grb", "lbc_13_06.nc"),
    ]
    assert tasks[0].renames["lnsp"] == "LNPS" and tasks[1].renames["sp"] == "PS"

    # No analysis for the start of the experiment
    with pytest.raises(preprocess.PreprocessError, match="No initial conditions"):
        preprocess.find_tasks(
            grib, tmp_path / "ic.nc", tmp_path / "lbc", "lbc_<d>_<h>.nc",
            START + dt.timedelta(hours=3), end, {},
        )  # fmt: skip


def test_missing_tools_are_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(preprocess.shutil, "which", lambda tool: None)
    with pytest.raises(preprocess.PreprocessError, match="cdo not found"):
        preprocess.tool_environment(tmp_path)

    monkeypatch.setattr(preprocess.shutil, "which", lambda tool: f"/bin/{tool}")
    monkeypatch.delenv("DWD_DEF", raising=False)
    with pytest.raises(preprocess.PreprocessError, match="dwd_grib_definitions"):
        preprocess.tool_environment(None)


def test_find_dart_program(tmp_path: Path):
    with pytest.raises(RuntimeError, match="obs_seq_to_netcdf"):
        obs_sequence.find_dart_program(tmp_path, "obs_seq_to_netcdf")
    work = tmp_path / "models" / "icon" / "work"
    work.mkdir(parents=True)
    (work / "obs_seq_to_netcdf").touch()
    assert obs_sequence.find_dart_program(tmp_path, "obs_seq_to_netcdf") == (
        work / "obs_seq_to_netcdf"
    )
