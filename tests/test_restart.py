import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import restart, update_bc

TIME = dt.datetime(2026, 3, 1, 18, tzinfo=dt.timezone.utc)
SHAPE = (1, 3, 4)


def make_file(path: Path, fields: dict[str, float], time: dt.datetime = TIME) -> None:
    """A file with a Times variable and constant (Time, y, x) fields"""

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("y", SHAPE[1])
        ds.createDimension("x", SHAPE[2])
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        update_bc.write_wrf_time(times, 0, time)
        for name, value in fields.items():
            ds.createVariable(name, "f4", ("Time", "y", "x"))[:] = np.full(SHAPE, value)


def test_field_names(tmp_path: Path):
    make_file(tmp_path / "wrfrst", {"U_1": 0, "U_2": 0, "QVAPOR": 0})

    with netCDF4.Dataset(tmp_path / "wrfrst") as ds:
        assert restart.field_names(ds, "U") == ["U_1", "U_2"]
        assert restart.field_names(ds, "QVAPOR") == ["QVAPOR"]
        assert restart.field_names(ds, "DUST_1") == []


def test_write_state_fills_both_time_levels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    make_file(tmp_path / "wrfrst", {"U_1": 0, "U_2": 1, "QVAPOR": 2, "DUST_1": 3})
    make_file(
        tmp_path / "analysis", {"U": 10, "QVAPOR": 20, "DUST_1": 30, "SEAS_1": 40}
    )

    with (
        netCDF4.Dataset(tmp_path / "wrfrst", "r+") as rst,
        netCDF4.Dataset(tmp_path / "analysis") as analysis,
    ):
        written = restart.write_state(rst, analysis, ["U", "QVAPOR", "SEAS_1", "PSFC"])

    assert written == ["U", "QVAPOR"]
    assert "SEAS_1 not in the restart file" in caplog.text
    assert "PSFC not in the source" in caplog.text
    with netCDF4.Dataset(tmp_path / "wrfrst") as rst:
        # The change (10 - 1) is added to the older level, which stays one unit behind
        np.testing.assert_array_equal(rst["U_1"][:], 9)
        np.testing.assert_array_equal(rst["U_2"][:], 10)
        np.testing.assert_array_equal(rst["QVAPOR"][:], 20)
        # Not asked for, left alone
        np.testing.assert_array_equal(rst["DUST_1"][:], 3)


def test_write_state_needs_matching_times(tmp_path: Path):
    make_file(tmp_path / "wrfrst", {"U_1": 0, "U_2": 0})
    make_file(tmp_path / "analysis", {"U": 1}, time=TIME + dt.timedelta(hours=6))

    with (
        netCDF4.Dataset(tmp_path / "wrfrst", "r+") as rst,
        netCDF4.Dataset(tmp_path / "analysis") as analysis,
    ):
        with pytest.raises(ValueError, match="Restart file is at"):
            restart.write_state(rst, analysis, ["U"])
