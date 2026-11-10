"""Tests for the streaming NetCDF writers used by `postprocess run`."""

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly.postprocess.streaming_writer import (
    StreamingEnsembleWriter,
    StreamingNetCDFWriter,
)
from wrf_ensembly.statistics import add_member_dimension, get_structure_from_xarray

TIMES = [b"2020-01-01_00:00:00", b"2020-01-01_01:00:00"]


def make_timestep(i: int) -> xr.Dataset:
    """One wrfout-like timestep with a numeric and a char (`Times`) variable."""
    t = np.array([np.datetime64("2020-01-01T00:00") + np.timedelta64(i, "h")])
    return xr.Dataset(
        {
            "T2": (("t", "y", "x"), np.full((1, 2, 3), float(i), dtype="f4")),
            "Times": (("t",), np.array([TIMES[i]])),
        },
        coords={"t": t},
    )


def test_ensemble_writer_handles_char_variables(tmp_path):
    n_members = 3
    first = make_timestep(0)
    template = add_member_dimension(get_structure_from_xarray(first), n_members)

    path = tmp_path / "ensemble.nc"
    with StreamingEnsembleWriter(path, template, n_members) as writer:
        for i in range(2):
            ds = make_timestep(i)
            time_val = ds["t"].values
            ds = ds.squeeze("t", drop=False)
            for m in range(n_members):
                data = {v: ds[v].values for v in ds.data_vars}
                writer.write_member(data, time_val, m)
            writer.finalize_timestep()

    with netCDF4.Dataset(path) as nc:
        assert nc["Times"].dimensions == ("t", "member")
        assert nc["Times"][:].tolist() == [[t.decode()] * n_members for t in TIMES]
        assert nc["T2"].shape == (2, n_members, 2, 3)
        np.testing.assert_array_equal(nc["T2"][1], 1.0)


def test_netcdf_writer_handles_char_variables(tmp_path):
    first = make_timestep(0)
    template = get_structure_from_xarray(first)

    path = tmp_path / "mean.nc"
    with StreamingNetCDFWriter(path, template) as writer:
        for i in range(2):
            ds = make_timestep(i)
            time_val = ds["t"].values
            ds = ds.squeeze("t", drop=False)
            writer.append_timestep({v: ds[v].values for v in ds.data_vars}, time_val)

    with netCDF4.Dataset(path) as nc:
        assert nc["Times"][:].tolist() == [t.decode() for t in TIMES]
        assert nc["t"][:].tolist() == [0, 60]


def test_missing_values_survive_quantization(tmp_path):
    """
    NaN in a variable with a fill value is stored as the fill value. Written as NaN,
    the lossy compression turns it into zero.
    """

    def timestep(i: int) -> xr.Dataset:
        ds = make_timestep(i).drop_vars("Times")
        ds["T2"].values[0, 0, 0] = np.nan
        ds["T2"].attrs["_FillValue"] = np.float32(netCDF4.default_fillvals["f4"])
        ds["static"] = (("y", "x"), np.full((2, 3), np.nan, dtype="f4"))
        ds["static"].attrs["_FillValue"] = np.float32(netCDF4.default_fillvals["f4"])
        return ds

    path = tmp_path / "mean.nc"
    with StreamingNetCDFWriter(
        path, get_structure_from_xarray(timestep(0)), significant_digits=3
    ) as writer:
        for i in range(2):
            ds = timestep(i)
            time_val = ds["t"].values
            ds = ds.squeeze("t", drop=False)
            writer.append_timestep({v: ds[v].values for v in ds.data_vars}, time_val)

    with xr.open_dataset(path) as ds:
        assert np.isnan(ds["T2"].values[:, 0, 0]).all()
        np.testing.assert_array_equal(ds["T2"].values[1, 1, :], 1.0)
        assert np.isnan(ds["static"].values).all()
