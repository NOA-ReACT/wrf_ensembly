"""The ICON step of the postprocessing pipeline (`wrf_ensembly.icon.processor`)"""

from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly import config
from wrf_ensembly.icon.processor import IconPostProcessor
from wrf_ensembly.processors import ProcessingContext, create_pipeline_from_config

from test_icon_model import template_text
from test_icon_regrid import limited_area_grid

NLEV = 4


def write_grid_file(path: Path):
    grid = limited_area_grid(20)
    with netCDF4.Dataset(path, "w") as ds:
        ds.uuidOfHGrid = "test-grid"
        ds.createDimension("cell", grid.ncells)
        ds.createDimension("nv", 3)
        ds.createVariable("clon", "f8", ("cell",))[:] = grid.clon
        ds.createVariable("clat", "f8", ("cell",))[:] = grid.clat
        ds.createVariable("clon_vertices", "f8", ("cell", "nv"))[:] = grid.vlon
        ds.createVariable("clat_vertices", "f8", ("cell", "nv"))[:] = grid.vlat
    return grid


def write_forecast(directory: Path, ncells: int) -> Path:
    """A forecast file and a first guess file the way ICON writes them"""

    directory.mkdir(parents=True)
    level = np.arange(NLEV, dtype=np.float32)[None, :, None]
    half_level = np.arange(NLEV + 1, dtype=np.float32)[None, :, None]
    ones = np.ones((1, 1, ncells), dtype=np.float32)
    ds = xr.Dataset(
        {
            # Level 0 is the model top
            "temp": (("time", "height", "ncells"), 200 + 10 * level * ones),
            "u": (("time", "height", "ncells"), 5 * ones * np.ones_like(level)),
            "w": (("time", "height_2", "ncells"), half_level * ones),
            "t_2m": (("time", "height_3", "ncells"), 290 * ones),
            "tot_prec": (("time", "ncells"), 2 * ones[0]),
            "height_bnds": (("height", "bnds"), np.zeros((NLEV, 2))),
        },
        coords={
            "time": [np.datetime64("2026-09-13T01:00:00")],
            "height": np.arange(1, NLEV + 1),
            "height_2": np.arange(1, NLEV + 2),
            "height_3": [2.0],
        },
        attrs={"title": "ICON simulation"},
    )
    path = directory / "icon_20260913T010000Z.nc"
    ds.to_netcdf(path)

    with netCDF4.Dataset(directory / "fg_20260913T060000Z.nc", "w") as fg:
        fg.createDimension("height_2", NLEV + 1)
        fg.createDimension("ncells_2", ncells)
        z_ifc = np.linspace(8000, 0, NLEV + 1)[:, None] * np.ones(ncells)
        fg.createVariable("z_ifc", "f4", ("height_2", "ncells_2"))[:] = z_ifc
    return path


def make_config(grid_file: Path, extra: str = "") -> config.Config:
    text = template_text().replace(
        'grid_file   = "/path/to/grid_DOM01.nc"', f'grid_file = "{grid_file}"'
    )
    return config.Config.from_toml(text + extra)


def run(tmp_path: Path, cfg: config.Config, forecast: Path) -> xr.Dataset:
    pipeline = create_pipeline_from_config(
        cfg.postprocess.processors, IconPostProcessor()
    )
    assert [p.name for p in pipeline.processors] == ["ICON post-processor"]
    context = ProcessingContext(
        member=0,
        cycle=0,
        input_file=forecast,
        output_file=Path("/dev/null"),
        config=cfg,
        cache_dir=tmp_path / "weights",
    )
    with xr.open_dataset(forecast) as ds:
        return pipeline.process(ds, context).load()


def test_layout_of_the_output(tmp_path: Path):
    grid = write_grid_file(tmp_path / "grid.nc")
    forecast = write_forecast(tmp_path / "member_00", grid.ncells)
    out = run(tmp_path, make_config(tmp_path / "grid.nc"), forecast)

    assert out["air_temperature"].dims == ("t", "z", "y", "x")
    assert out["t_2m"].dims == ("t", "y", "x")
    assert out.sizes["z"] == NLEV
    assert out["t"].values[0] == np.datetime64("2026-09-13T01:00:00")
    np.testing.assert_array_equal(out["longitude"].values, out["x"].values)
    assert 15.5 < out["x"].min() < 16.5 and 30.5 < out["y"].min() < 31.5
    assert "height_bnds" not in out and "temp" not in out and "u" not in out
    assert out.attrs["title"] == "ICON simulation"

    inside = np.isfinite(out["t_2m"].values[0])
    assert inside.mean() > 0.9
    # Outside of the domain is missing, and marked as such for the writers
    assert not inside.all()
    assert out["t_2m"].attrs["_FillValue"] == netCDF4.default_fillvals["f4"]
    assert out["t_2m"].attrs["_FillValue"].dtype == np.float32
    # The lowest level comes first now
    column = out["air_temperature"].values[0][:, inside][:, 0]
    np.testing.assert_allclose(column, [230, 220, 210, 200])
    np.testing.assert_allclose(out["wind_east"].values[0][:, inside], 5.0)
    # w was on half levels and is now on full levels
    np.testing.assert_allclose(
        out["w"].values[0][:, inside][:, 0], [3.5, 2.5, 1.5, 0.5]
    )
    # Height of the levels, from the first guess file
    heights = out["geopotential_height"].values[:, inside][:, 0]
    np.testing.assert_allclose(heights, [1000, 3000, 5000, 7000])
    np.testing.assert_allclose(out["tot_prec"].values[0][inside], 2.0, rtol=1e-6)

    # One weights file per method: precipitation is conservative in the template
    cached = sorted(f.name for f in (tmp_path / "weights").iterdir())
    assert len(cached) == 2
    assert "conservative" in cached[0] and "linear" in cached[1]


def test_target_grid_and_variable_filter(tmp_path: Path):
    grid = write_grid_file(tmp_path / "grid.nc")
    forecast = write_forecast(tmp_path / "member_00", grid.ncells)
    cfg = make_config(tmp_path / "grid.nc")
    assert cfg.icon is not None
    cfg.icon.regrid.bounds = (18.0, 22.0, 33.0, 37.0)
    cfg.icon.regrid.resolution = 0.5
    cfg.icon.regrid.method = "nearest"
    cfg.postprocess.variables_to_keep = ["air_temperature", "t_2m"]

    out = run(tmp_path, cfg, forecast)
    assert set(out.data_vars) == {"air_temperature", "t_2m"}
    assert (out.sizes["y"], out.sizes["x"]) == (8, 8)
    np.testing.assert_allclose(out["x"].values, np.arange(18.25, 22, 0.5))
    assert np.isfinite(out["t_2m"].values).all()
    assert out.attrs["regrid_method"] == "nearest"
