"""Map projection and extent of the model domain (`wrf_ensembly.domain`)"""

from importlib import resources
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from wrf_ensembly import config, domain  # noqa: E402
from wrf_ensembly.experiment.paths import ExperimentPaths  # noqa: E402
from wrf_ensembly.observations import plotting  # noqa: E402

from test_icon_processor import write_grid_file  # noqa: E402
from test_icon_regrid import global_grid  # noqa: E402


def read_template(name: str) -> config.Config:
    text = resources.files("wrf_ensembly.config_templates").joinpath(name).read_text()
    return config.Config.from_toml(text)


def test_projection():
    assert isinstance(domain.cartopy_crs(read_template("icon.toml")), ccrs.PlateCarree)
    assert isinstance(
        domain.cartopy_crs(read_template("default.toml")), ccrs.LambertConformal
    )


def test_icon_domain_bounds(tmp_path: Path):
    cfg = read_template("icon.toml")
    assert cfg.icon is not None
    cfg.icon.grid_file = tmp_path / "grid.nc"
    paths = ExperimentPaths(tmp_path, cfg)
    assert domain.domain_bounds(cfg, paths) is None  # no grid file yet

    write_grid_file(cfg.icon.grid_file)  # 8 x 8 degrees around 20 E, 35 N
    np.testing.assert_allclose(
        domain.domain_bounds(cfg, paths), (16.0, 24.0, 31.0, 39.0), atol=1e-6
    )


def test_icon_bounds_across_the_dateline():
    grid = global_grid(2)
    # Keep the cells around 180 E only
    near = np.abs(np.abs(np.degrees(grid.clon)) - 180) < 25
    near &= np.abs(np.degrees(grid.clat)) < 30
    for name in ("clon", "clat", "vlon", "vlat"):
        setattr(grid, name, getattr(grid, name)[near])
    west, east, south, north = domain.icon_grid_bounds(grid)
    assert east - west < 120
    assert west < 180 < east or west < -180 < east


def test_plot_obs_vs_triangular_grid(tmp_path: Path):
    grid = write_grid_file(tmp_path / "grid.nc")
    obs = pd.DataFrame(
        {
            "longitude": [20.1, 20.3, 50.0],
            "latitude": [35.1, 34.9, 35.0],
            "instrument": ["a", "a", "b"],
            "quantity": ["q", "q", "q"],
        }
    )
    fig = plotting.plot_obs_vs_triangular_grid(
        cell_lon=np.degrees(grid.clon),
        cell_lat=np.degrees(grid.clat),
        vertex_lon=np.degrees(grid.vlon),
        vertex_lat=np.degrees(grid.vlat),
        observations=obs,
        center=(35.0, 20.0),
        window_size=5,
    )
    ax = fig.axes[0]
    west, east, south, north = ax.get_extent(ccrs.PlateCarree())
    assert 17 < west < 20 < east < 23 and 32 < south < 35 < north < 38
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    # The observation at 50 E is outside of the view
    assert labels == ["Cell centres", "a - q"]
