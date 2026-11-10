"""
The model domain as the observations and the plots need it: a map projection, the
extent of the grid and which locations are inside it.

Postprocessed output and the observations in the database carry `x` and `y`
coordinates in the domain's projection. `create` gives the domain of an experiment,
also available as `Experiment.domain`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from wrf_ensembly import config, wrf
from wrf_ensembly.console import logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from wrf_ensembly.experiment.paths import ExperimentPaths


class Domain(ABC):
    def __init__(self, cfg: config.Config, paths: "ExperimentPaths"):
        self.cfg = cfg
        self.paths = paths

    @abstractmethod
    def cartopy_crs(self) -> ccrs.Projection:
        """
        The projection in which the model output has its `x` and `y` coordinates.

        Raises NotImplementedError if there is no cartopy equivalent for it.
        """

    def cartopy_crs_or_none(self) -> ccrs.Projection | None:
        """Like `cartopy_crs`, but None (plots then use PlateCarree) if there is none"""

        try:
            return self.cartopy_crs()
        except NotImplementedError:
            logger.warning(
                "Could not create cartopy projection for this domain, falling back to PlateCarree"
            )
            return None

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float] | None:
        """
        Extent of the domain as (x_min, x_max, y_min, y_max) in the projection of
        `cartopy_crs`, or None if the files to read it from do not exist yet.
        """

    @abstractmethod
    def locate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A copy of a dataframe of observations (`longitude`, `latitude`) with their
        `x` and `y` in the domain's projection and whether they are `in_domain`.

        Raises FileNotFoundError if the files that describe the grid do not exist yet.
        """

    @abstractmethod
    def plot_observations_on_grid(
        self,
        observations: pd.DataFrame,
        center: tuple[float, float] | None,
        window_size: int,
    ) -> "Figure":
        """
        Observation locations on top of the model grid, zoomed in to `window_size`
        grid points around `center` (latitude, longitude; default: the middle).
        """


class WrfDomain(Domain):
    def cartopy_crs(self) -> ccrs.Projection:
        assert self.cfg.domain_control is not None
        return wrf.get_wrf_cartopy_crs(self.cfg.domain_control)

    def _wrfinput(self) -> Path:
        """A wrfinput of the first cycle, for the extent of the grid"""

        assert self.cfg.data is not None
        if not self.cfg.data.per_member_meteorology:
            return self.paths.data_icbc / "wrfinput_d01_cycle_0"
        return self.paths.data_icbc / "member_00" / "wrfinput_d01_member_00_cycle_0"

    def bounds(self) -> tuple[float, float, float, float] | None:
        wrfinput_path = self._wrfinput()
        if not wrfinput_path.exists():
            logger.warning(f"No wrfinput file at {wrfinput_path}, cannot set map bounds")
            return None
        return wrf.get_spatial_domain_bounds(wrfinput_path)

    def locate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Imported here, the observations package is large
        from wrf_ensembly.observations.utils import project_locations_to_wrf

        wrfinput_path = self._wrfinput()
        if not wrfinput_path.exists():
            raise FileNotFoundError(
                f"wrfinput file not found at {wrfinput_path}, cannot trim observations spatially"
            )

        transformer = wrf.get_wrf_proj_transformer(self.cfg.domain_control)
        x_min, x_max, y_min, y_max = wrf.get_spatial_domain_bounds(wrfinput_path)

        df = project_locations_to_wrf(df, transformer)
        df["in_domain"] = (
            (df["x"] >= x_min)
            & (df["x"] <= x_max)
            & (df["y"] >= y_min)
            & (df["y"] <= y_max)
        )
        return df

    def plot_observations_on_grid(
        self,
        observations: pd.DataFrame,
        center: tuple[float, float] | None,
        window_size: int,
    ) -> "Figure":
        import xarray as xr

        from wrf_ensembly.observations import plotting

        # Load grid coordinates from wrfinput
        wrfinput_path = self.paths.ic_path(0, 0)
        if not wrfinput_path.exists():
            raise FileNotFoundError(f"wrfinput file not found at {wrfinput_path}")

        with xr.open_dataset(wrfinput_path) as ds:
            grid_lat = ds["XLAT"].isel(Time=0).values
            grid_lon = ds["XLONG"].isel(Time=0).values

        # Determine center grid index
        center_idx = None
        if center is not None:
            center_lat, center_lon = center
            # Find nearest grid point to the given lat/lon
            dist = (grid_lat - center_lat) ** 2 + (grid_lon - center_lon) ** 2
            center_idx = np.unravel_index(np.argmin(dist), dist.shape)
            logger.info(
                f"Centering on grid index {center_idx} (nearest to {center_lat}, {center_lon})"
            )

        return plotting.plot_obs_vs_grid(
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            observations=observations,
            proj=self.cartopy_crs(),
            center_idx=center_idx,
            window_size=window_size,
        )


class IconDomain(Domain):
    """
    ICON output is postprocessed to a regular lat/lon grid, so `x` and `y` are simply
    the longitude and latitude. The domain itself is the set of triangles of the grid.
    """

    def cartopy_crs(self) -> ccrs.Projection:
        return ccrs.PlateCarree()

    def _grid(self):
        # Imported here: the regridding code is only needed for ICON
        from wrf_ensembly.icon.regrid import IconGrid

        assert self.cfg.icon is not None
        if not self.cfg.icon.grid_file.exists():
            raise FileNotFoundError(f"No grid file at {self.cfg.icon.grid_file}")
        return IconGrid.from_file(self.cfg.icon.grid_file)

    def bounds(self) -> tuple[float, float, float, float] | None:
        try:
            grid = self._grid()
        except FileNotFoundError as e:
            logger.warning(f"{e}, cannot set map bounds")
            return None
        return icon_grid_bounds(grid)

    def locate(self, df: pd.DataFrame) -> pd.DataFrame:
        from wrf_ensembly.icon import regrid

        grid = self._grid()
        df = df.copy()
        df["x"] = df["longitude"]
        df["y"] = df["latitude"]
        # An observation is in the domain if it is inside one of the grid's triangles
        df["in_domain"] = (
            regrid.locate_cells(
                grid,
                np.radians(df["longitude"].to_numpy(dtype=float)),
                np.radians(df["latitude"].to_numpy(dtype=float)),
            )
            >= 0
        )
        return df

    def plot_observations_on_grid(
        self,
        observations: pd.DataFrame,
        center: tuple[float, float] | None,
        window_size: int,
    ) -> "Figure":
        from wrf_ensembly.observations import plotting

        grid = self._grid()
        # Longitudes near the centre of the grid, so that the cells of a domain across
        # the dateline stay together
        west, east, _, _ = icon_grid_bounds(grid)
        middle = 0.5 * (west + east)

        def wrap(lon):
            return middle + (np.degrees(lon) - middle + 180.0) % 360.0 - 180.0

        cell_lon = wrap(grid.clon)
        return plotting.plot_obs_vs_triangular_grid(
            cell_lon=cell_lon,
            cell_lat=np.degrees(grid.clat),
            vertex_lon=cell_lon[:, None]
            + (np.degrees(grid.vlon - grid.clon[:, None]) + 180.0) % 360.0
            - 180.0,
            vertex_lat=np.degrees(grid.vlat),
            observations=observations,
            center=center,
            window_size=window_size,
        )


def create(cfg: config.Config, paths: "ExperimentPaths") -> Domain:
    """The domain of an experiment, by `metadata.model`"""

    if cfg.metadata.model == "icon":
        return IconDomain(cfg, paths)
    return WrfDomain(cfg, paths)


def icon_grid_bounds(grid) -> tuple[float, float, float, float]:
    """
    Extent of an ICON grid in degrees, (west, east, south, north). The longitudes are
    continuous around the centre of the grid, so a domain across the dateline does not
    come out as the whole globe.
    """

    centre_lon = np.arctan2(np.sin(grid.clon).mean(), np.cos(grid.clon).mean())
    relative = (grid.vlon - centre_lon + np.pi) % (2 * np.pi) - np.pi
    lon = np.degrees(centre_lon + relative)
    lat = np.degrees(grid.vlat)
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
