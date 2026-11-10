"""
The model domain as the plots need it: a map projection and the extent of the grid.

Postprocessed output and the observations in the database carry `x` and `y`
coordinates in this projection.
"""

import cartopy.crs as ccrs
import numpy as np

from wrf_ensembly import config, wrf
from wrf_ensembly.console import logger
from wrf_ensembly.experiment.paths import ExperimentPaths


def cartopy_crs(cfg: config.Config) -> ccrs.Projection:
    """
    The projection in which the model output has its `x` and `y` coordinates.

    Raises NotImplementedError for WRF projections we have no cartopy equivalent for.
    """

    if cfg.is_icon:
        # ICON output is interpolated to a regular lat/lon grid
        return ccrs.PlateCarree()

    assert cfg.domain_control is not None
    return wrf.get_wrf_cartopy_crs(cfg.domain_control)


def cartopy_crs_or_none(cfg: config.Config) -> ccrs.Projection | None:
    """Like `cartopy_crs`, but None (plots then use PlateCarree) if there is none"""

    try:
        return cartopy_crs(cfg)
    except NotImplementedError:
        logger.warning(
            "Could not create cartopy projection for this domain, falling back to PlateCarree"
        )
        return None


def domain_bounds(
    cfg: config.Config, paths: ExperimentPaths
) -> tuple[float, float, float, float] | None:
    """
    Extent of the domain as (x_min, x_max, y_min, y_max) in the projection of
    `cartopy_crs`, or None if the files to read it from do not exist yet.
    """

    if cfg.icon is not None:
        # Imported here: the regridding code is only needed for ICON
        from wrf_ensembly.icon.regrid import IconGrid

        if not cfg.icon.grid_file.exists():
            logger.warning(
                f"No grid file at {cfg.icon.grid_file}, cannot set map bounds"
            )
            return None
        return icon_grid_bounds(IconGrid.from_file(cfg.icon.grid_file))

    assert cfg.data is not None
    if not cfg.data.per_member_meteorology:
        wrfinput_path = paths.data_icbc / "wrfinput_d01_cycle_0"
    else:
        wrfinput_path = paths.data_icbc / "member_00" / "wrfinput_d01_member_00_cycle_0"
    if not wrfinput_path.exists():
        logger.warning(f"No wrfinput file at {wrfinput_path}, cannot set map bounds")
        return None
    return wrf.get_spatial_domain_bounds(wrfinput_path)


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
