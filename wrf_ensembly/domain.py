"""
The model domain as the plots need it: a map projection and the extent of the grid.

Postprocessed output and the observations in the database carry `x` and `y`
coordinates in this projection.
"""

import cartopy.crs as ccrs

from wrf_ensembly import config, wrf
from wrf_ensembly.console import logger
from wrf_ensembly.experiment.paths import ExperimentPaths


def cartopy_crs(cfg: config.Config) -> ccrs.Projection:
    """
    The projection in which the model output has its `x` and `y` coordinates.

    Raises NotImplementedError for WRF projections we have no cartopy equivalent for.
    """

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

    assert cfg.data is not None
    if not cfg.data.per_member_meteorology:
        wrfinput_path = paths.data_icbc / "wrfinput_d01_cycle_0"
    else:
        wrfinput_path = paths.data_icbc / "member_00" / "wrfinput_d01_member_00_cycle_0"
    if not wrfinput_path.exists():
        logger.warning(f"No wrfinput file at {wrfinput_path}, cannot set map bounds")
        return None
    return wrf.get_spatial_domain_bounds(wrfinput_path)
