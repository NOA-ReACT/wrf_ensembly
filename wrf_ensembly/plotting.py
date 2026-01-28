import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from wrf_ensembly.config import PlotVariableConfig


def plot_forecast_vs_analysis(
    forecast_ds: xr.Dataset,
    analysis_ds: xr.Dataset,
    variable_cfg: PlotVariableConfig,
    arrangement: str = "horizontal",
    proj: ccrs.Projection | None = None,
    experiment_name: str = "",
    cycle: int = 0,
    time_str: str = "",
) -> Figure:
    """
    Create a three-panel comparison plot: forecast, analysis, and analysis - forecast.

    Args:
        forecast_ds: xarray Dataset with the forecast data.
        analysis_ds: xarray Dataset with the analysis data.
        variable_cfg: Configuration for the variable to plot.
        arrangement: Panel layout, either "horizontal" (1x3) or "vertical" (3x1).
        proj: Cartopy projection for the map panels. If None, uses PlateCarree.
        experiment_name: Name of the experiment for the figure title.
        cycle: Cycle index for the figure title.
        time_str: Time string for the figure title.

    Returns:
        A matplotlib Figure with three panels.
    """

    var_name = variable_cfg.name
    forecast_var = forecast_ds[var_name]
    analysis_var = analysis_ds[var_name]

    spatial_dims = {"south_north", "west_east", "x", "y"}
    time_dims = {"Time", "time", "t"}

    # Select vertical level if specified
    if variable_cfg.level is not None:
        level_dim = None
        for dim in forecast_var.dims:
            if dim not in spatial_dims and dim not in time_dims:
                level_dim = dim
                break
        if level_dim is not None:
            forecast_var = forecast_var.isel({level_dim: variable_cfg.level})
            analysis_var = analysis_var.isel({level_dim: variable_cfg.level})

    # Squeeze out time dimension if present (take first timestep)
    for dim in time_dims:
        if dim in forecast_var.dims:
            forecast_var = forecast_var.isel({dim: 0})
        if dim in analysis_var.dims:
            analysis_var = analysis_var.isel({dim: 0})

    # Use projected x/y coordinates for pcolormesh (postprocessed files store these
    # as 1D coordinates in the native map projection)
    has_xy = "x" in forecast_ds.coords and "y" in forecast_ds.coords

    diff_var = analysis_var - forecast_var

    map_proj = proj or ccrs.PlateCarree()

    if arrangement == "vertical":
        nrows, ncols = 3, 1
        figsize = (8, 18)
    else:
        nrows, ncols = 1, 3
        figsize = (22, 6)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": map_proj},
        figsize=figsize,
    )

    panels = [
        (
            "Forecast",
            forecast_var,
            variable_cfg.cmap,
            variable_cfg.vmin,
            variable_cfg.vmax,
        ),
        (
            "Analysis",
            analysis_var,
            variable_cfg.cmap,
            variable_cfg.vmin,
            variable_cfg.vmax,
        ),
        (
            "Analysis - Forecast",
            diff_var,
            variable_cfg.diff_cmap,
            variable_cfg.diff_vmin,
            variable_cfg.diff_vmax,
        ),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        if has_xy:
            mesh = ax.pcolormesh(
                forecast_ds["x"].values,
                forecast_ds["y"].values,
                data.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
        else:
            mesh = ax.pcolormesh(
                data.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        ax.coastlines()

        level_str = (
            f" (level {variable_cfg.level})" if variable_cfg.level is not None else ""
        )
        ax.set_title(f"{title}: {var_name}{level_str}")

        fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    suptitle_parts = []
    if experiment_name:
        suptitle_parts.append(experiment_name)
    suptitle_parts.append(f"Cycle {cycle}")
    if time_str:
        suptitle_parts.append(time_str)
    fig.suptitle(" | ".join(suptitle_parts), fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig
