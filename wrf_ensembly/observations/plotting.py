from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from wrf_ensembly.observations.utils import reconstruct_array

from .definitions import (
    INSTRUMENT_REGISTRY,
    QUANTITY_REGISTRY,
    Geometry,
    InstrumentSpec,
    QuantitySpec,
)


def _plot_geom_profile_curtain(
    ds: xr.Dataset,
    inst_spec: InstrumentSpec,
    qty_spec: QuantitySpec,
    plot_metadata: dict[str, np.ndarray],
    ax: Axes | None = None,
    **kwargs,
) -> Figure:
    fig, ax = (ax.figure, ax) if ax else plt.subplots(figsize=(10, 4))

    if inst_spec.y is None:
        raise ValueError("Trying to plot a 2D Curtain but AxisSpec for y is missing!")

    pcolormesh_kwargs = dict(
        cmap=qty_spec.cmap,
        vmin=qty_spec.vmin,
        vmax=qty_spec.vmax,
        robust=True,
        cbar_kwargs=dict(label=qty_spec.label),
    )
    pcolormesh_kwargs.update(kwargs)

    ds["value"].plot.pcolormesh(
        x=inst_spec.x.coord,
        y=inst_spec.y.coord,
        ax=ax,
        **pcolormesh_kwargs,
    )
    ax.set_xlabel(inst_spec.x.label)
    ax.set_ylabel(inst_spec.y.label)
    ax.set_title(inst_spec.label, fontsize=11)
    return fig


def _plot_geom_aeolus_windresults(
    ds: xr.Dataset,
    inst_spec: InstrumentSpec,
    qty_spec: QuantitySpec,
    plot_metadata: dict[str, np.ndarray],
    ax: Axes | None = None,
    **kwargs,
) -> Figure:
    """Curtain plot for Aeolus L2B wind results using PolyCollection quads.

    Each wind result is drawn as a rectangle whose x-extent is
    [time_start, time_stop] and y-extent is [alt_bottom, alt_top], coloured
    by the HLOS wind velocity.  This mirrors the approach used in the official
    Aeolus VirES notebooks (PolyCollection over individual quads) and handles
    the irregular horizontal spacing that makes pcolormesh unsuitable here.
    """
    fig, ax = (ax.figure, ax) if ax else plt.subplots(figsize=(12, 5))

    values = ds["value"].to_numpy()

    # Convert timestamps to matplotlib float dates for PolyCollection
    t0 = mdates.date2num(plot_metadata["time_start"])
    t1 = mdates.date2num(plot_metadata["time_stop"])
    y0 = plot_metadata["alt_bottom"].astype(float)
    y1 = plot_metadata["alt_top"].astype(float)

    # Build quad vertices: bottom-left → top-left → top-right → bottom-right
    quads = np.stack(
        [
            np.column_stack([t0, y0]),
            np.column_stack([t0, y1]),
            np.column_stack([t1, y1]),
            np.column_stack([t1, y0]),
        ],
        axis=1,
    )  # shape: (N, 4, 2)

    vmin = kwargs.pop("vmin", qty_spec.vmin)
    vmax = kwargs.pop("vmax", qty_spec.vmax)

    # Symmetric colorbar range for wind velocity when not explicitly set
    if vmin is None and vmax is None:
        abs_max = np.nanpercentile(np.abs(values), 98)
        vmin, vmax = -abs_max, abs_max

    coll = PolyCollection(
        quads,
        array=values,
        cmap=kwargs.pop("cmap", qty_spec.cmap),
        clim=(vmin, vmax),
        **kwargs,
    )
    ax.add_collection(coll)
    fig.colorbar(coll, ax=ax, label=f"{qty_spec.label} [{qty_spec.units}]", pad=0.01)

    ax.autoscale_view()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    ax.set_xlabel(inst_spec.x.label)
    ax.set_ylabel(inst_spec.y.label)
    ax.set_title(inst_spec.label, fontsize=11)

    return fig


GEOMETRY_PLOTTERS = {
    Geometry.PROFILE_CURTAIN: _plot_geom_profile_curtain,
    Geometry.AEOLUS_WINDRESULTS: _plot_geom_aeolus_windresults,
}


def plot_observations(df: pd.DataFrame, plot_kwargs: dict[str, Any] = {}) -> Figure:
    """
    Plots the given observations using the appropriate geometry plotter
    """

    instrument_quantity = df["instrument"] + "." + df["quantity"]
    if len(instrument_quantity.unique()) != 1:
        raise ValueError(
            "Only one value can be plotted at a time (one pair of instrument and quantity)"
        )
    if len(df["orig_filename"].unique()) != 1:
        raise ValueError("Only one source file can be plotted at a time")

    inst_spec = INSTRUMENT_REGISTRY[df.iloc[0]["instrument"]]
    qty_spec = QUANTITY_REGISTRY[df.iloc[0]["quantity"]]

    # Extract metadata fields declared by the instrument spec into flat arrays.
    # Each key becomes a numpy array aligned with the rows of df.
    plot_metadata: dict[str, np.ndarray] = {}
    if inst_spec.plot_metadata_keys:
        meta_series = df["metadata"].apply(pd.Series)
        for key in inst_spec.plot_metadata_keys:
            if key not in meta_series.columns:
                raise KeyError(
                    f"Instrument {inst_spec.label!r} requires metadata key {key!r} "
                    f"for plotting, but it is missing from the observation metadata."
                )
            plot_metadata[key] = meta_series[key].to_numpy()

    ds = reconstruct_array(df)
    return GEOMETRY_PLOTTERS[inst_spec.geometry](
        ds, inst_spec, qty_spec, plot_metadata, **plot_kwargs
    )


def plot_observation_locations_on_map(
    observations: pd.DataFrame,
    proj: ccrs.Projection | None,
    domain_bounds: tuple[float, float, float, float] | None = None,
    fig_kwargs: dict = {},
    subplot_kwargs: dict = {},
    ax_kwargs: dict = {},
) -> Figure:
    """
    Plot the locations of a set of observations on a map

    It will use the columns `latitude`, `longitude` from the DataFrame. If there is
    also columns for `instrument` and `quantity`, it will use them for a legend and to
    color the points.

    Args:
        observations: DataFrame with columns `latitude`, `longitude`, optionally
            `instrument` and `quantity`.
        proj: Cartopy projection to use for the map. If None, will use PlateCarree.
        domain_bounds: If given, will set the map extent to these bounds (min_x, max_x,
            min_y, max_y). If None, the extent will be set automatically to contain the
            observations. It must be in `proj` if provided, otherwise in PlateCarree.
        fig_kwargs: Additional keyword arguments to pass to `plt.figure()`
        subplot_kwargs: Additional keyword arguments to pass to `plt.subplot()`
        ax_kwargs: Additional keyword arguments to pass to `ax.scatter()`

    Returns:
        A matplotlib Figure object with the plot.
    """

    if "latitude" not in observations.columns:
        raise ValueError("DataFrame must have a 'latitude' column")
    if "longitude" not in observations.columns:
        raise ValueError("DataFrame must have a 'longitude' column")

    fig, ax = plt.subplots(
        subplot_kw={"projection": proj or ccrs.PlateCarree(), **subplot_kwargs},
        figsize=(10, 8),
        **fig_kwargs,
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

    if "instrument" in observations.columns and "quantity" in observations.columns:
        for (instrument, quantity), group in observations.groupby(
            ["instrument", "quantity"]
        ):
            label = f"{instrument} - {quantity}"
            ax.scatter(
                group["longitude"],
                group["latitude"],
                label=label,
                transform=ccrs.PlateCarree(),
                s=0.1,
                **ax_kwargs,
            )

        ax.legend(
            loc="lower left",
            title="Instrument - Quantity",
            fontsize=7,
            title_fontsize=8,
            markerscale=4,
            frameon=True,
            framealpha=0.8,
        )

    else:
        ax.scatter(
            observations["longitude"],
            observations["latitude"],
            transform=ccrs.PlateCarree(),
            s=0.1,
            **ax_kwargs,
        )

    if domain_bounds is not None:
        ax.set_extent(domain_bounds, crs=ccrs.PlateCarree())

    return fig


def plot_obs_vs_grid(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    observations: pd.DataFrame,
    proj: ccrs.Projection,
    center_idx: tuple[int, int] | None = None,
    window_size: int = 15,
) -> Figure:
    """
    Plot WRF model grid points alongside observation locations, zoomed in to show
    the relative density of observations vs. grid spacing. Useful for determining
    optimal superobbing bin sizes.

    Args:
        grid_lat: 2D array of grid latitudes (south_north, west_east), e.g. from
            wrfinput XLAT.
        grid_lon: 2D array of grid longitudes (south_north, west_east), e.g. from
            wrfinput XLONG.
        observations: DataFrame with columns `latitude`, `longitude`, and optionally
            `instrument` and `quantity`.
        proj: Cartopy projection for the WRF domain.
        center_idx: (i, j) index into the grid arrays to center the view on. If None,
            uses the center of the grid.
        window_size: Number of grid points to show in each direction from center
            (default 15, giving ~30x30 visible points).

    Returns:
        A matplotlib Figure object with the plot.
    """

    ny, nx = grid_lat.shape

    if center_idx is None:
        ci, cj = ny // 2, nx // 2
    else:
        ci, cj = center_idx

    # Compute index bounds for the window
    i_lo = max(0, ci - window_size)
    i_hi = min(ny, ci + window_size + 1)
    j_lo = max(0, cj - window_size)
    j_hi = min(nx, cj + window_size + 1)

    sub_lat = grid_lat[i_lo:i_hi, j_lo:j_hi]
    sub_lon = grid_lon[i_lo:i_hi, j_lo:j_hi]

    # Determine map extent from the subsetted grid (with a small padding)
    lat_min, lat_max = float(sub_lat.min()), float(sub_lat.max())
    lon_min, lon_max = float(sub_lon.min()), float(sub_lon.max())
    lat_pad = (lat_max - lat_min) * 0.05
    lon_pad = (lon_max - lon_min) * 0.05

    fig, ax = plt.subplots(
        subplot_kw={"projection": proj},
        figsize=(10, 10),
    )

    # Draw grid lines connecting grid points to show structure
    for i in range(sub_lat.shape[0]):
        ax.plot(
            sub_lon[i, :],
            sub_lat[i, :],
            color="grey",
            linewidth=0.3,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
    for j in range(sub_lat.shape[1]):
        ax.plot(
            sub_lon[:, j],
            sub_lat[:, j],
            color="grey",
            linewidth=0.3,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )

    # Plot grid points
    ax.scatter(
        sub_lon.ravel(),
        sub_lat.ravel(),
        color="grey",
        s=8,
        marker="s",
        label="Grid points",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # Filter observations to the visible area
    obs = observations[
        (observations["latitude"] >= lat_min - lat_pad)
        & (observations["latitude"] <= lat_max + lat_pad)
        & (observations["longitude"] >= lon_min - lon_pad)
        & (observations["longitude"] <= lon_max + lon_pad)
    ]

    # Plot observations, color-coded by instrument/quantity if available
    if "instrument" in obs.columns and "quantity" in obs.columns and not obs.empty:
        for (instrument, quantity), group in obs.groupby(["instrument", "quantity"]):
            ax.scatter(
                group["longitude"],
                group["latitude"],
                label=f"{instrument} - {quantity}",
                transform=ccrs.PlateCarree(),
                s=20,
                zorder=3,
            )
    elif not obs.empty:
        ax.scatter(
            obs["longitude"],
            obs["latitude"],
            label="Observations",
            transform=ccrs.PlateCarree(),
            s=20,
            zorder=3,
        )

    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines()
    ax.legend(
        loc="upper right",
        fontsize="small",
        frameon=True,
    )

    return fig
