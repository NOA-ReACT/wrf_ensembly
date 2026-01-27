import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


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
    ax.coastlines()

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

        # Create a nice legend outside the plot
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(1, 0.5),
            title="Instrument - Quantity",
            fontsize="small",
            title_fontsize="medium",
            frameon=False,
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
        ax.set_extent(domain_bounds, crs=proj or ccrs.PlateCarree())

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
